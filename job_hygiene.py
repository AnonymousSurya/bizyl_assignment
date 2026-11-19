#!/usr/bin/env python3
"""
job_hygiene.py — Bizyl Job Board Hygiene Pipeline

1. Soft-404 & HTML Cleanup     — detect expired/closed listings; unescape HTML entities
2. Contradiction Detection     — catch raw_location vs raw_page_snippet mismatches
3. UI Filter Mapping           — map active records to Bizyl enum values
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

# ── File Paths ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
JOBS_FILE = BASE / "dirty_jobs.json"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "cleaned_jobs.json"

# ── Inactive / Soft-404 Signal Patterns ───────────────────────────────────────
INACTIVE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"no longer available|has been filled", re.IGNORECASE),
        "Job no longer available or has been filled",
    ),
    (
        re.compile(r"position has been closed", re.IGNORECASE),
        "Position has been closed",
    ),
    (
        re.compile(r"404\s+Not\s+Found|does not exist", re.IGNORECASE),
        "Soft-404: Requisition ID does not exist",
    ),
    (
        re.compile(r"listing expired|no longer accepting", re.IGNORECASE),
        "Listing has expired",
    ),
]

# ── Contradiction Rules ────────────────────────────────────────────────────────
# (location_pattern, snippet_pattern, contradiction_note)
CONTRADICTION_RULES: list[tuple[re.Pattern, re.Pattern, str]] = [
    (
        re.compile(r"remote\s+anywhere|remote\s*[-–]\s*global|remote\s+worldwide", re.IGNORECASE),
        re.compile(r"must\s+reside|must\s+be\s+located|United\s+States|Canada", re.IGNORECASE),
        "raw_location claims 'Remote Anywhere' but snippet mandates US/Canada residency",
    ),
]

# ── Employment Type Normalisation ─────────────────────────────────────────────
# Maps raw employment_hint strings → Bizyl enum
EMPLOYMENT_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"intern(?:ship)?", re.IGNORECASE), "Internship"),
    (re.compile(r"part.?time", re.IGNORECASE), "Part-time"),
    (re.compile(r"contract|contractor|freelance", re.IGNORECASE), "Contract"),
    (re.compile(r"full.?time|permanent", re.IGNORECASE), "Full-time"),
]


def normalise_employment(hint: str) -> Optional[str]:
    for pat, label in EMPLOYMENT_MAP:
        if pat.search(hint):
            return label
    return None


# ── Remote Scope Mapping ───────────────────────────────────────────────────────

def map_remote_scope(location: str, contradiction: bool) -> str:
    loc = location.lower()
    if contradiction:
        # Location said "Remote Anywhere" but snippet restricts by country
        return "Remote (Restricted)"
    if re.search(r"remote\s*[-–]\s*global|remote\s+anywhere|remote\s+worldwide", loc):
        return "Remote (Global)"
    if re.search(r"remote", loc):
        # Remote but with explicit geographic restriction in location string
        return "Remote (Restricted)"
    if re.search(r"hybrid", loc):
        return "Hybrid"
    return "On-site"


# ── Domain Classifier ─────────────────────────────────────────────────────────
# Score each domain by keyword hits in title + snippet; highest wins.

DOMAIN_SIGNALS: dict[str, re.Pattern] = {
    "Information Technology": re.compile(
        r"engineer|developer|backend|frontend|data\s+engineer|machine\s+learning|"
        r"ML|AI(?!\s*\/)|\bNLP\b|cloud|software|python|django|pyspark|pipeline|"
        r"infrastructure|devops|tech",
        re.IGNORECASE,
    ),
    "Marketing": re.compile(
        r"marketing|paid\s+search|paid\s+social|SEO|SEM|growth|media\s+buyer|"
        r"\bMeta\b|google\s+ads|conversion|acquisition|social\s+media|buyer|"
        r"D2C|custom\s+audiences",
        re.IGNORECASE,
    ),
    "Consulting": re.compile(
        r"consulting|consultant|strategy|advisory|market\s+sizing|TAM|SAM|SOM|"
        r"porter|five\s+forces|issue\s+tree|MECE|due\s+diligence",
        re.IGNORECASE,
    ),
    "Finance": re.compile(
        r"finance|financial|FP&?A|accounting|DCF|WACC|financial\s+model|"
        r"unit\s+economics|CFO|investment|equity|valuation|run.?rate|forecast",
        re.IGNORECASE,
    ),
}


def classify_domain(title: str, snippet: str) -> str:
    combined = f"{title} {snippet}"
    scores = {domain: len(pat.findall(combined)) for domain, pat in DOMAIN_SIGNALS.items()}
    return max(scores, key=scores.get)


# ── Seniority Classifier ──────────────────────────────────────────────────────
# Rules checked in priority order; last rule is the default.

SENIORITY_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:intern(?:ship)?|graduate|entry.?level|junior)\b", re.IGNORECASE), "Intern/Entry"),
    (
        re.compile(
            r"\b(?:lead|head|director|VP|vice\s+president|principal|C-suite|chief)\b",
            re.IGNORECASE,
        ),
        "Lead/Executive",
    ),
    (re.compile(r"\b(?:senior|sr\.?|staff)\b", re.IGNORECASE), "Senior"),
]


def classify_seniority(title: str, snippet: str) -> str:
    combined = f"{title} {snippet}"
    for pattern, level in SENIORITY_RULES:
        if pattern.search(combined):
            return level
    # Experience-year signal as fallback
    exp = re.search(r"(\d+)\s*[-–]\s*(\d+)\s+years?", snippet, re.IGNORECASE)
    if exp:
        max_yrs = int(exp.group(2))
        if max_yrs <= 2:
            return "Intern/Entry"
        if max_yrs <= 6:
            return "Mid-Level"
        return "Senior"
    return "Mid-Level"


# ── Pydantic Output Model ──────────────────────────────────────────────────────


class CleanedJob(BaseModel):
    job_id: str
    is_active: bool
    inactive_reason: Optional[str]
    title_clean: str
    raw_location_clean: str
    contradiction_detected: bool
    contradiction_note: Optional[str]
    employment_type: Optional[str]
    remote_scope: Optional[str]
    mapped_domain: Optional[str]
    seniority: Optional[str]


# ── Core Processing ────────────────────────────────────────────────────────────


def process_job(record: dict) -> CleanedJob:
    raw_title: str = record["title"]
    raw_location: str = record["raw_location"]
    snippet: str = record["raw_page_snippet"]
    hint: str = record["employment_hint"]

    # Step 1 — Unescape HTML entities (&amp; &mdash; &ndash; etc.)
    title_clean = html.unescape(raw_title)
    location_clean = html.unescape(raw_location)

    # Step 2 — Inactive / Soft-404 detection
    inactive_reason: Optional[str] = None
    for pat, reason in INACTIVE_PATTERNS:
        if pat.search(snippet):
            inactive_reason = reason
            break

    if inactive_reason:
        return CleanedJob(
            job_id=record["job_id"],
            is_active=False,
            inactive_reason=inactive_reason,
            title_clean=title_clean,
            raw_location_clean=location_clean,
            contradiction_detected=False,
            contradiction_note=None,
            employment_type=None,
            remote_scope=None,
            mapped_domain=None,
            seniority=None,
        )

    # Step 3 — Contradiction detection (location vs snippet)
    contradiction = False
    contradiction_note: Optional[str] = None
    for loc_pat, snip_pat, note in CONTRADICTION_RULES:
        if loc_pat.search(raw_location) and snip_pat.search(snippet):
            contradiction = True
            contradiction_note = note
            break

    # Step 4 — Map to Bizyl UI filter enums
    employment_type = normalise_employment(hint)
    remote_scope = map_remote_scope(location_clean, contradiction)
    mapped_domain = classify_domain(title_clean, snippet)
    seniority = classify_seniority(title_clean, snippet)

    return CleanedJob(
        job_id=record["job_id"],
        is_active=True,
        inactive_reason=None,
        title_clean=title_clean,
        raw_location_clean=location_clean,
        contradiction_detected=contradiction,
        contradiction_note=contradiction_note,
        employment_type=employment_type,
        remote_scope=remote_scope,
        mapped_domain=mapped_domain,
        seniority=seniority,
    )


# ── Entry Point ────────────────────────────────────────────────────────────────


def main() -> None:
    jobs: list[dict] = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(exist_ok=True)

    results = []
    for record in jobs:
        cleaned = process_job(record)
        results.append(cleaned.model_dump())
        status = "ACTIVE  " if cleaned.is_active else "INACTIVE"
        print(f"  {record['job_id']}  [{status}]  {cleaned.title_clean}")
        if cleaned.contradiction_detected:
            print(f"           Contradiction: {cleaned.contradiction_note}")
        if cleaned.is_active:
            print(
                f"           {cleaned.employment_type} | {cleaned.remote_scope}"
                f" | {cleaned.mapped_domain} | {cleaned.seniority}"
            )

    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nOutput -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
