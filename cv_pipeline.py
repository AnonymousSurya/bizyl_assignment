#!/usr/bin/env python3
"""
cv_pipeline.py — Bizyl CV Processing Pipeline

Stage 1  Hierarchical Domain Gate:   classify primary + secondary domain/job-category
Stage 2  Scoped Skill Extraction:    regex pass (exact/synonym) + LLM pass (paraphrase)
         Pydantic guardrail rejects any LLM-hallucinated taxonomy node
Stage 3  S3 Flywheel Envelope:       wrap output for downstream fine-tuning accumulation
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# ── File Paths ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
TAXONOMY_FILE = BASE / "sandbox_taxonomy.json"
CVS_FILE = BASE / "synthetic_cvs.json"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "parsed_cvs.json"

# ── LLM Configuration ──────────────────────────────────────────────────────────
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")

# ── Domain locking threshold ───────────────────────────────────────────────────
SECONDARY_DOMAIN_MIN_HITS = 2

# ── Non-taxonomy skills to surface for admin review ────────────────────────────
OTHER_SKILL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?<![A-Za-z0-9])FastAPI(?![A-Za-z0-9])"), "FastAPI"),
    (re.compile(r"(?<![A-Za-z0-9])Pydantic(?![A-Za-z0-9])"), "Pydantic"),
    (re.compile(r"(?<![A-Za-z0-9])Redis(?![A-Za-z0-9])"), "Redis"),
    (re.compile(r"(?<![A-Za-z0-9])Docker(?![A-Za-z0-9])"), "Docker"),
    (re.compile(r"(?<![A-Za-z0-9])PostgreSQL(?![A-Za-z0-9])"), "PostgreSQL"),
    (re.compile(r"(?<![A-Za-z0-9])pandas(?![A-Za-z0-9])"), "pandas"),
    (re.compile(r"(?<![A-Za-z0-9])LangGraph(?![A-Za-z0-9])"), "LangGraph"),
    (re.compile(r"(?<![A-Za-z0-9])Looker\s+Studio(?![A-Za-z0-9])", re.IGNORECASE), "Looker Studio"),
    (re.compile(r"(?<![A-Za-z0-9])SQL(?![A-Za-z0-9\s]*(Query|Alchemy))"), "SQL"),
    (re.compile(r"(?<![A-Za-z0-9])Go-To-Market(?:\s+Strategy)?(?![A-Za-z0-9])"), "Go-To-Market Strategy"),
    (re.compile(r"(?<![A-Za-z0-9])A/B\s+(?:landing\s+page\s+)?test(?:s|ing)?(?![A-Za-z0-9])", re.IGNORECASE), "A/B Testing"),
    (re.compile(r"(?<![A-Za-z0-9])CAC(?![A-Za-z0-9])"), "Customer Acquisition Cost (CAC)"),
    (re.compile(r"(?<![A-Za-z0-9])B2B(?![A-Za-z0-9])"), "B2B"),
    (re.compile(r"(?<![A-Za-z0-9])B2C(?![A-Za-z0-9])"), "B2C"),
    (re.compile(r"(?<![A-Za-z0-9])CFA\s+Level\s+\d+(?![A-Za-z0-9])"), "CFA Level 1"),
    (re.compile(r"(?<![A-Za-z0-9])regex(?![A-Za-z0-9])", re.IGNORECASE), "Regex / Pattern Matching"),
]

# ── Pydantic Models ────────────────────────────────────────────────────────────


class TaxonomyPath(BaseModel):
    domain: str
    job_category: str
    job_title: str
    core_skill: str
    sub_skill: str
    matched_via: str


class DomainRouting(BaseModel):
    primary_domain: str
    primary_job_category: str
    secondary_domain: Optional[str] = None
    secondary_job_category: Optional[str] = None
    routing_confidence: str
    routing_signals: list[str]


class ModelExtractedProfile(BaseModel):
    mapped_skills: list[TaxonomyPath]
    other_skills_pending_approval: list[str]


class CVEnvelope(BaseModel):
    cv_id: str
    raw_text: str
    stage1_routing: DomainRouting
    model_extracted_profile: ModelExtractedProfile
    user_verified_profile: ModelExtractedProfile
    diff_log: list = Field(default_factory=list)


# ── LLM Client ─────────────────────────────────────────────────────────────────


def get_llm_client() -> tuple[Any, str]:
    """Returns (OpenAI client, model_name). Returns (None, '') if no API key set."""
    if not GROQ_API_KEY:
        print("  [LLM] No GROQ_API_KEY found — running regex-only mode")
        return None, ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
        print(f"  [LLM] Groq client ready — model: {LLM_MODEL}")
        return client, LLM_MODEL
    except ImportError:
        print("  [LLM] openai package not installed — running regex-only mode")
        return None, ""


# ── LLM Extraction + Guardrail ─────────────────────────────────────────────────


def llm_extract_skills(
    cv_text: str,
    locked_rows: list[dict],
    already_mapped_sub_skills: set[str],
    already_mapped_via: set[str],
    client: Any,
    model: str,
) -> tuple[list[TaxonomyPath], list[str], list[str]]:
    """
    Sends the CV text to the LLM with ONLY the locked domain's unmapped sub-skill
    nodes as allowed outputs. Returns:
      - new_paths:   TaxonomyPath objects for paraphrased skills the regex missed
      - new_other:   skills the LLM found but couldn't map to any node
      - rejected:    node names the LLM hallucinated (guardrail caught + discarded)

    Guardrail: every sub_skill in the LLM response is validated against the
    allowed_nodes set. Anything not in that set is rejected and moved to other_skills.
    """
    # Only offer nodes the regex pass hasn't already covered
    allowed_nodes = {row["Sub Skill"] for row in locked_rows} - already_mapped_sub_skills
    if not allowed_nodes:
        return [], [], []

    tax_index = {row["Sub Skill"]: row for row in locked_rows}
    allowed_list = "\n".join(f"- {s}" for s in sorted(allowed_nodes))

    # Tell LLM both canonical sub_skill names AND the CV-visible synonyms that are
    # already covered, so it doesn't re-report synonyms like "WACC Computation" or
    # "Python Requests" in unmatched_skills when they're already mapped.
    all_covered = already_mapped_sub_skills | already_mapped_via
    already_list = "\n".join(f"- {s}" for s in sorted(all_covered)) or "None"

    # /no_think disables Qwen 3 chain-of-thought tokens → clean JSON output
    prompt = f"""/no_think
You are a precise skill extraction engine for a professional taxonomy system.

CV TEXT:
{cv_text}

ALREADY COVERED TERMS (these are mapped — do NOT mention them in your output at all):
{already_list}

ALLOWED TAXONOMY NODES — your output sub_skill values MUST come EXACTLY from this list:
{allowed_list}

TASK:
Find skills mentioned (directly OR through paraphrase) in the CV text that are NOT
already covered above. For each skill found, map it to exactly one node from the ALLOWED list.

STRICT RULES:
1. Copy sub_skill values EXACTLY as they appear in the ALLOWED list — no modifications.
2. If a skill in the CV has no clear match in the ALLOWED list, add ONLY the specific
   skill or tool name (e.g. "FastAPI", "pandas") to unmatched_skills — NOT verbose
   sentences or descriptions.
3. Never invent, combine, abbreviate, or modify taxonomy node names.
4. If nothing new is found, return empty arrays.

Respond with ONLY valid JSON — no markdown, no explanation:
{{
  "matches": [
    {{"cv_text": "<phrase from CV>", "sub_skill": "<exact node from allowed list>"}}
  ],
  "unmatched_skills": ["<specific skill/tool name only — no sentences>"]
}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise skill extraction engine. Output only valid JSON with no markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        raw_text = response.choices[0].message.content

        # Strip Qwen thinking tags if model included them despite /no_think
        raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        # Strip markdown code fences
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text).strip()
        raw_text = re.sub(r"\s*```$", "", raw_text).strip()

        raw_json = json.loads(raw_text)

    except Exception as e:
        print(f"    [LLM] Error during API call or JSON parse: {e}")
        return [], [], []

    # ── Pydantic-style guardrail ───────────────────────────────────────────────
    new_paths: list[TaxonomyPath] = []
    new_other: list[str] = list(raw_json.get("unmatched_skills", []))
    rejected: list[str] = []

    for match in raw_json.get("matches", []):
        sub_skill = match.get("sub_skill", "").strip()
        cv_phrase = match.get("cv_text", "").strip()

        if sub_skill in allowed_nodes:
            # Valid — build full taxonomy path
            row = tax_index[sub_skill]
            new_paths.append(
                TaxonomyPath(
                    domain=row["Domain"],
                    job_category=row["Job Category"],
                    job_title=row["Job Title"],
                    core_skill=row["Core Skill"],
                    sub_skill=row["Sub Skill"],
                    matched_via=f"[LLM] {cv_phrase}",
                )
            )
        else:
            # Hallucinated node — reject + quarantine
            rejected.append(sub_skill)
            if cv_phrase:
                new_other.append(cv_phrase)

    return new_paths, new_other, rejected


# ── Taxonomy Index Builder ─────────────────────────────────────────────────────


def build_match_specs(taxonomy: list[dict]) -> list[dict]:
    """
    Builds regex match specs for every (Synonym, Sub Skill canonical) pair.
    Pattern: (?<![A-Za-z0-9]) TERM (?![A-Za-z0-9])
      Prevents 'Go' matching inside 'Google', 'SQL' inside 'SQLAlchemy', etc.
    """
    seen: set[tuple[str, str]] = set()
    specs: list[dict] = []
    for row in taxonomy:
        for match_term in [row["Synonym"], row["Sub Skill"]]:
            key = (row["Sub Skill"], match_term.lower())
            if key in seen:
                continue
            seen.add(key)
            escaped = re.escape(match_term)
            specs.append(
                {
                    "pattern": re.compile(
                        rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])",
                        re.IGNORECASE,
                    ),
                    "match_term": match_term,
                    "domain": row["Domain"],
                    "job_category": row["Job Category"],
                    "job_title": row["Job Title"],
                    "core_skill": row["Core Skill"],
                    "sub_skill": row["Sub Skill"],
                }
            )
    return specs


def scan_text(text: str, specs: list[dict]) -> list[dict]:
    matched: dict[tuple, dict] = {}
    for spec in specs:
        m = spec["pattern"].search(text)
        if m:
            key = (
                spec["domain"],
                spec["job_category"],
                spec["job_title"],
                spec["core_skill"],
                spec["sub_skill"],
            )
            if key not in matched:
                matched[key] = {**spec, "match_pos": m.start()}
    return list(matched.values())


# ── Broad keyword fallback for domain routing (used when regex gets 0 taxonomy hits) ──
# These are wider signals — not taxonomy synonyms — just enough to identify the domain
# so Stage 2 can inject the right taxonomy rows for the LLM pass.
DOMAIN_KEYWORD_FALLBACK: dict[str, re.Pattern] = {
    "Information Technology": re.compile(
        r"\b(engineer|developer|backend|frontend|goroutine|terraform|spark|flask|"
        r"django|kubernetes|docker|postgresql|pipeline|neural|pytorch|gradient\s+boosting|"
        r"SHAP|VPC|subnet|cloud|infrastructure|NLP|machine\s+learning|deep\s+learning|"
        r"REST\s+API|microservice|index\s+design|query\s+tuning)\b",
        re.IGNORECASE,
    ),
    "Marketing": re.compile(
        r"\b(marketing|paid\s+search|paid\s+social|SEO|SEM|growth|audience|campaign|"
        r"retargeting|conversion|acquisition|social\s+media|media\s+buyer)\b",
        re.IGNORECASE,
    ),
    "Consulting": re.compile(
        r"\b(consulting|consultant|strategy|advisory|market\s+sizing|"
        r"due\s+diligence|issue\s+tree|MECE|TAM|SAM|SOM)\b",
        re.IGNORECASE,
    ),
    "Finance": re.compile(
        r"\b(finance|financial|FP&A|accounting|DCF|WACC|financial\s+model|"
        r"valuation|run\s+rate|forecast|budget|revenue|unit\s+economics)\b",
        re.IGNORECASE,
    ),
}


def keyword_fallback_routing(text: str) -> tuple[str, str]:
    """
    Broad keyword scan used ONLY when taxonomy regex finds zero hits.
    Returns (primary_domain, primary_job_category).
    """
    scores = {d: len(pat.findall(text)) for d, pat in DOMAIN_KEYWORD_FALLBACK.items()}
    primary_domain = max(scores, key=scores.get)
    # Pick the first job category for that domain (single-category domains; IT gets Data&AI)
    from typing import Iterator
    cat_map = {
        "Information Technology": "Data and Artificial Intelligence",
        "Marketing": "Growth and Performance Media",
        "Consulting": "Strategy and Corporate Advisory",
        "Finance": "Corporate Finance and Strategic Planning",
    }
    return primary_domain, cat_map[primary_domain]


# ── Stage 1 — Hierarchical Domain Gate ────────────────────────────────────────


def route_domains(text: str, all_specs: list[dict]) -> tuple[DomainRouting, set[str]]:
    hits = scan_text(text, all_specs)

    domain_counts: dict[str, int] = {}
    domain_cat_counts: dict[str, dict[str, int]] = {}
    domain_first_pos: dict[str, int] = {}
    domain_signals: dict[str, list[str]] = {}

    for h in hits:
        d, cat, pos, term = h["domain"], h["job_category"], h["match_pos"], h["match_term"]
        domain_counts[d] = domain_counts.get(d, 0) + 1
        if d not in domain_cat_counts:
            domain_cat_counts[d] = {}
        domain_cat_counts[d][cat] = domain_cat_counts[d].get(cat, 0) + 1
        if d not in domain_first_pos or pos < domain_first_pos[d]:
            domain_first_pos[d] = pos
        domain_signals.setdefault(d, []).append(term)

    if not domain_counts:
        # Zero exact taxonomy hits — fall back to broad keyword signals for domain routing.
        # Stage 2 will rely entirely on the LLM pass to find paraphrased skills.
        primary_domain, primary_cat = keyword_fallback_routing(text)
        print(f"    [Stage 1] No exact taxonomy hits — keyword fallback → {primary_domain}")
        routing = DomainRouting(
            primary_domain=primary_domain,
            primary_job_category=primary_cat,
            routing_confidence="Low (keyword fallback — no exact taxonomy matches)",
            routing_signals=["keyword fallback used"],
        )
        locked_domains = {primary_domain}
        return routing, locked_domains

    ranked = sorted(
        domain_counts.items(),
        key=lambda x: (-x[1], domain_first_pos.get(x[0], 999_999)),
    )

    primary_domain = ranked[0][0]
    primary_cat = max(domain_cat_counts[primary_domain], key=domain_cat_counts[primary_domain].get)
    signals: list[str] = list(domain_signals[primary_domain])

    secondary_domain: Optional[str] = None
    secondary_cat: Optional[str] = None

    if len(ranked) > 1 and ranked[1][1] >= SECONDARY_DOMAIN_MIN_HITS:
        secondary_domain = ranked[1][0]
        secondary_cat = max(domain_cat_counts[secondary_domain], key=domain_cat_counts[secondary_domain].get)
        signals.append(
            f"secondary signals ({secondary_domain}): "
            + ", ".join(domain_signals[secondary_domain])
        )

    for d, cnt in ranked[2:]:
        if cnt >= SECONDARY_DOMAIN_MIN_HITS:
            signals.append(
                f"tertiary domain also locked ({d}, {cnt} hits): "
                + ", ".join(domain_signals[d])
            )

    top_count = ranked[0][1]
    second_count = ranked[1][1] if len(ranked) > 1 else 0
    confidence = "High" if (second_count == 0 or top_count >= 2 * second_count or top_count > second_count) else "Medium"

    routing = DomainRouting(
        primary_domain=primary_domain,
        primary_job_category=primary_cat,
        secondary_domain=secondary_domain,
        secondary_job_category=secondary_cat,
        routing_confidence=confidence,
        routing_signals=signals,
    )

    locked_domains: set[str] = {
        d for d, cnt in domain_counts.items() if cnt >= SECONDARY_DOMAIN_MIN_HITS
    }
    locked_domains.add(primary_domain)

    return routing, locked_domains


# ── Stage 2 — Scoped Skill Extraction + LLM + Quarantine ──────────────────────


def extract_skills(
    text: str,
    all_specs: list[dict],
    locked_domains: set[str],
    llm_client: Any = None,
    llm_model: str = "",
    locked_rows: Optional[list[dict]] = None,
) -> ModelExtractedProfile:
    """
    Pass 1 — Regex: exact synonym / canonical name matching with false-positive guards.
    Pass 2 — LLM:   paraphrase matching, constrained to locked taxonomy nodes only.
                    Pydantic guardrail rejects any hallucinated node name.
    """
    # ── Pass 1: Regex ──────────────────────────────────────────────────────────
    scoped_specs = [s for s in all_specs if s["domain"] in locked_domains]
    hits = scan_text(text, scoped_specs)

    mapped: list[TaxonomyPath] = [
        TaxonomyPath(
            domain=h["domain"],
            job_category=h["job_category"],
            job_title=h["job_title"],
            core_skill=h["core_skill"],
            sub_skill=h["sub_skill"],
            matched_via=h["match_term"],
        )
        for h in hits
    ]

    covered: set[str] = {m.sub_skill.lower() for m in mapped} | {m.matched_via.lower() for m in mapped}

    other: list[str] = []
    seen_other: set[str] = set()
    for pat, label in OTHER_SKILL_PATTERNS:
        if pat.search(text) and label not in seen_other:
            if not any(word.lower() in covered for word in label.split() if len(word) > 3):
                other.append(label)
                seen_other.add(label)

    # ── Pass 2: LLM ───────────────────────────────────────────────────────────
    if llm_client and locked_rows:
        already_mapped_sub_skills = {m.sub_skill for m in mapped}
        already_mapped_via = {m.matched_via for m in mapped}
        # Also exclude terms already in regex other_skills
        already_other_lower = {s.lower() for s in other}
        # Build a broad covered set for filtering LLM other output
        covered_lower = (
            {m.sub_skill.lower() for m in mapped}
            | {m.matched_via.lower() for m in mapped}
            | already_other_lower
        )

        llm_paths, llm_other, rejected = llm_extract_skills(
            text, locked_rows,
            already_mapped_sub_skills, already_mapped_via,
            llm_client, llm_model,
        )

        if rejected:
            print(f"    [LLM] Guardrail REJECTED {len(rejected)} hallucinated node(s): {rejected}")
        if llm_paths:
            print(f"    [LLM] Added {len(llm_paths)} new skill(s) via paraphrase matching")

        # Merge: regex results take precedence; LLM adds only genuinely new skills
        mapped_sub_skills = {m.sub_skill for m in mapped}
        for path in llm_paths:
            if path.sub_skill not in mapped_sub_skills:
                mapped.append(path)
                mapped_sub_skills.add(path.sub_skill)

        for skill in llm_other:
            skill = skill.strip()
            if not skill:
                continue
            # Skip if the skill text is already covered by a mapped term or synonym
            if skill.lower() in covered_lower:
                continue
            # Skip verbose descriptions (more than 6 words = sentence, not a skill name)
            if len(skill.split()) > 6:
                continue
            # Skip near-duplicates: if a core word (len > 4) in the new skill already
            # appears in an existing other_skill, it's likely covered
            # e.g. "regex validation" dup of "Regex / Pattern Matching" via "regex"
            new_words_lg = {w.lower() for w in skill.split() if len(w) > 4}
            existing_words_lg = {w.lower() for s in other for w in s.split() if len(w) > 4}
            if new_words_lg and new_words_lg & existing_words_lg:
                continue
            if skill not in seen_other:
                other.append(skill)
                seen_other.add(skill)

    return ModelExtractedProfile(
        mapped_skills=mapped,
        other_skills_pending_approval=other,
    )


# ── Stage 3 — S3 Flywheel Envelope ────────────────────────────────────────────


def build_envelope(cv: dict, routing: DomainRouting, profile: ModelExtractedProfile) -> CVEnvelope:
    return CVEnvelope(
        cv_id=cv["cv_id"],
        raw_text=cv["raw_text"],
        stage1_routing=routing,
        model_extracted_profile=profile,
        user_verified_profile=copy.deepcopy(profile),
        diff_log=[],
    )


# ── Entry Point ────────────────────────────────────────────────────────────────


def process_single_cv(
    cv: dict,
    taxonomy: list[dict],
    all_specs: list[dict],
    llm_client: Any,
    llm_model: str,
) -> dict:
    """Process one CV end-to-end. Designed to run in a thread — no shared mutable state."""
    routing, locked_domains = route_domains(cv["raw_text"], all_specs)
    locked_rows = [row for row in taxonomy if row["Domain"] in locked_domains]
    profile = extract_skills(
        cv["raw_text"], all_specs, locked_domains,
        llm_client, llm_model, locked_rows,
    )
    envelope = build_envelope(cv, routing, profile)

    sec = f" + {routing.secondary_domain}" if routing.secondary_domain else ""
    extra = locked_domains - {routing.primary_domain} - (
        {routing.secondary_domain} if routing.secondary_domain else set()
    )
    extra_str = f" + {', '.join(sorted(extra))} (tertiary)" if extra else ""
    print(
        f"  {cv['cv_id']}  ->  {routing.primary_domain}{sec}{extra_str}"
        f"  |  {len(profile.mapped_skills)} mapped"
        f"  |  {len(profile.other_skills_pending_approval)} quarantined"
    )
    return envelope.model_dump()


def main() -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    taxonomy: list[dict] = json.loads(TAXONOMY_FILE.read_text(encoding="utf-8"))
    cvs: list[dict] = json.loads(CVS_FILE.read_text(encoding="utf-8"))

    all_specs = build_match_specs(taxonomy)
    OUTPUT_DIR.mkdir(exist_ok=True)
    llm_client, llm_model = get_llm_client()

    print(f"\nProcessing {len(cvs)} CVs in parallel (one thread per CV) ...\n")
    t0 = time.time()

    # CVs are independent — run all LLM calls concurrently.
    # Within each CV the pipeline is still strictly sequential (regex → LLM).
    results_map: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(cvs)) as executor:
        futures = {
            executor.submit(process_single_cv, cv, taxonomy, all_specs, llm_client, llm_model): cv["cv_id"]
            for cv in cvs
        }
        for future in as_completed(futures):
            cv_id = futures[future]
            try:
                results_map[cv_id] = future.result()
            except Exception as e:
                print(f"  ERROR processing {cv_id}: {e}")

    # Preserve original CV order in output
    results = [results_map[cv["cv_id"]] for cv in cvs if cv["cv_id"] in results_map]

    elapsed = time.time() - t0
    print(f"\nAll CVs done in {elapsed:.1f}s  (sequential would be ~{elapsed * len(cvs) / max(len(results), 1):.1f}s)")

    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Output -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
