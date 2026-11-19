# Bizyl Backend & Applied NLP Engineering Challenge

## Setup & Run

### Prerequisites
- Conda (Python 3.12 environment)
- Groq API key (set in `.env`)

### Create environment and install dependencies
```bash
python -m venv venv          # Python 3.12
venv\Scripts\activate        # Windows
pip install pydantic openai python-dotenv
```

### Configure API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
LLM_MODEL=qwen/qwen3.8-27b
```

### Run Task 1 — CV Pipeline
```bash
python cv_pipeline.py
# Output: outputs/parsed_cvs.json
```
If `GROQ_API_KEY` is not set, the pipeline falls back to regex-only mode automatically.

### Run Task 2 — Job Hygiene
```bash
python job_hygiene.py
# Output: outputs/cleaned_jobs.json
```

---

## Task 1 Design Decisions

### Two-Pass Skill Extraction

**Pass 1 — Regex (deterministic):** Matches exact taxonomy synonyms and canonical sub-skill names using whole-phrase patterns. Zero hallucination risk. Catches everything that appears verbatim in the CV.

**Pass 2 — LLM (Groq / `qwen/qwen3.8-27b`):** Runs after regex on only the unmapped nodes. The LLM receives a constrained prompt listing only the locked domain's remaining sub-skill nodes as allowed outputs — it cannot invent new taxonomy nodes.

**Pydantic guardrail:** Every `sub_skill` returned by the LLM is validated against the actual taxonomy. Any node name not in the locked taxonomy is rejected and the CV phrase is moved to `other_skills_pending_approval` instead. This makes hallucinations impossible to reach the final output.

**Merge:** Regex results always take precedence. The LLM only adds skills it found via paraphrase that the regex missed. Near-duplicate other_skills are deduplicated via word-overlap check.

What the LLM added beyond regex in this run:
- CV_002: `"Cohort payback models"` (from "Built monthly run rate and cohort payback models in partnership with CFO") — not in taxonomy, correctly quarantined

### Domain Drift Prevention

**Stage 1 scores before Stage 2 injects.**  
Domain classification runs a full-taxonomy scan first. Only after locking domains does Stage 2 inject the scoped taxonomy rows. This prevents an IT synonym from accidentally matching inside a Finance CV's text just because the word happened to appear.

**Threshold-based multi-domain locking.**  
Any domain with ≥ 2 independent taxonomy hits is added to `locked_domains`. The `DomainRouting` model exposes `primary_domain` + `secondary_domain` (top 2 by score) for the API layer, but Stage 2 uses the full locked set. This correctly handles CV_002 (Marketing 4 hits + Consulting 4 hits + Finance 2 hits → all 3 locked) without forcing a binary primary/secondary choice.

**Tie-breaking by text position.**  
When two domains score equally (CV_002: Marketing == Consulting at 4 hits each), the domain whose first taxonomy term appears earliest in the CV text is elected primary. For CV_002 this correctly picks Marketing, since "Google Ads Smart Bidding" appears before "TAM SAM SOM Estimation" in the core competencies list.

### Token Collision Prevention

Patterns use **negative lookaround on `[A-Za-z0-9]`** on both sides:

```
(?<![A-Za-z0-9])  TERM  (?![A-Za-z0-9])
```

This blocks the two canonical failure modes cited in the brief:

| Risk | Why it's safe |
|------|--------------|
| `"Go"` inside `"Google Ads"` | Lookahead fails: `'o'` immediately follows `"Go"` in `"Google"` |
| `"SQL"` inside `"SQLAlchemy"` | Negative lookahead also excludes `(?![A-Za-z0-9\s]*(Query\|Alchemy))` |

**Whole-phrase matching only.** Multi-word synonyms like `"PySpark Pipelines"` or `"Dead Letter Queues"` are matched as atomic phrases — the code never tokenises a synonym into individual words and tests them separately, which is the root cause of most substring collision bugs.

**Synonyms before canonical names.** `build_match_specs` lists the Synonym spec before the Sub Skill canonical name spec for each row. `scan_text` deduplicates by taxonomy path, so when both match, the CV-visible surface form (`matched_via`) is preserved in the output — not the canonical name the candidate never wrote.

---

## Task 2 Design Decisions

**Inactive detection before all other logic.** Soft-404 / "no longer available" / "requisition does not exist" patterns are tested first; inactive records get `is_active: false` and no filter fields are populated, preventing meaningless enum values from appearing for dead listings.

**HTML unescaping happens before contradiction detection.** `html.unescape()` runs on title and location before any pattern matching, so `&amp;`, `&mdash;`, and `&ndash;` are resolved to their Unicode equivalents and can't interfere with regex patterns.

**Contradiction detection is location-vs-snippet, not keyword-only.** JB_101 has `"Remote Anywhere"` in `raw_location` but `"Must reside in the United States or Canada"` in `raw_page_snippet`. The rule requires both patterns to match — location pattern AND snippet pattern — preventing false positives from records that simply mention a country name in passing.

**`"Permanent"` → `"Full-time"` normalisation.** The employment_hint `"Permanent"` is a UK/ANZ convention meaning indefinite full-time employment. The normaliser maps it to `"Full-time"` rather than `null`.

---

## Task 3 — Bizyl Profile Snapshot Audit

> CV uploaded: Suryanarayan Mohapatra (AI/ML Engineer, 3 roles, 2 yr 6 mo total experience).  
> All issues below were observed directly after upload and profile generation on bizyl.com.

---

### Issue 1: Cross-domain taxonomy contamination injects a "Consulting" identity into an AI/ML CV

**Observed:** The Skill Matrix assigned domain **Consulting** as a secondary domain alongside Information Technology, surfacing job titles **"Economic Consultant"** and **"Data Analytics Consultant"**, and core skill **"Econometric Modeling"** — none of which appear anywhere in the uploaded CV. The candidate has zero consulting work history.

**Root cause:** The Consulting and Finance taxonomy lexicons share a large vocabulary with ML/Data Science — words like "modeling", "forecasting", "analysis", "platform", and "evaluation" appear as synonyms in both domains. Because domain routing fires on raw hit count with no minimum threshold, even two or three accidental collisions from ML prose (e.g. "statistical modeling" triggering a Consulting synonym) are enough to lock the Consulting domain and inject a full taxonomy path — job category, job title, and core skill — into the profile. The candidate never wrote a single consulting sentence, but the vocabulary overlap made the signal look credible to a threshold-free scorer.

**Fix (Python):**
```python
MIN_DOMAIN_HITS = 3  # require at least 3 independent hits before locking a domain

def route_domains(text: str, all_specs: list[MatchSpec]) -> tuple[DomainRouting, set[str]]:
    scores: dict[str, int] = defaultdict(int)
    for spec in all_specs:
        if spec.pattern.search(text):
            scores[spec.domain] += 1

    # Only lock domains that clear the minimum confidence threshold
    locked = {d for d, count in scores.items() if count >= MIN_DOMAIN_HITS}
    if not locked:
        locked = {max(scores, key=scores.get)}  # fallback: top domain only

    ranked = sorted(locked, key=lambda d: scores[d], reverse=True)
    return DomainRouting(
        primary_domain=ranked[0],
        secondary_domain=ranked[1] if len(ranked) > 1 else None,
    ), locked
```

---

### Issue 2: Achievements and certifications silently dropped — "ADDITIONAL INFORMATION" section not recognised

**Observed:** The Achievements tab shows **"No achievements yet"** and **"No certifications found"**, despite the CV listing the INSPIRE Scholarship (Dept. of Science & Technology, Govt. of India), Gold Badge in Python (HackerRank), and two certifications (Coursera Advanced ML, PW Skills Data Science Masters). All four are present under the section header labelled **"ADDITIONAL INFORMATION"**.

**Root cause:** The CV section parser only extracts achievements and certifications from sections whose header exactly matches a canonical list (`"Certifications"`, `"Awards"`, `"Achievements"`). The common variant `"ADDITIONAL INFORMATION"` — widely used by Indian and UK CV templates — is absent from that list, so the entire section is discarded silently. The candidate sees empty tabs with no indication that anything was detected and dropped.

**Fix (Python):**
```python
SECTION_ALIASES: dict[str, list[str]] = {
    "certifications": ["certifications", "credentials", "licenses", "courses"],
    "achievements":   ["achievements", "awards", "honors", "recognition", "activities"],
    # Catch-all for templates that lump everything under one section
    "additional":     ["additional information", "additional details", "other information"],
}

CERT_PATTERN  = re.compile(r"certif|credential|coursera|udemy|aws\s+certif", re.IGNORECASE)
AWARD_PATTERN = re.compile(r"award|scholarship|fellow|prize|badge|honor|gold|silver", re.IGNORECASE)

def extract_achievements_from_section(section_text: str) -> list[str]:
    """Scan any section text for certification/award signals, regardless of header name."""
    results = []
    for line in section_text.splitlines():
        line = line.strip(" •–-")
        if line and (CERT_PATTERN.search(line) or AWARD_PATTERN.search(line)):
            results.append(line)
    return results
```

---

### Issue 3: Auto-generated Professional Summary contains fabricated and duplicated content

**Observed:** The parsed summary shown on both the Profile Details page and the shareable Snapshot reads:  
*"…Stack: **Cloud Threat Detection Implementation**, Classical Time Series Modeling, Model Explainability Analysis, **Python Data Science Ecosystem, Python Data Science Ecosystem** [duplicated], AWS Infrastructure Engineering, Azure Cloud Engineering, Python Server Development, RESTful API Engineering, **Linux Storage LVM Management**."*

The CV contains none of "Cloud Threat Detection Implementation" or "Linux Storage LVM Management", and no phrase is repeated. The job title was also corrupted to **"Lead AI / MI Engineer"** (MI instead of ML).

**Root cause:** Instead of extracting the candidate's own written Professional Summary verbatim, Bizyl reconstructs the summary by passing taxonomy-matched node names through an LLM. The LLM fills gaps with plausible-sounding stack labels that were never in the source text, and the deduplication step runs only on exact string equality — it misses the duplicate after any whitespace normalisation difference.

**Fix (Python):**
```python
SUMMARY_RE = re.compile(
    r"(?:PROFESSIONAL\s+SUMMARY|SUMMARY|PROFILE|ABOUT\s+ME)\s*\n(.*?)(?=\n[A-Z][A-Z ]{2,}\n|\Z)",
    re.DOTALL | re.IGNORECASE,
)

def extract_verbatim_summary(cv_text: str) -> str | None:
    """Return the candidate's own written summary instead of reconstructing it from taxonomy nodes."""
    m = SUMMARY_RE.search(cv_text)
    if not m:
        return None
    raw = m.group(1).strip()
    # Collapse PDF line-wrap artefacts
    return re.sub(r"\s*\n\s*", " ", raw)
```

---

### Issue 4: Per-skill experience hardcoded to "1y" when the CV has no per-skill duration data

**Observed:** The Skill Matrix `EXP (Y)` column shows **1y** for virtually every skill (Anomaly Detection, FastAPI, AWS SageMaker, Time-Series Forecasting, semantic routing, etc.) regardless of how long the candidate has actually used each one. Skills the system couldn't attribute to any role show **0y**. The CV has no per-skill date annotations — it lists skills in a flat section and mentions them in work history prose — so neither value is derived from real data.

**Root cause:** When skills are listed in a flat section without date ranges, Bizyl cannot calculate per-skill experience. Rather than surfacing `null` or inferring duration from role overlap, it falls back to a hardcoded `1y` default for every matched skill. This produces a uniform, misleading signal that makes a 2.5-year senior engineer look indistinguishable from someone who picked up each skill yesterday.

**Fix (Python):**
```python
from datetime import date

def infer_skill_experience_years(
    skill_name: str,
    work_history: list[dict],  # [{title, start: date, end: date | None, bullets: str}]
) -> float | None:
    """
    Sum the duration of every role whose bullet points mention the skill.
    Returns None (not 0 or 1) if the skill cannot be attributed to any role.
    """
    total_days = 0
    today = date.today()
    for role in work_history:
        if skill_name.lower() in role["bullets"].lower():
            end = role["end"] or today
            total_days += (end - role["start"]).days
    if total_days == 0:
        return None  # unknown — show "—" in UI, not "0y" or "1y"
    return round(total_days / 365, 1)
```
