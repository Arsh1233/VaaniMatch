# ============================================================
# Prompt 4.3 — Fraud & Profile Inconsistency Detection Engine
# Focus: Temporal overlap, skill inflation (NER), deduplication
#        (Jaro-Winkler), Authenticity Score, Suspicion Report JSON.
# ============================================================

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Optional

import jellyfish            # pip install jellyfish — Jaro-Winkler
import spacy               # pip install spacy; python -m spacy download en_core_web_sm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

# Load the small English NLP model (downloaded separately)
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning(
        "spaCy model 'en_core_web_sm' not found. "
        "Run: python -m spacy download en_core_web_sm"
    )
    _nlp = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EmploymentPeriod:
    employer: str
    start: date
    end: date   # use date.today() if current role
    title: str = ""
    description: str = ""


@dataclass
class CandidateProfile:
    candidate_id: str
    full_name: str
    phone: str
    employment_history: list[EmploymentPeriod] = field(default_factory=list)
    self_reported_skills: dict[str, int] = field(default_factory=dict)
    # e.g. {"Python": 5, "React": 3}  — self-reported rating out of 5
    project_descriptions: list[str] = field(default_factory=list)


@dataclass
class SuspicionFlag:
    category: str        # "temporal_overlap" | "skill_inflation" | "duplicate"
    severity: str        # "HIGH" | "MEDIUM" | "LOW"
    detail: str
    penalty: int         # 0-100, deducted from authenticity score


@dataclass
class AuthenticityReport:
    candidate_id: str
    authenticity_score: int          # 0-100
    suspicion_flags: list[SuspicionFlag] = field(default_factory=list)
    is_duplicate_of: Optional[str] = None  # candidate_id of the likely duplicate

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=str)


# ---------------------------------------------------------------------------
# 1. Temporal Overlap Detection
# ---------------------------------------------------------------------------

def _detect_temporal_overlaps(history: list[EmploymentPeriod]) -> list[SuspicionFlag]:
    """
    Flag any pair of employment periods whose date ranges overlap.

    Algorithm: sort by start date, then for each pair check if
    start_j < end_i  (i.e., job j started before job i ended).
    """
    flags: list[SuspicionFlag] = []
    sorted_history = sorted(history, key=lambda e: e.start)

    for i, period_a in enumerate(sorted_history):
        for period_b in sorted_history[i + 1:]:
            if period_b.start < period_a.end:
                overlap_days = (period_a.end - period_b.start).days
                severity = "HIGH" if overlap_days > 30 else "MEDIUM"
                flags.append(
                    SuspicionFlag(
                        category="temporal_overlap",
                        severity=severity,
                        detail=(
                            f"'{period_a.employer}' ({period_a.start}→{period_a.end}) "
                            f"overlaps '{period_b.employer}' ({period_b.start}→{period_b.end}) "
                            f"by {overlap_days} days."
                        ),
                        penalty=20 if severity == "HIGH" else 10,
                    )
                )
    return flags


# ---------------------------------------------------------------------------
# 2. Skill Inflation Detection  (NER co-occurrence)
# ---------------------------------------------------------------------------

def _extract_skill_mentions(texts: list[str], skill_name: str) -> int:
    """
    Count the number of times `skill_name` appears in evidence texts using
    simple case-insensitive token matching. NER is used to filter to relevant
    technical / product entities.
    """
    if _nlp is None:
        # Fallback: dumb string count
        return sum(text.lower().count(skill_name.lower()) for text in texts)

    count = 0
    pattern = re.compile(re.escape(skill_name), re.IGNORECASE)
    for text in texts:
        doc = _nlp(text)
        # Check raw occurrences
        if pattern.search(text):
            count += len(pattern.findall(text))
        # Also look in recognised entities (ORG, PRODUCT, WORK_OF_ART covers tech)
        for ent in doc.ents:
            if ent.label_ in {"ORG", "PRODUCT", "WORK_OF_ART"}:
                if skill_name.lower() in ent.text.lower():
                    count += 1
    return count


def _detect_skill_inflation(
    self_reported_skills: dict[str, int],
    project_descriptions: list[str],
) -> list[SuspicionFlag]:
    """
    Compare self-reported proficiency (1-5 scale) against project evidence.
    Heuristic: a skill rated ≥4 should appear at least twice in project
    descriptions; rated 5 should appear at least four times.
    """
    flags: list[SuspicionFlag] = []
    for skill, rating in self_reported_skills.items():
        mentions = _extract_skill_mentions(project_descriptions, skill)
        required_mentions = 4 if rating == 5 else 2 if rating >= 4 else 1

        if mentions < required_mentions:
            severity = "HIGH" if rating == 5 and mentions == 0 else "MEDIUM"
            flags.append(
                SuspicionFlag(
                    category="skill_inflation",
                    severity=severity,
                    detail=(
                        f"'{skill}' self-rated {rating}/5 but found only "
                        f"{mentions} mention(s) in project descriptions "
                        f"(expected ≥{required_mentions})."
                    ),
                    penalty=15 if severity == "HIGH" else 8,
                )
            )
    return flags


# ---------------------------------------------------------------------------
# 3. Deduplication — Jaro-Winkler fuzzy matching
# ---------------------------------------------------------------------------

def _build_fingerprint(profile: CandidateProfile) -> str:
    """Combine name + phone + top skill into a dedup fingerprint string."""
    top_skill = (
        max(profile.self_reported_skills, key=profile.self_reported_skills.get)
        if profile.self_reported_skills
        else ""
    )
    return f"{profile.full_name.lower().strip()}|{profile.phone.strip()}|{top_skill.lower()}"


def detect_duplicates(
    target: CandidateProfile,
    existing_profiles: list[CandidateProfile],
    similarity_threshold: float = 0.92,
) -> Optional[str]:
    """
    Compare `target` against each existing profile using Jaro-Winkler similarity
    on the composite fingerprint string.

    Returns the candidate_id of the first suspected duplicate found, or None.
    """
    target_fp = _build_fingerprint(target)
    for existing in existing_profiles:
        existing_fp = _build_fingerprint(existing)
        similarity = jellyfish.jaro_winkler_similarity(target_fp, existing_fp)
        logger.debug(
            "Jaro-Winkler('%s', '%s') = %.3f",
            target_fp,
            existing_fp,
            similarity,
        )
        if similarity >= similarity_threshold:
            logger.warning(
                "Potential duplicate: %s ↔ %s (score=%.3f)",
                target.candidate_id,
                existing.candidate_id,
                similarity,
            )
            return existing.candidate_id
    return None


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyse_profile(
    profile: CandidateProfile,
    existing_profiles: Optional[list[CandidateProfile]] = None,
) -> AuthenticityReport:
    """
    Run all three fraud-detection algorithms against a candidate profile.

    Parameters
    ----------
    profile : CandidateProfile
        The candidate to analyse.
    existing_profiles : list[CandidateProfile], optional
        Pool of existing candidates for deduplication check.

    Returns
    -------
    AuthenticityReport
        Contains an authenticity_score (0-100) and a list of suspicion flags.
    """
    flags: list[SuspicionFlag] = []

    # --- Algorithm 1: Temporal overlaps ---
    overlap_flags = _detect_temporal_overlaps(profile.employment_history)
    flags.extend(overlap_flags)

    # --- Algorithm 2: Skill inflation ---
    inflation_flags = _detect_skill_inflation(
        profile.self_reported_skills, profile.project_descriptions
    )
    flags.extend(inflation_flags)

    # --- Algorithm 3: Deduplication ---
    duplicate_id: Optional[str] = None
    if existing_profiles:
        duplicate_id = detect_duplicates(profile, existing_profiles)
        if duplicate_id:
            flags.append(
                SuspicionFlag(
                    category="duplicate",
                    severity="HIGH",
                    detail=(
                        f"Profile is highly similar to existing candidate "
                        f"'{duplicate_id}' (Jaro-Winkler ≥ 0.92)."
                    ),
                    penalty=30,
                )
            )

    # --- Compute authenticity score ---
    total_penalty = min(100, sum(f.penalty for f in flags))
    score = max(0, 100 - total_penalty)

    report = AuthenticityReport(
        candidate_id=profile.candidate_id,
        authenticity_score=score,
        suspicion_flags=flags,
        is_duplicate_of=duplicate_id,
    )
    logger.info(
        "Authenticity report for %s: score=%d, flags=%d",
        profile.candidate_id,
        score,
        len(flags),
    )
    return report


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import date

    profile = CandidateProfile(
        candidate_id="CAND-001",
        full_name="Rahul Sharma",
        phone="+91-9876543210",
        employment_history=[
            EmploymentPeriod(
                employer="TechCorp India",
                start=date(2019, 6, 1),
                end=date(2022, 5, 31),
                title="Software Engineer",
                description="Built React dashboards.",
            ),
            EmploymentPeriod(
                employer="StartupXYZ",
                start=date(2022, 3, 1),   # Overlaps TechCorp by ~3 months
                end=date(2023, 12, 31),
                title="Senior Developer",
                description="Led a Node.js backend team.",
            ),
        ],
        self_reported_skills={"Python": 5, "Machine Learning": 4, "React": 3},
        project_descriptions=[
            "Developed a customer-facing portal using React and Redux.",
            "Optimised SQL queries; minor scripting in Python.",
        ],
    )

    existing = [
        CandidateProfile(
            candidate_id="CAND-099",
            full_name="Rahul Sharma",      # Same name
            phone="+91-9876543210",        # Same phone
            self_reported_skills={"Python": 5},
            project_descriptions=[],
        )
    ]

    report = analyse_profile(profile, existing_profiles=existing)
    print(report.to_json())
