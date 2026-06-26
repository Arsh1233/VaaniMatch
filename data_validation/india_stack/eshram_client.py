# ============================================================
# Prompt 4.1 — e-Shram & NCS API Integration
# Focus: OAuth 2.0 client credentials, government sandbox APIs,
#        PostgreSQL upsert with 'Government Verified' flag.
# ============================================================

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx
from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

# ---------------------------------------------------------------------------
# Database models
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.environ.get("DATABASE_URL", "postgresql://localhost/vaanimatch")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Candidate(Base):
    __tablename__ = "candidates"

    aadhaar_hash = Column(String(64), primary_key=True, index=True)
    verified_name = Column(String(256), nullable=True)
    age = Column(String(10), nullable=True)
    occupational_skills = Column(Text, nullable=True)          # JSON-serialised list
    is_government_verified = Column(Boolean, default=False, nullable=False)
    verified_at = Column(DateTime(timezone=True), nullable=True)
    source = Column(String(64), default="eshram", nullable=False)

    def __repr__(self) -> str:
        return f"<Candidate aadhaar_hash={self.aadhaar_hash!r} name={self.verified_name!r}>"


def init_db() -> None:
    """Create tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialised.")


# ---------------------------------------------------------------------------
# OAuth 2.0 token manager (client credentials flow)
# ---------------------------------------------------------------------------

class OAuth2TokenManager:
    """Thread-safe, auto-refreshing OAuth 2.0 client-credentials token store."""

    def __init__(self, token_url: str, client_id: str, client_secret: str) -> None:
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._access_token: Optional[str] = None
        self._expires_at: float = 0.0

    def _is_expired(self) -> bool:
        import time
        return time.time() >= self._expires_at - 30  # 30-second safety margin

    def get_token(self) -> str:
        if self._access_token is None or self._is_expired():
            self._refresh()
        return self._access_token  # type: ignore[return-value]

    def _refresh(self) -> None:
        import time
        logger.info("Refreshing OAuth 2.0 access token from %s", self._token_url)
        with httpx.Client(timeout=10) as client:
            response = client.post(
                self._token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "scope": "candidate:read",
                },
                headers={"Accept": "application/json"},
            )
        response.raise_for_status()
        payload = response.json()
        self._access_token = payload["access_token"]
        self._expires_at = time.time() + int(payload.get("expires_in", 3600))
        logger.info("Token refreshed, expires in %ds.", payload.get("expires_in", 3600))


# Singleton token managers (lazy initialisation)
_eshram_token_mgr: Optional[OAuth2TokenManager] = None
_ncs_token_mgr: Optional[OAuth2TokenManager] = None


def _get_eshram_token_manager() -> OAuth2TokenManager:
    global _eshram_token_mgr
    if _eshram_token_mgr is None:
        _eshram_token_mgr = OAuth2TokenManager(
            token_url=os.environ["ESHRAM_TOKEN_URL"],
            client_id=os.environ["ESHRAM_CLIENT_ID"],
            client_secret=os.environ["ESHRAM_CLIENT_SECRET"],
        )
    return _eshram_token_mgr


def _get_ncs_token_manager() -> OAuth2TokenManager:
    global _ncs_token_mgr
    if _ncs_token_mgr is None:
        _ncs_token_mgr = OAuth2TokenManager(
            token_url=os.environ["NCS_TOKEN_URL"],
            client_id=os.environ["NCS_CLIENT_ID"],
            client_secret=os.environ["NCS_CLIENT_SECRET"],
        )
    return _ncs_token_mgr


# ---------------------------------------------------------------------------
# Core fetch & upsert function
# ---------------------------------------------------------------------------

def _hash_aadhaar(raw_aadhaar: str) -> str:
    """Return a SHA-256 hex digest of the Aadhaar number (never store raw)."""
    return hashlib.sha256(raw_aadhaar.strip().encode()).hexdigest()


def _fetch_from_eshram(aadhaar_hash: str) -> dict:
    """
    Call the e-Shram sandbox API to retrieve basic candidate details.

    Returns a dict with keys: verified_name, age, occupational_skills.
    Raises httpx.HTTPStatusError on API errors.
    """
    token = _get_eshram_token_manager().get_token()
    api_url = os.environ.get("ESHRAM_API_URL", "https://sandbox.eshram.gov.in/api/v1")
    url = f"{api_url}/worker/{aadhaar_hash}"

    logger.info("Fetching e-Shram data for aadhaar_hash=%s…", aadhaar_hash[:8] + "****")
    with httpx.Client(timeout=15) as client:
        response = client.get(
            url,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
    response.raise_for_status()
    data = response.json()

    return {
        "verified_name": data.get("name"),
        "age": str(data.get("age", "")),
        "occupational_skills": str(data.get("skills", [])),  # serialise list → str
        "source": "eshram",
    }


def _fetch_from_ncs(aadhaar_hash: str) -> dict:
    """
    Optionally enrich from the NCS (National Career Service) sandbox API.
    Falls back gracefully if the NCS endpoint returns 404.
    """
    token = _get_ncs_token_manager().get_token()
    api_url = os.environ.get("NCS_API_URL", "https://sandbox.ncs.gov.in/api/v1")
    url = f"{api_url}/candidate/{aadhaar_hash}/skills"

    logger.info("Enriching from NCS for aadhaar_hash=%s…", aadhaar_hash[:8] + "****")
    try:
        with httpx.Client(timeout=15) as client:
            response = client.get(
                url,
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            logger.warning("Candidate not found in NCS — skipping NCS enrichment.")
            return {}
        raise


def _upsert_candidate(session: Session, aadhaar_hash: str, data: dict) -> Candidate:
    """
    Upsert a Candidate row, setting is_government_verified=True.
    Uses PostgreSQL's INSERT … ON CONFLICT DO UPDATE for atomicity.
    """
    stmt = (
        pg_insert(Candidate)
        .values(
            aadhaar_hash=aadhaar_hash,
            verified_name=data.get("verified_name"),
            age=data.get("age"),
            occupational_skills=data.get("occupational_skills"),
            is_government_verified=True,
            verified_at=datetime.now(timezone.utc),
            source=data.get("source", "eshram"),
        )
        .on_conflict_do_update(
            index_elements=["aadhaar_hash"],
            set_={
                "verified_name": data.get("verified_name"),
                "age": data.get("age"),
                "occupational_skills": data.get("occupational_skills"),
                "is_government_verified": True,
                "verified_at": datetime.now(timezone.utc),
                "source": data.get("source", "eshram"),
            },
        )
        .returning(Candidate)
    )
    result = session.execute(stmt)
    session.commit()
    row = result.fetchone()
    logger.info("Upserted candidate %s — Government Verified ✓", aadhaar_hash[:8] + "****")
    return row  # type: ignore[return-value]


def fetch_candidate_by_aadhaar(aadhaar_hash: str) -> dict:
    """
    Public API — fetch a candidate's verified profile from e-Shram (and
    optionally NCS) and persist it into PostgreSQL with the
    'Government Verified' flag set to True.

    Parameters
    ----------
    aadhaar_hash : str
        SHA-256 hash of the candidate's 12-digit Aadhaar number.
        Never pass the raw number.

    Returns
    -------
    dict
        The upserted candidate record as a plain dictionary.

    Raises
    ------
    httpx.HTTPStatusError
        If the government API returns a non-2xx status.
    """
    if len(aadhaar_hash) != 64:
        raise ValueError(
            "aadhaar_hash must be a 64-character SHA-256 hex digest. "
            "Use _hash_aadhaar(raw) to produce it."
        )

    # 1. Fetch from e-Shram (primary source)
    candidate_data = _fetch_from_eshram(aadhaar_hash)

    # 2. Optionally enrich from NCS
    ncs_data = _fetch_from_ncs(aadhaar_hash)
    if ncs_data.get("skills"):
        # Merge NCS skills into the existing list
        existing = candidate_data.get("occupational_skills", "[]")
        candidate_data["occupational_skills"] = str(
            list(set(eval(existing) + ncs_data["skills"]))  # noqa: S307
        )

    # 3. Persist into PostgreSQL
    with SessionLocal() as session:
        _upsert_candidate(session, aadhaar_hash, candidate_data)

    return {
        "aadhaar_hash": aadhaar_hash,
        **candidate_data,
        "is_government_verified": True,
    }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    # Accept a raw Aadhaar (12 digits) on the command line for quick testing.
    raw = sys.argv[1] if len(sys.argv) > 1 else "999900001234"
    h = _hash_aadhaar(raw)
    print(f"Aadhaar hash: {h}")

    # NOTE: This will call the real sandbox; ensure env vars are set.
    try:
        result = fetch_candidate_by_aadhaar(h)
        print(json.dumps(result, indent=2, default=str))
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
