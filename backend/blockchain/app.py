# ============================================================
# Prompt 4.4 — Blockchain Verification Microservice
# Focus: SHA-256 hashing, SQLite ledger (Hyperledger-style),
#        Flask /verify endpoint.
# ============================================================

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Ledger (SQLite) setup
# ---------------------------------------------------------------------------

LEDGER_PATH: str = os.environ.get(
    "LEDGER_PATH",
    str(Path(__file__).parent / "ledger.sqlite"),
)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(LEDGER_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_ledger() -> None:
    """Create the blocks table if it does not already exist."""
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blocks (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                block_index    INTEGER NOT NULL,
                previous_hash  TEXT    NOT NULL,
                certificate_hash TEXT  NOT NULL,
                block_hash     TEXT    NOT NULL UNIQUE,
                timestamp      TEXT    NOT NULL,
                nonce          INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        # Seed the genesis block if the ledger is empty
        row = conn.execute("SELECT COUNT(*) AS cnt FROM blocks").fetchone()
        if row["cnt"] == 0:
            genesis_hash = _compute_block_hash(
                block_index=0,
                previous_hash="0" * 64,
                certificate_hash="GENESIS",
                timestamp="2024-01-01T00:00:00+00:00",
                nonce=0,
            )
            conn.execute(
                """
                INSERT INTO blocks
                    (block_index, previous_hash, certificate_hash, block_hash, timestamp, nonce)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (0, "0" * 64, "GENESIS", genesis_hash, "2024-01-01T00:00:00+00:00", 0),
            )
            conn.commit()
            logger.info("Genesis block created: %s", genesis_hash)


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _compute_block_hash(
    block_index: int,
    previous_hash: str,
    certificate_hash: str,
    timestamp: str,
    nonce: int,
) -> str:
    """
    Deterministic SHA-256 hash of a block's canonical fields.
    Mirrors a simplified Hyperledger Fabric block header structure.
    """
    header = json.dumps(
        {
            "block_index": block_index,
            "previous_hash": previous_hash,
            "certificate_hash": certificate_hash,
            "timestamp": timestamp,
            "nonce": nonce,
        },
        sort_keys=True,
    )
    return _sha256(header)


def _get_latest_block(conn: sqlite3.Connection) -> sqlite3.Row:
    return conn.execute(
        "SELECT * FROM blocks ORDER BY block_index DESC LIMIT 1"
    ).fetchone()


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def anchor_certificate(certificate_hash: str) -> dict[str, Any]:
    """
    Add a new block to the ledger anchoring the given certificate_hash.

    Parameters
    ----------
    certificate_hash : str
        SHA-256 hash of the candidate's certificate document.

    Returns
    -------
    dict
        The newly created block record.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        latest = _get_latest_block(conn)
        new_index = latest["block_index"] + 1
        previous_hash = latest["block_hash"]

        block_hash = _compute_block_hash(
            block_index=new_index,
            previous_hash=previous_hash,
            certificate_hash=certificate_hash,
            timestamp=timestamp,
            nonce=0,
        )

        conn.execute(
            """
            INSERT INTO blocks
                (block_index, previous_hash, certificate_hash, block_hash, timestamp, nonce)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (new_index, previous_hash, certificate_hash, block_hash, timestamp, 0),
        )
        conn.commit()
        logger.info(
            "Anchored certificate — block #%d, block_hash=%s",
            new_index,
            block_hash[:16] + "…",
        )

    return {
        "block_index": new_index,
        "previous_hash": previous_hash,
        "certificate_hash": certificate_hash,
        "block_hash": block_hash,
        "timestamp": timestamp,
    }


def verify_certificate(certificate_hash: str) -> dict[str, Any]:
    """
    Verify a certificate_hash by looking it up in the ledger and
    recomputing the block hash to confirm integrity.

    Returns
    -------
    dict
        verified: bool, block_index, block_hash, timestamp (if found).
    """
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM blocks WHERE certificate_hash = ? ORDER BY block_index DESC LIMIT 1",
            (certificate_hash,),
        ).fetchone()

    if row is None:
        logger.info("Certificate not found in ledger: %s", certificate_hash[:16] + "…")
        return {
            "verified": False,
            "certificate_hash": certificate_hash,
            "detail": "Certificate hash not found in ledger.",
        }

    # Recompute and compare
    recomputed_hash = _compute_block_hash(
        block_index=row["block_index"],
        previous_hash=row["previous_hash"],
        certificate_hash=row["certificate_hash"],
        timestamp=row["timestamp"],
        nonce=row["nonce"],
    )

    is_valid = recomputed_hash == row["block_hash"]
    logger.info(
        "Certificate verification — valid=%s, block #%d",
        is_valid,
        row["block_index"],
    )
    return {
        "verified": is_valid,
        "certificate_hash": certificate_hash,
        "block_index": row["block_index"],
        "block_hash": row["block_hash"],
        "recomputed_hash": recomputed_hash,
        "timestamp": row["timestamp"],
        "detail": "Block hash verified." if is_valid else "Block hash mismatch — tampered!",
    }


# ---------------------------------------------------------------------------
# Flask endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health() -> Any:
    """Simple health check."""
    return jsonify({"status": "ok", "service": "blockchain-verification"})


@app.route("/anchor", methods=["POST"])
def anchor() -> Any:
    """
    POST /anchor
    Body: {"certificate_hash": "<sha256_hex>"}

    Adds a new block anchoring the certificate.
    """
    data = request.get_json(force=True, silent=True) or {}
    cert_hash = data.get("certificate_hash", "").strip()

    if not cert_hash or len(cert_hash) != 64:
        return jsonify({"error": "Provide a valid 64-character SHA-256 certificate_hash."}), 400

    result = anchor_certificate(cert_hash)
    return jsonify(result), 201


@app.route("/verify", methods=["GET", "POST"])
def verify() -> Any:
    """
    GET  /verify?certificate_hash=<hex>
    POST /verify  body: {"certificate_hash": "<hex>"}

    Recomputes the block hash for the stored record and returns
    Verified: True/False together with the block timestamp.
    """
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        cert_hash = data.get("certificate_hash", "").strip()
    else:
        cert_hash = request.args.get("certificate_hash", "").strip()

    if not cert_hash:
        return jsonify({"error": "certificate_hash parameter is required."}), 400

    result = verify_certificate(cert_hash)
    status_code = 200 if result["verified"] else 404
    return jsonify(result), status_code


@app.route("/ledger", methods=["GET"])
def ledger() -> Any:
    """
    GET /ledger?limit=20
    Returns the most recent blocks in the ledger (newest first).
    """
    limit = min(int(request.args.get("limit", 20)), 100)
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM blocks ORDER BY block_index DESC LIMIT ?", (limit,)
        ).fetchall()

    return jsonify([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _init_ledger()
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Blockchain Verification microservice on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
