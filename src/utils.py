"""
Utility functions untuk proyek CBR Putusan Wanprestasi.

Modul ini menyediakan helper functions yang digunakan di berbagai tahap:
- Setup logger
- Path management
- Sanitasi filename
- Safe HTTP requests dengan retry
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ─── Path Constants ──────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
PDF_DIR      = RAW_DIR / "pdf"
HTML_DIR     = RAW_DIR / "html"
TEXT_DIR     = RAW_DIR / "text"
CLEAN_DIR    = DATA_DIR / "cleaned"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR     = DATA_DIR / "eval"
RESULTS_DIR  = DATA_DIR / "results"
LOGS_DIR     = PROJECT_ROOT / "logs"


def ensure_dirs() -> None:
    """Pastikan semua folder data dan logs sudah ada."""
    for d in (PDF_DIR, HTML_DIR, TEXT_DIR, CLEAN_DIR, PROCESSED_DIR,
              EVAL_DIR, RESULTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ─── Logger Setup ────────────────────────────────────────────────────

def setup_logger(name: str, log_file: Optional[Path] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """Setup logger dengan format yang konsisten dan output ke file + console."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─── Filename Sanitization ───────────────────────────────────────────

def sanitize_filename(text: str, max_len: int = 100) -> str:
    """
    Sanitasi string menjadi nama file yang aman.

    Hilangkan karakter berbahaya, normalisasi unicode, batasi panjang.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "_", text).strip("_")
    return text[:max_len] if text else "unnamed"


def slug_no_perkara(no_perkara: str) -> str:
    """Konversi nomor perkara menjadi slug yang aman untuk filename."""
    return sanitize_filename(no_perkara, max_len=80)


# ─── HTTP Session dengan Retry ───────────────────────────────────────

def create_session(
    user_agent: str = "Mozilla/5.0 (Academic Research; CBR-UMM)",
    total_retries: int = 3,
    backoff_factor: float = 2.0,
) -> requests.Session:
    """
    Buat HTTP session dengan retry policy dan user-agent yang sesuai.

    Retry policy: retry pada status 429, 500, 502, 503, 504.
    Backoff exponential: 2s, 4s, 8s.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
    })

    retry_strategy = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def polite_sleep(seconds: float = 3.0) -> None:
    """Sleep untuk rate limiting yang etis."""
    time.sleep(seconds)
