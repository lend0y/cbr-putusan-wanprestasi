"""
Tahap 2: Ekstraksi teks dari PDF + Cleaning/Preprocessing.

Modul ini melakukan:
1. Membaca PDF dari data/raw/pdf/ → ekstrak teks → simpan ke data/raw/text/
2. Membersihkan teks: hapus header/footer berulang, nomor halaman, watermark
3. Normalisasi unicode, whitespace
4. Validasi: minimal 80% isi terbaca (proxy: word count >= 500)
5. Logging proses cleaning ke logs/cleaning.log

Output:
- data/raw/text/case_XXX.txt   (raw extracted)
- data/cleaned/case_XXX.txt    (cleaned + normalized)
- logs/cleaning.log
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    PDF_DIR, TEXT_DIR, CLEAN_DIR, RAW_DIR, LOGS_DIR,
    ensure_dirs, setup_logger,
)


MIN_WORD_COUNT = 500


@dataclass
class ProcessingResult:
    """Hasil pemrosesan satu PDF."""
    case_id: str
    pdf_file: str
    text_file: Optional[str]
    cleaned_file: Optional[str]
    word_count_raw: int
    word_count_clean: int
    extraction_status: str
    validation_status: str
    notes: Optional[str] = None


# ─── Ekstraksi PDF ─────────────────────────────────────────────────

def pdf_to_text(pdf_path: Path) -> tuple[str, str]:
    """
    Ekstrak teks dari PDF.

    Returns:
        (text, status) — status: "OK" | "EMPTY" | "ERROR"
    """
    try:
        text = extract_text(str(pdf_path))
        if not text or len(text.strip()) < 100:
            return "", "EMPTY"
        return text, "OK"
    except PDFSyntaxError as e:
        return "", f"ERROR: PDF syntax — {e}"
    except Exception as e:
        return "", f"ERROR: {e}"


# ─── Cleaning Functions ────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """Normalisasi karakter unicode (smart quotes, dll)."""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"',
        "\u2013": "-", "\u2014": "-", "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def remove_page_numbers(text: str) -> str:
    """Hapus pola nomor halaman."""
    text = re.sub(r"^\s*-?\s*\d{1,3}\s*-?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*Halaman\s+\d+\s+dari\s+\d+.*$", "", text,
                  flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^\s*Hal\.?\s*\d+.*$", "", text,
                  flags=re.MULTILINE | re.IGNORECASE)
    return text


def remove_repeated_headers_footers(text: str) -> str:
    """
    Hapus header/footer berulang yang sering muncul di putusan MA RI.

    Pattern yang umum:
    - "Disclaimer: Kepaniteraan Mahkamah Agung..."
    - "Direktori Putusan Mahkamah Agung Republik Indonesia"
    - "putusan.mahkamahagung.go.id"
    """
    patterns = [
        r"Disclaimer\s*\n?Kepaniteraan Mahkamah Agung.*?(?=\n\s*\n|\Z)",
        r"Mahkamah Agung Republik Indonesia\s*$",
        r"Direktori Putusan Mahkamah Agung Republik Indonesia\s*$",
        r"putusan\.mahkamahagung\.go\.id\s*$",
        r"^\s*Email\s*:\s*kepaniteraan@mahkamahagung\.go\.id.*$",
        r"^\s*Telp\s*:.*$",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)
    return text


def normalize_whitespace(text: str) -> str:
    """Normalisasi whitespace berlebih."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Pipeline pembersihan teks lengkap."""
    text = normalize_unicode(text)
    text = remove_page_numbers(text)
    text = remove_repeated_headers_footers(text)
    text = normalize_whitespace(text)
    return text


# ─── Validasi ──────────────────────────────────────────────────────

def validate_text(text: str, min_words: int = MIN_WORD_COUNT) -> tuple[str, str]:
    """
    Validasi kualitas teks hasil cleaning.

    Returns:
        (status, notes)
        status: "VALID" | "TOO_SHORT" | "SUSPICIOUS"
    """
    word_count = len(text.split())

    if word_count < min_words:
        return "TOO_SHORT", f"Hanya {word_count} kata (kemungkinan PDF scan)"

    # Cek apakah ada marker putusan standar
    required_markers = ["menimbang", "mengadili", "amar"]
    text_lower = text.lower()
    found = [m for m in required_markers if m in text_lower]
    if len(found) < 2:
        return "SUSPICIOUS", f"Marker putusan tidak lengkap: {found}"

    return "VALID", None


# ─── Pipeline Pemrosesan ───────────────────────────────────────────

def process_pdf(pdf_path: Path, logger) -> ProcessingResult:
    """Proses satu file PDF end-to-end."""
    case_id = pdf_path.stem.split("_")[0] + "_" + pdf_path.stem.split("_")[1]

    raw_text, status = pdf_to_text(pdf_path)
    if status != "OK":
        logger.warning(f"{case_id}: ekstraksi gagal — {status}")
        return ProcessingResult(
            case_id=case_id, pdf_file=str(pdf_path),
            text_file=None, cleaned_file=None,
            word_count_raw=0, word_count_clean=0,
            extraction_status=status, validation_status="N/A",
            notes="Tidak dapat ekstraksi teks",
        )

    text_file = TEXT_DIR / f"{case_id}.txt"
    text_file.write_text(raw_text, encoding="utf-8")
    word_count_raw = len(raw_text.split())

    cleaned = clean_text(raw_text)
    cleaned_file = CLEAN_DIR / f"{case_id}.txt"
    cleaned_file.write_text(cleaned, encoding="utf-8")
    word_count_clean = len(cleaned.split())

    val_status, notes = validate_text(cleaned)
    logger.info(
        f"{case_id}: extract=OK, clean={word_count_clean}w, validation={val_status}"
    )

    return ProcessingResult(
        case_id=case_id,
        pdf_file=str(pdf_path),
        text_file=str(text_file),
        cleaned_file=str(cleaned_file),
        word_count_raw=word_count_raw,
        word_count_clean=word_count_clean,
        extraction_status="OK",
        validation_status=val_status,
        notes=notes,
    )


def process_all() -> list[ProcessingResult]:
    """Proses semua PDF di data/raw/pdf/."""
    ensure_dirs()
    logger = setup_logger("cleaner", LOGS_DIR / "cleaning.log")
    logger.info("=" * 60)
    logger.info("Mulai ekstraksi + cleaning")
    logger.info("=" * 60)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.error("Tidak ada PDF di data/raw/pdf/. Jalankan scraping dulu.")
        return []

    logger.info(f"Ditemukan {len(pdf_files)} PDF untuk diproses")
    results: list[ProcessingResult] = []

    for pdf_path in tqdm(pdf_files, desc="Extracting"):
        result = process_pdf(pdf_path, logger)
        results.append(result)

    # Simpan ringkasan
    summary_file = RAW_DIR / "processing_summary.json"
    summary_file.write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Statistik
    total = len(results)
    valid = sum(1 for r in results if r.validation_status == "VALID")
    too_short = sum(1 for r in results if r.validation_status == "TOO_SHORT")
    failed = sum(1 for r in results if r.extraction_status != "OK")

    logger.info("=" * 60)
    logger.info(f"SELESAI. Total: {total}")
    logger.info(f"  VALID         : {valid}")
    logger.info(f"  TOO_SHORT     : {too_short}")
    logger.info(f"  EXTRACT_FAILED: {failed}")
    logger.info("=" * 60)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract + Clean PDF putusan")
    parser.add_argument("--min-words", type=int, default=MIN_WORD_COUNT,
                        help="Minimum word count untuk dianggap valid")
    args = parser.parse_args()

    global MIN_WORD_COUNT
    MIN_WORD_COUNT = args.min_words

    process_all()


if __name__ == "__main__":
    main()
