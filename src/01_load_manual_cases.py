"""
Tahap 1 (Manual Workflow): Load Manual Downloaded Cases.

Modul ini menggantikan 01_download_cases.py untuk workflow manual karena
Direktori MA RI memblokir scraping otomatis (HTTP 403).

Workflow:
1. PDF di-download manual oleh user → simpan di data/raw/pdf/case_XXX.pdf
2. Metadata di-input manual di data/raw/metadata.csv
3. Script ini: validasi konsistensi PDF vs metadata, convert ke metadata.json

Output:
- data/raw/metadata.json (versi terstruktur untuk pipeline downstream)
- logs/load_manual.log
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PDF_DIR, RAW_DIR, LOGS_DIR, ensure_dirs, setup_logger


@dataclass
class ManualCaseMetadata:
    """Metadata satu putusan dari manual download."""
    case_id: str
    no_perkara: str
    tanggal_putusan: str
    tahun: int
    pengadilan: str
    klasifikasi: str
    detail_url: str
    pdf_path: Optional[str]
    status: str
    error_msg: Optional[str] = None


def normalize_date(raw: str) -> str:
    """
    Normalisasi tanggal ke format ISO YYYY-MM-DD.

    Handle berbagai format input:
    - DD/MM/YYYY → 2020-02-05
    - D/M/YYYY → 2020-02-05
    - YYYY-MM-DD → unchanged
    """
    if not raw or pd.isna(raw):
        return ""

    s = str(raw).strip()

    # Coba parse berbagai format
    for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d"]:
        try:
            return pd.to_datetime(s, format=fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue

    # Fallback: biarkan pandas tebak
    try:
        return pd.to_datetime(s, dayfirst=True).strftime("%Y-%m-%d")
    except Exception:
        return s


def load_metadata_csv(csv_path: Path, logger) -> list[ManualCaseMetadata]:
    """Baca metadata.csv → validasi → convert ke list dataclass."""
    if not csv_path.exists():
        logger.error(f"metadata.csv tidak ditemukan di {csv_path}")
        return []

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    logger.info(f"Loaded {len(df)} baris dari {csv_path.name}")

    required_cols = ["case_id", "no_perkara", "tanggal_putusan", "tahun",
                     "pengadilan", "klasifikasi", "detail_url"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Kolom tidak lengkap di CSV. Missing: {missing}")
        return []

    cases: list[ManualCaseMetadata] = []
    for _, row in df.iterrows():
        case_id = row["case_id"].strip()
        if not case_id:
            continue

        # Cek apakah PDF ada
        pdf_file = PDF_DIR / f"{case_id}.pdf"
        if pdf_file.exists():
            status = "OK"
            pdf_path_str = str(pdf_file)
            err_msg = None
        else:
            status = "PDF_MISSING"
            pdf_path_str = None
            err_msg = f"File {pdf_file.name} tidak ditemukan di data/raw/pdf/"
            logger.warning(f"{case_id}: {err_msg}")

        # Parse tahun
        try:
            tahun = int(row["tahun"]) if row["tahun"] else 0
        except ValueError:
            tahun = 0
            logger.warning(f"{case_id}: tahun invalid → set 0")

        cases.append(ManualCaseMetadata(
            case_id=case_id,
            no_perkara=row["no_perkara"].strip(),
            tanggal_putusan=normalize_date(row["tanggal_putusan"]),
            tahun=tahun,
            pengadilan=row["pengadilan"].strip(),
            klasifikasi=row["klasifikasi"].strip() or "Wanprestasi",
            detail_url=row["detail_url"].strip(),
            pdf_path=pdf_path_str,
            status=status,
            error_msg=err_msg,
        ))

    return cases


def verify_pdf_inventory(cases: list[ManualCaseMetadata], logger) -> dict:
    """Cek inventory PDF di folder vs metadata."""
    # PDF yang ada di folder
    pdfs_on_disk = {p.stem for p in PDF_DIR.glob("*.pdf")}
    # case_id yang ada di metadata
    cases_in_meta = {c.case_id for c in cases}

    in_meta_not_disk = cases_in_meta - pdfs_on_disk
    on_disk_not_meta = pdfs_on_disk - cases_in_meta

    summary = {
        "total_metadata": len(cases_in_meta),
        "total_pdf": len(pdfs_on_disk),
        "match": len(cases_in_meta & pdfs_on_disk),
        "in_metadata_missing_pdf": sorted(in_meta_not_disk),
        "pdf_without_metadata": sorted(on_disk_not_meta),
    }

    logger.info(f"PDF di folder    : {summary['total_pdf']}")
    logger.info(f"Cases di metadata: {summary['total_metadata']}")
    logger.info(f"Match            : {summary['match']}")

    if in_meta_not_disk:
        logger.warning(f"Metadata tanpa PDF ({len(in_meta_not_disk)}): {list(in_meta_not_disk)[:5]}...")
    if on_disk_not_meta:
        logger.warning(f"PDF tanpa metadata ({len(on_disk_not_meta)}): {list(on_disk_not_meta)[:5]}...")

    return summary


def save_metadata_json(cases: list[ManualCaseMetadata], out_path: Path) -> None:
    """Simpan metadata ke JSON."""
    data = [asdict(c) for c in cases]
    out_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def run() -> list[ManualCaseMetadata]:
    """Eksekusi load manual cases end-to-end."""
    ensure_dirs()
    logger = setup_logger("load_manual", LOGS_DIR / "load_manual.log")

    logger.info("=" * 60)
    logger.info("Load Manual Cases — Workflow Manual Download")
    logger.info("=" * 60)

    csv_path = RAW_DIR / "metadata.csv"
    cases = load_metadata_csv(csv_path, logger)

    if not cases:
        logger.error("Tidak ada cases ter-load. Cek metadata.csv.")
        return []

    summary = verify_pdf_inventory(cases, logger)

    # Save metadata.json
    json_path = RAW_DIR / "metadata.json"
    save_metadata_json(cases, json_path)
    logger.info(f"Metadata tersimpan: {json_path}")

    # Save inventory summary
    summary_path = RAW_DIR / "inventory_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Final status
    ok_count = sum(1 for c in cases if c.status == "OK")
    logger.info("=" * 60)
    logger.info(f"SELESAI. Total cases: {len(cases)}, OK: {ok_count}")
    logger.info("=" * 60)

    return cases


if __name__ == "__main__":
    run()
