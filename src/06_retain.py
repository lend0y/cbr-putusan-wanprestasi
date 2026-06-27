"""
Tahap 5 CBR: Revise & Retain.

Mekanisme untuk menambahkan kasus baru yang sudah diverifikasi
ke dalam case base, sebagai siklus pembelajaran berkelanjutan.

Workflow:
1. Pengguna memberikan kasus baru + verified flag (apakah solusi sudah dicek manual)
2. Validasi: cek duplikasi case_id, validasi field wajib
3. Kalau lolos validasi DAN verified=True → append ke cases.csv
4. Log semua aksi ke logs/retain.log

Function: retain_case(new_case: dict, verified: bool)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROCESSED_DIR, LOGS_DIR, ensure_dirs, setup_logger


REQUIRED_FIELDS = [
    "case_id", "no_perkara", "tanggal_putusan", "tahun",
    "pengadilan", "klasifikasi", "kategori_solusi", "text_retrieval",
]


def _get_logger():
    return setup_logger("retain", LOGS_DIR / "retain.log")


def retain_case(new_case: dict, verified: bool,
                cases_path: Optional[Path] = None) -> dict:
    """
    Tambahkan kasus baru yang sudah diverifikasi ke dalam case base.

    Args:
        new_case: dict berisi metadata + content kasus baru.
                  Minimum harus punya field: case_id, no_perkara,
                  kategori_solusi, text_retrieval, dll.
        verified: True kalau prediksi solusi sudah diverifikasi manual.
                  Jika False, kasus DITOLAK (tidak ditambah ke case base).

    Returns:
        dict {status, message, case_id, new_total} dengan status:
        - "RETAINED"  : berhasil ditambah ke case base
        - "REJECTED"  : verified=False, kasus tidak ditambah
        - "DUPLICATE" : case_id sudah ada
        - "INVALID"   : field wajib tidak lengkap
    """
    ensure_dirs()
    logger = _get_logger()
    cases_path = cases_path or (PROCESSED_DIR / "cases.csv")

    case_id = new_case.get("case_id", "UNKNOWN")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Cek verified flag
    if not verified:
        logger.warning(f"[{timestamp}] {case_id}: REJECTED - belum diverifikasi manual")
        return {
            "status"   : "REJECTED",
            "message"  : "Kasus belum diverifikasi. Verifikasi dulu sebelum retain.",
            "case_id"  : case_id,
            "new_total": None,
        }

    # 2. Validasi field wajib
    missing = [f for f in REQUIRED_FIELDS if f not in new_case or not new_case.get(f)]
    if missing:
        logger.error(f"[{timestamp}] {case_id}: INVALID - missing fields {missing}")
        return {
            "status"   : "INVALID",
            "message"  : f"Field wajib tidak lengkap: {missing}",
            "case_id"  : case_id,
            "new_total": None,
        }

    # 3. Cek duplikasi
    if not cases_path.exists():
        logger.error(f"[{timestamp}] cases.csv tidak ditemukan: {cases_path}")
        return {
            "status"   : "ERROR",
            "message"  : "case base belum di-build (cases.csv tidak ada)",
            "case_id"  : case_id,
            "new_total": None,
        }

    df = pd.read_csv(cases_path)
    if case_id in df["case_id"].values:
        logger.warning(f"[{timestamp}] {case_id}: DUPLICATE - sudah ada di case base")
        return {
            "status"   : "DUPLICATE",
            "message"  : f"case_id '{case_id}' sudah ada di case base.",
            "case_id"  : case_id,
            "new_total": len(df),
        }

    # 4. Append ke case base
    new_row = {col: new_case.get(col, "") for col in df.columns}
    new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    new_df.to_csv(cases_path, index=False, encoding="utf-8")

    logger.info(f"[{timestamp}] {case_id}: RETAINED - case base sekarang {len(new_df)} cases")
    return {
        "status"   : "RETAINED",
        "message"  : f"Kasus {case_id} berhasil ditambahkan.",
        "case_id"  : case_id,
        "new_total": len(new_df),
    }


def demo_retain_workflow() -> list[dict]:
    """
    Demo workflow Revise & Retain dengan 3 skenario:
    1. Kasus baru valid + verified → RETAINED
    2. Kasus baru tidak verified → REJECTED
    3. Kasus duplikat → DUPLICATE
    """
    ensure_dirs()
    logger = _get_logger()
    logger.info("=" * 60)
    logger.info("DEMO: Revise & Retain Workflow")
    logger.info("=" * 60)

    results = []

    # Skenario 1: kasus baru valid + verified
    new_case_1 = {
        "case_id"          : "case_036_demo",
        "no_perkara"       : "999/Pdt.G/2026/PN.DEMO",
        "tanggal_putusan"  : "2026-01-15",
        "tahun"            : 2026,
        "pengadilan"       : "PN DEMO",
        "klasifikasi"      : "Wanprestasi",
        "kategori_solusi"  : "Dikabulkan",
        "text_retrieval"   : "Demo case untuk testing retain. Tergugat tidak melaksanakan kewajiban...",
        "pasal"            : "1243",
        "n_pasal"          : 1,
        "pihak_penggugat"  : "Penggugat [D.E.M.]",
        "pihak_tergugat"   : "Tergugat [O.N.E.]",
        "ringkasan_fakta"  : "Penggugat menggugat tergugat karena wanprestasi...",
        "argumen_hukum_utama": "Pertimbangan hukum demo...",
        "amar_putusan"     : "Mengabulkan gugatan penggugat...",
        "word_count"       : 500,
        "extraction_quality": "OK",
        "missing_fields"   : "",
        "detail_url"       : "https://demo.example.com",
        "text_clean_path"  : "data/cleaned/case_036_demo.txt",
    }
    r1 = retain_case(new_case_1, verified=True)
    logger.info(f"Skenario 1 (valid + verified): {r1['status']} - {r1['message']}")
    results.append({"skenario": "valid + verified", **r1})

    # Skenario 2: tidak verified
    new_case_2 = dict(new_case_1)
    new_case_2["case_id"] = "case_037_demo"
    r2 = retain_case(new_case_2, verified=False)
    logger.info(f"Skenario 2 (not verified): {r2['status']} - {r2['message']}")
    results.append({"skenario": "not verified", **r2})

    # Skenario 3: case_id duplikat
    r3 = retain_case(new_case_1, verified=True)   # case_id sama dengan #1
    logger.info(f"Skenario 3 (duplicate): {r3['status']} - {r3['message']}")
    results.append({"skenario": "duplicate", **r3})

    # Cleanup demo: hapus case_036_demo dari cases.csv supaya tidak mencemari evaluasi
    df = pd.read_csv(PROCESSED_DIR / "cases.csv")
    df_clean = df[~df["case_id"].astype(str).str.endswith("_demo")]
    if len(df_clean) < len(df):
        df_clean.to_csv(PROCESSED_DIR / "cases.csv", index=False, encoding="utf-8")
        logger.info(f"Cleanup demo: removed {len(df) - len(df_clean)} demo cases")

    logger.info("=" * 60)
    logger.info("DEMO selesai. Lihat logs/retain.log untuk jejak lengkap.")
    logger.info("=" * 60)
    return results


if __name__ == "__main__":
    demo_retain_workflow()
