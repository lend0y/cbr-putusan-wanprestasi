"""
Tahap 2 CBR: Case Representation (v2 - Robust Edition).

Versi ini memperbaiki masalah extraction rate rendah pada v1 dengan:
1. Pendekatan POSITIONAL HEURISTIC (split teks jadi 3 zona: opening, middle, tail)
2. Klasifikasi solusi dari TAIL text, bukan dari amar section yang mungkin gagal di-extract
3. Quality definition lebih masuk akal: hanya pasal & kategori_solusi yang critical
4. Section extraction punya fallback positional kalau regex marker gagal

Output:
- data/processed/cases.csv
- data/processed/cases.json
- logs/representation.log
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import RAW_DIR, CLEAN_DIR, PROCESSED_DIR, LOGS_DIR, ensure_dirs, setup_logger


@dataclass
class CaseRecord:
    case_id: str
    no_perkara: str
    tanggal_putusan: str
    tahun: int
    pengadilan: str
    klasifikasi: str
    detail_url: str

    pasal: str
    n_pasal: int
    pihak_penggugat: str
    pihak_tergugat: str

    ringkasan_fakta: str
    argumen_hukum_utama: str
    amar_putusan: str
    kategori_solusi: str

    word_count: int
    text_retrieval: str
    text_clean_path: str

    extraction_quality: str = "OK"
    missing_fields: str = ""


def anonymize_party(name: str, role: str) -> str:
    name = (name or "").strip()
    if not name:
        return f"{role} [anonim]"
    cleaned = re.sub(r"[^\w\s.,]", " ", name)
    tokens = [t for t in cleaned.split() if len(t) > 1][:3]
    if not tokens:
        return f"{role} [anonim]"
    initials = ".".join(t[0].upper() for t in tokens) + "."
    return f"{role} [{initials}]"


def extract_pasal(text: str) -> list[str]:
    pattern = r"[Pp]asal\s+(\d{1,4}(?:\s*ayat\s*\(?\d+\)?)?)"
    matches = re.findall(pattern, text)
    seen, result = set(), []
    for m in matches:
        clean = m.strip()
        if clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result[:20]


def extract_party_names(text: str) -> tuple[str, str]:
    head = text[:6000]
    patterns_penggugat = [
        r"penggugat[^\w]*?:?\s*([A-Z][A-Za-z0-9.\s,]{2,80})",
        r"pemohon\s+kasasi[^\w]*?:?\s*([A-Z][A-Za-z0-9.\s,]{2,80})",
        r"pembanding[^\w]*?:?\s*([A-Z][A-Za-z0-9.\s,]{2,80})",
    ]
    patterns_tergugat = [
        r"tergugat[^\w]*?:?\s*([A-Z][A-Za-z0-9.\s,]{2,80})",
        r"termohon\s+kasasi[^\w]*?:?\s*([A-Z][A-Za-z0-9.\s,]{2,80})",
        r"terbanding[^\w]*?:?\s*([A-Z][A-Za-z0-9.\s,]{2,80})",
    ]

    def first_match(patterns: list[str]) -> str:
        for pat in patterns:
            m = re.search(pat, head, re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                name = re.split(
                    r"\s+(?:beralamat|warga|tempat|alamat|umur|pekerjaan|melawan|lawan|kewarganegaraan|berkedudukan|selanjutnya|\;|\n)",
                    name,
                    maxsplit=1,
                )[0].strip()
                if 2 < len(name) < 80:
                    return name
        return ""

    p_raw = first_match(patterns_penggugat)
    t_raw = first_match(patterns_tergugat)

    return (
        anonymize_party(p_raw, "Penggugat"),
        anonymize_party(t_raw, "Tergugat"),
    )


def _split_zones(text: str) -> tuple[str, str, str]:
    """Split teks jadi opening (0-30%), middle (30-70%), tail (70-100%)."""
    n = len(text)
    return (
        text[: int(n * 0.30)],
        text[int(n * 0.30) : int(n * 0.70)],
        text[int(n * 0.70) :],
    )


def extract_facts(text: str) -> str:
    """Ekstrak ringkasan fakta. Fallback ke opening zone."""
    text_lower = text.lower()
    start_patterns = [
        r"tentang\s+duduk(?:nya)?\s+perkara",
        r"duduk\s+perkara",
        r"posita\s+gugatan",
        r"menimbang,?\s+bahwa\s+penggugat",
        r"menimbang,?\s+bahwa\s+pembanding",
        r"menimbang,?\s+bahwa\s+pemohon",
    ]
    end_patterns = [
        r"\n\s*tentang\s+pertimbangan",
        r"\n\s*pertimbangan\s+hukum",
        r"\n\s*mengadili",
        r"\n\s*m\s*e\s*n\s*g\s*a\s*d\s*i\s*l\s*i",
    ]

    start_pos = -1
    for pat in start_patterns:
        m = re.search(pat, text_lower)
        if m:
            start_pos = m.end()
            break

    if start_pos >= 0:
        end_pos = start_pos + 4000
        for pat in end_patterns:
            m = re.search(pat, text_lower[start_pos:])
            if m:
                end_pos = start_pos + m.start()
                break
        section = text[start_pos:end_pos].strip()
        section = re.sub(r"\n{2,}", "\n", section)
        if len(section.split()) > 50:
            return section[:4000]

    opening, _, _ = _split_zones(text)
    fallback = opening[1000:] if len(opening) > 1500 else opening
    return re.sub(r"\n{2,}", "\n", fallback).strip()[:4000]


def extract_legal_reasoning(text: str) -> str:
    """Ekstrak argumen hukum. Fallback ke middle zone."""
    text_lower = text.lower()
    start_patterns = [
        r"tentang\s+pertimbangan\s+hukum",
        r"pertimbangan\s+hukum",
        r"tentang\s+hukum",
        r"menimbang,?\s+bahwa\s+maksud",
        r"menimbang,?\s+bahwa\s+majelis",
        r"menimbang,?\s+bahwa\s+terhadap",
    ]
    end_patterns = [
        r"\n\s*mengadili[^a-z]",
        r"\n\s*memutuskan[^a-z]",
        r"\n\s*m\s*e\s*n\s*g\s*a\s*d\s*i\s*l\s*i",
        r"\n\s*demikianlah\s+diputuskan",
    ]

    start_pos = -1
    for pat in start_patterns:
        m = re.search(pat, text_lower)
        if m:
            start_pos = m.end()
            break

    if start_pos >= 0:
        end_pos = start_pos + 5000
        for pat in end_patterns:
            m = re.search(pat, text_lower[start_pos:])
            if m:
                end_pos = start_pos + m.start()
                break
        section = text[start_pos:end_pos].strip()
        section = re.sub(r"\n{2,}", "\n", section)
        if len(section.split()) > 100:
            return section[:5000]

    _, middle, _ = _split_zones(text)
    return re.sub(r"\n{2,}", "\n", middle).strip()[:5000]


def extract_verdict(text: str) -> str:
    """Ekstrak amar putusan dari occurrence TERAKHIR (di akhir dokumen)."""
    text_lower = text.lower()

    positions = [
        m.start() for m in re.finditer(r"\bmengadili\b|\bmemutuskan\b", text_lower)
    ]
    if positions:
        start_pos = positions[-1]  # YANG TERAKHIR, bukan pertama!
        end_pos = min(start_pos + 2500, len(text))
        tail_text = text[start_pos:end_pos]
        m = re.search(r"demikianlah?\s+diputus", tail_text, re.IGNORECASE)
        if m:
            end_pos = start_pos + m.start()
        section = text[start_pos:end_pos].strip()
        section = re.sub(r"\n{2,}", "\n", section)
        if len(section.split()) > 20:
            return section[:2500]

    _, _, tail = _split_zones(text)
    return re.sub(r"\n{2,}", "\n", tail).strip()[:2500]


def classify_solution(amar: str, full_text: str) -> str:
    """
    Klasifikasi solusi dengan strategi:
    Search di TAIL 40% terakhir dari teks (di sinilah amar selalu berada).
    """
    full_lower = full_text.lower()
    tail_start = int(len(full_lower) * 0.6)
    search_text = full_lower[tail_start:]

    # 0a. Perdamaian — cek paling awal
    if re.search(r"akta\s+perdamaian|kesepakatan\s+perdamaian", search_text):
        return "Perdamaian"
    if re.search(r"menghukum.{0,80}mentaati\s+kesepakatan", search_text):
        return "Perdamaian"

    # 0b. Dicabut
    if re.search(r"penetapan\s+pencabutan|pencabutan\s+(perkara|gugatan)", search_text):
        return "Dicabut"
    if re.search(r"mengabulkan.{0,80}pencabutan", search_text):
        return "Dicabut"
    if re.search(r"menyatakan.{0,80}gugatan.{0,30}dicabut", search_text):
        return "Dicabut"
    if re.search(r"perkara.{0,30}dicabut", search_text):
        return "Dicabut"

    # 1. Tidak Dapat Diterima (NO)
    if re.search(r"tidak\s+dapat\s+diterima|niet\s+ontvankelijk", search_text):
        return "Tidak Dapat Diterima"

    # ... sisanya (Dikabulkan Sebagian, Dikabulkan, Ditolak) tetap sama


def build_case_record(meta: dict, logger) -> Optional[CaseRecord]:
    case_id = meta["case_id"]
    if meta.get("status") != "OK":
        return None

    cleaned_path = CLEAN_DIR / f"{case_id}.txt"
    if not cleaned_path.exists():
        logger.warning(f"{case_id}: cleaned text tidak ditemukan")
        return None

    text = cleaned_path.read_text(encoding="utf-8")
    word_count = len(text.split())
    if word_count < 500:
        logger.warning(f"{case_id}: skip — terlalu pendek ({word_count} kata)")
        return None

    pasal_list = extract_pasal(text)
    penggugat, tergugat = extract_party_names(text)
    fakta = extract_facts(text)
    pertimbangan = extract_legal_reasoning(text)
    amar = extract_verdict(text)
    kategori = classify_solution(amar, text)

    # Quality assessment baru: hanya pasal & kategori yang critical
    critical_missing = []
    if not pasal_list:
        critical_missing.append("pasal")
    if kategori == "Tidak Teridentifikasi":
        critical_missing.append("kategori_solusi")

    soft_missing = []
    if not fakta or len(fakta.split()) < 30:
        soft_missing.append("ringkasan_fakta")
    if not pertimbangan or len(pertimbangan.split()) < 50:
        soft_missing.append("argumen_hukum_utama")
    if not amar or len(amar.split()) < 10:
        soft_missing.append("amar_putusan")

    all_missing = critical_missing + soft_missing
    if not critical_missing:
        quality = "OK"
    elif len(critical_missing) == 1:
        quality = "PARTIAL"
    else:
        quality = "FAILED"

    if all_missing:
        logger.info(f"{case_id}: quality={quality}, missing={all_missing}")

    text_retrieval = text  # full clean text, paling robust untuk TF-IDF

    return CaseRecord(
        case_id=case_id,
        no_perkara=meta.get("no_perkara", ""),
        tanggal_putusan=meta.get("tanggal_putusan", ""),
        tahun=meta.get("tahun", 0),
        pengadilan=meta.get("pengadilan", ""),
        klasifikasi=meta.get("klasifikasi", "Wanprestasi"),
        detail_url=meta.get("detail_url", ""),
        pasal=", ".join(pasal_list),
        n_pasal=len(pasal_list),
        pihak_penggugat=penggugat,
        pihak_tergugat=tergugat,
        ringkasan_fakta=fakta,
        argumen_hukum_utama=pertimbangan,
        amar_putusan=amar,
        kategori_solusi=kategori,
        word_count=word_count,
        text_retrieval=text_retrieval,
        text_clean_path=str(cleaned_path),
        extraction_quality=quality,
        missing_fields=", ".join(all_missing),
    )


def run() -> list[CaseRecord]:
    ensure_dirs()
    logger = setup_logger("representation", LOGS_DIR / "representation.log")
    logger.info("=" * 60)
    logger.info("Tahap 2 CBR: Case Representation (v2 - Robust)")
    logger.info("=" * 60)

    meta_path = RAW_DIR / "metadata.json"
    if not meta_path.exists():
        logger.error(f"metadata.json tidak ditemukan di {meta_path}")
        return []

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    logger.info(f"Memproses {len(metadata)} metadata entries...")

    records: list[CaseRecord] = []
    for meta in tqdm(metadata, desc="Building cases"):
        rec = build_case_record(meta, logger)
        if rec is not None:
            records.append(rec)

    if not records:
        logger.error("Tidak ada record berhasil dibuat.")
        return []

    save_outputs(records, logger)
    print_summary(records, logger)
    return records


def save_outputs(records: list[CaseRecord], logger) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    json_path = PROCESSED_DIR / "cases.json"
    json_path.write_text(
        json.dumps([asdict(r) for r in records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Tersimpan: {json_path}")

    csv_path = PROCESSED_DIR / "cases.csv"
    df = pd.DataFrame([asdict(r) for r in records])
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"Tersimpan: {csv_path}")


def print_summary(records: list[CaseRecord], logger) -> None:
    n = len(records)
    quality_count: dict[str, int] = {}
    kategori_count: dict[str, int] = {}
    for r in records:
        q = r.extraction_quality or "UNKNOWN"
        k = r.kategori_solusi or "UNKNOWN"
        quality_count[q] = quality_count.get(q, 0) + 1
        kategori_count[k] = kategori_count.get(k, 0) + 1

    logger.info("=" * 60)
    logger.info(f"SELESAI. Total cases: {n}")
    for q, cnt in sorted(quality_count.items()):
        logger.info(f"  Quality {str(q):8s}: {cnt}")
    logger.info("Distribusi kategori_solusi:")
    for kat, cnt in sorted(kategori_count.items(), key=lambda x: -x[1]):
        logger.info(f"  {str(kat):30s}: {cnt}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
