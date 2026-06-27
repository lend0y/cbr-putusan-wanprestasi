"""
Tahap 4 CBR: Case Solution Reuse.

Menggunakan top-k kasus termirip untuk memprediksi solusi (kategori_solusi)
kasus baru dengan dua strategi:

1. Majority Vote     - Pilih kategori yang paling sering muncul di top-k
2. Weighted Similarity - Bobot kategori berdasarkan similarity_score

Output:
- data/results/predictions.csv
- logs/predict.log
"""

from __future__ import annotations

import json
import sys
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import EVAL_DIR, RESULTS_DIR, LOGS_DIR, ensure_dirs, setup_logger


# ─── Retriever Loader (lazy) ─────────────────────────────────────────

def _load_module(name: str, path: Path):
    """Load module dengan nama file yang diawali angka."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_RETRIEVER = None


def _ensure_retriever():
    """Lazy-init CBRRetriever dari 04_retrieval.py."""
    global _RETRIEVER
    if _RETRIEVER is None:
        retr_path = Path(__file__).parent / "04_retrieval.py"
        retr_mod = _load_module("retr_mod_for_predict", retr_path)
        _RETRIEVER = retr_mod.CBRRetriever()
    return _RETRIEVER


# ─── Prediction Strategies ───────────────────────────────────────────

def majority_vote(retrieved: list[dict]) -> tuple[str, float, str]:
    """
    Majority Vote: pilih kategori_solusi yang paling sering muncul di top-k.

    Returns:
        (predicted_label, confidence, reasoning_text)
    """
    kategoris = [r["kategori_solusi"] for r in retrieved if r.get("kategori_solusi")]
    if not kategoris:
        return "Tidak Teridentifikasi", 0.0, "Top-k tidak punya label."

    counter = Counter(kategoris)
    predicted, count = counter.most_common(1)[0]
    confidence = count / len(kategoris)

    breakdown = ", ".join(f"{cnt}x {kat}" for kat, cnt in counter.most_common())
    reasoning = f"Dari {len(kategoris)} kasus top-k: {breakdown}. Mayoritas = {predicted}."
    return predicted, round(confidence, 4), reasoning


def weighted_similarity(retrieved: list[dict]) -> tuple[str, float, str]:
    """
    Weighted Similarity: bobot kategori berdasarkan similarity_score.
    Kategori dengan total bobot tertinggi menjadi prediksi.

    Returns:
        (predicted_label, confidence, reasoning_text)
    """
    weights: dict[str, float] = defaultdict(float)
    total_sim = 0.0

    for r in retrieved:
        kat = r.get("kategori_solusi") or "Tidak Teridentifikasi"
        sim = float(r.get("similarity_score", 0.0))
        weights[kat] += sim
        total_sim += sim

    if not weights or total_sim == 0:
        return "Tidak Teridentifikasi", 0.0, "Tidak ada similarity score yang valid."

    predicted = max(weights.items(), key=lambda x: x[1])[0]
    confidence = weights[predicted] / total_sim

    sorted_w = sorted(weights.items(), key=lambda x: -x[1])
    breakdown = ", ".join(f"{kat}={w:.3f}" for kat, w in sorted_w)
    reasoning = f"Bobot per kategori: {breakdown}. Tertinggi = {predicted} ({confidence:.2%})."
    return predicted, round(confidence, 4), reasoning


# ─── Main API ────────────────────────────────────────────────────────

def predict_outcome(query: str, k: int = 5,
                     method: str = "weighted_similarity") -> dict:
    """
    Prediksi kategori_solusi untuk kasus baru berdasarkan top-k retrieval.

    Args:
        query: deskripsi kasus baru (string)
        k: jumlah top kasus yang digunakan (default 5)
        method: 'majority_vote' atau 'weighted_similarity'

    Returns:
        dict berisi: predicted_solution, confidence_score, prediction_method,
        top_k_case_ids, similarity_scores, top_k_kategoris, reasoning, disclaimer.
    """
    retriever = _ensure_retriever()
    retrieved = retriever.retrieve(query, k=k)

    if method == "majority_vote":
        predicted, confidence, reasoning = majority_vote(retrieved)
    elif method == "weighted_similarity":
        predicted, confidence, reasoning = weighted_similarity(retrieved)
    else:
        raise ValueError(f"Unknown method: {method}. "
                         f"Gunakan 'majority_vote' atau 'weighted_similarity'.")

    return {
        "query"             : query,
        "predicted_solution": predicted,
        "confidence_score"  : confidence,
        "prediction_method" : method,
        "top_k_case_ids"    : [r["case_id"] for r in retrieved],
        "similarity_scores" : [r["similarity_score"] for r in retrieved],
        "top_k_kategoris"   : [r["kategori_solusi"] for r in retrieved],
        "reasoning"         : reasoning,
        "disclaimer"        : (
            "Hasil prediksi ini merupakan output sistem akademik berbasis CBR, "
            "BUKAN keputusan hukum final. Verifikasi oleh ahli hukum diperlukan."
        ),
    }


# ─── Batch Runner ────────────────────────────────────────────────────

def run() -> Optional[pd.DataFrame]:
    """Eksekusi predict_outcome pada semua query di queries.json."""
    ensure_dirs()
    logger = setup_logger("predict", LOGS_DIR / "predict.log")
    logger.info("=" * 60)
    logger.info("Tahap 4 CBR: Solution Reuse")
    logger.info("=" * 60)

    queries_path = EVAL_DIR / "queries.json"
    if not queries_path.exists():
        logger.error(f"queries.json tidak ditemukan: {queries_path}")
        return None

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(queries)} queries")

    retriever = _ensure_retriever()
    logger.info(f"Retriever ready ({len(retriever.df)} cases di base)")

    rows = []
    for q in queries:
        for method in ["majority_vote", "weighted_similarity"]:
            result = predict_outcome(q["query_text"], k=5, method=method)
            actual = q.get("expected_kategori", "")
            match = "OK" if result["predicted_solution"] == actual else "MISS"

            rows.append({
                "query_id"         : q["query_id"],
                "query_text"       : q["query_text"][:200],
                "predicted_solution": result["predicted_solution"],
                "actual_solution"  : actual,
                "match"            : match,
                "top_5_case_ids"   : ", ".join(result["top_k_case_ids"]),
                "similarity_scores": ", ".join(f"{s:.4f}" for s in result["similarity_scores"]),
                "prediction_method": method,
                "confidence_score" : result["confidence_score"],
                "source"           : q.get("source", ""),
                "notes"            : result["reasoning"],
            })

            logger.info(f"{q['query_id']:6s} [{method:20s}] pred={result['predicted_solution']:25s} "
                        f"actual={actual:25s} [{match}]")

    df = pd.DataFrame(rows)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "predictions.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"\nSaved: {output_path}")

    print_summary(df, logger)
    return df


def print_summary(df: pd.DataFrame, logger) -> None:
    """Print ringkasan perbandingan kedua metode."""
    logger.info("=" * 60)
    logger.info("PERBANDINGAN ACCURACY (hanya untuk query dengan ground truth)")
    logger.info("=" * 60)

    for method in df["prediction_method"].unique():
        sub = df[df["prediction_method"] == method]
        # Hanya hitung query yang punya ground truth (synthetic)
        with_gt = sub[(sub["actual_solution"] != "") & (sub["source"] == "synthetic")]
        if len(with_gt) > 0:
            hits = (with_gt["match"] == "OK").sum()
            total = len(with_gt)
            acc = hits / total
            logger.info(f"  {method:22s}: {hits}/{total} ({acc:.2%}) accuracy on synthetic queries")

    logger.info("=" * 60)


if __name__ == "__main__":
    run()
