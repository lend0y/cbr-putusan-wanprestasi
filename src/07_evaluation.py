"""
Tahap 6 CBR: Model Evaluation.

Evaluasi komprehensif untuk retrieval dan prediksi:

RETRIEVAL METRICS:
- Precision@k, Recall@k, F1@k
- Hit Rate@k
- Mean Reciprocal Rank (MRR)
- Top-1 Accuracy

PREDICTION METRICS:
- Accuracy, Precision, Recall, F1 (weighted + macro)
- Confusion Matrix
- Per-class metrics

ERROR ANALYSIS:
- Failed queries dengan reasoning
- Pola kegagalan (query pendek, class imbalance, dll)
- Rekomendasi perbaikan

Output:
- data/eval/retrieval_metrics.csv
- data/eval/prediction_metrics.csv
- data/eval/error_analysis.csv
- reports/figures/*.png
"""

from __future__ import annotations

import json
import sys
import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import EVAL_DIR, PROCESSED_DIR, RESULTS_DIR, LOGS_DIR, ensure_dirs, setup_logger


# ─── Lazy loaders ───────────────────────────────────────────────────

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_RETRIEVER = None


def _ensure_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        retr_path = Path(__file__).parent / "04_retrieval.py"
        retr_mod = _load_module("retr_for_eval", retr_path)
        _RETRIEVER = retr_mod.CBRRetriever()
    return _RETRIEVER


# ─── Retrieval Evaluation ────────────────────────────────────────────

def eval_retrieval(queries: list[dict], k: int = 5) -> dict:
    """
    Evaluasi performa retrieval pada query uji.

    Metrics dihitung hanya untuk query yang punya ground_truth_case_ids.

    Returns:
        dict berisi rata-rata metrics + per-query details.
    """
    retriever = _ensure_retriever()
    per_query = []

    p_at_k_list = []
    r_at_k_list = []
    hit_list    = []
    rr_list     = []
    top1_list   = []

    for q in queries:
        gt = set(q.get("ground_truth_case_ids", []))
        if not gt:
            # Skip query tanpa ground truth (manual queries)
            continue

        retrieved = retriever.retrieve(q["query_text"], k=k)
        retrieved_ids = [r["case_id"] for r in retrieved]

        # Precision@k
        relevant_in_topk = set(retrieved_ids) & gt
        p_at_k = len(relevant_in_topk) / k

        # Recall@k
        r_at_k = len(relevant_in_topk) / len(gt) if gt else 0.0

        # F1@k
        f1_at_k = (2 * p_at_k * r_at_k / (p_at_k + r_at_k)) if (p_at_k + r_at_k) > 0 else 0.0

        # Hit Rate@k
        hit = 1 if relevant_in_topk else 0

        # MRR
        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in gt:
                rr = 1.0 / rank
                break

        # Top-1
        top1 = 1 if retrieved_ids[0] in gt else 0

        per_query.append({
            "query_id"          : q["query_id"],
            "ground_truth"      : ", ".join(sorted(gt)),
            "retrieved_top_k"   : ", ".join(retrieved_ids),
            "precision_at_k"    : round(p_at_k, 4),
            "recall_at_k"       : round(r_at_k, 4),
            "f1_at_k"           : round(f1_at_k, 4),
            "hit_rate_at_k"     : hit,
            "reciprocal_rank"   : round(rr, 4),
            "top1_accuracy"     : top1,
        })

        p_at_k_list.append(p_at_k)
        r_at_k_list.append(r_at_k)
        hit_list.append(hit)
        rr_list.append(rr)
        top1_list.append(top1)

    if not per_query:
        return {"average": None, "per_query": []}

    average = {
        "k"               : k,
        "n_queries"       : len(per_query),
        "precision_at_k"  : round(float(np.mean(p_at_k_list)), 4),
        "recall_at_k"     : round(float(np.mean(r_at_k_list)), 4),
        "hit_rate_at_k"   : round(float(np.mean(hit_list)), 4),
        "mrr"             : round(float(np.mean(rr_list)), 4),
        "top1_accuracy"   : round(float(np.mean(top1_list)), 4),
    }

    return {"average": average, "per_query": per_query}


# ─── Prediction Evaluation ───────────────────────────────────────────

def eval_prediction(predictions_csv: Path, only_synthetic: bool = True) -> dict:
    """
    Evaluasi predict_outcome() pada hasil di predictions.csv.

    Returns dict per metode dengan accuracy + per-class metrics.
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    df = pd.read_csv(predictions_csv)
    if only_synthetic:
        df = df[df["source"] == "synthetic"]

    df = df[df["actual_solution"].notna() & (df["actual_solution"] != "")]

    results = {}
    for method in df["prediction_method"].unique():
        sub = df[df["prediction_method"] == method]
        y_true = sub["actual_solution"].values
        y_pred = sub["predicted_solution"].values

        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        results[method] = {
            "name"              : method,
            "n_samples"         : len(sub),
            "accuracy"          : round(acc, 4),
            "precision_weighted": round(report["weighted avg"]["precision"], 4),
            "recall_weighted"   : round(report["weighted avg"]["recall"], 4),
            "f1_weighted"       : round(report["weighted avg"]["f1-score"], 4),
            "precision_macro"   : round(report["macro avg"]["precision"], 4),
            "recall_macro"      : round(report["macro avg"]["recall"], 4),
            "f1_macro"          : round(report["macro avg"]["f1-score"], 4),
        }

    return results


# ─── Error Analysis ──────────────────────────────────────────────────

def error_analysis(predictions_csv: Path, queries_path: Path) -> list[dict]:
    """
    Identifikasi prediksi yang gagal dan kategorikan jenis errornya.
    """
    df = pd.read_csv(predictions_csv)
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    queries_map = {q["query_id"]: q for q in queries}

    errors = []

    # Loop tiap prediksi yang salah
    misses = df[df["match"] == "MISS"]
    for _, row in misses.iterrows():
        q = queries_map.get(row["query_id"], {})
        query_text = q.get("query_text", "")
        word_count = len(query_text.split())

        # Kategorikan jenis error
        error_type = []
        if word_count < 30:
            error_type.append("query_too_short")

        # Cek class imbalance
        if row["predicted_solution"] == "Tidak Dapat Diterima":
            error_type.append("majority_class_bias")

        # Cek apakah confidence rendah
        if row["confidence_score"] < 0.5:
            error_type.append("low_confidence")

        # Cek apakah manual query (kemungkinan terlalu generik)
        if q.get("source") == "manual":
            error_type.append("manual_query_generic")

        if not error_type:
            error_type.append("unknown")

        errors.append({
            "query_id"          : row["query_id"],
            "source"            : row["source"],
            "method"            : row["prediction_method"],
            "actual"            : row["actual_solution"],
            "predicted"         : row["predicted_solution"],
            "confidence"        : row["confidence_score"],
            "query_word_count"  : word_count,
            "error_types"       : ", ".join(error_type),
            "top_5_case_ids"    : row["top_5_case_ids"],
        })

    return errors


# ─── Main Pipeline ───────────────────────────────────────────────────

def run() -> dict:
    """Eksekusi evaluasi end-to-end."""
    ensure_dirs()
    logger = setup_logger("evaluation", LOGS_DIR / "evaluation.log")
    logger.info("=" * 60)
    logger.info("Tahap 6 CBR: Model Evaluation")
    logger.info("=" * 60)

    # 1. Load queries
    queries_path = EVAL_DIR / "queries.json"
    if not queries_path.exists():
        logger.error(f"queries.json tidak ditemukan: {queries_path}")
        return {}

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(queries)} queries")

    # 2. Evaluate retrieval
    logger.info("\n--- Retrieval Evaluation ---")
    retrieval_result = eval_retrieval(queries, k=5)
    if retrieval_result["average"]:
        avg = retrieval_result["average"]
        logger.info(f"  N queries evaluated : {avg['n_queries']}")
        logger.info(f"  Precision@5         : {avg['precision_at_k']:.4f}")
        logger.info(f"  Recall@5            : {avg['recall_at_k']:.4f}")
        logger.info(f"  Hit Rate@5          : {avg['hit_rate_at_k']:.4f}")
        logger.info(f"  MRR                 : {avg['mrr']:.4f}")
        logger.info(f"  Top-1 Accuracy      : {avg['top1_accuracy']:.4f}")

    # Save retrieval metrics
    retrieval_df = pd.DataFrame(retrieval_result["per_query"])
    if not retrieval_df.empty:
        # Add average row
        avg_row = {"query_id": "AVERAGE"}
        for col in retrieval_df.select_dtypes(include=[np.number]).columns:
            avg_row[col] = round(retrieval_df[col].mean(), 4)
        retrieval_df = pd.concat([retrieval_df, pd.DataFrame([avg_row])], ignore_index=True)
    retrieval_df.to_csv(EVAL_DIR / "retrieval_metrics.csv", index=False, encoding="utf-8")
    logger.info(f"  Saved: {EVAL_DIR / 'retrieval_metrics.csv'}")

    # 3. Evaluate prediction
    logger.info("\n--- Prediction Evaluation ---")
    pred_csv = RESULTS_DIR / "predictions.csv"
    if not pred_csv.exists():
        logger.error(f"predictions.csv tidak ditemukan. Jalankan Tahap 4 dulu.")
        return {}

    pred_metrics = eval_prediction(pred_csv, only_synthetic=True)
    for method, m in pred_metrics.items():
        logger.info(f"  [{method}]")
        for k, v in m.items():
            logger.info(f"    {k:25s}: {v}")

    # Save prediction metrics
    pred_metrics_df = pd.DataFrame(list(pred_metrics.values()))
    pred_metrics_df.to_csv(EVAL_DIR / "prediction_metrics.csv", index=False, encoding="utf-8")
    logger.info(f"  Saved: {EVAL_DIR / 'prediction_metrics.csv'}")

    # 4. Error Analysis
    logger.info("\n--- Error Analysis ---")
    errors = error_analysis(pred_csv, queries_path)
    err_df = pd.DataFrame(errors)
    err_df.to_csv(EVAL_DIR / "error_analysis.csv", index=False, encoding="utf-8")
    logger.info(f"  Total errors    : {len(errors)}")
    logger.info(f"  Saved: {EVAL_DIR / 'error_analysis.csv'}")

    # Breakdown error types
    if errors:
        type_counter = Counter()
        for e in errors:
            for et in e["error_types"].split(", "):
                type_counter[et] += 1
        logger.info(f"  Error types breakdown:")
        for et, cnt in type_counter.most_common():
            logger.info(f"    {et:30s}: {cnt}")

    logger.info("=" * 60)
    logger.info("Evaluation selesai.")
    logger.info("=" * 60)

    return {
        "retrieval": retrieval_result,
        "prediction": pred_metrics,
        "errors": errors,
    }


if __name__ == "__main__":
    run()
