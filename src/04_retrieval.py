"""
Tahap 3 CBR: Case Retrieval.

Pipeline:
1. Load cases.csv → build TF-IDF model
2. Function retrieve(query, k=5) → cosine similarity ranking
3. Train SVM + Naive Bayes untuk klasifikasi kategori_solusi
4. Generate test queries (5 synthetic + 2 manual) → queries.json
5. Save model artifacts untuk re-use di tahap berikutnya

Output:
- data/eval/queries.json
- data/processed/tfidf_model.pkl
- data/processed/svm_model.pkl
- data/processed/nb_model.pkl
- data/processed/tfidf_matrix.pkl
- logs/retrieval.log
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROCESSED_DIR, EVAL_DIR, LOGS_DIR, ensure_dirs, setup_logger


# ─── Constants ───────────────────────────────────────────────────────

# Stopwords Indonesia kecil — kata umum yang tidak informatif untuk retrieval
INDO_STOPWORDS = {
    "yang", "dan", "di", "dari", "ke", "untuk", "pada", "dengan", "atau",
    "ini", "itu", "adalah", "tidak", "akan", "telah", "sudah", "bahwa",
    "oleh", "sebagai", "para", "kami", "kita", "saya", "anda", "mereka",
    "dia", "ia", "yaitu", "yakni", "agar", "supaya", "jika", "ketika",
    "saat", "waktu", "hari", "tanggal", "bulan", "tahun",
    "tersebut", "diatas", "dibawah", "tentang", "terhadap", "kepada",
    "dalam", "antara", "secara", "serta", "juga", "namun", "tetapi",
    "atas", "bagi", "bagian", "berdasarkan", "berikut", "demikian",
}

TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "max_features": 3000,
    "min_df": 1,
    "max_df": 0.85,
    "sublinear_tf": True,
    "stop_words": list(INDO_STOPWORDS),
}


# ─── Retrieval Engine ────────────────────────────────────────────────

class CBRRetriever:
    """
    Main retrieval engine: TF-IDF + Cosine Similarity.
    Menyediakan fungsi retrieve() untuk mencari top-k kasus termirip.
    """

    def __init__(self, cases_path: Optional[Path] = None):
        cases_path = cases_path or (PROCESSED_DIR / "cases.csv")
        if not cases_path.exists():
            raise FileNotFoundError(f"cases.csv tidak ditemukan: {cases_path}")

        self.df = pd.read_csv(cases_path)
        # Pastikan text_retrieval ada dan tidak null
        self.df["text_retrieval"] = self.df["text_retrieval"].fillna("").astype(str)

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._fit()

    def _preprocess(self, text: str) -> str:
        """Preprocessing minimal untuk query/dokumen."""
        text = (text or "").lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _fit(self) -> None:
        """Build TF-IDF model dari corpus case base."""
        corpus = [self._preprocess(t) for t in self.df["text_retrieval"]]
        self.vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """
        Cari top-k kasus paling mirip dengan query.

        Args:
            query: deskripsi kasus baru (string)
            k: jumlah kasus yang dikembalikan

        Returns:
            list of dict berisi: ranking, case_id, similarity_score, dll.
        """
        q_clean = self._preprocess(query)
        q_vec = self.vectorizer.transform([q_clean])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]

        top_k_idx = np.argsort(sims)[::-1][:k]
        results = []
        for rank, idx in enumerate(top_k_idx, start=1):
            case = self.df.iloc[idx]
            results.append({
                "ranking"          : rank,
                "case_id"          : str(case["case_id"]),
                "no_perkara"       : str(case.get("no_perkara", "")),
                "similarity_score" : float(round(sims[idx], 4)),
                "kategori_solusi"  : str(case.get("kategori_solusi", "")),
                "pasal"            : str(case.get("pasal", "")),
                "ringkasan_fakta"  : str(case.get("ringkasan_fakta", ""))[:300],
                "amar_putusan"     : str(case.get("amar_putusan", ""))[:300],
                "source_url"       : str(case.get("detail_url", "")),
            })
        return results


# ─── Classifier Training ─────────────────────────────────────────────

def train_classifiers(retriever: CBRRetriever,
                       test_size: float = 0.2,
                       random_state: int = 42) -> dict:
    """
    Train SVM dan Naive Bayes classifiers untuk prediksi kategori_solusi.

    Returns dict berisi model + data split + predictions.
    """
    df = retriever.df
    X = retriever.tfidf_matrix
    y = df["kategori_solusi"].fillna("Tidak Teridentifikasi").values

    # Stratify hanya jika setiap label punya ≥ 2 sampel
    from collections import Counter
    label_counts = Counter(y)
    can_stratify = all(c >= 2 for c in label_counts.values())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if can_stratify else None,
    )

    # SVM (Linear) — robust untuk klasifikasi teks
    svm = LinearSVC(random_state=random_state, max_iter=2000, C=1.0)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)

    # Naive Bayes — fast baseline
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)

    return {
        "svm": svm,
        "nb": nb,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "svm_pred": svm_pred,
        "nb_pred": nb_pred,
        "can_stratify": can_stratify,
    }


def evaluate_classifier(y_true, y_pred, name: str = "model") -> dict:
    """Hitung metrik evaluasi: accuracy, precision, recall, F1 (weighted + macro)."""
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "name"              : name,
        "accuracy"          : round(acc, 4),
        "precision_weighted": round(report["weighted avg"]["precision"], 4),
        "recall_weighted"   : round(report["weighted avg"]["recall"], 4),
        "f1_weighted"       : round(report["weighted avg"]["f1-score"], 4),
        "precision_macro"   : round(report["macro avg"]["precision"], 4),
        "recall_macro"      : round(report["macro avg"]["recall"], 4),
        "f1_macro"          : round(report["macro avg"]["f1-score"], 4),
    }


# ─── Test Queries Generation ─────────────────────────────────────────

def generate_test_queries(retriever: CBRRetriever, logger) -> list[dict]:
    """
    Generate 7 test queries: 5 synthetic + 2 manual.

    Synthetic: ambil ringkasan_fakta dari 5 case berbeda kategori →
               ground truth = case yang sama (validasi self-retrieval).

    Manual: skenario hipotetis berdasarkan domain wanprestasi.
    """
    df = retriever.df
    queries: list[dict] = []

    # 5 SYNTHETIC — coba ambil dari kategori berbeda untuk diversity
    used_case_ids: set[str] = set()
    kategoris = df["kategori_solusi"].value_counts().index.tolist()

    for kat in kategoris:
        if len(queries) >= 5:
            break
        subset = df[(df["kategori_solusi"] == kat) &
                    (~df["case_id"].isin(used_case_ids))]
        if len(subset) == 0:
            continue
        case = subset.iloc[0]

        # Gunakan ringkasan_fakta sebagai query (truncate 500 char)
        fakta = str(case.get("ringkasan_fakta", "")).strip()
        if not fakta or len(fakta) < 100:
            fakta = str(case.get("text_retrieval", ""))[:500]
        query_text = fakta[:500]

        queries.append({
            "query_id"             : f"Q{len(queries)+1:03d}",
            "query_text"           : query_text,
            "ground_truth_case_ids": [str(case["case_id"])],
            "expected_kategori"    : str(case["kategori_solusi"]),
            "source"               : "synthetic",
            "notes"                : f"Generated from {case['case_id']} ringkasan_fakta",
        })
        used_case_ids.add(case["case_id"])

    # 2 MANUAL — skenario wanprestasi umum
    manual_queries = [
        {
            "query_id"             : f"Q{len(queries)+1:03d}",
            "query_text"           : (
                "Penggugat dan tergugat telah menandatangani perjanjian kredit "
                "untuk pembelian rumah. Namun setelah jangka waktu yang disepakati, "
                "tergugat tidak melunasi sisa angsuran pinjaman yang menjadi "
                "kewajibannya kepada penggugat. Penggugat memohon agar tergugat "
                "dihukum membayar sisa hutang beserta bunga."
            ),
            "ground_truth_case_ids": [],   # untuk validasi manual
            "expected_kategori"    : "Dikabulkan",
            "source"               : "manual",
            "notes"                : "Skenario wanprestasi kredit perumahan",
        },
        {
            "query_id"             : f"Q{len(queries)+2:03d}",
            "query_text"           : (
                "Para pihak menandatangani perjanjian jual beli kendaraan bermotor. "
                "Penggugat telah membayar lunas harga yang disepakati, tetapi "
                "tergugat tidak menyerahkan kendaraan beserta dokumen kepemilikan "
                "sesuai dengan kesepakatan. Tergugat juga tidak memberikan alasan "
                "yang jelas atas keterlambatan tersebut."
            ),
            "ground_truth_case_ids": [],
            "expected_kategori"    : "Dikabulkan",
            "source"               : "manual",
            "notes"                : "Skenario wanprestasi jual beli barang",
        },
    ]
    queries.extend(manual_queries)

    logger.info(f"Generated {len(queries)} test queries "
                f"({sum(1 for q in queries if q['source']=='synthetic')} synthetic + "
                f"{sum(1 for q in queries if q['source']=='manual')} manual)")
    return queries


# ─── Persistence ─────────────────────────────────────────────────────

def save_artifacts(retriever: CBRRetriever, classifiers: dict, logger) -> None:
    """Simpan model artifacts untuk re-use di tahap berikutnya."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "tfidf_model.pkl"  : retriever.vectorizer,
        "tfidf_matrix.pkl" : retriever.tfidf_matrix,
        "svm_model.pkl"    : classifiers["svm"],
        "nb_model.pkl"     : classifiers["nb"],
    }
    for name, obj in artifacts.items():
        with open(PROCESSED_DIR / name, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Saved: {PROCESSED_DIR / name}")


def save_queries(queries: list[dict], logger) -> None:
    """Simpan queries.json."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    queries_path = EVAL_DIR / "queries.json"
    queries_path.write_text(
        json.dumps(queries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Saved: {queries_path}")


# ─── Main Pipeline ───────────────────────────────────────────────────

def run() -> dict:
    """Eksekusi Tahap 3 end-to-end."""
    ensure_dirs()
    logger = setup_logger("retrieval", LOGS_DIR / "retrieval.log")
    logger.info("=" * 60)
    logger.info("Tahap 3 CBR: Case Retrieval")
    logger.info("=" * 60)

    # 1. Build retriever
    retriever = CBRRetriever()
    logger.info(f"Loaded {len(retriever.df)} cases")
    logger.info(f"TF-IDF matrix shape: {retriever.tfidf_matrix.shape}")

    # 2. Generate test queries
    queries = generate_test_queries(retriever, logger)
    save_queries(queries, logger)

    # 3. Train classifiers
    classifiers = train_classifiers(retriever)
    logger.info(f"Train size: {classifiers['X_train'].shape[0]}, "
                f"Test size: {classifiers['X_test'].shape[0]}, "
                f"Stratified: {classifiers['can_stratify']}")

    # 4. Evaluate
    svm_metrics = evaluate_classifier(classifiers["y_test"],
                                       classifiers["svm_pred"], "SVM")
    nb_metrics = evaluate_classifier(classifiers["y_test"],
                                      classifiers["nb_pred"], "Naive Bayes")

    logger.info("\nSVM Performance:")
    for k, v in svm_metrics.items():
        logger.info(f"  {k:25s}: {v}")
    logger.info("\nNaive Bayes Performance:")
    for k, v in nb_metrics.items():
        logger.info(f"  {k:25s}: {v}")

    # 5. Save artifacts
    save_artifacts(retriever, classifiers, logger)

    logger.info("=" * 60)
    logger.info("Tahap 3 selesai. Output:")
    logger.info(f"  - data/eval/queries.json ({len(queries)} queries)")
    logger.info(f"  - data/processed/tfidf_model.pkl")
    logger.info(f"  - data/processed/svm_model.pkl")
    logger.info(f"  - data/processed/nb_model.pkl")
    logger.info("=" * 60)

    return {
        "retriever": retriever,
        "classifiers": classifiers,
        "queries": queries,
        "svm_metrics": svm_metrics,
        "nb_metrics": nb_metrics,
    }


if __name__ == "__main__":
    run()
