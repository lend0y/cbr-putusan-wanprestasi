# CBR Putusan Wanprestasi

> Sistem **Case-Based Reasoning (CBR)** untuk analisis putusan pengadilan perkara **Perdata Wanprestasi** menggunakan data dari Direktori Putusan Mahkamah Agung Republik Indonesia.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Tim

| Nama                          | NIM              |
| ----------------------------- | ---------------- |
| **Ahmad Qayyim**              | 202310370311286  |
| **Bintang Mars Satria Tuhu**  | 202310370311410  |

**Mata Kuliah:** Penalaran Komputer — SubCPMK-3
**Program Studi:** Teknik Informatika, Universitas Muhammadiyah Malang
**Semester:** Genap 2025/2026

---

## 📂 Akses Data & Resources

| Resource         | Link                                                                                            |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| **GitHub Repo**  | [github.com/lend0y/cbr-putusan-wanprestasi](https://github.com/lend0y/cbr-putusan-wanprestasi)  |
| **Google Drive** | [Folder Project (PDF + Artifacts)](https://drive.google.com/drive/folders/1hRMmqmx8xnmmamOKVMdeNdJ1DwTG-wES?usp=sharing) |

> **Catatan untuk Reviewer/Dosen:** Folder Google Drive berisi **35 PDF putusan asli** beserta hasil ekstraksi `.txt` yang tidak di-commit ke GitHub karena ukuran file (PDF & cleaned text di-gitignore). Repository GitHub berisi seluruh source code, notebook, hasil terstruktur (`cases.csv`, `cases.json`, `queries.json`), model artifacts (`.pkl`), metrik evaluasi, dan visualisasi.

---

## 1. Deskripsi Singkat

Proyek ini mengimplementasikan siklus lengkap **Case-Based Reasoning (CBR)** untuk domain hukum perdata wanprestasi, dengan corpus 32 putusan dari Direktori Putusan Mahkamah Agung Republik Indonesia. Sistem menyediakan fungsi:

- **Retrieval** kasus serupa berdasarkan deskripsi kasus baru
- **Klasifikasi** kategori solusi (Dikabulkan / Ditolak / Tidak Dapat Diterima / dll)
- **Prediksi** outcome dengan dua strategi: Majority Vote dan Weighted Similarity
- **Revise & Retain** untuk memperluas case base secara iteratif

## 2. Latar Belakang

Pengambilan keputusan hukum sering memerlukan referensi ke putusan-putusan terdahulu yang serupa (yurisprudensi). Pendekatan Case-Based Reasoning meniru pola penalaran ini secara komputasional: ketika dihadapkan dengan kasus baru, sistem mencari kasus-kasus historis yang paling mirip lalu menggunakan outcome-nya sebagai dasar rekomendasi.

Direktori Putusan Mahkamah Agung RI menyediakan ribuan putusan publik yang dapat dijadikan basis pengetahuan, namun belum banyak tools yang memanfaatkannya untuk eksperimen NLP berbahasa Indonesia di domain hukum.

## 3. Tujuan

1. Membangun corpus terstruktur dari 30+ putusan perdata wanprestasi.
2. Mengimplementasikan siklus CBR lengkap (Retrieve → Reuse → Revise → Retain).
3. Mengevaluasi performa retrieval dan prediksi menggunakan metrik standar.
4. Mendokumentasikan keterbatasan sistem secara kritis untuk perbaikan iteratif.

## 4. Domain Perkara

**Perdata Wanprestasi** dipilih sebagai domain dengan pertimbangan:

- Label hasil putusan relatif konsisten (Kabul / Tolak / Sebagian / NO)
- Format dokumen mostly text-based (bukan scan)
- Risiko privasi minimal (cukup masking nama pihak)
- Pola "fakta → pasal → amar" cukup jelas untuk ekstraksi otomatis

Dalam praktek, ditemukan juga kategori tambahan: **Perdamaian** (akta perdamaian) dan **Dicabut** (penetapan pencabutan), sehingga total 6 kategori solusi diklasifikasi.

## 5. Siklus CBR yang Diimplementasikan

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    RETRIEVE     │ -> │      REUSE      │ -> │     REVISE      │ -> │     RETAIN      │
│ TF-IDF + Cosine │    │ Majority Vote / │    │ Verifikasi      │    │ Append ke       │
│ Top-K Cases     │    │ Weighted Sim.   │    │ manual oleh user│    │ case base       │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 6. Struktur Repository

```
cbr-putusan-wanprestasi/
├── data/
│   ├── raw/
│   │   ├── pdf/            # 35 PDF putusan (gitignored, tersedia di Drive)
│   │   ├── html/           # halaman detail (jika ada)
│   │   ├── text/           # hasil ekstraksi mentah (gitignored, tersedia di Drive)
│   │   ├── metadata.csv    # input manual dari user
│   │   └── metadata.json   # generated dari metadata.csv
│   ├── cleaned/            # teks bersih per case (gitignored, tersedia di Drive)
│   ├── processed/
│   │   ├── cases.csv       # case base terstruktur
│   │   ├── cases.json
│   │   ├── tfidf_model.pkl
│   │   ├── tfidf_matrix.pkl
│   │   ├── svm_model.pkl
│   │   └── nb_model.pkl
│   ├── eval/
│   │   ├── queries.json
│   │   ├── retrieval_metrics.csv
│   │   ├── prediction_metrics.csv
│   │   └── error_analysis.csv
│   └── results/
│       └── predictions.csv
├── logs/                    # log file scraping/cleaning/retain/eval
├── notebooks/               # Jupyter notebooks per tahap (01-05)
├── src/                     # Python scripts modular (01-07 + utils.py)
├── reports/figures/         # visualisasi hasil
├── .gitignore
├── LICENSE                  # MIT
├── README.md
└── requirements.txt
```

## 7. Requirements

- **Python** 3.9 atau lebih baru
- Dependencies lihat `requirements.txt`

## 8. Instalasi

```bash
# 1. Clone repository
git clone https://github.com/lend0y/cbr-putusan-wanprestasi.git
cd cbr-putusan-wanprestasi

# 2. (Opsional) Buat virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# atau
.\venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download data dari Google Drive
#    (PDF putusan tidak di-commit ke GitHub karena ukuran file)
#    Link: https://drive.google.com/drive/folders/1hRMmqmx8xnmmamOKVMdeNdJ1DwTG-wES
#    Letakkan PDF di data/raw/pdf/ dan cleaned text di data/cleaned/
```

## 9. Cara Menjalankan Pipeline End-to-End

### Opsi A: Via Google Colab (recommended)

Buka notebook di Colab secara berurutan:

```
notebooks/01_case_base.ipynb           # Tahap 1: Build case base
notebooks/02_case_representation.ipynb # Tahap 2: Representasi terstruktur
notebooks/03_retrieval.ipynb           # Tahap 3: TF-IDF + SVM/NB
notebooks/04_solution_reuse.ipynb      # Tahap 4: Prediksi outcome
notebooks/05_evaluation.ipynb          # Tahap 5+6: Retain + Evaluasi
```

### Opsi B: Via Script Lokal

```bash
# Tahap 1: Load metadata + extract & clean PDF
python src/01_load_manual_cases.py
python src/02_extract_and_clean.py

# Tahap 2: Case Representation
python src/03_build_case_representation.py

# Tahap 3: Retrieval + Classifier Training
python src/04_retrieval.py

# Tahap 4: Solution Reuse (predict_outcome batch run)
python src/05_predict.py

# Tahap 5: Revise & Retain (demo workflow)
python src/06_retain.py

# Tahap 6: Evaluation
python src/07_evaluation.py
```

### Catatan Penting Tentang Data Acquisition

Direktori Putusan MA RI memblokir scraping otomatis (HTTP 403, anti-bot protection). Workflow yang dipakai di proyek ini adalah **download manual** via browser, kemudian metadata diinput ke `data/raw/metadata.csv` mengikuti template. Script `01_load_manual_cases.py` mem-validate konsistensi PDF vs metadata sebelum pipeline berjalan.

**Untuk reproduce hasil tanpa download manual:** semua PDF asli + hasil ekstraksi text tersedia di [Google Drive folder](https://drive.google.com/drive/folders/1hRMmqmx8xnmmamOKVMdeNdJ1DwTG-wES?usp=sharing). Tinggal download dan letakkan di `data/raw/pdf/` dan `data/cleaned/`.

## 10. Cara Menggunakan `retrieve()`

```python
import importlib.util
import sys

# Load retrieval module (filename diawali angka, jadi pakai importlib)
spec = importlib.util.spec_from_file_location("retr_mod", "src/04_retrieval.py")
retr_mod = importlib.util.module_from_spec(spec)
sys.modules["retr_mod"] = retr_mod
spec.loader.exec_module(retr_mod)

# Inisialisasi retriever (auto-load cases.csv + build TF-IDF)
retriever = retr_mod.CBRRetriever()

# Search top-5 kasus termirip
results = retriever.retrieve(
    query="Tergugat tidak membayar angsuran kredit sesuai perjanjian",
    k=5
)

for r in results:
    print(f"[{r['ranking']}] {r['case_id']} | "
          f"sim={r['similarity_score']:.4f} | "
          f"kategori={r['kategori_solusi']}")
```

**Output format** (per item):

```python
{
    'ranking': 1,
    'case_id': 'case_007',
    'no_perkara': '151/Pdt.G/2020',
    'similarity_score': 0.7234,
    'kategori_solusi': 'Dikabulkan',
    'pasal': '1243, 1338',
    'ringkasan_fakta': '...',
    'amar_putusan': '...',
    'source_url': 'https://putusan3.mahkamahagung.go.id/...'
}
```

## 11. Cara Menggunakan `predict_outcome()`

```python
spec = importlib.util.spec_from_file_location("predict_mod", "src/05_predict.py")
predict_mod = importlib.util.module_from_spec(spec)
sys.modules["predict_mod"] = predict_mod
spec.loader.exec_module(predict_mod)

# Prediksi dengan weighted similarity (default & recommended)
result = predict_mod.predict_outcome(
    query="Penggugat menggugat tergugat karena wanprestasi perjanjian jual beli...",
    k=5,
    method='weighted_similarity'   # atau 'majority_vote'
)

print(f"Predicted: {result['predicted_solution']}")
print(f"Confidence: {result['confidence_score']:.2%}")
print(f"Reasoning: {result['reasoning']}")
```

**Output format:**

```python
{
    'query': '...',
    'predicted_solution': 'Dikabulkan',
    'confidence_score': 0.6148,
    'prediction_method': 'weighted_similarity',
    'top_k_case_ids': ['case_007', 'case_024', 'case_005', 'case_006', 'case_010'],
    'similarity_scores': [0.72, 0.65, 0.58, 0.51, 0.42],
    'top_k_kategoris': ['Dikabulkan', 'Dikabulkan', 'Ditolak', 'Dikabulkan', 'Dikabulkan'],
    'reasoning': 'Bobot per kategori: Dikabulkan=1.65, Ditolak=0.58. ...',
    'disclaimer': 'Hasil prediksi ini merupakan output sistem akademik...'
}
```

## 12. Metode Evaluasi

### Retrieval Metrics

- **Precision@k** — Proporsi cases relevan di top-k retrieved
- **Recall@k** — Proporsi ground truth yang ditemukan di top-k
- **F1@k** — Harmonic mean dari Precision dan Recall
- **Hit Rate@k** — Apakah minimal 1 ground truth muncul di top-k
- **MRR (Mean Reciprocal Rank)** — Rata-rata 1/rank dari hit pertama
- **Top-1 Accuracy** — Apakah ground truth muncul di rank #1

### Prediction Metrics

- **Accuracy** — Proporsi prediksi yang benar
- **Precision / Recall / F1** (weighted + macro)
- **Confusion Matrix** untuk inspeksi per-class

### Test Queries

7 query uji disusun di `data/eval/queries.json`:
- **5 synthetic** queries dari ringkasan_fakta case berbeda kategori (ground truth = case sumber)
- **2 manual** queries skenario wanprestasi umum (untuk demo, tanpa ground truth tetap)

## 13. Hasil Evaluasi

### Retrieval Performance (averaged on synthetic queries)

| Metric          | Score   | Catatan                                 |
| --------------- | ------- | --------------------------------------- |
| Precision@5     | 0.2000  | Wajar — 1 ground truth dari 5 retrieved |
| Recall@5        | 1.0000  | Semua ground truth selalu masuk top-5   |
| F1@5            | 0.3333  | Mengikuti precision (1 GT per query)    |
| Hit Rate@5      | 1.0000  | 100% query hit di top-5                 |
| **MRR**         | **0.9000** | Rata-rata rank #1.1                  |
| **Top-1 Acc**   | **0.8000** | 4/5 query → GT di rank #1            |

### Prediction Performance (5 synthetic queries)

| Method                | Accuracy | F1 (weighted) | F1 (macro) |
| --------------------- | -------- | ------------- | ---------- |
| Majority Vote         | 0.800    | 0.733         | 0.733      |
| **Weighted Similarity** | **1.000** | **1.000**   | **1.000**  |

**Kesimpulan:** Weighted Similarity unggul karena memberikan bobot proporsional terhadap similarity score, sehingga case yang lebih mirip lebih berpengaruh pada prediksi final.

## 14. Error Analysis

### Temuan Utama

**4 dari 14 prediksi** (28.6%) salah, semuanya pada manual queries (Q006, Q007). Jenis error yang teridentifikasi:

| Error Type             | Frekuensi | Penjelasan                                       |
| ---------------------- | --------- | ------------------------------------------------ |
| `majority_class_bias`  | 4         | Predicted "Tidak Dapat Diterima" (kelas mayoritas) |
| `manual_query_generic` | 4         | Query manual terlalu generik                     |
| `low_confidence`       | 2         | Confidence < 0.5 saat prediksi salah             |

### Diskusi Kritis

1. **Class imbalance ekstrem.** 53% case base bertipe "Tidak Dapat Diterima", menyebabkan retrieval cenderung mengembalikan case kategori tersebut untuk query generik.
2. **Kemiripan kosakata ≠ kemiripan substansi hukum.** TF-IDF menangkap pola kata, tetapi tidak memahami konteks legal. Dua kasus dengan pasal sama dapat memiliki outcome berbeda.
3. **Dataset kecil (32 cases).** Test set hanya ~7 case, menyebabkan metrik classifier kurang stabil.
4. **Manual query bersifat ambigu.** Query seperti "tergugat tidak membayar kredit" cocok untuk kategori Dikabulkan, Ditolak, maupun NO — tergantung detail formal gugatan.

### Rekomendasi Perbaikan

- Tambah ukuran dataset (target: 100+ cases) untuk distribusi lebih representatif
- Class balancing (SMOTE atau class weights di classifier)
- Embedding semantik (IndoBERT atau Sentence-BERT) sebagai komplemen TF-IDF
- Hybrid retrieval — gabung TF-IDF dengan metadata filter (pasal, pengadilan)
- Synonym mapping istilah hukum untuk variasi terminologi
- Cross-validation k-fold untuk metric classifier yang lebih reliable

## 15. Keterbatasan Sistem

1. **Bukan sistem hukum produksi.** Output sistem ini adalah hasil analisis akademik, bukan rekomendasi hukum.
2. **Bias dataset.** Sample 32 case tidak representatif terhadap seluruh putusan wanprestasi Indonesia.
3. **Ekstraksi rule-based.** Regex tidak menjamin akurasi 100% pada variasi format putusan.
4. **Tidak ada IndoBERT.** Karena keterbatasan resource Colab Free, embedding semantik tidak digunakan.
5. **Anonimisasi parsial.** Nama pihak di-mask, tetapi data publik MA RI tetap dapat ditelusuri lewat `detail_url`.

## 16. Etika & Disclaimer

> **Disclaimer:** Sistem ini dibangun untuk tujuan akademik dan analisis awal. **Hasil retrieval maupun prediksi BUKAN keputusan hukum final.** Konsultasi dengan ahli hukum tetap diperlukan untuk interpretasi yang valid.

### Aspek Etika yang Diperhatikan

- **Sumber data publik:** Putusan diambil dari Direktori Putusan MA RI yang memang dipublikasikan terbuka.
- **Anonimisasi:** Nama pihak di case base diubah menjadi inisial (e.g. "Penggugat [A.S.]") untuk meminimalkan privasi.
- **Tidak ada scraping otomatis:** Karena MA RI menerapkan anti-bot protection (HTTP 403), data dikumpulkan **manual via browser** dengan menghormati rate dan kebijakan situs.
- **Transparansi keterbatasan:** Semua error dan limitasi sistem didokumentasikan terbuka di repository.

## 17. Pembagian Tugas Tim

### Ahmad Qayyim (202310370311286)

- Setup repository GitHub + struktur folder
- Implementasi seluruh pipeline kode:
  - `src/utils.py` — helper functions
  - `src/01_load_manual_cases.py` — load metadata workflow
  - `src/02_extract_and_clean.py` — ekstraksi & cleaning PDF
  - `src/03_build_case_representation.py` — case representation
  - `src/04_retrieval.py` — TF-IDF + Cosine + SVM/NB
  - `src/05_predict.py` — predict_outcome dengan dua strategi
  - `src/06_retain.py` — retain_case workflow
  - `src/07_evaluation.py` — evaluasi & error analysis
- Implementasi notebook untuk demo dan visualisasi (01-05)
- Iterasi pengembangan classify_solution dari v1 → v2 → v3
- Eksekusi pipeline & training di Google Colab
- Debugging dan optimisasi sistem retrieval & klasifikasi

### Bintang Mars Satria Tuhu (202310370311410)

- Download manual seluruh 35 PDF putusan dari Direktori MA RI (case_001 - case_035)
- Pengisian dan validasi `metadata.csv` untuk semua case
- Organisasi struktur folder data (raw/pdf, raw/text)
- Penulisan dokumentasi proyek (README.md + komentar inline)
- Diskusi metodologi dan analisis hasil bersama tim

## 18. Referensi

1. **Aamodt, A., & Plaza, E.** (1994). Case-Based Reasoning: Foundational Issues, Methodological Variations, and System Approaches. *AI Communications*, 7(1), 39-59.
2. **Bench-Capon, T., & Sartor, G.** (2003). A model of legal reasoning with cases incorporating theories and values. *Artificial Intelligence*, 150(1-2), 97-143.
3. **scikit-learn documentation** — TF-IDF, Cosine Similarity, Linear SVC, Multinomial Naive Bayes
4. **Mahkamah Agung Republik Indonesia** — Direktori Putusan ([putusan3.mahkamahagung.go.id](https://putusan3.mahkamahagung.go.id/))

## License

[MIT License](LICENSE) © 2026 Ahmad Qayyim, Bintang Mars Satria Tuhu

---

**Repository:** [github.com/lend0y/cbr-putusan-wanprestasi](https://github.com/lend0y/cbr-putusan-wanprestasi)
**Google Drive (PDF + Artifacts):** [Folder Project](https://drive.google.com/drive/folders/1hRMmqmx8xnmmamOKVMdeNdJ1DwTG-wES?usp=sharing)
