# CBR Putusan Wanprestasi

Sistem Case-Based Reasoning (CBR) sederhana untuk analisis putusan pengadilan perkara Perdata Wanprestasi, menggunakan data dari Direktori Putusan Mahkamah Agung Republik Indonesia.

> **Status:** 🚧 Work in progress — saat ini di Tahap 1 (Membangun Case Base)

## Tim

- **Ahmad Qayyim** — 202310370311286
- **Bintang Mars Satria Tuhu** — 202310370311410

Mata Kuliah: Penalaran Komputer — SubCPMK-3
Universitas Muhammadiyah Malang, 2025/2026

## Quick Start

```bash
# 1. Clone repo
git clone https://github.com/lend0y/cbr-putusan-wanprestasi.git
cd cbr-putusan-wanprestasi

# 2. Install dependencies
pip install -r requirements.txt

# 3. Scraping + Extraction (Tahap 1)
python src/01_download_cases.py --target 35
python src/02_extract_and_clean.py
```

Atau jalankan di **Google Colab**: buka `notebooks/01_case_base.ipynb`.

## Struktur Project

```
cbr-putusan-wanprestasi/
├── data/        # Raw, cleaned, processed data
├── src/         # Python scripts per tahap CBR
├── notebooks/   # Jupyter notebooks Colab
├── logs/        # Log scraping, cleaning, retain
├── reports/     # Figures, hasil evaluasi
└── tests/       # Unit tests
```

## Progress

- [x] Tahap 1: Case Base (scraping + extraction + cleaning)
- [ ] Tahap 2: Case Representation
- [ ] Tahap 3: Case Retrieval
- [ ] Tahap 4: Solution Reuse
- [ ] Tahap 5: Revise & Retain
- [ ] Tahap 6: Evaluation

README lengkap akan ditulis di akhir setelah semua tahap selesai.

## License

MIT
