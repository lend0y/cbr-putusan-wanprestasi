"""
Tahap 1: Scraping Direktori Putusan Mahkamah Agung RI untuk kasus Wanprestasi.

Modul ini melakukan:
1. Pencarian putusan dengan keyword "wanprestasi" di Direktori MA RI
2. Pengumpulan URL detail page dari hasil pencarian
3. Download PDF + simpan metadata setiap putusan
4. Logging proses scraping ke logs/scraping.log

Output:
- data/raw/pdf/case_XXX.pdf
- data/raw/html/case_XXX.html (halaman detail)
- data/raw/metadata.json (metadata semua kasus)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    PDF_DIR, HTML_DIR, RAW_DIR, LOGS_DIR,
    ensure_dirs, setup_logger, create_session, polite_sleep,
    slug_no_perkara,
)


BASE_URL   = "https://putusan3.mahkamahagung.go.id"
SEARCH_URL = f"{BASE_URL}/search.html"
DEFAULT_QUERY = "wanprestasi"
DEFAULT_TARGET = 35


@dataclass
class CaseMetadata:
    """Metadata satu putusan."""
    case_id: str
    no_perkara: str
    tanggal_register: Optional[str]
    tanggal_putusan: Optional[str]
    pengadilan: Optional[str]
    klasifikasi: Optional[str]
    detail_url: str
    pdf_url: Optional[str]
    pdf_path: Optional[str]
    html_path: Optional[str]
    status: str
    error_msg: Optional[str] = None


class CourtScraper:
    """Scraper untuk Direktori Putusan Mahkamah Agung RI."""

    def __init__(self, query: str = DEFAULT_QUERY,
                 target: int = DEFAULT_TARGET,
                 delay: float = 3.0):
        self.query    = query
        self.target   = target
        self.delay    = delay
        self.session  = create_session()
        self.logger   = setup_logger("scraper", LOGS_DIR / "scraping.log")
        self.results: list[CaseMetadata] = []

    def search_page(self, page: int) -> list[str]:
        """Ambil halaman search dan kembalikan list URL detail page."""
        params = {"q": self.query, "page": page}
        try:
            resp = self.session.get(SEARCH_URL, params=params, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            self.logger.error(f"Gagal akses halaman search {page}: {e}")
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        # Direktori MA RI menggunakan tag <strong><a> untuk link putusan
        # Format URL detail: /direktori/putusan/[hash].html
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/direktori/putusan/" in href and href.endswith(".html"):
                full_url = urljoin(BASE_URL, href)
                if full_url not in links:
                    links.append(full_url)

        self.logger.info(f"Halaman {page}: ditemukan {len(links)} link putusan")
        return links

    def collect_detail_urls(self, max_pages: int = 10) -> list[str]:
        """Kumpulkan URL detail page dari beberapa halaman search."""
        all_urls = []
        for page in range(1, max_pages + 1):
            urls = self.search_page(page)
            if not urls:
                self.logger.warning(f"Tidak ada link di halaman {page}, stop pagination.")
                break
            for u in urls:
                if u not in all_urls:
                    all_urls.append(u)
            self.logger.info(f"Total URL terkumpul: {len(all_urls)}")
            if len(all_urls) >= self.target:
                break
            polite_sleep(self.delay)
        return all_urls[: self.target]

    def parse_detail(self, html: str) -> dict:
        """Parse metadata dari halaman detail putusan."""
        soup = BeautifulSoup(html, "lxml")
        meta: dict = {
            "no_perkara": None,
            "tanggal_register": None,
            "tanggal_putusan": None,
            "pengadilan": None,
            "klasifikasi": None,
            "pdf_url": None,
        }

        # Tabel metadata biasanya berisi <td> label dan <td> value
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            label = cells[0].get_text(strip=True).lower()
            value = cells[1].get_text(strip=True)
            if "nomor" in label and not meta["no_perkara"]:
                meta["no_perkara"] = value
            elif "tanggal register" in label:
                meta["tanggal_register"] = value
            elif "tanggal musyawarah" in label or "tanggal putusan" in label:
                meta["tanggal_putusan"] = value
            elif "pengadilan" in label and "tingkat" not in label:
                meta["pengadilan"] = value
            elif "klasifikasi" in label:
                meta["klasifikasi"] = value

        # Cari link PDF — biasanya class "btn" dengan href ke /direktori/download_file/
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/direktori/download_file/" in href or href.endswith(".pdf"):
                meta["pdf_url"] = urljoin(BASE_URL, href)
                break

        return meta

    def download_pdf(self, pdf_url: str, save_path: Path) -> bool:
        """Download PDF dari URL ke file system."""
        try:
            resp = self.session.get(pdf_url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_kb = save_path.stat().st_size / 1024
            if size_kb < 10:
                self.logger.warning(f"PDF terlalu kecil ({size_kb:.1f} KB): {save_path.name}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Gagal download PDF {pdf_url}: {e}")
            return False

    def scrape_one(self, idx: int, detail_url: str) -> CaseMetadata:
        """Scrape satu putusan: detail page + metadata + PDF."""
        case_id = f"case_{idx:03d}"

        # Fetch detail page
        try:
            resp = self.session.get(detail_url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            self.logger.error(f"{case_id}: gagal fetch detail: {e}")
            return CaseMetadata(
                case_id=case_id, no_perkara="", tanggal_register=None,
                tanggal_putusan=None, pengadilan=None, klasifikasi=None,
                detail_url=detail_url, pdf_url=None, pdf_path=None,
                html_path=None, status="FAILED_DETAIL", error_msg=str(e),
            )

        html_path = HTML_DIR / f"{case_id}.html"
        html_path.write_text(resp.text, encoding="utf-8")

        meta = self.parse_detail(resp.text)
        no_perkara = meta.get("no_perkara") or case_id
        slug = slug_no_perkara(no_perkara)

        pdf_url  = meta.get("pdf_url")
        pdf_path: Optional[Path] = None
        status   = "OK"
        err_msg  = None

        if pdf_url:
            polite_sleep(self.delay)
            pdf_path = PDF_DIR / f"{case_id}_{slug}.pdf"
            if not self.download_pdf(pdf_url, pdf_path):
                status  = "FAILED_PDF"
                err_msg = "PDF download gagal atau ukuran terlalu kecil"
                pdf_path = None
        else:
            status  = "NO_PDF_LINK"
            err_msg = "Link PDF tidak ditemukan di detail page"

        return CaseMetadata(
            case_id=case_id,
            no_perkara=no_perkara,
            tanggal_register=meta.get("tanggal_register"),
            tanggal_putusan=meta.get("tanggal_putusan"),
            pengadilan=meta.get("pengadilan"),
            klasifikasi=meta.get("klasifikasi"),
            detail_url=detail_url,
            pdf_url=pdf_url,
            pdf_path=str(pdf_path) if pdf_path else None,
            html_path=str(html_path),
            status=status,
            error_msg=err_msg,
        )

    def run(self) -> list[CaseMetadata]:
        """Eksekusi scraping end-to-end."""
        self.logger.info("=" * 60)
        self.logger.info(f"Mulai scraping: query='{self.query}', target={self.target}")
        self.logger.info("=" * 60)

        ensure_dirs()
        detail_urls = self.collect_detail_urls()

        if not detail_urls:
            self.logger.error("Tidak ada URL terkumpul. Stop.")
            return []

        self.logger.info(f"Akan scrape {len(detail_urls)} putusan...")

        # Track no_perkara untuk hindari duplikat
        seen_no_perkara: set[str] = set()

        for idx, url in enumerate(tqdm(detail_urls, desc="Scraping"), start=1):
            meta = self.scrape_one(idx, url)

            if meta.no_perkara in seen_no_perkara:
                meta.status = "DUPLICATE"
                meta.error_msg = "no_perkara sudah ada"
                self.logger.warning(f"{meta.case_id}: DUPLIKAT {meta.no_perkara}")
            else:
                if meta.no_perkara:
                    seen_no_perkara.add(meta.no_perkara)
                self.logger.info(f"{meta.case_id}: {meta.status} — {meta.no_perkara}")

            self.results.append(meta)
            polite_sleep(self.delay)

        self._save_metadata()
        self._print_summary()
        return self.results

    def _save_metadata(self) -> None:
        """Simpan metadata ke JSON."""
        meta_file = RAW_DIR / "metadata.json"
        data = [asdict(m) for m in self.results]
        meta_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        self.logger.info(f"Metadata tersimpan: {meta_file}")

    def _print_summary(self) -> None:
        """Print ringkasan hasil scraping."""
        total = len(self.results)
        ok    = sum(1 for m in self.results if m.status == "OK")
        failed = total - ok
        self.logger.info("=" * 60)
        self.logger.info(f"SELESAI. Total: {total}, OK: {ok}, Gagal: {failed}")
        self.logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scraping Direktori MA RI - Wanprestasi")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Search keyword")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help="Jumlah putusan yang di-scrape")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Delay antar request (detik)")
    args = parser.parse_args()

    scraper = CourtScraper(query=args.query, target=args.target, delay=args.delay)
    scraper.run()


if __name__ == "__main__":
    main()
