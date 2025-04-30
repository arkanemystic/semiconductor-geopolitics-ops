#!/usr/bin/env python3
"""
scrape_by_company.py

Usage:
    python scrape_by_company.py urls_companies.txt /path/to/output_root

Reads a tab-delimited file where each line has:
    URL[TAB]Company[...optional extras]

Skips header lines beginning with "URL", then for each row:
  - Creates output_root/Company/ if needed
  - Downloads and parses the article at URL
  - Saves its text to:
        output_root/Company/NNN_<safe_url_basename>.txt
"""

import os
import re
import sys
import argparse
import logging
from newspaper import Article

def sane_filename(s: str, maxlen: int = 80) -> str:
    """Make a filesystem-safe name from any string."""
    name = re.sub(r'[^0-9a-zA-Z]+', '_', s.strip().lower())
    return name[:maxlen].strip('_') or "item"

def parse_input(path: str):
    """
    Read the input file, yield (url, company) tuples.
    Skips blank lines and headers starting with 'URL'.
    Splits on tabs, ignores extra columns.
    """
    with open(path, encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.lower().startswith('url'):
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                logging.warning(f"[line {line_num}] malformed, skipping: {line}")
                continue
            url, company = parts[0].strip(), parts[1].strip() or "Unknown"
            yield url, company

def scrape_and_save(url: str, company: str, out_root: str, idx: int):
    """Download `url` and save its text into out_root/company/."""
    try:
        art = Article(url)
        art.download()
        art.parse()
        text = art.text.strip()
        if not text:
            raise ValueError("no text extracted")
    except Exception as e:
        logging.error(f"[{idx:03d}] Failed to fetch {url!r}: {e}")
        return

    # prepare directories
    company_dir = os.path.join(out_root, sane_filename(company))
    os.makedirs(company_dir, exist_ok=True)

    # build filename
    base = sane_filename(url, maxlen=60)
    fname = f"{idx:03d}_{base}.txt"
    out_path = os.path.join(company_dir, fname)

    try:
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(text)
        logging.info(f"[{idx:03d}] Saved {out_path}")
    except Exception as e:
        logging.error(f"[{idx:03d}] Could not write file {out_path!r}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Scrape URLs grouped by company into txt files."
    )
    parser.add_argument("input_file", help="Tab-delimited file of URL[TAB]Company")
    parser.add_argument("output_root", help="Root directory for output folders")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # ensure output root exists
    os.makedirs(args.output_root, exist_ok=True)

    entries = list(parse_input(args.input_file))
    if not entries:
        logging.error("No valid URL/company pairs found. Exiting.")
        sys.exit(1)

    for idx, (url, company) in enumerate(entries, start=1):
        scrape_and_save(url, company, args.output_root, idx)

if __name__ == "__main__":
    main()
