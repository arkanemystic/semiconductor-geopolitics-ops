#!/usr/bin/env python3
"""
Improved Web Scraping and Sentiment Analysis Script
- Works with tab-separated TXT input or Excel
- Uses Newspaper3k with HTTP prefetch
- Falls back to BeautifulSoup and Readability-lxml
- Rotates multiple User-Agents
"""

import os
import re
import time
import random
import logging
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article, ArticleException, Config
from readability import Document  # pip install readability-lxml
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from tqdm import tqdm

# ----- Configuration -----
# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='scraper_log.txt'
)

# NLTK stopwords
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    stop_words = set(stopwords.words('english'))
except Exception as e:
    logging.error(f"Failed to download NLTK data: {e}")
    stop_words = set()

# Newspaper3k config
config = Config()
config.request_timeout = 20
config.fetch_images = False
config.memoize_articles = False

# Rotate user agents\USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 15_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:102.0) Gecko/20100101 Firefox/102.0'
]

# Keywords for sentiment weighting
keywords = [
    'taiwan', 'china', 'semiconductor', 'supply chain', 'risk', 'threat', 'growth',
    'chip', 'factory', 'production', 'shortage', 'investment', 'manufacturing',
    'innovation', 'disruption', 'global', 'market', 'demand', 'silicon', 'foundry'
]

# ----- Utility Functions -----

def fetch_html(url):
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.text


def clean_text(text: str) -> str:
    if not text:
        return ''
    clean = re.sub(r'[^a-zA-Z\s]', ' ', text)
    clean = re.sub(r'\s+', ' ', clean).strip().lower()
    tokens = [w for w in clean.split() if w not in stop_words]
    return ' '.join(tokens)


def extract_with_bs4(url):
    try:
        html = fetch_html(url)
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text().strip() for p in paragraphs)
        if not text or len(text) < 200:
            for container in soup.find_all(['article', 'main', 'div', 'section']):
                ct = container.get_text(separator=' ', strip=True)
                if len(ct) > 200:
                    text = ct
                    break
        return clean_text(text)
    except Exception as e:
        logging.error(f"BS4 fallback failed for {url}: {e}")
        return None


def extract_with_readability(html: str) -> str:
    try:
        doc = Document(html)
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        return clean_text(text)
    except Exception as e:
        logging.error(f"Readability fallback failed: {e}")
        return None


def get_clean_text(url: str) -> str:
    max_attempts = 3
    for attempt in range(1, max_attempts+1):
        try:
            html = fetch_html(url)
            article = Article(url, config=config)
            article.download(input_html=html)
            article.parse()
            raw = article.text or ''
            if len(raw) < 100:
                raise ArticleException('Raw text too short')
            return clean_text(raw)
        except (ArticleException, requests.RequestException) as e:
            logging.warning(f"[Attempt {attempt}] newspaper3k failed for {url}: {e}")
            time.sleep(2 * attempt)

    text_bs4 = extract_with_bs4(url)
    if text_bs4 and len(text_bs4) > 100:
        return text_bs4

    try:
        html = fetch_html(url)
        text_rd = extract_with_readability(html)
        if text_rd and len(text_rd) > 100:
            return text_rd
    except:
        pass

    logging.error(f"All extraction methods failed for {url}")
    return None


def count_keyword_mentions(text: str, keywords: list) -> int:
    if not text:
        return 0
    count = 0
    for kw in keywords:
        if ' ' in kw:
            count += text.count(kw)
        else:
            count += len(re.findall(rf"\b{re.escape(kw)}\b", text))
    return count


def analyze_weighted_sentiment(text: str, keywords: list):
    if not text or len(text) < 100:
        return 'Insufficient Content', None
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        kcount = count_keyword_mentions(text, keywords)
        density = kcount / (len(text.split()) + 1e-6)
        boost = 1 + min(density*10, 0.5)
        adj = max(min(polarity * boost, 1.0), -1.0)
        if adj > 0.1:
            sent = 'Positive'
        elif adj < -0.05:
            sent = 'Negative'
        else:
            sent = 'Neutral'
        return sent, adj
    except Exception as e:
        logging.error(f"Sentiment analysis error: {e}")
        return 'Error', None


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc
    except:
        return 'unknown'


def process_articles(input_path: str, output_path: str):
    # Load input file (TXT or XLSX)
    if input_path.lower().endswith('.txt'):
        df = pd.read_csv(input_path, sep='\t', header=0, usecols=['URL', 'Company'])
        df['URL'] = df['URL'].astype(str).str.strip()
        df['Company'] = df['Company'].astype(str).str.strip()
    else:
        df = pd.read_excel(input_path, sheet_name='Articles')

    urls = df['URL'].tolist()
    comps = df['Company'].tolist()

    results = []
    domain_failures = {}
    success = fail = 0

    for idx, (url, comp) in enumerate(tqdm(zip(urls, comps), total=len(urls))):
        if not isinstance(url, str) or not url.strip():
            results.append({'Company': comp, 'URL': url, 'Sentiment': 'Invalid URL', 'Polarity Score': None, 'Text': None, 'Keywords Found': 0})
            fail += 1
            continue

        domain = get_domain(url)
        time.sleep(random.uniform(1, 3))
        text = get_clean_text(url)
        if text:
            kcount = count_keyword_mentions(text, keywords)
            sent, score = analyze_weighted_sentiment(text, keywords)
            results.append({'Company': comp, 'URL': url, 'Sentiment': sent, 'Polarity Score': score, 'Text': text, 'Keywords Found': kcount})
            success += 1
        else:
            domain_failures[domain] = domain_failures.get(domain, 0) + 1
            results.append({'Company': comp, 'URL': url, 'Sentiment': 'Extraction Failed', 'Polarity Score': None, 'Text': None, 'Keywords Found': 0})
            fail += 1

        if (idx + 1) % 5 == 0 or idx == len(urls) - 1:
            pd.DataFrame(results).to_csv(output_path, index=False)

    logging.info(f"Success rate: {success}/{len(urls)}")
    for dom, cnt in domain_failures.items():
        if cnt > 1:
            logging.info(f"  - {dom}: {cnt} failures")

    return results, success, fail


if __name__ == '__main__':
    inp = 'urls_companies.txt'
    out = 'sentiment_articles_improved.csv'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    print("Starting scraping...")
    res, s, f = process_articles(inp, out)
    print(f"Done: {s}/{s+f} succeeded, {f} failed.")
    print(f"Results saved to {out}")
