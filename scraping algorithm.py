#!/usr/bin/env python3
"""
Improved Web Scraping and Sentiment Analysis Script
- Uses a hardcoded Excel input file ('urls_companies.xlsx')
- Uses Newspaper3k with HTTP prefetch
- Falls back to BeautifulSoup and Readability-lxml
- Rotates multiple User-Agents
- Outputs parse/failure percentages and per-URL keyword frequencies in CSV
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
default_config = Config()
default_config.request_timeout = 20
default_config.fetch_images = False
default_config.memoize_articles = False

# Rotate User-Agents
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 15_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:102.0) Gecko/20100101 Firefox/102.0'
]

# Keywords for sentiment weighting
keywords = [
    'taiwan','china','semiconductor','supply chain','risk','threat','growth',
    'chip','factory','production','shortage','investment','manufacturing',
    'innovation','disruption','global','market','demand','silicon','foundry'
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


def extract_with_bs4(url: str) -> str:
    try:
        html = fetch_html(url)
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script','style','nav','header','footer','aside']): tag.decompose()
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text().strip() for p in paragraphs)
        if len(text) < 200:
            for c in soup.find_all(['article','main','div','section']):
                ct = c.get_text(' ',strip=True)
                if len(ct)>200:
                    text=ct; break
        return clean_text(text)
    except Exception as e:
        logging.error(f"BS4 fallback failed for {url}: {e}")
        return None


def extract_with_readability(html: str) -> str:
    try:
        doc=Document(html)
        summary=doc.summary()
        soup=BeautifulSoup(summary,'html.parser')
        return clean_text(soup.get_text(' ',strip=True))
    except Exception as e:
        logging.error(f"Readability fallback failed: {e}")
        return None


def get_clean_text(url: str) -> str:
    for i in range(3):
        try:
            html=fetch_html(url)
            art=Article(url,config=default_config)
            art.download(input_html=html); art.parse()
            raw=art.text or ''
            if len(raw)<100: raise ArticleException('Short')
            return clean_text(raw)
        except Exception as e:
            logging.warning(f"Attempt {i+1} failed for {url}: {e}")
            time.sleep(2*(i+1))
    bs4=extract_with_bs4(url)
    if bs4: return bs4
    try: return extract_with_readability(fetch_html(url))
    except: pass
    logging.error(f"All methods failed for {url}")
    return None


def count_keyword_mentions(text: str, kw: str)->int:
    return text.count(kw) if ' ' in kw else len(re.findall(rf"\b{re.escape(kw)}\b", text))


def analyze_weighted_sentiment(text: str)->tuple:
    blob=TextBlob(text);
    pol=blob.sentiment.polarity
    total_kw=sum(count_keyword_mentions(text, k) for k in keywords)
    dens=total_kw/(len(text.split())+1e-6)
    boost=1+min(dens*10,0.5)
    score=max(min(pol*boost,1.0),-1.0)
    sent=('Positive' if score>0.1 else 'Negative' if score<-0.05 else 'Neutral')
    return sent,score


def get_domain(url: str)->str:
    try: return urlparse(url).netloc
    except: return 'unknown'


def process_articles(out_csv: str):
    df=pd.read_excel('urls_companies (1).xlsx',sheet_name='Articles')
    results=[];fails={};s=f=0
    for idx,(url,comp) in enumerate(tqdm(df[['URL','Company']].values, total=len(df))):
        row={'Company':comp,'URL':url}
        if not isinstance(url,str) or not url.strip():
            row.update({'Sentiment':'Invalid URL','Polarity':None,'Text':None,'KeywordsFound':0})
            for k in keywords: row[k]=0
            fails['invalid']=fails.get('invalid',0)+1; f+=1
        else:
            time.sleep(random.uniform(1,3))
            txt=get_clean_text(url)
            if txt:
                kw_sum=0
                for k in keywords:
                    cnt=count_keyword_mentions(txt,k)
                    row[k]=cnt; kw_sum+=cnt
                sent,score=analyze_weighted_sentiment(txt)
                row.update({'Sentiment':sent,'Polarity':score,'Text':txt,'KeywordsFound':kw_sum})
                s+=1
            else:
                dom=get_domain(url); fails[dom]=fails.get(dom,0)+1; f+=1
                row.update({'Sentiment':'Extraction Failed','Polarity':None,'Text':None,'KeywordsFound':0})
                for k in keywords: row[k]=0
        results.append(row)
        if (idx+1)%5==0 or idx==len(df)-1:
            pd.DataFrame(results).to_csv(out_csv,index=False)
    total=s+f
    print(f"Done: {s}/{total} succeeded, {f}/{total} failed.")
    print(f"CSV saved to {out_csv}")
    return results,s,f

if __name__=='__main__':
    os.makedirs('.',exist_ok=True)
    print("Starting...")
    process_articles('sentiment_articles_output.csv')
