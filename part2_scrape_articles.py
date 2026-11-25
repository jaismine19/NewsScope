"""
CSIS 4260 - Assignment 2
Part 2 (step 1): collect 50+ articles from NPR Technology.

Uses requests + BeautifulSoup (the library recommended in Part 1) to:
  1. collect article links from the section page (plus one more page of
     the listing for good measure),
  2. fetch each article and extract its title, date and body text.

Polite scraping: identifies itself with a normal browser user-agent,
waits ~0.6 s between requests and only hits public pages once.

The raw collection is cached in data/articles_raw.json so the text
analysis step can be rerun without hitting the site again.

Usage:
    .venv/bin/python part2_scrape_articles.py            # use cache if present
    .venv/bin/python part2_scrape_articles.py --refresh  # re-scrape
"""

import argparse
import json
import os
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PATH = os.path.join(DATA_DIR, "articles_raw.json")

SECTION_URL = "https://www.npr.org/sections/technology/"
LISTING_PAGES = [
    "https://www.npr.org/sections/technology/",
    "https://www.npr.org/sections/business/",
    "https://www.npr.org/sections/science/",
]
TARGET = 55  # a comfortable margin above the required 50
DELAY = 0.6

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

ARTICLE_RE = re.compile(r"https://www\.npr\.org/20\d\d/\d\d/\d\d/[^\"'\s#?]+")


def collect_article_urls():
    urls, seen = [], set()
    for listing in LISTING_PAGES:
        try:
            r = requests.get(listing, headers=HEADERS, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  !! listing page failed: {listing} ({e})", file=sys.stderr)
            continue
        soup = BeautifulSoup(r.text, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if ARTICLE_RE.fullmatch(href) and href not in seen:
                seen.add(href)
                urls.append(href)
        print(f"  listing {listing}: {len(seen)} unique article urls so far")
        time.sleep(DELAY)
    return urls


def fetch_article(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else ""
    if not title:
        og = soup.find("meta", property="og:title")
        title = og["content"].strip() if og and og.get("content") else ""

    date = ""
    meta = soup.find("meta", property="article:published_time")
    if meta and meta.get("content"):
        date = meta["content"][:10]
    if not date:
        t = soup.find("time", attrs={"datetime": True})
        if t:
            date = t["datetime"][:10]

    body = soup.find(id="storytext") or soup.find("article") or soup.body
    paras = []
    for p in body.find_all("p") if body else []:
        # skip image captions / photo credits (NPR wraps them in
        # .caption / .credit containers and marks them with "hide caption")
        cls = " ".join(p.get("class", []))
        parent_cls = " ".join(p.parent.get("class", [])) if p.parent else ""
        if "caption" in (cls + " " + parent_cls).lower():
            continue
        if "credit" in (cls + " " + parent_cls).lower():
            continue
        t = p.get_text(strip=True)
        if t and "hide caption" not in t.lower():
            paras.append(t)
    text = "\n".join(paras)
    if not text and body:
        text = body.get_text(" ", strip=True)

    return {"url": url, "title": title, "date": date,
            "text": text, "word_count": len(text.split())}


def scrape(refresh):
    if os.path.exists(RAW_PATH) and not refresh:
        with open(RAW_PATH) as f:
            articles = json.load(f)
        print(f"Using cached collection: {len(articles)} articles")
        return

    urls = collect_article_urls()
    print(f"\nFetching up to {min(len(urls), TARGET)} article pages...")
    articles, seen = [], set()
    for i, url in enumerate(urls):
        if len(articles) >= TARGET:
            break
        if url in seen:
            continue
        seen.add(url)
        try:
            art = fetch_article(url)
            if len(art["text"].split()) >= 100:  # skip teasers / empty pages
                articles.append(art)
                print(f"  [{len(articles):2d}] {art['title'][:70]}")
        except requests.RequestException as e:
            print(f"  !! failed: {url} ({e})", file=sys.stderr)
        time.sleep(DELAY)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(RAW_PATH, "w") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(articles)} articles -> {RAW_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-scrape instead of using the cache")
    args = parser.parse_args()
    scrape(args.refresh)
