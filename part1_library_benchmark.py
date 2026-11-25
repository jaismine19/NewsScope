"""
CSIS 4260 - Assignment 2
Part 1: Web scraping - library research and benchmark.

Compares two popular scraping approaches on the same target pages:

  1. requests + BeautifulSoup  (fetch HTML, parse the static DOM)
  2. Playwright                 (drive a real headless Chromium browser)

Benchmarks:
  A. Fetch the same 10 NPR article pages with each approach and measure
     success rate, wall-clock time, bytes downloaded and extracted text.
  B. A JavaScript-rendering demo on a public test site
     (https://quotes.toscrape.com/js/): the quotes only appear after JS
     runs, so BeautifulSoup sees zero quotes while Playwright sees all ten.
     This shows exactly when a full browser is worth the extra cost.

Outputs:
    benchmarks/part1_results.json   raw measurements
    benchmarks/part1_charts.png     chart used in the README
"""

import json
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(BASE_DIR, "benchmarks")
SECTION_URL = "https://www.npr.org/sections/technology/"
JS_DEMO_URL = "https://quotes.toscrape.com/js/"
RESULTS_JSON = os.path.join(BENCH_DIR, "part1_results.json")
CHART_PNG = os.path.join(BENCH_DIR, "part1_charts.png")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

ARTICLE_RE = re.compile(r"https://www\.npr\.org/20\d\d/\d\d/\d\d/[^\"'\s]+")
DELAY = 0.5  # polite pause between requests


def collect_article_urls(n=10):
    """Grab the first n unique article links from the NPR tech section page."""
    r = requests.get(SECTION_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    urls, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ARTICLE_RE.fullmatch(href) and href not in seen:
            seen.add(href)
            urls.append(href)
        if len(urls) >= n:
            break
    return urls


def fetch_with_bs4(url):
    """requests + BeautifulSoup: fetch and extract title + article text."""
    t0 = time.perf_counter()
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else ""
    body = soup.find(id="storytext") or soup.find("article") or soup.body
    text = body.get_text(" ", strip=True) if body else ""
    return {
        "ok": True,
        "seconds": time.perf_counter() - t0,
        "bytes": len(r.content),
        "title": title,
        "text_len": len(text),
    }


def fetch_with_playwright(url, page):
    """Playwright: drive a real browser to the page and read the rendered DOM."""
    t0 = time.perf_counter()
    resp = page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    if not page:
        raise RuntimeError("no page")
    title = page.title()
    body = page.locator("#storytext, article, body").first
    text = body.inner_text(timeout=30_000)
    content = page.content()
    return {
        "ok": True,
        "seconds": time.perf_counter() - t0,
        "bytes": len(content.encode()),
        "title": title,
        "text_len": len(text),
        "status": resp.status if resp else None,
    }


def bench_fetch(urls):
    bs4_rows = []
    for u in urls:
        try:
            bs4_rows.append(fetch_with_bs4(u))
        except Exception as e:  # noqa: BLE001
            bs4_rows.append({"ok": False, "seconds": None, "error": str(e)[:80]})
        time.sleep(DELAY)

    from playwright.sync_api import sync_playwright
    pw_rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for u in urls:
            try:
                pw_rows.append(fetch_with_playwright(u, page))
            except Exception as e:  # noqa: BLE001
                pw_rows.append({"ok": False, "seconds": None, "error": str(e)[:80]})
            time.sleep(DELAY)
        browser.close()
    return bs4_rows, pw_rows


def js_demo():
    """JS-rendering demo: content that only exists after JavaScript runs."""
    out = {}
    # BeautifulSoup sees the raw HTML (no quotes, the page renders them via JS)
    try:
        r = requests.get(JS_DEMO_URL, headers=HEADERS, timeout=30)
        soup = BeautifulSoup(r.text, "lxml")
        out["bs4_quotes"] = len(soup.select(".quote"))
    except Exception as e:  # noqa: BLE001
        out["bs4_quotes"] = None
        out["bs4_error"] = str(e)[:80]

    # Playwright waits for the JS to finish, then counts the same elements
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(JS_DEMO_URL, wait_until="networkidle", timeout=60_000)
            page.wait_for_selector(".quote", timeout=15_000)
            out["playwright_quotes"] = page.locator(".quote").count()
        except Exception as e:  # noqa: BLE001
            out["playwright_quotes"] = None
            out["playwright_error"] = str(e)[:80]
        browser.close()
    return out


def summarize_rows(rows, label):
    ok = [r for r in rows if r.get("ok")]
    times = [r["seconds"] for r in ok]
    return {
        "label": label,
        "attempted": len(rows),
        "succeeded": len(ok),
        "mean_seconds": float(sum(times) / len(times)) if times else None,
        "median_seconds": float(sorted(times)[len(times) // 2]) if times else None,
        "total_seconds": float(sum(times)) if times else None,
        "mean_bytes": float(sum(r["bytes"] for r in ok) / len(ok)) if ok else None,
        "mean_text_len": float(sum(r["text_len"] for r in ok) / len(ok)) if ok else None,
    }


def make_chart(bs4_sum, pw_sum, js):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    labels = ["requests +\nBeautifulSoup", "Playwright\n(browser)"]
    vals = [bs4_sum["mean_seconds"], pw_sum["mean_seconds"]]
    colors = ["#4C72B0", "#DD8452"]
    bars = axes[0].bar(labels, vals, color=colors)
    axes[0].set_ylabel("mean seconds per page")
    axes[0].set_title("Fetching the same 10 NPR article pages")
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}s",
                     ha="center", va="bottom")
    axes[0].grid(axis="y", alpha=0.3)

    libs = ["BeautifulSoup\n(no JS)", "Playwright\n(runs JS)"]
    counts = [js["bs4_quotes"], js["playwright_quotes"]]
    bars = axes[1].bar(libs, counts, color=["#4C72B0", "#DD8452"])
    axes[1].set_ylabel("quotes found")
    axes[1].set_title("JS-rendered page: quotes.toscrape.com/js")
    for b, v in zip(bars, counts):
        axes[1].text(b.get_x() + b.get_width() / 2, v, str(v),
                     ha="center", va="bottom")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("requests + BeautifulSoup vs Playwright", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(CHART_PNG, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(BENCH_DIR, exist_ok=True)
    urls = collect_article_urls(10)
    print(f"Benchmarking {len(urls)} article pages")
    for u in urls[:3]:
        print("  ", u)

    bs4_rows, pw_rows = bench_fetch(urls)
    bs4_sum = summarize_rows(bs4_rows, "requests+BeautifulSoup")
    pw_sum = summarize_rows(pw_rows, "Playwright")
    js = js_demo()

    payload = {
        "section_url": SECTION_URL,
        "urls": urls,
        "bs4": bs4_sum,
        "playwright": pw_sum,
        "js_demo": js,
        "notes": (
            "Same 10 pages fetched with both approaches, 0.5s polite delay, "
            "Chromium launched once for Playwright and reused across pages."
        ),
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nrequests + BeautifulSoup:", json.dumps(bs4_sum, indent=2))
    print("\nPlaywright:", json.dumps(pw_sum, indent=2))
    print("\nJS demo:", json.dumps(js, indent=2))

    make_chart(bs4_sum, pw_sum, js)
    print(f"\nResults -> {RESULTS_JSON}")
    print(f"Chart   -> {CHART_PNG}")


if __name__ == "__main__":
    main()
