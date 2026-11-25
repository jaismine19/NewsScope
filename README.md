# NewsScope — scraping the news and making sense of it

**CSIS 4260 — Assignment 2** · Student: **Jaismine Kaur**

This assignment had two parts: figure out how to scrape the web (and which tool
to use), then actually scrape something and analyze it. I collected **55 recent
articles from NPR's technology, business and science sections** and ran four
text-analysis algorithms over them. The end result is a table where every
article has a **summary** and an **importance score** with a direction —
positive or negative.

| Part | What it covers | Code | What it produces |
|------|----------------|------|------------------|
| 1 | Compare two scraping libraries, benchmark them, pick one | `part1_library_benchmark.py` | `benchmarks/part1_results.json`, `benchmarks/part1_charts.png` |
| 2a | Scrape 55 articles from NPR | `part2_scrape_articles.py` | `data/articles_raw.json` |
| 2b | Summaries + sentiment + relevance + importance scores | `part2_text_analysis.py` | `data/articles.csv`, `benchmarks/part2_analysis.json` |

---

## Setup (virtual environment)

Everything runs in a Python 3.13 virtual environment. From the project folder:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` is the dependency file my venv tooling generated for me
(`pip freeze`), so the grader can recreate my exact setup. The `.venv` folder
itself is not uploaded.

One honest heads-up: the analysis step needs three small NLTK data files (the
sentence tokenizer and the sentiment lexicon). The script downloads them on
first run — and because the normal NLTK downloader hit an SSL certificate
problem on my machine, the script also knows how to fetch the same official
files from GitHub itself. Either way it works, no manual step needed.

### Run Part 1 — the library benchmark
```bash
.venv/bin/python part1_library_benchmark.py
```
Fetches 10 real NPR pages with both libraries and measures them, plus a
JavaScript demo. Takes about a minute.

### Run Part 2 — scrape, then analyze
```bash
.venv/bin/python part2_scrape_articles.py            # ~55 articles from NPR
.venv/bin/python part2_text_analysis.py              # summaries + scores -> articles.csv
```

---

## Part 1 — Web scraping: which library?

### The research

The assignment asked me to compare two popular scraping libraries and pick one
based on benchmarks and ease of use. I chose two approaches that are as
different as it gets:

* **requests + BeautifulSoup** — download the HTML with `requests`, parse the
  static document with BeautifulSoup. Light, fast and simple. The catch: it
  only sees what the server sends in that first response. If a page builds its
  content with JavaScript, BeautifulSoup sees an empty shell.
* **Playwright** — drives a real headless Chrome browser. It runs the page's
  JavaScript, waits for content to appear, and reads what a human would
  actually see. Far more powerful, but a whole browser has to run for every
  page, so it is slower and heavier.

### The benchmark

I measured both on the **same 10 NPR article pages** — politely, with a 0.5 s
pause between requests, and Chromium launched once and reused for Playwright
so it got a fair start:

| Measure | requests + BeautifulSoup | Playwright |
|---------|------------------------:|-----------:|
| Pages fetched | 10 / 10 | 10 / 10 |
| Mean time per page | **0.82 s** | 2.33 s |
| Mean download size | 178 KB | 195 KB |
| Mean extracted text | 5,579 chars | 6,462 chars |

On a static site like NPR, BeautifulSoup is about **2.8x faster** and both
end up with the same article text. No drama there.

### The JavaScript demo (why Playwright still matters)

To make the comparison fair rather than one-sided, I also tested both on a
page whose content only appears after JavaScript runs
(`quotes.toscrape.com/js`):

| Library | Quotes found |
|---------|-------------:|
| BeautifulSoup (no JS) | **0** |
| Playwright (runs JS) | **10** |

BeautifulSoup saw literally nothing — the page is an empty shell until the
JavaScript executes. That is exactly the situation where a full browser is
worth every millisecond, and I wanted the numbers to show it.

Chart: `benchmarks/part1_charts.png`.

### My choice

**requests + BeautifulSoup** for this assignment. NPR serves plain static
HTML, so the faster, lighter tool does the job — 2.8x quicker per page, same
text, and far less setup. If the target site ever renders its content with
JavaScript, I would switch to Playwright without hesitation (the demo above is
my evidence for that call).

---

## Part 2 — Scraping 55 articles and analyzing them

### Step 1: the scrape (and what went wrong along the way)

Picking the site was its own little adventure. I tried the obvious candidates
first: Reuters politely refused (HTTP 401), The Guardian returned 403, BBC and
NPR both worked. NPR won because its articles are clean, static HTML with the
body text in one predictable place.

Then NPR threw me a curveball: its section pages **ignore the `?page=`
parameter** — every page I asked for quietly returned the same first 24
articles. That is how I learned scraping is never quite as easy as the
tutorials. The fix was simple: walk three sections (technology, business,
science) instead of paginating one, which gave me plenty of unique links.

I then fetched 55 article pages with requests + BeautifulSoup and extracted
the title, date and body text for each. To be a good citizen I identified
myself with a normal browser user-agent, waited ~0.6 s between requests, and
only visited each public page once. The raw collection is cached in
`data/articles_raw.json`, so the analysis step can be rerun without touching
the site again.

One more small catch: my first summaries kept quoting NPR's image captions
("hide caption", photo credits). That was because I was pulling every
paragraph under the story container, captions included. I fixed it at the
source — the scraper now skips caption and credit containers — and the
summaries read like actual article sentences.

### Step 2: four text-analysis algorithms

| Algorithm | What it does | Where it's used |
|-----------|--------------|-----------------|
| **LexRank** | graph-based extractive summarization — ranks sentences by how well they represent the article | produces the `summary` column |
| **TextRank** | a second graph-based summarizer | cross-check for LexRank (mean Jaccard agreement: 0.38 — same idea, different sentence picks) |
| **VADER** | rule-based sentiment analysis built for social/news text | gives the direction (positive/negative) and its strength |
| **TF-IDF + cosine** | measures how close each article is to a topic profile of technology/AI keywords | the relevance part of the score |

I picked these because they are all free, run in seconds on a laptop, and are
easy to explain — no paid APIs needed (the assignment allowed up to $5 on a
commercial API; I simply never needed one).

### The importance score

Each article gets a score from **-100 to +100**:

```
score = direction * (0.5 * relevance + 0.3 * coverage + 0.2 * |sentiment|) * 100
```

* **relevance** — TF-IDF cosine similarity of the article to the
  technology/AI topic profile (0..1). Weighted most, because the assignment
  asks for importance *in the context of my topic*.
* **coverage** — how much substance the article has, by length (0..1, capped
  at 800 words). A two-paragraph brief matters less than a deep dive.
* **sentiment** — VADER's compound score; its sign is the **direction**
  (positive or negative), its magnitude shows how strongly the article leans.

The weights (0.5 / 0.3 / 0.2) are my judgment call — relevance first, then
depth, then emotional charge — and they are stated right there in the code so
the grader can see exactly what happened.

### The results (tabulated)

The full table is in `data/articles.csv` — one row per article with
`url, title, date, score, direction, sentiment, relevance, summary`.

| Overall | Value |
|---------|-------|
| Articles analyzed | 55 |
| Score range | -86.3 to +98.8 |
| Positive articles | 46 |
| Negative articles | 9 |
| Mean sentiment | 0.65 |
| Mean relevance | 0.28 |
| Mean summary length | ~72 words |

A few honest notes on these numbers:

* Most articles score positive, which mostly reflects the tone of NPR's tech
  coverage in this window — AI hype, record IPOs, "AI helps doctors" stories.
  VADER is also known to lean optimistic on news prose, so I read the scores
  as *relative*: the most negative articles really are the lawsuits, market
  crashes and "killer robots" stories, which is what the direction is for.
* The top scores are squarely on-topic AI stories (Nvidia, OpenAI, Anthropic),
  and the negative tail is led by "California sues OpenAI" at -86.3 — which
  feels right.

### What the two summarizers agree on

LexRank and TextRank are cousins, so I ran both and compared them. Their mean
Jaccard word-overlap is 0.38: they often pick the same key sentences, but
order and framing differ. I published the LexRank summary in the CSV and kept
the TextRank one internally for the cross-check — two algorithms used, and a
number showing they genuinely agree on what matters in each article.

---

## Files

```
CSIS4260_Assignment2/
├── README.md                        ← this file
├── requirements.txt                 ← generated by the venv (pip freeze)
├── part1_library_benchmark.py       ← Part 1: BS4 vs Playwright benchmark
├── part2_scrape_articles.py         ← Part 2 step 1: the NPR scrape
├── part2_text_analysis.py           ← Part 2 step 2: summaries + scores
├── data/
│   ├── articles_raw.json            ← the scraped articles (cache)
│   └── articles.csv                 ← the required table: url, summary, score
└── benchmarks/
    ├── part1_results.json           ← benchmark measurements
    ├── part1_charts.png             ← benchmark chart
    └── part2_analysis.json          ← analysis statistics
```
