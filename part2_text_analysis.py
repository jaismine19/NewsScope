"""
CSIS 4260 - Assignment 2
Part 2 (step 2): text analysis of the scraped articles.

Four algorithms are applied to every article:

  1. LexRank          - extractive summarization (produces the "summary")
  2. TextRank         - a second extractive summarizer, used to cross-check
                        LexRank (both are graph-based sentence rankers)
  3. VADER            - rule-based sentiment analysis -> direction
                        (positive / negative) and magnitude
  4. TF-IDF + cosine  - topic relevance: how close each article is to a
                        technology/AI topic profile built from keywords

The importance score combines relevance, coverage (article length) and
sentiment magnitude, and takes the direction from VADER:

    score = direction * (0.5*relevance + 0.3*coverage + 0.2*|sentiment|) * 100

so scores range from -100 (strongly negative, off-topic-ish) to
+100 (strongly positive, very relevant).

Outputs:
    articles.csv                 url / title / date / score / direction /
                                 sentiment / relevance / summary  (one row
                                 per article, the required tabulated form)
    benchmarks/part2_analysis.json   supporting stats (summary lengths,
                                 algorithm agreement, score distribution)
"""

import json
import os
import re
import statistics

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
BENCH_DIR = os.path.join(BASE_DIR, "benchmarks")
RAW_PATH = os.path.join(DATA_DIR, "articles_raw.json")
CSV_PATH = os.path.join(DATA_DIR, "articles.csv")
ANALYSIS_JSON = os.path.join(BENCH_DIR, "part2_analysis.json")

TOPIC = "technology"
TOPIC_PROFILE = (
    "artificial intelligence machine learning AI model algorithm chip "
    "semiconductor Nvidia OpenAI chatbot robot software data privacy "
    "social media app startup innovation computing cyber security cloud "
    "automation digital technology"
)
SUMMARY_SENTENCES = 3
MAX_WORDS = 800  # a "full coverage" article length for the coverage factor


def ensure_nltk_data():
    """Make sure the small NLTK data files exist.

    Tries the normal NLTK downloader first. If that fails (e.g. SSL
    certificate problems on some machines), falls back to fetching the
    official files from GitHub with requests and placing them where
    NLTK looks for them.
    """
    import io
    import zipfile

    import nltk
    import requests as _requests

    base = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/"
    data_home = nltk.data.path[0]

    def has(check_path):
        try:
            nltk.data.find(check_path)
            return True
        except LookupError:
            return False

    # punkt sentence tokenizers (sumy needs these as extracted folders)
    for zip_name, check in [("tokenizers/punkt.zip", "tokenizers/punkt"),
                            ("tokenizers/punkt_tab.zip", "tokenizers/punkt_tab")]:
        if has(check):
            continue
        try:
            nltk.download(check.split("/")[-1], quiet=True)
        except Exception:  # noqa: BLE001
            pass
        if has(check):
            continue
        r = _requests.get(base + zip_name, timeout=60)
        r.raise_for_status()
        target = os.path.join(data_home, "tokenizers")
        os.makedirs(target, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(target)

    # VADER's lexicon: VADER reads it from a zip file on disk
    vader_zip = os.path.join(data_home, "sentiment", "vader_lexicon.zip")
    if not os.path.isfile(vader_zip):
        try:
            nltk.download("vader_lexicon", quiet=True)
        except Exception:  # noqa: BLE001
            pass
        if not os.path.isfile(vader_zip):
            r = _requests.get(base + "sentiment/vader_lexicon.zip", timeout=60)
            r.raise_for_status()
            os.makedirs(os.path.dirname(vader_zip), exist_ok=True)
            with open(vader_zip, "wb") as fh:
                fh.write(r.content)


ensure_nltk_data()


CAPTION_MARKERS = (
    "hide caption", "toggle caption", "getty images", "screenshot by",
    "via ap", "courtesy of", "caption", "photo credit",
)


def clean(text):
    text = re.sub(r"\s+", " ", text or "")
    # drop sentences that are just image captions / photo credits
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [p for p in parts
            if not any(m in p.lower() for m in CAPTION_MARKERS)]
    if kept:
        return " ".join(kept).strip()
    return text.strip()


def summarize(text, summarizer_cls):
    """Extractive summary with sumy; fall back to leading sentences."""
    text = clean(text)
    if len(text.split()) < 30:
        return text
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = summarizer_cls()
        sentences = summarizer(parser.document, SUMMARY_SENTENCES)
        return " ".join(str(s) for s in sentences)
    except Exception:  # noqa: BLE001
        sents = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sents[:SUMMARY_SENTENCES])


def main():
    os.makedirs(BENCH_DIR, exist_ok=True)
    with open(RAW_PATH) as f:
        articles = json.load(f)
    print(f"Analyzing {len(articles)} articles")

    # ---------------- VADER sentiment ----------------
    sid = SentimentIntensityAnalyzer()
    for a in articles:
        a["text"] = clean(a["text"])
        a["sentiment"] = sid.polarity_scores(a["text"])["compound"]

    # ---------------- TF-IDF topic relevance ----------------
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
    tfidf = vectorizer.fit_transform([a["text"] for a in articles])
    topic_vec = vectorizer.transform([TOPIC_PROFILE])
    rel = cosine_similarity(tfidf, topic_vec).ravel()
    for a, r in zip(articles, rel):
        a["relevance"] = float(np.clip(r * 6, 0, 1))  # rescale: raw cosines are tiny

    # ---------------- summaries (LexRank + TextRank cross-check) ----------------
    for a in articles:
        a["summary_lexrank"] = summarize(a["text"], LexRankSummarizer)
        a["summary_textrank"] = summarize(a["text"], TextRankSummarizer)

    # ---------------- importance score ----------------
    for a in articles:
        coverage = min(1.0, a["word_count"] / MAX_WORDS)
        importance = (0.5 * a["relevance"]
                      + 0.3 * coverage
                      + 0.2 * abs(a["sentiment"]))
        a["direction"] = "positive" if a["sentiment"] >= 0 else "negative"
        a["score"] = round((1 if a["sentiment"] >= 0 else -1) * importance * 100, 1)

    # ---------------- tabulated result ----------------
    rows = [{
        "url": a["url"],
        "title": a["title"],
        "date": a["date"],
        "score": a["score"],
        "direction": a["direction"],
        "sentiment": round(a["sentiment"], 3),
        "relevance": round(a["relevance"], 3),
        "summary": a["summary_lexrank"],
    } for a in articles]
    df = pd.DataFrame(rows).sort_values("score", key=abs, ascending=False)
    df.to_csv(CSV_PATH, index=False)

    # agreement between the two summarizers (Jaccard on sentence words)
    def jaccard(s1, s2):
        w1, w2 = set(s1.lower().split()), set(s2.lower().split())
        return len(w1 & w2) / len(w1 | w2) if w1 | w2 else 0.0

    overlaps = [jaccard(a["summary_lexrank"], a["summary_textrank"]) for a in articles]
    stats = {
        "n_articles": len(articles),
        "score_range": [float(df["score"].min()), float(df["score"].max())],
        "positive": int((df["direction"] == "positive").sum()),
        "negative": int((df["direction"] == "negative").sum()),
        "mean_sentiment": float(df["sentiment"].mean()),
        "mean_relevance": float(df["relevance"].mean()),
        "mean_summary_words": float(df["summary"].str.split().str.len().mean()),
        "lexrank_textrank_mean_jaccard": float(statistics.mean(overlaps)),
        "topic": TOPIC,
        "score_formula": "direction * (0.5*relevance + 0.3*coverage + 0.2*|sentiment|) * 100",
    }
    with open(ANALYSIS_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    # ---------------- pretty table for the terminal ----------------
    print("\n" + "=" * 100)
    print(f"Top 10 by |score| (of {len(df)} articles)")
    print("=" * 100)
    show = df.head(10)
    for _, r in show.iterrows():
        arrow = "\u2191" if r["direction"] == "positive" else "\u2193"
        print(f"{arrow} {r['score']:>6.1f}  {r['title'][:58]:58}  {r['url']}")
        print(f"     summary: {r['summary'][:150]}")
    print("\n" + json.dumps(stats, indent=2))
    print(f"\nSaved -> {CSV_PATH}")


if __name__ == "__main__":
    main()
