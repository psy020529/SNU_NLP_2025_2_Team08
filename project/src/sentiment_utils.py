"""
Utility module for sentiment scoring.
Supports two backends:
  1. VADER (fast, lexicon-based)
  2. Transformer (DistilBERT fine-tuned on SST-2)
"""

import re

# -----------------------------
# Loaders
# -----------------------------
def get_vader():
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()

def get_transformer():
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

# -----------------------------
# Scoring interface
# -----------------------------
def compute_sentiment(text: str, method="vader", analyzer=None):
    """
    Return a sentiment polarity score between -1.0 and +1.0
    """
    if not text or not isinstance(text, str):
        return 0.0

    text = text.strip()
    if method == "vader":
        if analyzer is None:
            analyzer = get_vader()
        score = analyzer.polarity_scores(text)["compound"]
        return round(score, 4)

    elif method == "transformer":
        if analyzer is None:
            analyzer = get_transformer()
        res = analyzer(text[:512])[0]
        polarity = res["score"] if res["label"] == "POSITIVE" else -res["score"]
        return round(float(polarity), 4)

    else:
        raise ValueError(f"Unknown method: {method}")


# -----------------------------
# Helper for lexical stats
# -----------------------------
def lexical_stats(text: str):
    tokens = re.findall(r"\w+", text.lower())
    n = len(tokens)
    unique = len(set(tokens)) if n else 0
    ttr = unique / n if n else 0
    return {"length": n, "ttr": ttr}