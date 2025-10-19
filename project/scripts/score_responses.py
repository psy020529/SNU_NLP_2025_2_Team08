# right now, just length / diversity(ttr) / pos-neg word counts

import argparse, json
from pathlib import Path
import pandas as pd
from project.src.sentiment_utils import compute_sentiment, lexical_stats, get_vader, get_transformer

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def analyze(text, method, analyzer):
    stats = lexical_stats(text)
    polarity = compute_sentiment(text, method=method, analyzer=analyzer)
    stats["polarity"] = polarity
    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="jsonl files from run_infer.py")
    ap.add_argument("--out_csv", default="project/outputs/scores/scores.csv")
    ap.add_argument("--method", choices=["vader", "transformer"], default="vader",
                    help="Sentiment analysis backend to use")
    args = ap.parse_args()

    # initialize analyzer once
    analyzer = get_vader() if args.method == "vader" else get_transformer()

    records = []
    for path in args.inputs:
        path = Path(path)
        for rec in read_jsonl(path):
            metrics = analyze(rec.get("response", ""), args.method, analyzer)
            records.append({
                "file": path.name,
                "id": rec.get("id"),
                "topic": rec.get("topic"),
                "frame": rec.get("frame"),
                **metrics
            })

    df = pd.DataFrame(records)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] saved scores → {out_path}\n")

    summary = df.groupby("frame")[["length","ttr","polarity"]].mean().round(3)
    print(summary)

if __name__ == "__main__":
    main()