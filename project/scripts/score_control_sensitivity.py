import argparse, json
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer, util

from project.src.sentiment_utils import compute_sentiment, lexical_stats, get_vader, get_transformer

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def analyze(text, method, analyzer):
    stats = lexical_stats(text)
    stats["polarity"] = compute_sentiment(text, method=method, analyzer=analyzer)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="project/outputs")
    ap.add_argument("--out_csv", default="project/outputs/sensitivity.csv")
    ap.add_argument("--method", choices=["vader", "transformer"], default="vader",
                    help="Sentiment analysis backend (vader or transformer)")
    args = ap.parse_args()

    analyzer = get_vader() if args.method == "vader" else get_transformer()

    print(f"[INFO] Loading jsonl files from {args.input_dir}")
    paths = list(Path(args.input_dir).rglob("*.jsonl"))
    if not paths:
        print("[WARN] No .jsonl files found.")
        return

    rows = []
    for p in paths:
        for rec in read_jsonl(p):
            metrics = analyze(rec.get("response", ""), args.method, analyzer)
            rows.append({
                "file": p.name,
                "frame": rec.get("frame"),
                "topic": rec.get("topic", ""),
                "prompt": rec.get("prompt", ""),
                "response": rec.get("response", ""),
                **metrics
            })

    df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved raw metrics → {out_path}")

    # ---------- Embedding Similarity ----------
    print("[INFO] Calculating embedding-based similarities ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    sim_records = []
    for prompt, group in df.groupby("prompt"):
        if len(group) < 2:
            continue
        emb = model.encode(group["response"].tolist(), convert_to_tensor=True)
        sim = util.pytorch_cos_sim(emb, emb)
        avg_sim = sim.mean().item()
        sim_records.append({
            "prompt": prompt,
            "avg_embedding_similarity": round(avg_sim, 4),
            "frames": list(group["frame"].unique())
        })

    sim_df = pd.DataFrame(sim_records)
    sim_df.to_csv("project/outputs/embedding_similarity.csv", index=False)
    print("[OK] Saved embedding similarity scores.")

    # ---------- Summary ----------
    print("\n=== Control Sensitivity Summary ===")
    summary = df.groupby("frame")[["length", "ttr", "polarity"]].agg(["mean", "std"]).round(3)
    print(summary)

    print("\nAverage embedding similarity per prompt:")
    print(sim_df.head())

if __name__ == "__main__":
    main()