import argparse, json, re
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# 간단 긍부정 단어 목록
POS_WORDS = {"good","benefit","positive","improve","advantage","growth","safe","support"}
NEG_WORDS = {"bad","risk","negative","harm","problem","danger","fail","threat"}

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def analyze(text):
    tokens = re.findall(r"\w+", text.lower())
    n = len(tokens)
    unique = len(set(tokens)) if n else 0
    ttr = unique / n if n else 0
    pos = sum(w in POS_WORDS for w in tokens)
    neg = sum(w in NEG_WORDS for w in tokens)
    polarity = (pos - neg) / max(1, n)
    return {"len": n, "ttr": ttr, "polarity": polarity}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="project/outputs")
    ap.add_argument("--out_csv", default="project/outputs/sensitivity.csv")
    args = ap.parse_args()

    # 모든 jsonl 파일 읽기
    paths = list(Path(args.input_dir).rglob("*.jsonl"))
    rows = []
    for p in paths:
        for rec in read_jsonl(p):
            meta = {"file": p.name, "frame": rec.get("frame"), "topic": rec.get("topic", ""), "prompt": rec.get("prompt", ""), "response": rec.get("response", "")}
            meta.update(analyze(rec.get("response", "")))
            rows.append(meta)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved raw metrics → {args.out_csv}")

    # ---------------- Embedding Similarity ----------------
    print("[Info] Calculating embedding-based similarities ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # prompt별 그룹화
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
            "frames": list(group["frame"].unique()),
        })

    sim_df = pd.DataFrame(sim_records)
    sim_df.to_csv("project/outputs/embedding_similarity.csv", index=False)
    print("[OK] Saved embedding similarity scores.")

    # ---------------- Summary ----------------
    print("\n=== Control Sensitivity Summary ===")
    summary = df.groupby("frame")[["len", "ttr", "polarity"]].agg(["mean", "std"]).round(3)
    print(summary)
    print("\nAverage embedding similarity per prompt:")
    print(sim_df.head())

if __name__ == "__main__":
    main()