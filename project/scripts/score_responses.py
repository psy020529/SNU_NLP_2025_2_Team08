# right now, just length / diversity(ttr) / pos-neg word counts

import argparse, json, re
from pathlib import Path
import pandas as pd

# 긍/부정 단어 placeholder, 나중에 모듈 임포트 해야할듯. 
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
    frame_score = (pos - neg) / max(1, n)
    return {"length": n, "ttr": ttr, "pos": pos, "neg": neg, "frame_score": frame_score}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="jsonl files from run_infer.py")
    ap.add_argument("--out_csv", default="project/outputs/scores/scores.csv")
    args = ap.parse_args()

    records = []
    for path in args.inputs:
        path = Path(path)
        for rec in read_jsonl(path):
            metrics = analyze(rec.get("response", ""))
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

    summary = df.groupby("frame")[["length","ttr","pos","neg","frame_score"]].mean().round(3)
    print(summary)

if __name__ == "__main__":
    main()