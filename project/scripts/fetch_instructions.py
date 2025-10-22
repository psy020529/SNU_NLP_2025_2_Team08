import argparse, json, random
from pathlib import Path

from datasets import load_dataset
import pandas as pd

def save_jsonl(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] saved → {out_path}  ({len(rows)} rows)")


# -------------------------
# LaMini-instruction (MBZUAI)
# -------------------------
def fetch_lamini(limit=10000, seed=42, split="train"):
    ds = load_dataset("MBZUAI/LaMini-instruction", split=split)
    # Columns: instruction, response (a.k.a. "output" in some mirrors)
    cols = ds.column_names
    inst_col = "instruction"
    resp_col = "response" if "response" in cols else ("output" if "output" in cols else None)
    if resp_col is None:
        raise ValueError(f"Cannot find response/output column in {cols}")
    # filter empty & English-ish (best-effort)
    df = ds.to_pandas()
    df = df[df[inst_col].astype(str).str.len() > 0]
    df = df[df[resp_col].astype(str).str.len() > 0]
    # sample
    if limit and len(df) > limit:
        df = df.sample(n=limit, random_state=seed)
    rows = [{"instruction": a, "response": b} for a, b in zip(df[inst_col], df[resp_col])]
    return rows


# -------------------------
# OpenAssistant / OASST1 (pairs)
# -------------------------
def fetch_oasst1_pairs(limit=5000, seed=42, lang="en", split="train"):
    """
    Build (instruction, response) pairs:
      - instruction: a prompter message
      - response: its direct assistant reply
    """
    ds = load_dataset("OpenAssistant/oasst1", split=split)
    df = ds.to_pandas()

    # Common columns in oasst1: "text","role","lang","message_id","parent_id"
    need_cols = {"text", "role", "lang"}
    if not need_cols.issubset(set(df.columns)):
        raise ValueError(f"Unexpected columns in OASST1: {df.columns}")

    # Filter language
    if "lang" in df.columns and lang:
        df = df[df["lang"] == lang]

    # Keep minimal columns
    keep = ["text", "role", "lang"]
    if "message_id" in df.columns: keep.append("message_id")
    if "parent_id" in df.columns: keep.append("parent_id")
    df = df[keep].copy()

    # Index for fast lookup by parent_id
    # Strategy: find assistant rows, map to their parent prompter rows.
    if "parent_id" in df.columns and "message_id" in df.columns:
        by_id = df.set_index("message_id")
        pairs = []
        for idx, row in df.iterrows():
            if row["role"] != "assistant": 
                continue
            pid = row.get("parent_id", None)
            if pd.isna(pid) or pid not in by_id.index:
                continue
            parent = by_id.loc[pid]
            if parent.get("role", None) != "prompter":
                continue
            inst = str(parent.get("text", "")).strip()
            resp = str(row.get("text", "")).strip()
            if inst and resp:
                pairs.append({"instruction": inst, "response": resp})
    else:
        # Fallback: very conservative pairing by adjacent rows (rarely needed)
        pairs, prev = [], None
        for _, r in df.iterrows():
            if r["role"] == "prompter":
                prev = r["text"]
            elif r["role"] == "assistant" and prev:
                inst = str(prev).strip()
                resp = str(r["text"]).strip()
                if inst and resp:
                    pairs.append({"instruction": inst, "response": resp})
                prev = None

    random.Random(seed).shuffle(pairs)
    if limit and len(pairs) > limit:
        pairs = pairs[:limit]
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["lamini", "oasst1"], required=True)
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--lang", type=str, default="en", help="OASST1 language filter")
    ap.add_argument("--out_dir", type=str, default="project/data/instructions")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "lamini":
        rows = fetch_lamini(limit=args.limit, seed=args.seed, split=args.split)
        out_path = out_dir / f"lamini_{args.limit}.jsonl"
        save_jsonl(rows, out_path)

    elif args.dataset == "oasst1":
        rows = fetch_oasst1_pairs(limit=args.limit, seed=args.seed, lang=args.lang, split=args.split)
        out_path = out_dir / f"oasst1_{args.lang}_{args.limit}.jsonl"
        save_jsonl(rows, out_path)


if __name__ == "__main__":
    main()