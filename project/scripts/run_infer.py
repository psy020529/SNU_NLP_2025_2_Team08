import argparse, json
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from project.src.prompts import make_system_prompt

def load(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    return tok, mdl, device


def generate_batch(texts, tok, mdl, device, genkw=None):
    genkw = genkw or {"max_new_tokens": 256, "temperature": 0.7}
    batch = tok(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = mdl.generate(**batch, **genkw)
    return tok.batch_decode(out, skip_special_tokens=True)


def load_data(path: str):
    """Auto-detect format: CSV or JSONL (instruction set)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if p.suffix == ".csv":
        df = pd.read_csv(p)
        # expected columns: id, topic, prompt
        if "prompt" not in df.columns:
            raise ValueError("CSV file must contain a 'prompt' column.")
        return df, "csv"

    elif p.suffix in [".json", ".jsonl"]:
        # instruction-response format
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    rec = json.loads(line)
                    if "instruction" in rec:
                        rows.append({"id": i, "prompt": rec["instruction"], "topic": "instruction"})
                    elif "prompt" in rec:
                        rows.append({"id": i, "prompt": rec["prompt"], "topic": "prompt"})
        df = pd.DataFrame(rows)
        return df, "jsonl"

    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="project/data/prompts.csv",
                    help="Path to prompts.csv or instruction dataset (.jsonl)")
    ap.add_argument("--frame", default="NEUTRAL", choices=["NEUTRAL", "PRO", "CON"])
    ap.add_argument("--model_id", default="google/flan-t5-small")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df, fmt = load_data(args.data_path)
    if args.limit is not None:
        df = df.head(args.limit)
        print(f"[INFO] Using {len(df)} samples (limited)")
        
    sys_prompt = make_system_prompt(args.frame)

    inputs = [
        f"{sys_prompt}\n\nUser: {row.prompt}\nAssistant:"
        for _, row in df.iterrows()
    ]

    tok, mdl, device = load(args.model_id)
    outputs = generate_batch(inputs, tok, mdl, device)

    out_dir = Path("project/outputs/baseline_responses" if args.frame == "NEUTRAL"
                   else "project/outputs/manipulated_responses")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.frame.lower()}_{fmt}.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for i, ((_, row), resp) in enumerate(zip(df.iterrows(), outputs), start=1):
            rec = {
                "id": int(getattr(row, "id", i)),
                "topic": getattr(row, "topic", ""),
                "frame": args.frame,
                "prompt": row.prompt,
                "response": resp
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] saved → {out_path}")


if __name__ == "__main__":
    main()