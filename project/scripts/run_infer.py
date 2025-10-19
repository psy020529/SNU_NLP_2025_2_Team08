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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts_csv", default="project/data/prompts.csv")
    ap.add_argument("--frame", default="NEUTRAL", choices=["NEUTRAL","PRO","CON"])
    ap.add_argument("--model_id", default="google/flan-t5-small")
    args = ap.parse_args()

    df = pd.read_csv(args.prompts_csv)
    sys_prompt = make_system_prompt(args.frame)

    inputs = [
        f"{sys_prompt}\n\nUser: {row.prompt}\nAssistant:"
        for _, row in df.iterrows()
    ]

    tok, mdl, device = load(args.model_id)
    outputs = generate_batch(inputs, tok, mdl, device)

    out_dir = Path("project/outputs/baseline_responses" if args.frame=="NEUTRAL"
                   else "project/outputs/manipulated_responses")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.frame.lower()}.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for i, ((_, row), resp) in enumerate(zip(df.iterrows(), outputs), start=1):
            rec = {"id": int(getattr(row, "id", i)),
                   "topic": getattr(row, "topic", ""),
                   "frame": args.frame,
                   "prompt": row.prompt,
                   "response": resp}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] saved → {out_path}")

if __name__ == "__main__":
    main()