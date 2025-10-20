"""
Mini LoRA fine-tuning test for CPU / limited resources.
Runs with ~500 samples for 1 short epoch to verify the pipeline.
"""
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch


def load_small_dataset(path, limit=500):
    """Load JSONL instruction dataset and sample subset."""
    ds = load_dataset("json", data_files=path)["train"]
    if len(ds) > limit:
        ds = ds.shuffle(seed=42).select(range(limit))
    print(f"[INFO] Loaded {len(ds)} samples from {path}")
    return ds


def tokenize_dataset(ds, tokenizer):
    def preprocess(ex):
        model_in = tokenizer(
            ex["instruction"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            ex["response"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        model_in["labels"] = labels["input_ids"]
        return model_in

    return ds.map(preprocess, batched=True, remove_columns=ds.column_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="google/flan-t5-small")
    ap.add_argument("--dataset_path", default="project/data/instructions/lamini_10000.jsonl")
    ap.add_argument("--output_dir", default="project/models/flan_t5_lora_mini")
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    ds = load_small_dataset(args.dataset_path, limit=args.limit)
    tokenized = tokenize_dataset(ds, tokenizer)

    # 2. Load model + LoRA config
    print("[INFO] Loading base model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Training setup
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    args_train = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=0.5,      # half epoch
        learning_rate=1e-4,
        logging_steps=10,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=tokenized,
        data_collator=collator,
    )

    print("[INFO] Starting mini LoRA fine-tuning (sanity check)...")
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[OK] Mini LoRA model saved to {args.output_dir}")


if __name__ == "__main__":
    main()