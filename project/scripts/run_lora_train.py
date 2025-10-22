"""
Simple LoRA fine-tuning script for Flan-T5 using instruction data (e.g., LaMini).
"""
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import torch


def load_instruction_dataset(path_or_name: str, limit: int = 10000):
    """
    path_or_name: either Hugging Face dataset name or local JSONL path
    """
    if Path(path_or_name).exists():
        ds = load_dataset("json", data_files=path_or_name)
    else:
        ds = load_dataset(path_or_name)

    ds = ds["train"] if "train" in ds else ds
    if limit and len(ds) > limit:
        ds = ds.shuffle(seed=42).select(range(limit))
    print(f"[INFO] Loaded dataset: {len(ds)} samples")

    def preprocess(ex):
        return {"instruction": ex["instruction"], "response": ex["response"]}
    return ds.map(preprocess)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="google/flan-t5-small")
    ap.add_argument("--dataset_path", default="project/data/instructions/lamini_10000.jsonl")
    ap.add_argument("--output_dir", default="project/models/flan_t5_lora")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=10000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------- Load dataset ---------
    dataset = load_instruction_dataset(args.dataset_path, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    def tokenize_fn(batch):
        inputs = tokenizer(
            batch["instruction"],
            max_length=256,
            truncation=True,
            padding="max_length"
        )
        outputs = tokenizer(
            batch["response"],
            max_length=256,
            truncation=True,
            padding="max_length"
        )
        inputs["labels"] = outputs["input_ids"]
        return inputs

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    # --------- Model + LoRA ---------
    print(f"[INFO] Loading model {args.model_id}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)

    # LoRA 설정
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],  # attention weight matrices
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # --------- Training ---------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_steps=500,
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator
    )

    print("[INFO] Starting LoRA fine-tuning ...")
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[OK] Saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()