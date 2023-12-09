import argparse
import json
from typing import Dict, List, Optional

import evaluate
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="glue")
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--text_fields", nargs="+", default=["sentence"])
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--shuffle_dataset", action="store_true", default=False)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--metric", type=str, default="accuracy")
    parser.add_argument("--eval_split", type=str, default="validation")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(args.dataset_name, args.dataset_config_name)
    num_classes = dataset["train"].features["label"].num_classes

    if args.train_file is not None:
        dataset["train"] = load_from_disk(args.train_file)["train"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess_function(examples):
        parts = [examples[field] for field in args.text_fields]
        text = " [SEP] ".join(parts)
        return tokenizer(text, truncation=True)

    dataset = dataset.map(preprocess_function, batched=False)
    if args.shuffle_dataset:
        dataset = dataset.shuffle(seed=args.seed)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    metric = evaluate.load(args.metric)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_classes
    ).to(device)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        save_strategy=args.save_strategy,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=False,
        report_to="wandb",
        logging_steps=args.logging_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset[args.eval_split],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
