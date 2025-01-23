#!/usr/bin/env python3

import sys
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score
import torch


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.tensor(logits).argmax(dim=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def main():
    if len(sys.argv) != 2:
        sys.exit(1)

    subset_path = sys.argv[1]
    if not os.path.isfile(subset_path):
        sys.exit(1)

    # load dataset and prepare
    dataset = load_dataset("json", data_files=subset_path, split="train")
    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    dataset["eval"] = dataset.pop("test")

    # load tokenizer and tokenize
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    # load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    # training arguments
    # controls training parameters like batch size, epochs, and saving behavior
    training_args = TrainingArguments(
        output_dir="bert_finetuned",
        per_device_train_batch_size=8,
        num_train_epochs=2,
        save_strategy="no",
        logging_strategy="no",
        evaluation_strategy="no",
        disable_tqdm=True,
        report_to=[],
        log_level="error"
    )

    # initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # train and evaluate
    trainer.train()
    eval_results = trainer.evaluate(tokenized_datasets["eval"])
    print(eval_results["accuracy"])

    # save the model
    trainer.save_model("bert_finetuned")


if __name__ == "__main__":
    main()
