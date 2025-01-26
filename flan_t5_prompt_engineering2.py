import sys
import argparse
import random
import os

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset


def load_dataset(json_path: str) -> pd.DataFrame:
    # Read line-delimited JSON
    df = pd.read_json(json_path, lines=True)

    # Convert numeric labels to string labels
    # if label==1 => 'positive', label==0 => 'negative'
    df["label"] = df["label"].apply(lambda x: "positive" if x == 1 else "negative")

    return df


def stratified_sample(df: pd.DataFrame, n: int = 50, seed: int = 42) -> pd.DataFrame:
    # This method returns a stratified sample, so it's balanced in different labels
    random.seed(seed)

    # Number of samples per class
    half_n = n // 2

    # Split DataFrame by label
    df_positive = df[df["label"].str.lower() == "positive"]
    df_negative = df[df["label"].str.lower() == "negative"]

    # Sample from each class
    sampled_pos = df_positive.sample(half_n, random_state=seed)
    sampled_neg = df_negative.sample(half_n, random_state=seed)

    # Combine and shuffle
    sampled = pd.concat([sampled_pos, sampled_neg])
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)

    return sampled


def create_zero_shot_prompt(review_text: str) -> str:
    # Updated Zero-shot classification prompt
    return (
        "Dive into the emotions of this movie review: Is it 'positive' or 'negative'?\n"
        f"{review_text}"
    )


def create_few_shot_prompt(review_text: str) -> str:
    # Few-shot classification using 5 examples
    return (
        "Here are examples of movie reviews and their sentiment:\n\n"
        "'I absolutely loved this movie!' → positive\n"
        "'This film was a complete waste of time.' → negative\n"
        "'The cinematography was stunning and the story was gripping.' → positive\n"
        "'I found the movie boring and the characters unrelatable.' → negative\n"
        "'An inspiring journey with stellar performances.' → positive\n\n"
        "Now classify this review:\n"
        f"{review_text}"
    )


def create_instruction_prompt(review_text: str) -> str:
    # Instruction-based classification prompt with more detailed instructions
    return (
        "You are an expert in sentiment analysis for movie reviews. Carefully read the following review and determine its sentiment. "
        "Your response should be either 'positive' or 'negative' only, without any additional commentary or explanation:\n"
        f"{review_text}"
    )


def generate_prediction(model, tokenizer, prompt: str, device: str = "cpu") -> str:
    # Tokenize with truncation to ensure input fits the model's max length
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

    # If not exactly 'positive' or 'negative', return 'invalid'
    if "positive" in pred_text and "negative" in pred_text:
        return "invalid"
    elif "positive" in pred_text:
        return "positive"
    elif "negative" in pred_text:
        return "negative"
    else:
        return "invalid"


def fine_tune_model(df: pd.DataFrame, model_name: str, output_dir: str, seed: int = 42):
    # Prepare dataset for fine-tuning
    dataset = Dataset.from_pandas(df)

    # Define the input and target
    def preprocess_function(examples):
        inputs = examples["text"]
        targets = examples["label"]
        return {"input_text": inputs, "target_text": targets}

    tokenized_dataset = dataset.map(preprocess_function, remove_columns=["text", "label"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenize inputs and targets
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=512,
            truncation=True,
        )
        labels = tokenizer(
            examples["target_text"],
            max_length=10,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = tokenized_dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_steps=10_000,
        save_total_limit=2,
        seed=seed,
        logging_steps=500,
        evaluation_strategy="no",
        overwrite_output_dir=True,  # Always overwrite
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser(description="FLAN-T5 Prompt Engineering for IMDb Sentiment Classification (Line-Delimited JSON).")
    parser.add_argument("json_file", type=str, help="Path to your line-delimited JSON file (e.g. imdb_subset.json).")
    parser.add_argument("output_file", type=str, help="Path to write the classification results (e.g., results.txt).")
    parser.add_argument("--sample_size", type=int, default=50, help="Number of reviews to sample (25 pos, 25 neg).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Ensure the output_file has a .txt extension
    if not args.output_file.endswith('.txt'):
        args.output_file += '.txt'

    # Load dataset
    df = load_dataset(args.json_file)

    # Stratified sample
    df_sample = stratified_sample(df, n=args.sample_size, seed=args.seed)

    # Fine-Tune the model every time
    fine_tuned_model_dir = "flan-t5-small-finetuned-imdb"
    print("Fine-tuning the model...")
    fine_tune_model(df_sample, "google/flan-t5-small", fine_tuned_model_dir, seed=args.seed)
    print("Fine-tuning completed and model saved.")

    # Load fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    zero_shot_preds = []
    few_shot_preds = []
    instr_preds = []
    true_labels = []
    output_lines = []

    # Define prompts used
    zero_shot_prompt_example = "Dive into the emotions of this movie review: Is it 'positive' or 'negative'?\n{review_text}"
    few_shot_prompt_example = (
        "Here are examples of movie reviews and their sentiment:\n\n"
        "'I absolutely loved this movie!' → positive\n"
        "'This film was a complete waste of time.' → negative\n"
        "'The cinematography was stunning and the story was gripping.' → positive\n"
        "'I found the movie boring and the characters unrelatable.' → negative\n"
        "'An inspiring journey with stellar performances.' → positive\n\n"
        "Now classify this review:\n{review_text}"
    )
    instruction_prompt_example = (
        "You are an expert in sentiment analysis for movie reviews. Carefully read the following review and determine its sentiment. "
        "Your response should be either 'positive' or 'negative' only, without any additional commentary or explanation:\n{review_text}"
    )

    # Classify each review
    for i, row in df_sample.iterrows():
        text = row["text"]
        label = row["label"].strip().lower()
        true_labels.append(label)

        # Zero-shot classification
        z_prompt = create_zero_shot_prompt(text)
        z_pred = generate_prediction(model, tokenizer, z_prompt, device=device)
        zero_shot_preds.append(z_pred)

        # Few-shot classification
        f_prompt = create_few_shot_prompt(text)
        f_pred = generate_prediction(model, tokenizer, f_prompt, device=device)
        few_shot_preds.append(f_pred)

        # Instruction-based classification
        i_prompt = create_instruction_prompt(text)
        i_pred = generate_prediction(model, tokenizer, i_prompt, device=device)
        instr_preds.append(i_pred)

        # Save details
        output_lines.append(f"Review {i + 1}: {text}")
        output_lines.append(f"Review {i + 1} true label: {label}")
        output_lines.append(f"Review {i + 1} zero-shot: {z_pred}")
        output_lines.append(f"Review {i + 1} few-shot: {f_pred}")
        output_lines.append(f"Review {i + 1} instruction-based: {i_pred}")
        output_lines.append("")

    # Append the prompts used
    output_lines.append("=== Prompts Used ===\n")

    output_lines.append("**Zero-Shot Prompt:**")
    output_lines.append(zero_shot_prompt_example.replace("{review_text}", "{review_text}"))
    output_lines.append("")

    output_lines.append("**Few-Shot Prompt:**")
    output_lines.append(few_shot_prompt_example.replace("{review_text}", "{review_text}"))
    output_lines.append("")

    output_lines.append("**Instruction-Based Prompt:**")
    output_lines.append(instruction_prompt_example.replace("{review_text}", "{review_text}"))
    output_lines.append("")

    # Accuracy calculation
    def accuracy(preds, golds):
        correct = sum(p == g for p, g in zip(preds, golds))
        return correct / len(golds) if golds else 0.0

    zero_acc = accuracy(zero_shot_preds, true_labels)
    few_acc = accuracy(few_shot_preds, true_labels)
    instr_acc = accuracy(instr_preds, true_labels)

    # Append accuracies to output
    output_lines.append("=== Accuracy Results ===")
    output_lines.append(f"Zero-shot accuracy: {zero_acc:.2f}")
    output_lines.append(f"Few-shot accuracy: {few_acc:.2f}")
    output_lines.append(f"Instruction-based accuracy: {instr_acc:.2f}")
    output_lines.append("")

    # Write to file with error handling
    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for line in output_lines:
                f.write(line + "\n")
        print("=== ACCURACY SUMMARY ===")
        print(f"Zero-shot accuracy       : {zero_acc:.2f}")
        print(f"Few-shot accuracy        : {few_acc:.2f}")
        print(f"Instruction-based accuracy: {instr_acc:.2f}")
        print(f"\nAll results saved to {args.output_file}")
    except PermissionError:
        print(f"PermissionError: Cannot write to '{args.output_file}'. Please check the file path and permissions.")


if __name__ == "__main__":
    main()
