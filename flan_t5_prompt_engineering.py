import sys
import argparse
import random

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_dataset(json_path: str) -> pd.DataFrame:
    #read line-delimited JSON
    df = pd.read_json(json_path, lines=True)

    #convert numeric labels to string labels
    #if label==1 => 'positive', label==0 => 'negative'
    df["label"] = df["label"].apply(lambda x: "positive" if x == 1 else "negative")

    return df


def stratified_sample(df: pd.DataFrame, n: int = 50, seed: int = 42) -> pd.DataFrame:
    #this method reutrns a stratified sample, so it's balanced in diffrent labels
    random.seed(seed)

    #number of samples per class
    half_n = n // 2

    #split by label
    df_positive = df[df["label"].str.lower() == "positive"]
    df_negative = df[df["label"].str.lower() == "negative"]

    #sample from each
    sampled_pos = df_positive.sample(half_n, random_state=seed)
    sampled_neg = df_negative.sample(half_n, random_state=seed)

    #combine and shuffle
    sampled = pd.concat([sampled_pos, sampled_neg])
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)

    return sampled


def create_zero_shot_prompt(review_text: str) -> str:
    #zero-shot classification
    return (
        "Classify the following movie review as 'positive' or 'negative':\n"
        f"{review_text}"
    )


def create_few_shot_prompt(review_text: str) -> str:
    #few-shot classifciation using 2 examples
    return (
        "Here are examples of movie reviews and their sentiment:\n\n"
        "'I absolutely loved this movie!' → positive\n"
        "'This film was a complete waste of time.' → negative\n"
        "Now classify this review:\n"
        f"{review_text}"
    )


def create_instruction_prompt(review_text: str) -> str:

    #instruction based classification prompt
    return (
        "You are a sentiment classification expert. Please read the following movie review and "
        "determine whether it is 'positive' or 'negative.' Be concise and respond only with the exact word "
        "'positive' or 'negative':\n"
        f"{review_text}")


def generate_prediction(model, tokenizer, prompt: str, device: str = "cpu") -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

    #if not exactly positive or negative, is invalid
    if "positive" in pred_text and "negative" in pred_text:
        return "invalid"
    elif "positive" in pred_text:
        return "positive"
    elif "negative" in pred_text:
        return "negative"
    else:
        return "invalid"


def main():
    parser = argparse.ArgumentParser(description="FLAN-T5 Prompt Engineering for IMDb Sentiment Classification (Line-Delimited JSON).")
    parser.add_argument("json_file", type=str, help="Path to your line-delimited JSON file (e.g. imdb_subset.json).")
    parser.add_argument("output_file", type=str, help="Path to write the classification results.")
    parser.add_argument("--sample_size", type=int, default=50, help="Number of reviews to sample (25 pos, 25 neg).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    #load dataset
    df = load_dataset(args.json_file)

    #stratified sample of 50 as explained
    df_sample = stratified_sample(df, n=args.sample_size, seed=args.seed)

    #load flan-t5-small
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    zero_shot_preds = []
    few_shot_preds = []
    instr_preds = []
    true_labels = []
    output_lines = []

    #classify each review
    for i, row in df_sample.iterrows():
        text = row["text"]
        label = row["label"].strip().lower()
        true_labels.append(label)

        #zero-shot classification
        z_prompt = create_zero_shot_prompt(text)
        z_pred = generate_prediction(model, tokenizer, z_prompt, device=device)
        zero_shot_preds.append(z_pred)

        #few-shot classificiaton
        f_prompt = create_few_shot_prompt(text)
        f_pred = generate_prediction(model, tokenizer, f_prompt, device=device)
        few_shot_preds.append(f_pred)

        #instruction-based classification
        i_prompt = create_instruction_prompt(text)
        i_pred = generate_prediction(model, tokenizer, i_prompt, device=device)
        instr_preds.append(i_pred)

        #save details
        output_lines.append(f"Review {i + 1}: {text}")
        output_lines.append(f"Review {i + 1} true label: {label}")
        output_lines.append(f"Review {i + 1} zero-shot: {z_pred}")
        output_lines.append(f"Review {i + 1} few-shot: {f_pred}")
        output_lines.append(f"Review {i + 1} instruction-based: {i_pred}")
        output_lines.append("")

    #accuracy calc
    def accuracy(preds, golds):
        correct = sum(p == g for p, g in zip(preds, golds))
        return correct / len(golds) if golds else 0.0

    zero_acc = accuracy(zero_shot_preds, true_labels)
    few_acc = accuracy(few_shot_preds, true_labels)
    instr_acc = accuracy(instr_preds, true_labels)

    #work on accuracies
    output_lines.append("Accuracy Results")
    output_lines.append(f"Zero-shot accuracy: {zero_acc:.2f}")
    output_lines.append(f"Few-shot accuracy: {few_acc:.2f}")
    output_lines.append(f"Instruction-based accuracy: {instr_acc:.2f}")
    output_lines.append("")

    #write to file
    with open(args.output_file, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    print("=== ACCURACY SUMMARY ===")
    print(f"Zero-shot accuracy       : {zero_acc:.2f}")
    print(f"Few-shot accuracy        : {few_acc:.2f}")
    print(f"Instruction-based accuracy: {instr_acc:.2f}")
    print(f"\nAll results saved to {args.output_file}")


if __name__ == "__main__":
    main()
