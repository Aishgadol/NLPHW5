import os
import sys
import torch
import argparse
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

#setpadtoken
def tokenize_imdb_reviews(dataset, tokenizer, max_length=150):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    def tokenize_func(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    tokenized = dataset.map(tokenize_func, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    return tokenized

#trainmodel
def train_sentiment_model(
        imdb_subset_path,
        sentiment_label,
        output_dir,
        max_length=150,
        num_epochs=3,
        batch_size=4
):
    dataset = load_dataset("json", data_files=imdb_subset_path, split="train")
    filtered_dataset = dataset.filter(lambda x: x["label"] == sentiment_label).select(range(100))
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized_data = tokenize_imdb_reviews(filtered_dataset, tokenizer, max_length=max_length)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10000,
        save_total_limit=2,
        logging_steps=50,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_data,
        data_collator=collator
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir

#generatereviews
def generate_reviews(model_dir, prompt="The movie was", num_reviews=5):
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    reviews_list = []
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    att_mask = inputs.ne(tokenizer.pad_token_id)
    with torch.no_grad():
        for _ in range(num_reviews):
            out = model.generate(
                input_ids=inputs,
                attention_mask=att_mask,
                max_length=50,
                min_length=10,
                temperature=0.79,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            reviews_list.append(text)
    return reviews_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("imdb_subset_path", type=str, help="path to imdb subset json")
    parser.add_argument("generated_reviews_path", type=str, help="path to save reviews txt")
    parser.add_argument("models_output_dir", type=str, help="path to save models")
    args = parser.parse_args()

    os.makedirs(args.models_output_dir, exist_ok=True)
    pos_model_dir = os.path.join(args.models_output_dir, "gpt2_positive")
    neg_model_dir = os.path.join(args.models_output_dir, "gpt2_negative")

    print("training positive model.")
    train_sentiment_model(
        imdb_subset_path=args.imdb_subset_path,
        sentiment_label=1,
        output_dir=pos_model_dir
    )
    print("training negative model.")
    train_sentiment_model(
        imdb_subset_path=args.imdb_subset_path,
        sentiment_label=0,
        output_dir=neg_model_dir
    )

    print("generating reviews...")
    pos_reviews = generate_reviews(pos_model_dir, prompt="The movie was", num_reviews=5)
    neg_reviews = generate_reviews(neg_model_dir, prompt="The movie was", num_reviews=5)

    with open(args.generated_reviews_path, "w", encoding="utf-8") as f:
        f.write("Reviews generated by positive model:\n")
        for i, review in enumerate(pos_reviews, 1):
            f.write(f"{i}. {review}\n\n")
        f.write("\nReviews generated by negative model:\n")
        for i, review in enumerate(neg_reviews, 1):
            f.write(f"{i}. {review}\n\n")

    print("done.")

if __name__ == "__main__":
    main()
