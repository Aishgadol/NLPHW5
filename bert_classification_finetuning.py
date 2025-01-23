import os
import sys
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification
)
from sklearn.metrics import accuracy_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}


def tokenize(batch, tokenizer):
    return tokenizer(batch['text'], padding='max_length', truncation=True)


def main():
    argv = sys.argv
    if len(argv) != 2:
        print("invalid input")
        sys.exit(1)

    imdb_subset_path = argv[1]

    #subset loading
    if os.path.exists(imdb_subset_path):
        subset = load_dataset(
            "json",
            data_files=imdb_subset_path,
            split="train"
        )


    # Part B: sentiment analysis
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # tokenize everything
    tokenized_subset = subset.map(lambda sample: tokenize(sample, tokenizer), batched=True)

    # split into train & test (80/20)
    train_test = tokenized_subset.train_test_split(test_size=0.2, seed=42)
    train_subset = train_test['train']
    test_subset = train_test['test']

    saved_model_dir = "./saved_model"
    if os.path.exists(saved_model_dir):
        # Load the model and tokenizer6+66666666666666666
        model = AutoModelForSequenceClassification.from_pretrained(saved_model_dir)
    else:
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        # create training arguments (epochs=4)
        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=4,
            seed=42,
            logging_steps=10,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=test_subset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        # train the model
        trainer.train()

        # Save the trained model and tokenizer
        model.save_pretrained(saved_model_dir)
        tokenizer.save_pretrained(saved_model_dir)
        print(f"Model and tokenizer saved to {saved_model_dir}")

    # create trainer for evaluation
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./results_eval"),
        eval_dataset=test_subset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Evaluate the model
    results = trainer.evaluate(test_subset)
    print(f'Accuracy on the test set: {results["eval_accuracy"]}')


if __name__ == "__main__":
    main()
