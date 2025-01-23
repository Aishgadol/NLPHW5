import os
import sys
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    PreTrainedModel,
    AutoModelForSequenceClassification
)
from sklearn.metrics import accuracy_score


# metrics function, as shown in class
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)
    # Return metrics as a dictionary
    return {"accuracy": accuracy}

def tokenize(batch, tokenizer):
    # Batch tokenize the text
    return tokenizer(batch['text'], padding='max_length', truncation=True)

def main():
    argv = sys.argv

    if len(argv) != 2:
        print("Error! number of input arguments does not fit expectation:", \
              "path to IMDB subset")
        sys.exit(1)

    IMDB_subset_path = argv[1]

    #------------ Part A - subset loading/creating -------------
    if os.path.exists(IMDB_subset_path):
        subset = load_from_disk(IMDB_subset_path)  # load existing subset
    else:
        dataset = load_dataset("imdb")
        subset = dataset["train"].shuffle(seed=42).select(range(500))
        subset.save_to_disk("imdb_subset")      # create a new subset if no subset exists
    

    #------------ Part B - sentiment analysis --------------------

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # tokenize everything
    tokenized_subset = subset.map(lambda sample: tokenize(sample, tokenizer), batched=True)

    # split into train & test
    train_test = tokenized_subset.train_test_split(test_size=0.2, seed=42)
    train_subset = train_test['train']
    test_subset = train_test['test']

    # TODO: remove the if when submitting
    # in case we already trained the model
    saved_model_dir = "./saved_model"
    if os.path.exists(saved_model_dir):
        # Load the model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(saved_model_dir)
    else:
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        
        # create training arguments, as shown in class
        training_args = TrainingArguments(
            output_dir="./results",          
            eval_strategy="epoch",
            save_strategy="epoch",                  
            per_device_train_batch_size=8,  
            per_device_eval_batch_size=8,   
            num_train_epochs=3,             
            seed=42,                        
            logging_steps=10,               
            load_best_model_at_end=True,
        )

        # create trainer fro training, as shown in class
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
        save_directory = "./saved_model"
        model.save_pretrained(save_directory)  # Save model weights and config
        tokenizer.save_pretrained(save_directory)  # Save tokenizer
        print(f"Model and tokenizer saved to {save_directory}")


    # create trainer for evaluation, as shown in class
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./results"),
        eval_dataset=test_subset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Evaluate the model
    results = trainer.evaluate(test_subset)
    print(f'Accuracy on the test set: {results["eval_accuracy"]}')


    




if __name__ == "__main__":
    main()
