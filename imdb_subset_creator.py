
import os
import sys
from datasets import load_dataset, Dataset

def main():
    subset_file = "imdb_subset.json"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    subset_path = os.path.join(script_dir, subset_file)

    if os.path.isfile(subset_path):
        try:
            subset = load_dataset("json", data_files=subset_path, split='train')
        except Exception as e:
            sys.exit(f"Failed to load subset: {e}")
    else:
        try:
            dataset = load_dataset("imdb")
            subset = dataset["train"].shuffle(seed=42).select(range(500))
            subset.to_json(subset_path, orient="records", lines=True)
        except Exception as e:
            sys.exit(f"Error creating or saving subset: {e}")

if __name__ == "__main__":
    main()
