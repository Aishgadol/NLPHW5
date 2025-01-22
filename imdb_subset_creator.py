#!/usr/bin/env python3

import os
import sys
from datasets import load_dataset, load_from_disk

def main():
    subset_dir = "imdb_subset"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    subset_path = os.path.join(script_dir, subset_dir)

    if os.path.exists(subset_path):
        print("Loading existing subset...")
        try:
            subset = load_from_disk(subset_path)
        except Exception as e:
            print(f"Failed to load subset: {e}")
            sys.exit(1)
    else:
        print("Creating subset...")
        try:
            dataset = load_dataset("imdb")
            subset = dataset["train"].shuffle(seed=42).select(range(500))
            subset.save_to_disk(subset_path)
            print("Subset saved.")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    for i in range(5):
        example = subset[i]
        label = "Positive" if example['label'] == 1 else "Negative"
        print(f"\nReview {i+1}: {label}\n{example['text'][:500]}...")

if __name__ == "__main__":
    main()
