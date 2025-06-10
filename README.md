# NLPHW5: BERT, FLAN-T5 and GPT-2 Experiments

This repository contains a collection of scripts for sentiment analysis and text generation on a small IMDb movie review subset.  Models are fine-tuned or prompted with Hugging Face Transformers to demonstrate classification and generation techniques.

## Purpose & Objectives

- Train BERT models for binary sentiment classification.
- Explore FLAN-T5 prompt engineering (zero-shot, few-shot and instruction-based prompts) and optional fine-tuning.
- Fine-tune GPT-2 on positive and negative reviews to synthesize new text.
- Measure classification accuracy and inspect generated reviews to assess model behavior.

## Architecture & Module Breakdown

```
imdb_subset_creator.py  →  Creates or loads a 500-example subset from the IMDb dataset.
bert_classification_finetuning.py / itay_bert.py
    - Tokenize reviews, split train/test and fine‑tune BERT.
    - Results and model weights are stored in ./results and ./saved_model.
flan_t5_prompt_engineering.py
    - Load flan‑t5-small and classify reviews using crafted prompts.
flan_t5_prompt_engineering2.py
    - Fine‑tune flan‑t5-small on the sampled subset and then classify with three prompt styles.
gpt_generation_finetuning.py / gpt_generation_finetuning2.py
    - Fine‑tune GPT‑2 separately on positive and negative reviews and generate new sentences.
x.py
    - Simple utility to count words from stdin (standalone example).
```

Typical workflow: `imdb_subset_creator.py` prepares the dataset → BERT and GPT‑2 scripts load that subset for training → T5 scripts either fine‑tune or run prompts → results are written to `.txt` logs while models are saved into subdirectories for later reuse.

## Installation & Environment Setup

1. Install Python 3.10+ and CUDA capable GPU drivers (CUDA ≥ 11.7 recommended).
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

No additional system packages are required beyond a standard build environment with Git and GCC.

## Usage & Examples

### 1. BERT Classification

```bash
python bert_classification_finetuning.py imdb_subset.json
```
Outputs training logs and prints final accuracy, e.g. `Accuracy on the test set: 0.84`.  The trained model is stored in `saved_model/`.

### 2. FLAN‑T5 Prompt Engineering and Fine‑Tuning

```bash
python flan_t5_prompt_engineering2.py imdb_subset.json flan_t5_results.txt --sample_size 50 --seed 42
```
Generates classification results using zero‑shot, few‑shot and instruction prompts.  The fine‑tuned model is saved under `flan-t5-small-finetuned-imdb/` and accuracy metrics are echoed on the console.

### 3. GPT‑2 Generation

```bash
python gpt_generation_finetuning2.py imdb_subset.json generated_text.txt models/
```
Trains positive and negative GPT‑2 models and writes generated reviews to `generated_text.txt`.

## Outputs & Artifacts

- `saved_model/` – BERT weights and tokenizer.
- `results/`, `results_eval/` – Hugging Face Trainer logs.
- `flan-t5-small-finetuned-imdb/` – fine‑tuned FLAN‑T5 model.
- `models/` – GPT‑2 positive and negative model directories.
- `generated_text.txt`, `flan_t5_results.txt`, `dif_prompts_flan_t5_results.txt` – evaluation logs and generated text.

Paths are created automatically by the scripts and can be cleaned between runs.

## Development & Contribution Workflow

- Run `pytest` before committing to ensure all unit tests (if any) pass.
- Follow `flake8` and `black` style conventions; run `flake8` for linting and `black .` for formatting.
- Submit pull requests with descriptive messages and reference the relevant script or dataset changes.

## Project Status & Roadmap

Status: **alpha** – demonstration code for coursework.  Future work may include larger dataset experiments, hyper‑parameter sweeps and automated evaluation suites.

## License & Attribution

No license is provided in this repository.  It is recommended to release the code under the MIT License.  Portions of the dataset are derived from the IMDb corpus available through the `datasets` library.

