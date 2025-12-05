# README.md

## compare_tokenizers

This project provides a unified experimental framework to compare three tokenizer strategies—**Byte-level**, **HuggingFace BPE**, and a **Custom BPE implementation**—across multiple languages (English and Chinese). It trains a small GPT-style model on Wikipedia streaming data and evaluates tokenization compression ratio, perplexity, BPC, and inference speed.

### Features

* **Byte-level tokenizer** (256 fixed vocab)
* **HuggingFace BPE tokenizer** (trainable, with special tokens)
* **Custom GPT-2–style BPE tokenizer** (full merge-learning implementation)
* **Unified GPT training loop** for fair comparison
* **Compression ratio metrics** (tokens per character)
* **Bits-per-character (BPC)** evaluation
* **Inference speed measurement** across tokenizers

### Usage

Run all experiments:

```bash
python compare_tokenizers.py --mode all
```

Run a specific tokenizer mode:

```bash
python compare_tokenizers.py --mode hf_bpe
```

### Output

Results are saved to the `compare_runs/` directory, including:

* tokenizer files
* training curves
* inference metrics
* final `{lang}_{mode}.json` result summaries

### Jupyter Notebooks

The repository includes:

* **main_results.ipynb** — Visualizations, training/validation curves, BPC comparisons, tokenizer efficiency plots and inference speed analysis generated from experiment outputs.

## Requirements

See `requirements.txt` file.
