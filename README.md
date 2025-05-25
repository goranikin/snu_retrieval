# SPECTER2 Fine-tuning and Evaluation

This repository contains code and configuration for fine-tuning and evaluating the SPECTER2 scientific document retrieval model using custom datasets. The project leverages adapters for efficient domain adaptation and supports both training and evaluation workflows.

## Features

- Fine-tune SPECTER2 with custom scientific datasets using triplet loss.
- Evaluate retrieval performance with various metrics and dataset splits.
- Modular codebase with Hydra-based configuration.
- Jupyter notebooks for exploratory analysis and benchmarking.
- Integration with VESSL for scalable training and evaluation.

## Installation

1. **Clone the repository** and set up a Python 3.12+ environment.
2. **Install dependencies** (preferably using [uv](https://github.com/astral-sh/uv)):
```bash
pip install --upgrade uv
uv sync
```

---

# LitSearch dataset augmentation process

The LitSearch dataset consists of 597 queries for paper retrieval.  
Fine-tuning a model with such a small dataset has limitations, like overfitting and reduced quality and reliability of model performance.

Therefore, it is necessary to augment the dataset.

**Dataset Augmentation Process**

1. Query rewriting using citation sentences.
2. Filtering based on word overlap and quality.

The steps above are performed by an LLM.

3. Evaluating the performance of the newly generated queries.
- Models: BM25, SPECTER2
- Goal: To compare the retrieval performance with existing models. (Not too high or low.)


## trial

First, I use qwen3:14b for generating queries.

The trials of prompt engineering are in "Prompt.md".
