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
   uv sync
