
# Fine-tuning Pipeline for Scientific Document Retrieval

This directory provides a unified pipeline for fine-tuning three state-of-the-art retrieval models—**DPR**, **SPECTER2**, and **SPLADE**—on custom scientific datasets. Each model is organized in its own subdirectory, with modular code for training, evaluation, and model management.

## Supported Models

- **DPR (Dense Passage Retrieval)**
  - [facebook/dpr-question_encoder-single-nq-base](https://huggingface.co/facebook/dpr-question_encoder-single-nq-base)
  - Dense dual-encoder architecture for open-domain question answering and retrieval.

- **SPECTER2**
  - [allenai/specter2](https://huggingface.co/allenai/specter2)
  - Transformer-based model for scientific document representation, leveraging citation-informed contrastive learning.

- **SPLADE (Sparse Lexical and Expansion Model)**
  - [naver/splade-v3](https://huggingface.co/naver/splade-v3)
  - Sparse neural retrieval model combining lexical and expansion-based representations for efficient and effective search.

## Directory Structure

- `dpr/`, `specter2/`, `splade/`
  - Each contains a `finetune.py` script for model-specific fine-tuning.
  - Additional modules (e.g., `model.py`, `trainer.py`) define model architectures and training logic.

- `base/`
  - Shared components for all models:
    - `data.py`: Dataset and data loader definitions (e.g., triplet datasets for retrieval).
    - `loss.py`: Loss functions (e.g., triplet margin loss).
    - `retrieval.py`: Abstract retrieval classes and utilities.

## Fine-tuning Workflow

Each model’s `finetune.py` script follows a similar workflow:
1. **Load Data**: Reads triplet-formatted training data (query, positive, negative).
2. **Initialize Model**: Loads the pre-trained model from HuggingFace.
3. **Set Up Trainer**: Prepares the training loop, optimizer, and evaluation logic.
4. **Train and Validate**: Fine-tunes the model and saves the best checkpoint.

See each model’s directory for detailed configuration and usage.

## Performance

- Values in parentheses () indicate pre-fine-tuning scores.  
- Values outside parentheses indicate post-fine-tuning scores.
- DPR have to use after fine-tuning (by paper).

### spec0 Recall@20

| Model     | inline           | manual           | avg             |
|-----------|------------------|------------------|-----------------|
| DPR       | 0.1467           | 0.1143           | 0.1394          |
| SPECTER2  | 0.5142 (0.3814)  | 0.4571 (0.2000)  | 0.5013 (0.3404) |
| SPLADE    | 0.3783 (0.5333)  | 0.4714 (0.6285)  | 0.4228 (0.5788) |

---

### spec1 Recall

| Model     | inline@5         | inline@20        | manual@5        | manual@20       | avg@20           |
|-----------|------------------|------------------|------------------|------------------|------------------|
| DPR       | 0.0671           | 0.1169           | 0.0190           | 0.0948           | 0.1063           |
| SPECTER2  | 0.4113 (0.3377)  | 0.6255 (0.4502)  | 0.4882 (0.3460)  | 0.6730 (0.5118)  | 0.6482 (0.3416)  |
| SPLADE    | 0.2532 (0.5108)  | 0.4156 (0.6450)  | 0.3886 (0.6493)  | 0.5118 (0.7725)  | 0.4616 (0.7059)  |

---


## References

- [DPR Paper](https://arxiv.org/abs/2004.04906)
- [SPECTER2 Paper](https://arxiv.org/abs/2305.14722)
- [SPLADE Paper](https://arxiv.org/abs/2104.06967)

---

**Note:** For more details on each model’s fine-tuning process, refer to the respective `finetune.py` and supporting modules in each subdirectory.
