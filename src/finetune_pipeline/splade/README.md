# SPLADE Fine-tuning Pipeline

This directory contains the code and configuration for fine-tuning the **SPLADE** (Sparse Lexical and Expansion Model) on custom scientific retrieval datasets. SPLADE is a sparse neural retrieval model that leverages the MLM (Masked Language Modeling) head of transformer models (e.g., BERT) to generate sparse document and query representations suitable for efficient lexical search.

---

## Overview

- **Model**: SPLADE uses a transformer backbone (e.g., BERT) with its MLM head to produce sparse, high-dimensional representations for queries and documents.
- **Training Objective**: The model is fine-tuned using a triplet loss (query, positive, negative) to encourage relevant documents to be closer to the query in the sparse space, with additional regularization (FLOPS/L0) to promote sparsity.
- **Pipeline**: The pipeline is modular, with clear separation between data loading, model definition, training logic, and configuration.

---

## Key Components & Workflow

### 1. MLM Head for Sparse Representation

- The `Splade` model class (see `models/transformer_rep.py`) wraps a transformer with an MLM head (`AutoModelForMaskedLM`).
- For each input (query or document), the model computes the logits from the MLM head.
- Sparse representation is generated as:
  ```
  sparse_vec = aggregation(torch.log(1 + relu(logits)) * attention_mask)
  ```
  - **Aggregation**: Either `sum` or `max` over the sequence dimension, resulting in a single sparse vector per input.
  - This vector is high-dimensional (vocab size, e.g., 30,522 for BERT) and mostly zeros, enabling efficient sparse search.

### 2. Fine-tuning Pipeline (`finetune.py`)

- Loads triplet-formatted data (`query`, `positive`, `negative`).
- Initializes the SPLADE model and tokenizer.
- Sets up regularization (FLOPS or L0) to enforce sparsity.
- Instantiates the `SpladeTrainer`, which manages the training loop.
- During training:
  - Each batch is tokenized and passed through the model to obtain sparse representations.
  - **TripletMarginLoss** is computed to ensure the query is closer to the positive than the negative.
  - **Regularization loss** (FLOPS/L0) is added to promote sparsity.
  - The model is updated via backpropagation.
- Validation is performed periodically, and the best model is saved.

### 3. Data, Tokenizer, and Model Integration

- **Data**: Expects a list of dictionaries with keys: `query`, `positive_title`, `positive_abstract`, `negative_title`, `negative_abstract`.
- **Tokenizer**: HuggingFace tokenizer matching the transformer backbone.
- **Model**: SPLADE model outputs sparse vectors for both queries and documents, used directly in the loss computation.

---

## File Structure

- `finetune.py`  
  Main entry point for fine-tuning SPLADE. Handles configuration, data loading, model/trainer setup, and training loop.

- `models/transformer_rep.py`  
  Contains the SPLADE model class, which wraps a transformer with an MLM head and implements the sparse encoding logic.

- `models/models_utils.py`  
  Utility for instantiating the correct SPLADE model variant based on configuration.

- `trainer.py`  
  Implements the `SpladeTrainer` class, which manages batching, loss computation (triplet + regularization), optimizer/scheduler, validation, and checkpointing.

- `losses/regularization.py`  
  Implements regularization losses (FLOPS, L0, etc.) and their schedulers.

- `datasets/dataloaders.py`, `datasets/datasets.py`  
  Utilities for loading and batching triplet data.

- `conf/splade_conf.yaml`  
  Configuration file for model, training, and regularization parameters.

---

## Training Process: Step-by-Step

1. **Configuration**  
   Set all hyperparameters and model options in `conf/splade_conf.yaml`.

2. **Data Preparation**  
   Prepare your triplet data in JSON format, with each entry containing a query, positive, and negative document (title + abstract).

3. **Run Fine-tuning**  
   Execute:
   ```
   python finetune.py
   ```
   This will:
   - Load and shuffle the data.
   - Split into train/validation sets.
   - Initialize the SPLADE model and tokenizer.
   - Set up regularization and the trainer.
   - Train the model, periodically evaluating on the validation set and saving the best checkpoint.

4. **Model Output**  
   The best model and tokenizer are saved to the specified output directory.

---

## SPLADE Model Details

- **Sparse Representation**:  
  The MLM head produces logits for each token position. These are aggregated (sum or max) across the sequence, after applying `log(1 + relu(logits))` and masking out padding tokens. The result is a sparse vector over the vocabulary.

- **Triplet Loss**:  
  For each batch, the model computes representations for the query, positive, and negative documents. The triplet margin loss encourages the query to be closer to the positive than the negative in the sparse space.

- **Regularization**:  
  - **FLOPS**: Penalizes the average squared activation, encouraging sparsity.
  - **L0**: Penalizes the number of non-zero entries.
  - Regularization weights are scheduled during training.

---

## Customization

- **Aggregation**:  
  Change the `agg` parameter (`sum` or `max`) in the model config to control how token-level logits are aggregated.

- **Regularization**:  
  Switch between FLOPS and L0 regularization via the config.

- **Model Variants**:  
  Use different SPLADE variants (`splade`, `splade_doc`, etc.) by changing `matching_type` in the config.

---

## References

- [SPLADE Paper](https://arxiv.org/abs/2104.06967)
- [Official SPLADE GitHub](https://github.com/naver/splade)

---

**For further details, see the code in each module and the configuration file. If you wish to extend or modify the pipeline (e.g., for different data formats or loss functions), the modular structure makes it straightforward.**
