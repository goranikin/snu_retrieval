# DPR Fine-tuning Pipeline

This directory contains the code and configuration for fine-tuning the **DPR** (Dense Passage Retrieval) model on custom scientific retrieval datasets. DPR is a dense retrieval model based on BERT, designed for efficient and effective open-domain question answering and document retrieval.

---

## Overview

- **Model**: DPR uses two separate BERT-based encoders—one for queries and one for documents (papers)—forming a dual-encoder architecture.
- **Training Objective**: The model is fine-tuned using a triplet loss (query, positive, negative) to ensure that relevant documents are closer to the query in the embedding space than irrelevant ones.
- **Pipeline**: The pipeline is modular, with clear separation between data loading, model definition, training logic, and configuration.

---

## Key Components & Workflow

### 1. Dual-Encoder Architecture

- The DPR model consists of two independent BERT encoders:
  - **Query Encoder**: Encodes the input query into a dense vector.
  - **Document Encoder**: Encodes the document (paper) into a dense vector.
- During retrieval, the similarity (typically dot product) between the query and document vectors is used to rank documents.

### 2. Fine-tuning Pipeline (`finetune.py`)

- Loads triplet-formatted data (`query`, `positive`, `negative`).
- Initializes the DPR model and tokenizer.
- Instantiates the `DPRTrainer`, which manages the training loop.
- During training:
  - Each batch is tokenized and passed through the query and document encoders to obtain dense representations.
  - **TripletMarginLoss** is computed to ensure the query is closer to the positive than the negative.
  - The model is updated via backpropagation.
- Validation is performed periodically, and the best model is saved.

### 3. Data, Tokenizer, and Model Integration

- **Data**: Expects a list of dictionaries with keys: `query`, `positive_title`, `positive_abstract`, `negative_title`, `negative_abstract`.
- **Tokenizer**: HuggingFace tokenizer matching the BERT backbone.
- **Model**: DPR model outputs dense vectors for both queries and documents, used directly in the loss computation.

---

## File Structure

- `finetune.py`  
  Main entry point for fine-tuning DPR. Handles configuration, data loading, model/trainer setup, and training loop.

- `model.py`  
  Contains the DPR model class, implementing the dual-encoder architecture.

- `trainer.py`  
  Implements the `DprTrainer` class, which manages batching, loss computation (triplet), optimizer/scheduler, validation, and checkpointing.

---

## Training Process: Step-by-Step

1. **Data Preparation**  
   Prepare your triplet data in JSON format, with each entry containing a query, positive, and negative document (title + abstract).

2. **Run Fine-tuning**  
   Execute:
   ```
   python finetune.py
   ```
   This will:
   - Load and shuffle the data.
   - Split into train/validation sets.
   - Initialize the DPR model and tokenizer.
   - Set up the trainer.
   - Train the model, periodically evaluating on the validation set and saving the best checkpoint.

3. **Model Output**  
   The best model and tokenizer are saved to the specified output directory.

---

## DPR Model Details

- **Dense Representation**:  
  Each encoder (query and document) produces a dense vector (typically 768-dim for BERT-base) for its input.
- **Triplet Loss**:  
  For each batch, the model computes representations for the query, positive, and negative documents. The triplet margin loss encourages the query to be closer to the positive than the negative in the dense space.
- **Dual Encoder**:  
  The query and document encoders do not share weights, allowing them to specialize for their respective inputs.

---

## Customization

- **Encoder Backbones**:  
  You can use different BERT variants for the query and document encoders by changing the config.
- **Loss Function**:  
  The default is triplet margin loss, but you can implement or swap in other loss functions as needed.
- **Batching and Data**:  
  The data pipeline is modular and can be adapted for other formats or negative sampling strategies.

---

## References

- [DPR Paper](https://arxiv.org/abs/2004.04906)
- [Official DPR GitHub](https://github.com/facebookresearch/DPR)

---

**For further details, see the code in each module and the configuration file. The modular structure makes it straightforward to extend or modify the pipeline for your needs.**
