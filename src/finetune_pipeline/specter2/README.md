# SPECTER2 Fine-tuning Pipeline

This directory contains the code and configuration for fine-tuning **SPECTER2** on custom scientific retrieval datasets. SPECTER2 is a dense retrieval model based on a transformer backbone (e.g., BERT), designed for scientific document representation and retrieval. It leverages adapters to specialize the encoding process for queries and documents, while sharing the main transformer body.

---

## Overview

- **Model**: SPECTER2 uses a transformer backbone (e.g., BERT) with adapters to produce dense, high-dimensional representations for queries and documents.
- **Training Objective**: The model is fine-tuned using a triplet loss (query, positive, negative) to encourage relevant documents to be closer to the query in the dense space.
- **Pipeline**: The pipeline is modular, with clear separation between data loading, model definition, training logic, and configuration.

---

## Key Components & Workflow

### 1. Adapter-based Dense Representation

- The SPECTER2 model class wraps a transformer backbone (e.g., BERT) and attaches adapters for both query and document encoding.
- When encoding a query or a document, the same transformer body is used, but different adapters are activated depending on the input type (query or paper).
- The output is a dense vector (typically the [CLS] token embedding) representing the input.

### 2. Fine-tuning Pipeline (`finetune.py`)

- Loads triplet-formatted data (`query`, `positive`, `negative`).
- Initializes the SPECTER2 model and tokenizer.
- Instantiates the trainer, which manages the training loop.
- During training:
  - Each batch is tokenized and passed through the model to obtain dense representations.
  - **TripletMarginLoss** is computed to ensure the query is closer to the positive than the negative.
  - The model is updated via backpropagation.
- Validation is performed periodically, and the best model is saved.

### 3. Data, Tokenizer, and Model Integration

- **Data**: Expects a list of dictionaries with keys: `query`, `positive_title`, `positive_abstract`, `negative_title`, `negative_abstract`.
- **Tokenizer**: HuggingFace tokenizer matching the transformer backbone.
- **Model**: SPECTER2 model outputs dense vectors for both queries and documents, used directly in the loss computation.

---

## File Structure

- `finetune.py`  
  Main entry point for fine-tuning SPECTER2. Handles configuration, data loading, model/trainer setup, and training loop.

- `model.py`  
  Contains the SPECTER2 model class, which wraps a transformer with adapters and implements the dense encoding logic.

- `trainer.py`  
  Implements the trainer class, which manages batching, loss computation (triplet), optimizer/scheduler, validation, and checkpointing.

---

## Training Process: Step-by-Step

1. **Configuration**  
   Set all hyperparameters and model options in `conf/specter2_conf.yaml`.

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
   - Initialize the SPECTER2 model and tokenizer.
   - Set up the trainer.
   - Train the model, periodically evaluating on the validation set and saving the best checkpoint.

4. **Model Output**  
   The best model and tokenizer are saved to the specified output directory.

---

## SPECTER2 Model Details

- **Dense Representation**:  
  The transformer backbone produces hidden states for each token. The [CLS] token embedding is typically used as the dense representation for the input.
- **Adapters**:  
  SPECTER2 uses adapters to specialize the encoding process for queries and documents. The main transformer weights are shared, but the adapters are switched depending on the input type.
- **Triplet Loss**:  
  For each batch, the model computes representations for the query, positive, and negative documents. The triplet margin loss encourages the query to be closer to the positive than the negative in the dense space.

---

## Customization

- **Adapters**:  
  You can modify or add adapters for different input types or tasks.
- **Model Variants**:  
  You can experiment with different transformer backbones or adapter configurations by changing the config.
- **Loss Function**:  
  The loss function can be swapped or extended for other retrieval objectives.

---

## References

- [SPECTER2 Paper](https://arxiv.org/abs/2305.14722)
- [Official SPECTER2 HuggingFace Model](https://huggingface.co/allenai/specter2)

---

**For further details, see the code in each module and the configuration file. The modular structure makes it straightforward to extend or modify the pipeline for your needs.**
