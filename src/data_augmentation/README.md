# [LitSearch] Data Augmentation & Triplet Construction Pipeline

## Directory Structure

```
src/data_augmentation/
├── generate_query/
│   ├── extracting_ctss.py
│   ├── llm.py
│   ├── main.py
│   ├── prompt.py
│   └── query_generator.py
├── make_triplet/
│   ├── make_triplet.py
│   └── *.ipynb
└── README.md
```

## Overview

This pipeline augments the [LitSearch] dataset and constructs training triplets for retrieval model training. The process consists of two main stages:

1. **Query Generation**: Extracting citation context and generating synthetic queries using LLMs.
2. **Triplet Construction**: Creating (query, positive, negative) triplets using both easy and hard negatives, following the methodology from the SPECTER paper.

---

## 1. Query Generation

All query generation and preprocessing are handled in the `generate_query/` directory.

- **extracting_ctss.py**: Extracts structured citation information from the LitSearch dataset, including source/citation paper IDs, citation context, and metadata (title, abstract).
- **prompt.py**: Defines and manages prompts for the LLM.
- **llm.py**: Interfaces with the Qwen3:14B LLM to generate queries.
- **query_generator.py**: Coordinates feeding citation info and prompts to the LLM and collects generated queries.
- **main.py**: Orchestrates the full query generation pipeline.

The output is a set of synthetic queries paired with their corresponding citation information, ready for triplet construction.

Example of extracted data:
```json
{
  "index": 0,
  "source_corpus_id": 252715594,
  "ref_id": "b57",
  "citation_corpus_id": 238582653,
  "start": 11564,
  "end": 11568,
  "title": "VECTOR-QUANTIZED IMAGE MODELING WITH IMPROVED VQGAN",
  "abstract": "...",
  "prev": "...",
  "curr": "...",
  "next": "..."
}
```

Prompt engineering details, including trials and challenges, are documented in `Prompt.md`.

## 2. Triplet Construction

Triplet construction is performed in the `make_triplet/` directory, primarily via `make_triplet.py` (Jupyter/IPython notebook format).

### Methodology

- **Approach**: Follows the triplet construction strategy described in the SPECTER paper.
- **Triplet Structure**: Each sample consists of a (query, positive, negative) triplet, used for training with triplet margin loss.
- **Positive**: The citation paper referenced by the source paper (i.e., the ground-truth citation).
- **Negatives**:
  - **Easy Negative**: Randomly sampled papers from the dataset that are not cited by the source paper.
  - **Hard Negative**: Papers cited by the positive (citation) paper, but not cited by the source paper itself.
- **Negative Ratio**: Easy and hard negatives are mixed in a 3:2 ratio. This ratio is based on empirical findings from AllenAI, who observed that using only hard negatives can cause similar papers to be pushed too far apart in the embedding space, which is detrimental to training. Mixing easy and hard negatives in this proportion helps maintain a balance and improves model performance.

### Implementation

- The triplet construction process takes the generated queries and citation information, and for each query-positive pair, samples negatives according to the above strategy.
- The final triplets are formatted to match the original LitSearch dataset structure for compatibility.

---

## Model Used

- **Qwen3:14B**: All query generation is performed using the Qwen3:14B large language model.

## Notes

- Large JSON files containing raw or augmented data are not included in this repository to conserve space.

---
[LitSearch]: https://huggingface.co/datasets/princeton-nlp/LitSearch
