# %%
import json
import random

from datasets import load_dataset

# %%
with open(
    "./citation_corpus_info_index_with_cited_index_and_hard_negative.json",
    "r",
) as f:
    hard_negative_index_data = json.load(f)

# %%
query_data = load_dataset("princeton-nlp/LitSearch", "query", split="full")
corpus_clean_data = load_dataset(
    "princeton-nlp/LitSearch", "corpus_clean", split="full"
)
corpus_s2orc_data = load_dataset(
    "princeton-nlp/LitSearch", "corpus_s2orc", split="full"
)

# %%
simple_hard_negative_data = [
    {
        "source_corpus_id": item["source_corpus_id"],
        "citation_corpus_id": item["citation_corpus_info"][0]["citation_corpus_id"],
        "cited_by_citation": item["citation_corpus_info"][0]["cited_by_citation"],
        "hard_negative": item["citation_corpus_info"][0]["hard_negative"],
    }
    for item in hard_negative_index_data
    if item["citation_corpus_info"]
]

# %%
all_corpus_ids = set(corpus_clean_data["corpusid"])

hard_negative_dict = {
    item["source_corpus_id"]: item for item in simple_hard_negative_data
}

# %%
with open("./citation_info3.json", "r") as f:
    citation_info_data = json.load(f)

filtered_citation_info = []
for info in citation_info_data:
    src_id = info["source_corpus_id"]
    citation_id = info["citation_corpus_id"]

    hard_neg = []
    if src_id in hard_negative_dict:
        hard_neg = [
            cid
            for cid in hard_negative_dict[src_id]["hard_negative"]
            if cid in all_corpus_ids
        ]
    info["hard_negative"] = hard_neg

    exclude_ids = {src_id, citation_id}
    if src_id in hard_negative_dict:
        exclude_ids.update(hard_negative_dict[src_id]["hard_negative"])
        exclude_ids.update(hard_negative_dict[src_id]["cited_by_citation"])
    easy_neg_candidates = list(all_corpus_ids - exclude_ids)
    if len(easy_neg_candidates) >= 3:
        easy_neg = random.sample(easy_neg_candidates, 3)
    else:
        easy_neg = []
    info["easy_negative"] = easy_neg

    if len(hard_neg) >= 2 and len(easy_neg) >= 3:
        filtered_citation_info.append(info)
# %%
with open("./citation_info_with_negatives.json", "w") as f:
    json.dump(filtered_citation_info, f, indent=2)
# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
