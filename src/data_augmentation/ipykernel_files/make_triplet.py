# %%
import json

with open(
    "./citation_corpus_info_index_with_cited_index_and_hard_negative2.json",
    "r",
) as f:
    hard_negative_index_data = json.load(f)

with open(
    "./final_generating_query_data.json",
    "r",
) as f:
    generated_query_data = json.load(f)

print(len(hard_negative_index_data), len(generated_query_data))
# %%
from datasets import load_dataset

query_data = load_dataset("princeton-nlp/LitSearch", "query", split="full")
corpus_clean_data = load_dataset(
    "princeton-nlp/LitSearch", "corpus_clean", split="full"
)
corpus_s2orc_data = load_dataset(
    "princeton-nlp/LitSearch", "corpus_s2orc", split="full"
)

# %%
import json

simple_generated_query_data = [
    {
        "source_corpus_id": item["source_corpus_id"],
        "citation_corpus_id": item["citation_corpus_id"],
        "query": item["query"],
    }
    for item in generated_query_data
]

with open("./simple_generated_query_data.json", "w") as f:
    json.dump(simple_generated_query_data, f, indent=2)

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

with open("./simple_hard_negative_data.json", "w") as f:
    json.dump(simple_hard_negative_data, f, indent=2)

# %%
import random

all_corpus_ids = set(corpus_clean_data["corpusid"])

hard_negative_dict = {
    item["source_corpus_id"]: item for item in simple_hard_negative_data
}

for item in simple_generated_query_data:
    src_id = item["source_corpus_id"]

    if src_id in hard_negative_dict:
        hard_neg_list = hard_negative_dict[src_id]["hard_negative"]
        valid_hard_neg = [cid for cid in hard_neg_list if cid in all_corpus_ids]
        sampled_hard_neg = random.sample(valid_hard_neg, min(10, len(valid_hard_neg)))
        item["hard_negative"] = sampled_hard_neg
    else:
        item["hard_negative"] = []

    exclude_ids = {item["source_corpus_id"], item["citation_corpus_id"]}
    if src_id in hard_negative_dict:
        exclude_ids.update(hard_negative_dict[src_id]["cited_by_citation"])

    easy_neg_candidates = list(all_corpus_ids - exclude_ids)
    sampled_easy_neg = random.sample(
        easy_neg_candidates, min(10, len(easy_neg_candidates))
    )
    item["easy_negative"] = sampled_easy_neg

with open("./pre_triplet.json", "w") as f:
    json.dump(simple_generated_query_data, f, indent=2)
# %%
len(simple_generated_query_data)
# %%
filtered_data = [
    item
    for item in simple_generated_query_data
    if len(item.get("hard_negative", [])) >= 2
    and len(item.get("easy_negative", [])) >= 3
]

with open("./filtered_pre_triplet.json", "w") as f:
    json.dump(filtered_data, f, indent=2)

# %%
len(filtered_data)


# %%
import json
import random

with open("./citation_info2.json", "r") as f:
    citation_info_data = json.load(f)

filtered_triplets = []
for info in citation_info_data:
    src_id = info["source_corpus_id"]
    citation_id = info["citation_corpus_id"]

    # hard_negative 찾기 및 corpus_clean_data에 존재하는 것만 남기기
    hard_neg = []
    if src_id in hard_negative_dict:
        hard_neg = [
            cid
            for cid in hard_negative_dict[src_id]["hard_negative"]
            if cid in all_corpus_ids
        ]
    # 2개 이상만 진행
    if len(hard_neg) < 2:
        continue

    # easy_negative 후보 만들기 (source, citation, hard_negative, cited_by_citation 모두 제외)
    exclude_ids = {src_id, citation_id}
    if src_id in hard_negative_dict:
        exclude_ids.update(hard_negative_dict[src_id]["hard_negative"])
        exclude_ids.update(hard_negative_dict[src_id]["cited_by_citation"])
    easy_neg_candidates = list(all_corpus_ids - exclude_ids)
    if len(easy_neg_candidates) < 3:
        continue
    easy_neg = random.sample(easy_neg_candidates, 3)

    # 결과 저장
    filtered_triplets.append(
        {
            "source_corpus_id": src_id,
            "citation_corpus_id": citation_id,
            "hard_negative": hard_neg,
            "easy_negative": easy_neg,
            # 필요하다면 info의 다른 필드도 추가 가능
        }
    )

with open("./filtered_triplet_candidates.json", "w") as f:
    json.dump(filtered_triplets, f, indent=2)
# %%
len(filtered_triplets)

# %%
len(citation_info_data)

# %%
