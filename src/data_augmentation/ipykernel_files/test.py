# %%
import json

with open(
    "../jsons_for_triplet/citation_corpus_info_index_with_cited_index_and_hard_negative.json",
    "r",
) as f:
    data1 = json.load(f)

with open(
    "../jsons/filtered_extracted_citation_info_and_queries_title_2.json",
    "r",
) as f:
    data2 = json.load(f)

print(len(data1), len(data2))
