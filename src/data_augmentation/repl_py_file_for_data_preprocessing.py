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
from typing import List, Tuple

from datasets import Dataset


def get_clean_corpusid(item: dict) -> int:
    return item["corpusid"]


def get_clean_title(item: dict) -> str:
    return item["title"]


def get_clean_abstract(item: dict) -> str:
    return item["abstract"]


def get_clean_title_abstract(item: dict) -> str:
    title = get_clean_title(item)
    abstract = get_clean_abstract(item)
    return f"Title: {title}\nAbstract: {abstract}"


def get_clean_full_paper(item: dict) -> str:
    return item["full_paper"]


def get_clean_paragraph_indices(item: dict) -> List[Tuple[int, int]]:
    text = get_clean_full_paper(item)
    paragraph_indices = []
    paragraph_start = 0
    paragraph_end = 0
    while paragraph_start < len(text):
        paragraph_end = text.find("\n\n", paragraph_start)
        if paragraph_end == -1:
            paragraph_end = len(text)
        paragraph_indices.append((paragraph_start, paragraph_end))
        paragraph_start = paragraph_end + 2
    return paragraph_indices


def get_clean_text(item: dict, start_idx: int, end_idx: int) -> str:
    text = get_clean_full_paper(item)
    assert start_idx >= 0 and end_idx >= 0
    assert start_idx <= end_idx
    assert end_idx <= len(text)
    return text[start_idx:end_idx]


def get_clean_paragraphs(item: dict, min_words: int = 10) -> List[str]:
    paragraph_indices = get_clean_paragraph_indices(item)
    paragraphs = [
        get_clean_text(item, paragraph_start, paragraph_end)
        for paragraph_start, paragraph_end in paragraph_indices
    ]
    paragraphs = [
        paragraph for paragraph in paragraphs if len(paragraph.split()) >= min_words
    ]
    return paragraphs


def get_clean_citations(item: dict) -> List[int]:
    return item["citations"]


def get_clean_dict(data: Dataset) -> dict:
    return {get_clean_corpusid(item): item for item in data}


def create_kv_pairs(data: List[dict], key: str) -> dict:
    if key == "title_abstract":
        kv_pairs = {
            get_clean_title_abstract(record): get_clean_corpusid(record)
            for record in data
        }
    elif key == "full_paper":
        kv_pairs = {
            get_clean_full_paper(record): get_clean_corpusid(record) for record in data
        }
    elif key == "paragraphs":
        kv_pairs = {}
        for record in data:
            corpusid = get_clean_corpusid(record)
            paragraphs = get_clean_paragraphs(record)
            for paragraph_idx, paragraph in enumerate(paragraphs):
                kv_pairs[paragraph] = (corpusid, paragraph_idx)
    else:
        raise ValueError("Invalid key")
    return kv_pairs


# %%
import json

with open(
    "./jsons/filtered_extracted_citation_info_and_queries_title_2.json", "r"
) as f:
    data = json.load(f)

# %%
data[0]


# %%
def find_citation_paper_info(
    source_corpus_id: int,
    target_corpus_id_list: list[int],
    str_bibentry_list: str,
    str_bibref_list: str,
    index: int,
) -> list[dict]:
    if str_bibentry_list is None or str_bibref_list is None:
        return []

    bibentry_list: list[dict] = json.loads(str_bibentry_list)
    bibref_list: list[dict] = json.loads(str_bibref_list)

    matched_bid_list = []
    for entry in bibentry_list:
        bib_corpus_id = entry.get("attributes", {}).get("matched_paper_id")
        bib_ref_id = entry.get("attributes", {}).get("id")
        if bib_corpus_id in target_corpus_id_list and bib_ref_id is not None:
            matched_bid_list.append({"corpus_id": bib_corpus_id, "bib_id": bib_ref_id})

    bibid_to_corpusid = {item["bib_id"]: item["corpus_id"] for item in matched_bid_list}

    result = []
    for ref in bibref_list:
        ref_id = (
            ref.get("attributes", {}).get("ref_id") if "attributes" in ref else None
        )
        if ref_id in bibid_to_corpusid:
            result.append(
                {
                    "index": index,
                    "source_corpus_id": source_corpus_id,
                    "ref_id": ref_id,
                    "citation_corpus_id": bibid_to_corpusid[ref_id],
                    "start": ref["start"],
                    "end": ref["end"],
                }
            )

    return result


# %%
from tqdm import tqdm

result_list = []

# 1007 makes 597 data.
for i in tqdm(range(3000), desc="Processing citation information"):
    source_corpus_id = corpus_clean_data[i]["corpusid"]
    citation_corpus_ids = corpus_clean_data[i]["citations"]

    result = find_citation_paper_info(
        source_corpus_id,
        citation_corpus_ids,
        corpus_s2orc_data[i]["content"]["annotations"]["bibentry"],
        corpus_s2orc_data[i]["content"]["annotations"]["bibref"],
        i,
    )

    result_list.append(result)

not_null = 0

real_result_list = []

for result in result_list:
    if result != []:
        real_result_list.append(result)
        not_null += 1
# %%
len(real_result_list)
# %%
from collections import defaultdict


def flatten_citation_data(nested_list):
    result = []
    # (index, source_corpus_id)로 그룹핑
    group_dict = defaultdict(list)
    for inner_list in nested_list:
        for d in inner_list:
            key = (d["index"], d["source_corpus_id"])
            group_dict[key].append(
                {
                    "citation_corpus_id": d["citation_corpus_id"],
                    "start": d["start"],
                    "end": d["end"],
                }
            )
    # 결과 구조로 변환
    for (index, source_corpus_id), citation_list in group_dict.items():
        result.append(
            {
                "index": index,
                "source_corpus_id": source_corpus_id,
                "citation_corpus_info": citation_list,
            }
        )
    return result


# %%
new_data = flatten_citation_data(real_result_list)

# %%
with open("./jsons_for_triplet/citation_corpus_info_index.json", "w") as f:
    json.dump(new_data, f, indent=2)

# %%
citation_id_to_citations = {}

for i in range(len(corpus_s2orc_data)):
    # 논문 id
    paper_id = (
        corpus_s2orc_data[i]["corpusid"] if "corpusid" in corpus_s2orc_data[i] else None
    )
    # 인용 논문 id 리스트
    citations = []
    # bibentry와 bibref가 모두 있어야 함
    bibentry = corpus_s2orc_data[i]["content"]["annotations"].get("bibentry")
    if bibentry and paper_id is not None:
        # find_citation_paper_info를 활용
        # bibentry, bibref는 json string일 수 있으니, 필요시 json.loads
        if isinstance(bibentry, str):
            bibentry = json.loads(bibentry)
        # bibentry에서 matched_paper_id 추출
        for entry in bibentry:
            matched_id = entry.get("attributes", {}).get("matched_paper_id")
            if matched_id is not None:
                citations.append(matched_id)
    citation_id_to_citations[paper_id] = citations

# %%
for item in new_data:
    for citation_info in item["citation_corpus_info"]:
        citation_id = citation_info["citation_corpus_id"]
        # citation_id가 인용하는 논문 id 리스트
        cited_ids = citation_id_to_citations.get(citation_id, [])
        citation_info["cited_by_citation"] = cited_ids

# %%
with open(
    "./jsons_for_triplet/citation_corpus_info_index_with_cited_index.json", "w"
) as f:
    json.dump(new_data, f, indent=2)

# %%
for item in new_data:
    source_id = item["source_corpus_id"]
    cited_ids = citation_id_to_citations.get(source_id, [])
    item["cited_ids"] = cited_ids
with open(
    "./jsons_for_triplet/citation_corpus_info_index_with_cited_index.json", "w"
) as f:
    json.dump(new_data, f, indent=2)
# %%
# This is a test code for checking data is correct
for idx, item in enumerate(new_data):
    cited_ids_set = set(item.get("cited_ids", []))
    for citation_info in item["citation_corpus_info"]:
        citation_corpus_id = citation_info["citation_corpus_id"]
        if citation_corpus_id not in cited_ids_set:
            print("The data is incorrect!")
# %%
for idx, item in enumerate(new_data):
    cited_ids_set = set(item.get("cited_ids", []))
    for citation_info in item["citation_corpus_info"]:
        cited_by_citation_set = set(citation_info.get("cited_by_citation", []))
        hard_negative = list(cited_by_citation_set - cited_ids_set)
        citation_info["hard_negative"] = hard_negative
# %%
with open(
    "./jsons_for_triplet/citation_corpus_info_index_with_cited_index_and_hard_negative.json",
    "w",
) as f:
    json.dump(new_data, f, indent=2)

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
