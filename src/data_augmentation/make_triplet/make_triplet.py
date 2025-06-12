# 이 파일은 IDE의 REPL 기능으로 사용하기 위한 파일입니다. ipykernel을 설치한 후 설정한 뒤, jupyter 파일처럼 사용하면 됩니다.
# %%
import json
import random
from typing import List, Tuple

# %%
from datasets import Dataset, load_dataset
from tqdm import tqdm

query_data = load_dataset("princeton-nlp/LitSearch", "query", split="full")
corpus_clean_data = load_dataset(
    "princeton-nlp/LitSearch", "corpus_clean", split="full"
)
corpus_s2orc_data = load_dataset(
    "princeton-nlp/LitSearch", "corpus_s2orc", split="full"
)


# %%
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
with open(
    "../jsons/filtered_extracted_citation_info_and_queries_title_2.json", "r"
) as f:
    data = json.load(f)


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
result_list = []

for i in tqdm(range(6000), desc="Processing citation information"):
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
for item in new_data:
    source_id = item["source_corpus_id"]
    cited_ids = citation_id_to_citations.get(source_id, [])
    item["cited_ids"] = cited_ids
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
    "./citation_corpus_info_index_with_cited_index_and_hard_negative.json",
    "w",
) as f:
    json.dump(new_data, f, indent=2)

# %%
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

query_data = load_dataset("princeton-nlp/LitSearch", "query", split="full")
corpus_clean_data = load_dataset(
    "princeton-nlp/LitSearch", "corpus_clean", split="full"
)
corpus_s2orc_data = load_dataset(
    "princeton-nlp/LitSearch", "corpus_s2orc", split="full"
)

# %%
my_generated_query_data = [
    {
        "source_corpus_id": item["source_corpus_id"],
        "citation_corpus_id": item["citation_corpus_id"],
        "query": item["query"],
    }
    for item in generated_query_data
]

# %%
temp = [
    {
        "source_corpus_id": item["source_corpus_id"],
        "citation_corpus_id": item["citation_corpus_info"][0]["citation_corpus_id"],
        "cited_by_citation": item["citation_corpus_info"][0]["cited_by_citation"],
        "hard_negative": item["citation_corpus_info"][0]["hard_negative"],
    }
    for item in hard_negative_index_data
    if item["citation_corpus_info"]
]

my_generated_query_data = temp

# %%

all_corpus_ids = set(corpus_clean_data["corpusid"])

hard_negative_dict = {
    item["source_corpus_id"]: item for item in my_generated_query_data
}

for item in my_generated_query_data:
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
    json.dump(my_generated_query_data, f, indent=2)
# %%
len(my_generated_query_data)
# %%
filtered_data = [
    item
    for item in my_generated_query_data
    if len(item.get("hard_negative", [])) >= 2
    and len(item.get("easy_negative", [])) >= 3
]

with open("./filtered_pre_triplet.json", "w") as f:
    json.dump(filtered_data, f, indent=2)

# %%
len(filtered_data)


# %%
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
# make pre-triplet with query data
with open(
    "./final_generated_query_data.json",
    "r",
) as f:
    generated_query_data = json.load(f)
# %%
corpusid2meta = {
    row["corpusid"]: {"title": row["title"], "abstract": row["abstract"]}
    for row in corpus_clean_data
}

# %%
triplets = []

for item in generated_query_data:
    query = item["query"]
    positive = item["citation_corpus_id"]
    hard_samples = random.sample(item["hard_negative"], 2)
    easy_samples = random.sample(item["easy_negative"], 3)
    negatives = hard_samples + easy_samples

    for neg in negatives:
        # positive/negative의 title/abstract를 찾아서 추가
        pos_meta = corpusid2meta.get(positive, {"title": "", "abstract": ""})
        neg_meta = corpusid2meta.get(neg, {"title": "", "abstract": ""})

        triplets.append(
            {
                "query": query,
                "positive_title": pos_meta["title"],
                "positive_abstract": pos_meta["abstract"],
                "negative_title": neg_meta["title"],
                "negative_abstract": neg_meta["abstract"],
            }
        )

# %%
triplets
# %%

len(triplets)
# %%
with open("./triplet_data.json", "w") as f:
    json.dump(triplets, f, indent=2)
