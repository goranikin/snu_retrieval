import argparse
import json
import os
import re

import nltk
from datasets import load_dataset
from tqdm import tqdm


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


def extract_citation_context(text, citation):
    nltk.download("punkt_tab")
    sentences = nltk.sent_tokenize(text)

    for idx, sent in enumerate(sentences):
        if citation in sent:
            prev_sent = sentences[idx - 1] if idx > 0 else None
            curr_sent = sent
            next_sent = sentences[idx + 1] if idx < len(sentences) - 1 else None
            return prev_sent, curr_sent, next_sent
    return None, None, None


def filtering_papers(full_corpus_data: list[dict], corpus_ids: set[int]):
    def is_target_batch(batch):
        return [cid in corpus_ids for cid in batch["corpusid"]]

    filtered = full_corpus_data.filter(is_target_batch, batched=True, batch_size=1000)
    return filtered


# def check_title_is_in_source_text(
#     source_text: str, title_list: list[str]
# ) -> list[bool]:
#     results = []

#     for title in title_list:
#         match = re.search(re.escape(title), source_text)
#         if match:
#             print(f"'{title}' found at index {match.start()}")
#             results.append(True)
#         else:
#             print(f"'{title}' not found")
#             results.append(False)

#     return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        type=str,
        default="./jsons/citation_info.json",
        help="Output file path for citation info JSON",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    corpus_clean_data = load_dataset(
        "princeton-nlp/LitSearch", "corpus_clean", split="full"
    )
    corpus_s2orc_data = load_dataset(
        "princeton-nlp/LitSearch", "corpus_s2orc", split="full"
    )

    result_list = []

    # 1007 makes 597 data.
    for i in tqdm(range(1200), desc="Processing citation information"):
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
            real_result_list.append(result[0])
            not_null += 1

    # extracing title and abstract from citation corpus
    for info in real_result_list:
        target_paper = filtering_papers(
            corpus_s2orc_data, set([info["citation_corpus_id"]])
        )[0]

        title_json = target_paper["content"]["annotations"].get("title")
        if title_json is not None:
            try:
                title_annotation = json.loads(title_json)[0]
                start, end = title_annotation["start"], title_annotation["end"]
                title = target_paper["content"]["text"][start:end]
            except Exception:
                title = None
        else:
            title = None

        abstract_json = target_paper["content"]["annotations"].get("abstract")
        if abstract_json is not None:
            try:
                abstract_annotation = json.loads(abstract_json)[0]
                start, end = abstract_annotation["start"], abstract_annotation["end"]
                abstract = target_paper["content"]["text"][start:end]
            except Exception:
                abstract = None
        else:
            abstract = None

        info["title"] = title
        info["abstract"] = abstract

    # extracting citation sentences
    for result in tqdm(real_result_list, desc="Extracting citations context"):
        paper = corpus_s2orc_data[result["index"]]
        citation = paper["content"]["text"][result["start"] : result["end"]]
        snippet = paper["content"]["text"][
            result["start"] - 1000 : result["end"] + 1000
        ]
        prev, curr, next_ = extract_citation_context(snippet, citation)
        result["prev"], result["curr"], result["next"] = prev, curr, next_

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(real_result_list, f, ensure_ascii=False, indent=2)
