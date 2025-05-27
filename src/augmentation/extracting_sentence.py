import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm
from utils import (
    extract_citation_context,
    extract_content_with_corpusid,
    find_citation_paper_info,
)

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
        target_paper = extract_content_with_corpusid(
            corpus_s2orc_data, set([info["citation_corpus_id"]])
        )[0]

        # Title 추출 시 NoneType 에러 방지
        title_json = target_paper["content"]["annotations"].get("title")
        if title_json is not None:
            try:
                title_annotation = json.loads(title_json)[0]
                start, end = title_annotation["start"], title_annotation["end"]
                title = target_paper["content"]["text"][start:end]
            except Exception as e:
                title = None
        else:
            title = None

        # Abstract 추출 시 NoneType 에러 방지
        abstract_json = target_paper["content"]["annotations"].get("abstract")
        if abstract_json is not None:
            try:
                abstract_annotation = json.loads(abstract_json)[0]
                start, end = abstract_annotation["start"], abstract_annotation["end"]
                abstract = target_paper["content"]["text"][start:end]
            except Exception as e:
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
