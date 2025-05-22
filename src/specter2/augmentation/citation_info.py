import json

from utils import find_citation_paper_info

from datasets import load_dataset

if __name__ == "__main__":
    query_data = load_dataset("princeton-nlp/LitSearch", "query", split="full")
    corpus_clean_data = load_dataset(
        "princeton-nlp/LitSearch", "corpus_clean", split="full"
    )
    corpus_s2orc_data = load_dataset(
        "princeton-nlp/LitSearch", "corpus_s2orc", split="full"
    )

    result_list = []

    # 1007 makes 597 data.
    for i in range(1007):
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

    with open("/output/citation_info.json", "w", encoding="utf-8") as f:
        json.dump(real_result_list, f, ensure_ascii=False, indent=2)
