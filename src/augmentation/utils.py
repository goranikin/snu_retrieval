import json

import nltk


def find_citation_paper_info(
    source_corpus_id: int,
    filtered_corpus_id_list: list[int],
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
        if bib_corpus_id in filtered_corpus_id_list and bib_ref_id is not None:
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


# def extract_content_with_corpusid(full_corpus_data: list[dict], corpus_ids: set[int]):

#     def is_target_batch(batch):
#         return [cid in corpus_ids for cid in batch["corpusid"]]

#     filtered = full_corpus_data.filter(is_target_batch, batched=True, batch_size=1000)

#     return filtered

# def extract_title(filtered_list: list[dict]) -> list[str]:

#     title_list = []

#     for filtered in filtered_list:
#         title_annotation = filtered['content']['annotations']['title'][0]

#         start, end = title_annotation['start'], title_annotation['end']
#         title = filtered['content']['text'][start:end]
#         title_list.append(title)

#     return title_list

# def check_title_is_in_source_text(source_text: str, title_list: list[str]) -> list[bool]:

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
