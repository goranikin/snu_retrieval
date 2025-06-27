# utilities for processing LitSearch dataset
def get_clean_corpusid(item: dict) -> int:
    return item["corpusid"]


def get_clean_title(item: dict) -> str:
    return item["title"]


def get_clean_abstract(item: dict) -> str:
    return item["abstract"]


def get_clean_title_abstract(item: dict) -> str:
    title = get_clean_title(item)
    abstract = get_clean_abstract(item)
    return f"{title} [SEP] {abstract}"


def create_kv_pairs(data) -> dict:
    kv_pairs = {
        get_clean_title_abstract(record): get_clean_corpusid(record) for record in data
    }
    return kv_pairs
