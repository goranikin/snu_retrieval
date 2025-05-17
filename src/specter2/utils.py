from torch.utils.data import Dataset


def get_clean_corpusid(item: dict) -> int:
    return item["corpusid"]


def get_clean_title(item: dict) -> str:
    return item["title"]


def get_clean_abstract(item: dict) -> str:
    return item["abstract"]


def get_clean_title_abstract(item: dict, tokenizer) -> str:
    title = get_clean_title(item)
    abstract = get_clean_abstract(item)
    return title + tokenizer.sep_token + abstract


def create_kv_pairs(data: Dataset, key: str, tokenizer) -> dict:
    return {
        get_clean_title_abstract(record, tokenizer): get_clean_corpusid(record)
        for record in data
    }
