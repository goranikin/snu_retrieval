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
from typing import Any, List, Tuple

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
kv_pairs = create_kv_pairs(corpus_clean_data, "title_abstract")

# %%
import os
import pickle
from enum import Enum
from typing import Any, List, Tuple

from tqdm import tqdm


class TextType(Enum):
    KEY = 1
    QUERY = 2


class KVStore:
    def __init__(self, index_name: str, index_type: str) -> None:
        self.index_name = index_name
        self.index_type = index_type

        self.keys = []
        self.encoded_keys = []
        self.values = []

    def __len__(self) -> int:
        return len(self.keys)

    def _encode(self, text: str, type: TextType) -> Any:
        return self._encode_batch([text], type, show_progress_bar=False)[0]

    def _encode_batch(
        self, texts: List[str], type: TextType, show_progress_bar: bool = True
    ) -> List[Any]:
        raise NotImplementedError

    def _query(self, encoded_query: Any, n: int) -> List[int]:
        raise NotImplementedError

    def clear(self) -> None:
        self.keys = []
        self.encoded_keys = []
        self.values = []

    def create_index(self, key_value_pairs: List[Tuple[str, Any]]) -> None:
        if len(self.keys) > 0:
            raise ValueError(
                "Index is not empty. Please create a new index or clear the existing one."
            )

        for key, value in tqdm(
            key_value_pairs.items(), desc=f"Creating {self.index_name} index"
        ):
            self.keys.append(key)
            self.values.append(value)
        self.encoded_keys = self._encode_batch(self.keys, TextType.KEY)

    def query(self, query_text: str, n: int, return_keys: bool = False) -> List[Any]:
        encoded_query = self._encode(query_text, TextType.QUERY)
        indices = self._query(encoded_query, n)
        if return_keys:
            results = [(self.keys[i], self.values[i]) for i in indices]
        else:
            results = [self.values[i] for i in indices]
        return results

    def save(self, dir_name: str) -> None:
        save_dict = {}
        for key, value in self.__dict__.items():
            if key[0] != "_":
                save_dict[key] = value

        print(
            f"Saving index to {os.path.join(dir_name, f'{self.index_name}.{self.index_type}')}"
        )
        os.makedirs(dir_name, exist_ok=True)
        with open(
            os.path.join(dir_name, f"{self.index_name}.{self.index_type}"), "wb"
        ) as file:
            pickle.dump(save_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path: str) -> None:
        if len(self.keys) > 0:
            raise ValueError(
                "Index is not empty. Please create a new index or clear the existing one before loading from disk."
            )

        print(f"Loading index from {file_path}...")
        with open(file_path, "rb") as file:
            pickle_data = pickle.load(file)

        for key, value in pickle_data.items():
            setattr(self, key, value)


# %%
import nltk
import numpy as np
from rank_bm25 import BM25Okapi


class BM25(KVStore):
    def __init__(self, index_name: str):
        super().__init__(index_name, "bm25")

        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("punkt_tab")

        self._tokenizer = nltk.word_tokenize
        self._stop_words = set(nltk.corpus.stopwords.words("english"))
        self._stemmer = nltk.stem.PorterStemmer().stem
        self.index = None  # BM25 index

    def _encode_batch(
        self, texts: List[str], type: TextType, show_progress_bar: bool = True
    ) -> List[str]:
        # lowercase, tokenize, remove stopwords, and stem
        tokens_list = []
        for text in tqdm(texts, disable=not show_progress_bar):
            tokens = self._tokenizer(text.lower())
            tokens = [token for token in tokens if token not in self._stop_words]
            tokens = [self._stemmer(token) for token in tokens]
            tokens_list.append(tokens)
        return tokens_list

    def _query(self, encoded_query: List[str], n: int) -> List[int]:
        top_indices = np.argsort(self.index.get_scores(encoded_query))[::-1][
            :n
        ].tolist()
        return top_indices

    def clear(self) -> None:
        super().clear()
        self.index = None

    def create_index(self, key_value_pairs: List[Tuple[str, Any]]) -> None:
        super().create_index(key_value_pairs)
        self.index = BM25Okapi(self.encoded_keys)

    def load(self, dir_name: str):
        super().load(dir_name)
        self._tokenizer = nltk.word_tokenize
        self._stop_words = set(nltk.corpus.stopwords.words("english"))
        self._stemmer = nltk.stem.PorterStemmer().stem
        return self


# %%
bm25_model = BM25("Title_Abstract_BM25")

# %%
bm25_model.create_index(kv_pairs)

# %%
import json

with open("./final_generated_query_data.json", "r") as f:
    data = json.load(f)

# %%
query_set = [element for element in data]
for query in tqdm(query_set):
    query_text = query["query"]
    top_k = bm25_model.query(query_text, 20)
    query["retrieved"] = top_k


# %%
def calculate_recall(corpusids: list, retrieved: list, k: int):
    top_k = retrieved[:k]
    intersection = set(corpusids) & set(top_k)
    return len(intersection) / len(corpusids) if corpusids else 0.0


# %%
import pandas as pd

query_set_df = pd.DataFrame(query_set)
all_recall_at20 = []

for _, query in query_set_df.iterrows():
    r20 = calculate_recall([query["citation_corpus_id"]], query["retrieved"], 20)
    all_recall_at20.append(r20)

mean_recall_at20 = np.mean(all_recall_at20)

mean_recall_at20

# %%


# %%


# %%
