from enum import Enum
from typing import Any, Dict, List

import torch


class TextType(Enum):
    KEY = 1
    QUERY = 2


class Retrieval:
    def __init__(self, index_name: str = "", index_type: str = "") -> None:
        self.index_name = index_name
        self.index_type = index_type

        self.keys = []
        self.values = []
        self.index = None
        self.faiss_index = None

    def __len__(self) -> int:
        return len(self.keys)

    def clear(self) -> None:
        self.keys = []
        self.values = []
        self.index = None
        self.faiss_index = None

    def _encode_text(self, input_ids, attention_mask, adapter_type="proximity"):
        """
        input: tokenized value
        output: embedding value
        """
        raise NotImplementedError

    def encode_query(self, query_text: str):
        """
        embed query text
        """
        raise NotImplementedError

    def encode_paper(self, title: str, abstract: str):
        """
        embed paper using the concat of title and abstract
        """
        raise NotImplementedError

    def _query(self, query_embedding: torch.Tensor, top_k: int = 10) -> List[int]:
        raise NotImplementedError

    def query(self, query_text: str, n: int, return_keys: bool = False) -> List[Any]:
        query_embedding = self.encode_query(query_text)
        indices = self._query(query_embedding, n)
        if return_keys:
            results = [(self.keys[i], self.values[i]) for i in indices]
        else:
            results = [self.values[i] for i in indices]
        return results

    def _encode_paper_batch(
        self, text_list: list[str], show_progress_bar: bool = False
    ) -> Any:
        """
        embed batch of papers using the concat of title and abstract
        """
        raise NotImplementedError

    def create_index(self, key_value_pairs: Dict[str, int]) -> None:
        if len(self.keys) > 0:
            raise ValueError(
                "Index is not empty. Please create a new index or clear the existing one."
            )

        for key, value in key_value_pairs.items():
            self.keys.append(key)
            self.values.append(value)
