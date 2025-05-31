from enum import Enum
from typing import Any, Dict, List

import torch


class TextType(Enum):
    KEY = 1
    QUERY = 2


class Retrieval:
    def __init__(self, index_name: str, index_type: str) -> None:
        self.index_name = index_name
        self.index_type = index_type

        self.keys = []
        self.values = []

    def __len__(self) -> int:
        return len(self.keys)

    def _get_embeddings(
        self, textList: list[str], type: TextType, show_progress_bar: bool = False
    ) -> Any:
        raise NotImplementedError

    def _query(self, query_embedding: torch.Tensor, top_k: int = 10) -> List[int]:
        raise NotImplementedError

    def query(self, query_text: str, n: int, return_keys: bool = False) -> List[Any]:
        embedding_query = self._get_embeddings([query_text], TextType.QUERY)
        indices = self._query(embedding_query, n)
        if return_keys:
            results = [(self.keys[i], self.values[i]) for i in indices]
        else:
            results = [self.values[i] for i in indices]
        return results

    def clear(self) -> None:
        self.keys = []
        self.encoded_keys = []
        self.values = []

    def create_index(self, key_value_pairs: Dict[str, int]) -> None:
        if len(self.keys) > 0:
            raise ValueError(
                "Index is not empty. Please create a new index or clear the existing one."
            )

        for key, value in key_value_pairs.items():
            self.keys.append(key)
            self.values.append(value)
