import os

import faiss
import torch
from adapters import AutoAdapterModel
from retrieval import Retrieval
from tqdm import tqdm
from transformers import AutoTokenizer


class Specter2(Retrieval):
    def __init__(self, base_model_name="allenai/specter2_base", device=None):
        self.keys = []
        self.values = []
        self.index = None
        self.faiss_index = None

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")

        self.model.load_adapter("allenai/specter2", source="hf", load_as="proximity")
        self.model.load_adapter(
            "allenai/specter2_adhoc_query", source="hf", load_as="adhoc_query"
        )

        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if "adapters.adhoc_query" in name:
                param.requires_grad = True

        self.model.to(self.device)

    def parameters(self):
        return self.model.parameters()

    def clear(self):
        super().clear()
        self.index = None
        self.faiss_index = None

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_adapter(output_dir, "adhoc_query")

        self.tokenizer.save_pretrained(output_dir)

        print(f"어댑터가 {output_dir}에 저장되었습니다.")

    def _encode_text(self, input_ids, attention_mask, adapter_type="proximity"):
        """
        adapter_type: query -> "adhoc_query", paper -> "proximity"
        """
        self.model.set_active_adapters(adapter_type)

        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        )
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

    def encode_query(self, query_text: str):
        self.model.eval()
        with torch.no_grad():
            tokens = self.tokenizer(
                query_text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            return self._encode_text(
                tokens["input_ids"],
                tokens["attention_mask"],
                adapter_type="adhoc_query",
            )

    def encode_paper(self, title: str, abstract: str):
        self.model.eval()
        with torch.no_grad():
            text = title + self.tokenizer.sep_token + abstract
            tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            return self._encode_text(
                tokens["input_ids"], tokens["attention_mask"], adapter_type="proximity"
            )

    def _query(self, query_embedding: torch.Tensor, top_k: int = 20) -> list[int]:
        if self.faiss_index is None:
            raise ValueError(
                "FAISS index has not been created yet. Call create_index first."
            )

        query_vector = query_embedding.numpy()
        faiss.normalize_L2(query_vector)
        distances, indices = self.faiss_index.search(query_vector, top_k)

        return indices[0].tolist()

    def query(self, query_text: str, n: int, return_keys: bool = False) -> list:
        query_embedding = self.encode_query(query_text)
        indices = self._query(query_embedding, n)

        if return_keys:
            results = [(self.keys[i], self.values[i]) for i in indices]
        else:
            results = [self.values[i] for i in indices]

        return results

    def _encode_paper_batch(
        self, textList: list[str], show_progress_bar: bool = True
    ) -> torch.Tensor:
        self.model.set_active_adapters("proximity")
        batch_size = 256
        embeddings = []

        should_show_progress = show_progress_bar

        iterator = range(0, len(textList), batch_size)
        if should_show_progress:
            iterator = tqdm(iterator, desc="Processing document embeddings")

        for i in iterator:
            batch_texts = textList[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)

            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            embeddings.append(batch_embeddings)

            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings

    def create_index(self, key_value_pairs: dict[str, int]) -> None:
        super().create_index(key_value_pairs)
        self.index = self._encode_paper_batch(self.keys)

        vector_dim = self.index.shape[1]
        index_flat = faiss.IndexFlatIP(vector_dim)
        index_vectors = self.index.numpy()
        faiss.normalize_L2(index_vectors)

        index_flat.add(index_vectors)
        self.faiss_index = index_flat
