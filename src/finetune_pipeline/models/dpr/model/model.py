import os

import faiss
import torch
from tqdm import tqdm
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)

from finetune_pipeline.models.dpr.model.retrieval import Retrieval


class Dpr(Retrieval):
    def __init__(self, base_model_name="facebook/dpr-", device=None):
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

        query_model_name = base_model_name + "question_encoder-single-nq-base"
        paper_model_name = base_model_name + "ctx_encoder-single-nq-base"

        self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            query_model_name
        )
        self.paper_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            paper_model_name
        )
        self.query_encoder = DPRQuestionEncoder.from_pretrained(query_model_name).to(
            device
        )
        self.paper_encoder = DPRContextEncoder.from_pretrained(paper_model_name).to(
            device
        )

    def parameters(self):
        return list(self.query_encoder.parameters()) + list(
            self.paper_encoder.parameters()
        )

    def clear(self):
        super().clear()
        self.index = None
        self.faiss_index = None

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.query_encoder.save_pretrained(os.path.join(output_dir, "query_encoder"))
        self.paper_encoder.save_pretrained(os.path.join(output_dir, "paper_encoder"))
        self.query_tokenizer.save_pretrained(
            os.path.join(output_dir, "query_tokenizer")
        )
        self.paper_tokenizer.save_pretrained(
            os.path.join(output_dir, "paper_tokenizer")
        )
        print(f"DPR 모델과 토크나이저가 {output_dir}에 저장되었습니다.")

    def _encode_text(self, input_ids, attention_mask, adapter_type="query"):
        if adapter_type == "query":
            outputs = self.query_encoder(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
            )
        elif adapter_type == "paper":
            outputs = self.paper_encoder(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
            )
        embeddings = outputs.pooler_output
        return embeddings

    def encode_query(self, query_text: str):
        self.query_encoder.eval()
        with torch.no_grad():
            tokens = self.query_tokenizer(
                query_text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            return self._encode_text(
                tokens["input_ids"], tokens["attention_mask"], adapter_type="query"
            )

    def encode_paper(self, title: str, abstract: str):
        self.paper_encoder.eval()
        with torch.no_grad():
            text = title + self.paper_tokenizer.sep_token + abstract
            tokens = self.paper_tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            return self._encode_text(
                tokens["input_ids"], tokens["attention_mask"], adapter_type="paper"
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
        batch_size = 256
        embeddings = []

        should_show_progress = show_progress_bar

        iterator = range(0, len(textList), batch_size)
        if should_show_progress:
            iterator = tqdm(iterator, desc="Processing document embeddings")

        for i in iterator:
            batch_texts = textList[i : i + batch_size]
            encoded = self.paper_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.paper_encoder(**encoded)

            batch_embeddings = outputs.pooler_output
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
