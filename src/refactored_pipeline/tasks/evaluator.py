import faiss
import numpy as np
import torch

from refactored_pipeline.models.base import Specter2Encoder


class FaissIndexer:
    def __init__(self, encoder: Specter2Encoder, device=None):
        """
        encoder: Specter2Encoder
        device: "cuda" or "mps" or "cpu"
        """
        self.encoder = encoder
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

    def build_faiss_index(
        self,
        documents: list[str],
        index_path: str,
    ) -> tuple[faiss.IndexFlatIP, np.ndarray]:
        """
        documents: list of document texts
        return: (faiss index, document embeddings)
        """

        with torch.no_grad():
            doc_embs = self.encoder.encode(documents, is_q=False)
            if isinstance(doc_embs, tuple):
                doc_embs = doc_embs[0]
            doc_embs = doc_embs.cpu().detach().numpy()

        faiss.normalize_L2(doc_embs)

        dim = doc_embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(doc_embs)
        if index_path:
            faiss.write_index(index, index_path)
        return index, doc_embs


class FaissRetriever:
    def __init__(self, encoder: Specter2Encoder, index_path):
        self.encoder = encoder
        self.index = faiss.read_index(index_path)

    def retrieve(self, query, top_k=20):
        if self.index is None:
            raise ValueError("FAISS index has not been created yet.")
        q_emb = self.encoder.encode(query, is_q=True)
        if isinstance(q_emb, tuple):
            q_emb = q_emb[0]
        q_emb = q_emb.cpu().detach().numpy()

        faiss.normalize_L2(q_emb)
        distances, indices = self.index.search(q_emb, top_k)
        return indices[0].tolist(), distances[0].tolist()
