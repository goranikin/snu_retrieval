import numpy as np

from refactored_pipeline.utils.data_processing_utils import create_kv_pairs
from refactored_pipeline.utils.retrieval_utils import mean_recall, retrieve_for_dataset


def indexing(encoder, documents, faiss_index_path):
    from refactored_pipeline.tasks.evaluator import FaissIndexer

    indexer = FaissIndexer(encoder=encoder)
    indexer.build_faiss_index(documents, faiss_index_path)


def retreival(encoder, query_data, corpusid_list_path, faiss_index_path, top_k=20):
    from refactored_pipeline.tasks.evaluator import FaissRetriever

    retriever = FaissRetriever(encoder=encoder, index_path=faiss_index_path)
    corpusid_list = np.load(corpusid_list_path).tolist()

    return retrieve_for_dataset(retriever, query_data, corpusid_list, k=top_k)


def main():
    from datasets import load_dataset

    from refactored_pipeline.models.base import Specter2Encoder

    corpusid_list_path = "./corpusid_list.npy"
    faiss_index_path = "./LitSearch_index.faiss"
    model_name = "allenai/specter2_base"
    dataset_name = "princeton-nlp/LitSearch"
    top_k = 20

    corpus_clean_data = load_dataset(dataset_name, "corpus_clean", split="full")
    query_data = load_dataset(dataset_name, "query", split="full")

    kv_pairs = create_kv_pairs(corpus_clean_data)

    documents = list(kv_pairs.keys())
    corpusid_list = list(kv_pairs.values())

    np.save(corpusid_list_path, np.array(corpusid_list))

    encoder = Specter2Encoder(model_name_or_dir=model_name)

    indexing(encoder, documents, faiss_index_path)

    query_data_with_retrieved = retreival(
        encoder, query_data, corpusid_list_path, faiss_index_path, top_k
    )

    recall = mean_recall(query_data_with_retrieved, k=top_k)

    print(f"Mean Recall@{top_k}: {recall:.4f}")


if __name__ == "__main__":
    main()
