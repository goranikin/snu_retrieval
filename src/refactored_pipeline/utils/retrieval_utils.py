import numpy as np
from tqdm import tqdm


def retrieve_for_dataset(retriever, query_datasets, corpusid_list, k=20):
    def retrieval_fn(example):
        retrieved_indices, _ = retriever.retrieve([example["query"]], top_k=k)
        retrieved_corpusids = [corpusid_list[i] for i in retrieved_indices]
        example["retrieved"] = retrieved_corpusids
        return example

    items = list(query_datasets)
    results = []
    for example in tqdm(items, desc="Retrieval"):
        results.append(retrieval_fn(example))
    return results


def calculate_recall(corpusids: list, retrieved: list, k: int):
    top_k = retrieved[:k]
    intersection = set(corpusids) & set(top_k)
    return len(intersection) / len(corpusids) if corpusids else 0.0


def mean_recall(dataset, k):
    recalls = [
        calculate_recall(example["corpusids"], example["retrieved"], k)
        for example in dataset
    ]
    return np.mean(recalls) if recalls else 0.0
