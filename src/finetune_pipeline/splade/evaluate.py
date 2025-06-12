import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

from splade.models.models_utils import get_model
from transformers import AutoTokenizer


def encode_sparse(
    model, tokenizer, texts, device, max_length=512, is_query=False, batch_size=32
):
    all_sparse = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Encoding corpus",
            total=(len(texts) + batch_size - 1) // batch_size,
        ):
            batch_texts = texts[i : i + batch_size]
            tokens = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            if is_query:
                sparse = model(q_kwargs=tokens)["q_rep"].cpu()
            else:
                sparse = model(d_kwargs=tokens)["d_rep"].cpu()
            all_sparse.append(sparse)
    return torch.cat(all_sparse, dim=0)


def build_sparse_index(model, tokenizer, corpus_texts, device, batch_size=32):
    doc_sparse = encode_sparse(
        model, tokenizer, corpus_texts, device, is_query=False, batch_size=batch_size
    )
    # shape: (num_docs, vocab_size)
    return doc_sparse.numpy()


def retrieve_topk(query_sparse, doc_sparse, top_k=20):
    # query_sparse: (vocab_size,)
    # doc_sparse: (num_docs, vocab_size)
    scores = np.dot(doc_sparse, query_sparse)
    topk_idx = np.argsort(-scores)[:top_k]
    return topk_idx.tolist()


def calculate_recall(gt_ids, retrieved_ids, k):
    top_k = retrieved_ids[:k]
    intersection = set(gt_ids) & set(top_k)
    return len(intersection) / len(gt_ids) if gt_ids else 0.0


def mean_recall(dataset, k):
    recalls = [
        calculate_recall(example["corpusids"], example["retrieved"], k)
        for example in dataset
    ]
    return np.mean(recalls) if recalls else 0.0


def main():
    query_data = load_dataset("princeton-nlp/LitSearch", "query", split="full")
    corpus_data = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full")

    def classify(example):
        if example["specificity"] == 0:
            if example["query_set"].startswith("inline"):
                return "spec0_inline"
            else:
                return "spec0_manual"
        else:
            if example["query_set"].startswith("inline"):
                return "spec1_inline"
            else:
                return "spec1_manual"

    query_data = query_data.map(lambda x: {"class": classify(x)})

    spec0_ds = query_data.filter(lambda x: x["specificity"] == 0)
    spec1_ds = query_data.filter(lambda x: x["specificity"] == 1)
    spec0_inline_ds = query_data.filter(lambda x: x["class"] == "spec0_inline")
    spec0_manual_ds = query_data.filter(lambda x: x["class"] == "spec0_manual")
    spec1_inline_ds = query_data.filter(lambda x: x["class"] == "spec1_inline")
    spec1_manual_ds = query_data.filter(lambda x: x["class"] == "spec1_manual")

    print(
        spec0_ds.num_rows,
        spec1_ds.num_rows,
        spec0_inline_ds.num_rows,
        spec0_manual_ds.num_rows,
        spec1_inline_ds.num_rows,
        spec1_manual_ds.num_rows,
    )
    model_dir = "splade/output"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # config 예시: {"model_type_or_dir": model_dir, "matching_type": "splade", ...}
    config = {
        "model_type_or_dir": "naver/splade-v3",
        "model_type_or_dir_q": "naver/splade-v3",  # 또는 None
        "matching_type": "splade",
    }
    init_dict = {
        "model_type_or_dir": "naver/splade-v3",
        "model_type_or_dir_q": "naver/splade-v3",
        "agg": "max",
        "fp16": False,
    }
    print("Load model")
    model = get_model(config, init_dict)
    model.load_state_dict(
        torch.load("splade/output/pytorch_model.bin", map_location="cpu")
    )
    model.to(device)

    print("Create index")
    corpus_texts = [
        f"{item['title']}{tokenizer.sep_token}{item['abstract']}"
        for item in corpus_data
    ]
    doc_sparse = build_sparse_index(model, tokenizer, corpus_texts, device)

    def retrieval_fn(example):
        query_sparse = encode_sparse(
            model, tokenizer, [example["query"]], device, is_query=True
        )[0].numpy()
        retrieved_idx = retrieve_topk(query_sparse, doc_sparse, top_k=20)
        # corpusids는 corpus_data의 인덱스와 1:1 매칭된다고 가정
        example["retrieved"] = [corpus_data[i]["corpusid"] for i in retrieved_idx]
        return example

    print("retrieval query")
    spec0_inline_ds = spec0_inline_ds.map(retrieval_fn)
    spec0_manual_ds = spec0_manual_ds.map(retrieval_fn)
    spec1_inline_ds = spec1_inline_ds.map(retrieval_fn)
    spec1_manual_ds = spec1_manual_ds.map(retrieval_fn)
    spec0_ds = spec0_ds.map(retrieval_fn)
    spec1_ds = spec1_ds.map(retrieval_fn)

    spec0_inline_recall20 = mean_recall(spec0_inline_ds, 20)
    spec0_manual_recall20 = mean_recall(spec0_manual_ds, 20)
    spec0_avg_recall20 = mean_recall(spec0_ds, 20)

    spec1_inline_recall5 = mean_recall(spec1_inline_ds, 5)
    spec1_inline_recall20 = mean_recall(spec1_inline_ds, 20)
    spec1_manual_recall5 = mean_recall(spec1_manual_ds, 5)
    spec1_manual_recall20 = mean_recall(spec1_manual_ds, 20)
    spec1_avg_recall20 = mean_recall(spec1_ds, 20)

    print(f"spec0 inline Recall@20: {spec0_inline_recall20:.4f}")
    print(f"spec0 manual Recall@20: {spec0_manual_recall20:.4f}")
    print(f"spec0 avg Recall@20:    {spec0_avg_recall20:.4f}")
    print()
    print(f"spec1 inline Recall@5:  {spec1_inline_recall5:.4f}")
    print(f"spec1 inline Recall@20: {spec1_inline_recall20:.4f}")
    print(f"spec1 manual Recall@5:  {spec1_manual_recall5:.4f}")
    print(f"spec1 manual Recall@20: {spec1_manual_recall20:.4f}")
    print(f"spec1 avg Recall@20:    {spec1_avg_recall20:.4f}")


if __name__ == "__main__":
    main()
