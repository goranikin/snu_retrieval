import glob
import json
import os
import random

import pandas as pd
from datasets import load_dataset
from model import SPECTER2QueryAdapterFinetuner
from tqdm import tqdm

import hydra
from omegaconf import DictConfig


def evaluate_model(model, test_data, query_data, k_values=[1, 5, 10, 20]):
    query_df = pd.DataFrame(
        {"query": query_data["query"], "corpusids": query_data["corpusids"]}
    )

    test_questions = [item["question"] for item in test_data]
    filtered_query_df = query_df[query_df["query"].isin(test_questions)]

    if test_data:
        print(f"Evaluating {len(test_data)} samples")
        query_types = {}
        for item in test_data:
            query = item["question"]
            query_row = filtered_query_df[filtered_query_df["query"] == query]
            if not query_row.empty:
                query_set = query_row.iloc[0].get("query_set", "unknown")
                query_types[query_set] = query_types.get(query_set, 0) + 1
        for query_type, count in query_types.items():
            print(f"  - {query_type}: {count} samples")

    results = {}
    result = []

    for k in k_values:
        total_recall = 0
        count = 0

        test_progress = tqdm(
            enumerate(test_data), total=len(test_data), desc=f"Evaluating Recall@{k}"
        )

        for i, item in test_progress:
            query = item["question"]
            top_k_results = model.query(query, k)

            query_row = filtered_query_df[filtered_query_df["query"] == query]
            if not query_row.empty:
                true_corpus_ids = query_row.iloc[0]["corpusids"]
                if isinstance(true_corpus_ids, list):
                    true_corpus_ids_flat = true_corpus_ids
                else:
                    true_corpus_ids_flat = [true_corpus_ids]

                intersection = set(true_corpus_ids_flat) & set(top_k_results)
                recall = (
                    len(intersection) / len(true_corpus_ids_flat)
                    if true_corpus_ids_flat
                    else 0
                )

                total_recall += recall
                count += 1

                current_avg_recall = total_recall / count if count > 0 else 0
                test_progress.set_postfix({"avg_recall": f"{current_avg_recall:.4f}"})

        if count > 0:
            avg_recall = total_recall / count
            results[f"Recall@{k}"] = avg_recall
            result.append(avg_recall)
        else:
            results[f"Recall@{k}"] = 0
            result.append(0)

    return results


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 데이터 로드 및 분할 (test set만 필요)
    datasets_dir = cfg.train.datasets_dir

    json_files = glob.glob(os.path.join(datasets_dir, "*.json"))

    data_dict = {}
    for file_path in json_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        if cfg.train.test_key in file_name or "test" in file_name:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data_dict[file_name] = data

    # test_data key 자동 탐색 (예: spec0_test, spec1_test 등)
    test_keys = [k for k in data_dict.keys() if "test" in k]
    test_data = []
    for k in test_keys:
        test_data.extend(data_dict[k])
    random.shuffle(test_data)

    print(
        f"Testing: {len(test_data)} samples "
        + " ".join([f"({k}: {len(data_dict[k])})" for k in test_keys])
    )

    # 저장된 모델/어댑터/토크나이저 불러오기
    model = SPECTER2QueryAdapterFinetuner(base_model_name=cfg.train.output_dir)

    print("Evaluating model on test set...")

    query_data = load_dataset("princeton-nlp/LitSearch", "query", split="full")

    # Specificity별 성능 평가
    test_spec_0 = [item for item in test_data if item.get("specificity", 0) == 0]
    test_spec_1 = [item for item in test_data if item.get("specificity", 0) == 1]

    def is_inline(item):
        return "inline" in item.get("query_set", "")

    def is_manual(item):
        return "manual" in item.get("query_set", "")

    test_spec_0_inline = [item for item in test_spec_0 if is_inline(item)]
    test_spec_0_manual = [item for item in test_spec_0 if is_manual(item)]
    test_spec_1_inline = [item for item in test_spec_1 if is_inline(item)]
    test_spec_1_manual = [item for item in test_spec_1 if is_manual(item)]

    print("\nSpecificity와 Query type 조합 평가")
    spec_0_inline_performance = evaluate_model(model, test_spec_0_inline, query_data)
    print(f"Specificity 0 + Inline performance: {spec_0_inline_performance}")

    spec_0_manual_performance = evaluate_model(model, test_spec_0_manual, query_data)
    print(f"Specificity 0 + Manual performance: {spec_0_manual_performance}")

    spec_1_inline_performance = evaluate_model(model, test_spec_1_inline, query_data)
    print(f"Specificity 1 + Inline performance: {spec_1_inline_performance}")

    spec_1_manual_performance = evaluate_model(model, test_spec_1_manual, query_data)
    print(f"Specificity 1 + Manual performance: {spec_1_manual_performance}")

    overall_performance = evaluate_model(model, test_data, query_data)
    print(f"Overall test performance: {overall_performance}")

    spec_0_performance = evaluate_model(model, test_spec_0, query_data)
    print(f"Specificity 0 performance: {spec_0_performance}")

    spec_1_performance = evaluate_model(model, test_spec_1, query_data)
    print(f"Specificity 1 performance: {spec_1_performance}")


if __name__ == "__main__":
    main()  # type: ignore
