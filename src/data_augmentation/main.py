import argparse

from query_generator import query_generator
from tqdm import tqdm

from data_augmentation.llm import extract_query
from utils.io_utils import load_json, save_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="./jsons/citation_info2.json",
        help="Input file path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./jsons/final_generating_query_data.json",
        help="Output file path for citation info JSON",
    )
    args = parser.parse_args()

    data = load_json(args.input_path)

    filtered = [
        item
        for item in data
        if all(
            item.get(key) is not None
            for key in ["prev", "curr", "next", "title", "abstract"]
        )
    ]
    queries = query_generator(data)

    query_map = {item["index"]: extract_query(item["query"]) for item in queries}
    for item in tqdm(data, desc="Adding queries to data"):
        idx = item["index"]
        if idx in query_map:
            item["query"] = query_map[idx]

    save_json(args.output_path, data)
