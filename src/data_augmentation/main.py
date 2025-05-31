import argparse

from postprocess import extract_query
from query_generator import generate_queries
from tqdm import tqdm

from utils.io_utils import load_json, save_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="./jsons/citation_info.json",
        help="Input file path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./jsons/citation_info_with_query.json",
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
    queries = generate_queries(data)

    query_map = {item["index"]: extract_query(item["query"]) for item in queries}
    for item in tqdm(data, desc="Adding queries to data"):
        idx = item["index"]
        if idx in query_map:
            item["query"] = query_map[idx]

    save_json(args.output_path, data)
