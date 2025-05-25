import argparse
import json
import os

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
    default="./jsons/filtered_citation_info.json",
    help="Output file path for citation info JSON",
)
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)


with open(args.input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

filtered = [
    item
    for item in data
    if all(item.get(key) is not None for key in ["prev", "curr", "next"])
]

with open(args.output_path, "w", encoding="utf-8") as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)
