import re

import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:14b"


def make_prompt(prev, curr, next_):
    return (
        "Below are three consecutive sentences from a research paper.\n"
        "Given these, generate a natural language query in the style of a researcher asking, for example, 'Are there any research papers on ...', 'Are there any studies that ...', or 'Are there any tools or resources for ...'.\n"
        "The query should be broad, abstract, and exploratory, aiming to discover relevant literature based on the overall context, not just the current sentence.\n"
        "Do not copy or closely paraphrase the current sentence; instead, focus on the general research topic, methods, or concepts implied by the context.\n"
        "Output only the search query itself, without any unnecessary prefixes or explanations.\n\n"
        "Example 1:\n"
        "Previous sentence: Deep generative models have shown remarkable success in image synthesis tasks.\n"
        "Current sentence: Recent advances leverage adversarial training to improve sample quality.\n"
        "Next sentence: However, training instability remains a significant challenge.\n"
        "Query: Are there any research papers on generative models for image synthesis and adversarial training methods?\n\n"
        "Example 2:\n"
        "Previous sentence: Temporal consistency is crucial for video generation.\n"
        "Current sentence: Several approaches use optical flow to enforce smooth transitions between frames.\n"
        "Next sentence: Despite these efforts, artifacts still occur in challenging scenarios.\n"
        "Query: Are there any studies that explore methods for improving temporal consistency in video generation using optical flow?\n\n"
        "Example 3:\n"
        "Previous sentence: Self-supervised learning has gained popularity in representation learning.\n"
        "Current sentence: Contrastive loss functions are commonly used to train such models.\n"
        "Next sentence: These methods have been applied to various domains including vision and language.\n"
        "Query: Are there any research papers on self-supervised representation learning with contrastive loss in vision and language?\n\n"
        "Now, generate a query for the following context:\n"
        f"Previous sentence: {prev}\n"
        f"Current sentence: {curr}\n"
        f"Next sentence: {next_}\n"
        "Query:"
    )


def query_ollama(prompt):
    response = requests.post(
        OLLAMA_API_URL, json={"model": MODEL, "prompt": prompt, "stream": False}
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def generate_queries(data):
    from tqdm import tqdm

    queries = []
    for item in tqdm(data, desc="Generating LLM queries"):
        prev = item["prev"]
        curr = item["curr"]
        next_ = item["next"]
        prompt = make_prompt(prev, curr, next_)
        query = query_ollama(prompt)
        queries.append(
            {
                "index": item["index"],
                "citation_corpus_id": item["citation_corpus_id"],
                "query": query,
            }
        )
    return queries


def extract_query(llm_response):
    cleaned = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL)
    cleaned = cleaned.strip()
    if "\n" in cleaned:
        cleaned = cleaned.split("\n")[-1].strip()
    return cleaned


if __name__ == "__main__":
    import json

    from tqdm import tqdm

    with open("src/augmentation/jsons/filtered_citation_info.json") as f:
        data = json.load(f)
    queries = generate_queries(data)

    query_map = {item["index"]: extract_query(item["query"]) for item in queries}
    for item in tqdm(data, desc="Adding queries to data"):
        idx = item["index"]
        if idx in query_map:
            item["query"] = query_map[idx]

    with open("/output/filtered_citation_info_with_query.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
