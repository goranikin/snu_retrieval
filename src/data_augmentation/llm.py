import re

import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:14b"


def query_ollama(prompt):
    response = requests.post(
        OLLAMA_API_URL, json={"model": MODEL, "prompt": prompt, "stream": False}
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def extract_query(llm_response):
    cleaned = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL)
    cleaned = cleaned.strip()
    if "\n" in cleaned:
        cleaned = cleaned.split("\n")[-1].strip()
    return cleaned
