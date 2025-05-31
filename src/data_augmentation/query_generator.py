from prompt import make_prompt
from ollama import query_ollama


def query_generator(data):
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
