from llm import query_ollama
from prompt import make_prompt


def query_generator(data):
    from tqdm import tqdm

    queries = []
    for item in tqdm(data, desc="Generating LLM queries"):
        prev = item["prev"]
        curr = item["curr"]
        next_ = item["next"]
        title = item["title"]
        abstract = item["abstract"]
        prompt = make_prompt(prev, curr, next_, title, abstract)
        query = query_ollama(prompt)
        queries.append(
            {
                "index": item["index"],
                "citation_corpus_id": item["citation_corpus_id"],
                "query": query,
            }
        )
    return queries
