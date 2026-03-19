import re
from typing import List
from llm import generate_response


def expand_query(question: str) -> List[str]:

    prompt = f"""
You are a search query expansion assistant.

Generate 3 different variations of the user query to improve document retrieval.

STRICT RULES:
- DO NOT introduce unrelated meanings
- DO NOT guess abbreviations unless obvious
- Stay in same domain as the query
- Keep queries close to original meaning
- If unsure, keep query unchanged

Return ONLY a comma-separated list.
User Query:
{question}
"""

    response = generate_response(prompt)

    # Split by comma OR newline, then clean up numbers like "1. " and strip whitespace
    raw_queries = re.split(r'[,\n]', response)
    queries: List[str] = []
    for q in raw_queries:
        clean_q = re.sub(r'^\d+\.\s*', '', q.strip()).strip()
        # Also remove quotes if the LLM added them
        clean_q = clean_q.strip('"').strip("'")
        if clean_q:
            queries.append(clean_q)

    # fallback
    if not queries or len(queries) < 1:
        return [question]

    return [queries[i] for i in range(min(len(queries), 3))]


if __name__ == "__main__":
    expanded = expand_query("what is RAG pipeline?")
    print("Expanded queries:")
    for i, q in enumerate(expanded):
        print(f"{i+1}. {q}")
