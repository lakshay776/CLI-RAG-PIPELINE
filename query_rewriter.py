from llm import generate_response


def rewrite_query(question):

    prompt = f"""
You are a query rewriting assistant.

Rewrite the user query to make it clearer for document retrieval.

STRICT RULES:
- Do NOT introduce new concepts
- Do NOT assume missing context
- ONLY rephrase the given query
- If the query is vague, return it unchanged
- Return ONLY the rewritten query (no explanations)
- Output ONLY the search query. No preamble.
- Example: "what is this docs abt?" -> "summary of all documents"

User Query:
{question}
Rewritten Query:
"""

    response = generate_response(prompt).strip()

    # 🔥 Cleanup (important)
    response = response.replace("Rewritten Query:", "").strip()


    # 🔥 Fallback safety
    if not response or len(response) < 5:
        return question
    print(response)

    return response


if __name__ == "__main__":
    print(rewrite_query("what is the capital of india"))