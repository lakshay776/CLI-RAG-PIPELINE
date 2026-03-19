from llm import generate_response


def rerank_chunks(query, chunks, top_n=3):

    formatted_chunks = "\n\n".join([
        f"{i+1}. {chunk['text'][:300]}"
        for i, chunk in enumerate(chunks)
    ])

    prompt = f"""
You are an AI assistant.

A user asked a question:
"{query}"

Here are some retrieved document chunks:

{formatted_chunks}

Select the {top_n} most relevant chunks for answering the question.

Return ONLY the numbers of the best chunks in order (example: 2,5,1)
"""

    response = generate_response(prompt)

    try:
        indices = [int(i.strip())-1 for i in response.text.split(",")]
    except:
        return chunks[:top_n]  # fallback

    ranked = []
    for i in indices:
        if 0 <= i < len(chunks):
            ranked.append(chunks[i])

    return ranked