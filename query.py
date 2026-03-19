from vector_store import VectorStore
from embeddings import get_embedding
from llm import generate_response
from reranker import rerank_chunks

store = VectorStore(384) # all-MiniLM-L6-v2 dimension is 384
store.load("db")    


while True:

    question = input("\nAsk: ")

    query_embedding = get_embedding(question)

    # Step 1: Retrieve more chunks
    context_chunks = store.search(query_embedding, k=8, threshold=0.15)

    if not context_chunks:
        print("\nNo relevant context found.")
        continue

    # Step 2: Re-rank
    reranked_chunks = rerank_chunks(question, context_chunks, top_n=3)

    # Step 3: Build context
    context = "\n\n".join([c["text"] for c in reranked_chunks])
    sources = list(set([c["source"] for c in reranked_chunks]))

    # Debug: FAISS output
    # print("\n--- Retrieved (FAISS) ---")
    # for r in context_chunks:
    #     print(f"\nScore: {r['score']:.3f}")
    #     print(r["text"][:120])

    # # Debug: Reranked output
    # print("\n--- After Re-ranking ---")
    # for r in reranked_chunks:
    #     print(r["text"][:120])

    # Step 4: Prompt
    prompt = f"""
Use the provided context to answer the question concisely.
If the context is completely unrelated to the question, say "I don't know based on the provided documents."
If the question is about the nature or summary of these documents, provide a summary based on all provided context chunks.
Do NOT use outside knowledge.

Context:
{context}

Question:
{question}

The answer should be concise (max 50 words).
"""

    # Step 5: Generate response
    response = generate_response(prompt)

    print("\nAnswer:\n", response)
    print("\nSources:", ", ".join(sources))