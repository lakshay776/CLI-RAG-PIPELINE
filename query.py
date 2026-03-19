from vector_store import VectorStore
from embeddings import get_embedding
from llm import generate_response
from reranker import rerank_chunks
from query_expander import expand_query

# Initialize store
store = VectorStore(384)
store.load("db")


while True:

    question = input("\nAsk: ").strip()

    # ---------------------------
    # 🔥 Step 1: Expand Query
    # ---------------------------
    expanded_queries = expand_query(question)

    # Always include original query
    if question not in expanded_queries:
        expanded_queries.insert(0, question)

    # ---------------------------
    # 🔥 Step 2: Clean Queries
    # ---------------------------
    clean_queries = []

    for q in expanded_queries:
        if isinstance(q, str) and len(q.strip()) > 3 and "unknown" not in q.lower():
            clean_queries.append(q.strip())

    # fallback safety
    if not clean_queries:
        clean_queries = [question]

    # limit number of queries
    clean_queries = clean_queries[0:3]

    print("\nSearching for:")
    for q in clean_queries:
        print("-", q)

    # ---------------------------
    # 🔥 Step 3: Retrieve Chunks
    # ---------------------------
    all_chunks = []

    for q in clean_queries:
        emb = get_embedding(q)
        results = store.search(emb, k=8, threshold=0.15)
        all_chunks.extend(results)

    # Deduplicate (text + source)
    unique_chunks = {
        (c["text"], c["source"]): c for c in all_chunks
    }.values()

    context_chunks = list(unique_chunks)

    if not context_chunks:
        print("\nNo relevant context found.")
        continue

    # ---------------------------
    # 🔥 Step 4: Re-rank
    # ---------------------------
    reranked_chunks = rerank_chunks(question, context_chunks, top_n=3)

    if not reranked_chunks:
        print("\nNo relevant context after reranking.")
        continue

    context_text = " ".join([c["text"] for c in reranked_chunks])

    if len(context_text) < 100:
        print("\nInsufficient context to answer.")
        continue

    # ---------------------------
    # 🔥 Step 5: Build Context
    # ---------------------------
    context = "\n\n".join([c["text"] for c in reranked_chunks])
    sources = list(set([c["source"] for c in reranked_chunks]))

    # ---------------------------
    # 🔥 Step 6: Prompt (Guardrails)
    # ---------------------------
    prompt = f"""
You are a strict AI assistant.

Answer ONLY using the provided context.

Rules:
- Do NOT use any external knowledge
- If the answer is not clearly present in the context, say:
  "I don't know based on the provided documents."
- Keep the answer under 50 words

Context:
{context}

Question:
{question}
"""

    # ---------------------------
    # 🔥 Step 7: Generate Answer
    # ---------------------------
    response = generate_response(prompt)

    # ---------------------------
    # 🔥 Step 8: Guardrails
    # ---------------------------
    if "I don't know" not in response and len(context.strip()) < 40:
        print("\n⚠️ Response may be hallucinated. Skipping.")
        continue

    avg_score = sum(c["score"] for c in reranked_chunks) / len(reranked_chunks)

    if avg_score < 0.25:
        print("\n⚠️ Low confidence retrieval. Skipping answer.")
        continue

    # ---------------------------
    # ✅ Final Output
    # ---------------------------
    print("\nAnswer:\n", response)
    print("\nSources:", ", ".join(sources))