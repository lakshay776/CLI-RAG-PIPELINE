from vector_store import VectorStore
from embeddings import get_embedding
from llm import generate_response
store = VectorStore(768)
store.load("db/index.faiss")

while True:
    question = input("\nAsk: ")
    query_embedding = get_embedding(question)
    context_chunks = store.search(query_embedding, 3)
    context = "\n\n".join([c["text"] for c in context_chunks])
    sources = list(set([c["source"] for c in context_chunks]))
    prompt = f"""
    Answer the question using the context below. If you don't know the answer, say you don't know. 
    Context:
    {context}

    Question:
    {question}

    the answer should be in just 50 words

    """

    response = generate_response(prompt)
    print("\nAnswer:\n", response)
    print("\nSources:", ", ".join(sources))