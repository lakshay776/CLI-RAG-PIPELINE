from loader import load_documents
from chunker import chunk_text
from embeddings import get_embedding
from vector_store import VectorStore
import os

docs = load_documents("documents")
chunks = []
chunks_metadata = []

for doc in docs:
    file_chunks = chunk_text(doc)
    chunks.extend(file_chunks)

embeddings = []

for chunk in chunks:
    emb = get_embedding(chunk["text"])
    embeddings.append(emb)

if not embeddings:
    print("No documents found or no text extracted. Exiting.")
    exit(1)

dim = len(embeddings[0])

# We should probably update VectorStore to handle metadata if we want to retrieve filenames later.
# For now, let's keep it simple and just index the text.
store = VectorStore(dim)
store.add(embeddings, chunks)
os.makedirs("db", exist_ok=True)
store.save("db")
print("documents indexed successfully")
