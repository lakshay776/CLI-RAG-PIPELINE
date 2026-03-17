import faiss
import numpy as np
import pickle

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.data = []

    def add(self, embeddings, texts):
        # fix: FAISS requires a float32 numpy array, not a plain Python list
        vectors = np.array(embeddings, dtype=np.float32)
        self.index.add(vectors)
        self.data.extend(texts)

    def search(self, query_emb, k=5):
        query_vector = np.array([query_emb], dtype=np.float32)
        distances, indices = self.index.search(query_vector, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue # FAISS returned fewer than k results
            data_item = self.data[idx]
            if isinstance(data_item, dict):
                results.append({
                    "text": data_item.get("text", ""),
                    "source": data_item.get("source", "Unknown"),
                    "distance": distances[0][i]
                })
            else:
                results.append({
                    "text": data_item,
                    "source": "Unknown",
                    "distance": distances[0][i]
                })
        return results

    def save(self, path):
        # fix: path must be a file, not a directory
        faiss.write_index(self.index, path + "/index.faiss")
        with open(path + "/index.texts", "wb") as f:
            pickle.dump(self.data, f)

    def load(self, path):
        self.index = faiss.read_index(path)
        with open(path.replace("index.faiss", "index.texts"), "rb") as f:
            self.data = pickle.load(f)