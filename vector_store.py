import faiss
import numpy as np
import pickle
import os

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.data = []

    def add(self, embeddings, texts):
        vectors = np.array(embeddings, dtype=np.float32)
        self.index.add(vectors)
        self.data.extend(texts)

    def search(self, query_emb, k=5, threshold=0.3):
        query_vector = np.array([query_emb], dtype=np.float32)
        distances, indices = self.index.search(query_vector, k)




        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            distance = distances[0][i]
            if distance > 1e10:
                continue
            similarity = distance # For IndexFlatIP, distance is the inner product (similarity)

            if similarity < threshold:
                continue

            data_item = self.data[idx]

            if isinstance(data_item, dict):
                results.append({
                    "text": data_item.get("text",""),
                    "source": data_item.get("source","Unknown"),
                    "score": similarity,
                    "distance": distance
                })
            else:
                results.append({
                    "text": data_item,
                    "source": "Unknown",
                    "score": similarity,
                    "distance": distance
                })

        return results

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "index.texts"), "wb") as f:
            pickle.dump(self.data, f)

    def load(self, path):
        index_path = os.path.join(path, "index.faiss")
        data_path = os.path.join(path, "index.texts")

        self.index = faiss.read_index(index_path)

        with open(data_path, "rb") as f:
            self.data = pickle.load(f)