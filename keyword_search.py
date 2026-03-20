from rank_bm25 import BM25Okapi
import re


class KeywordSearch:

    def __init__(self, documents):

        self.docs = documents

        # 🔥 Better tokenization (removes punctuation)
        self.texts = [doc["text"] for doc in documents]
        tokenized = [self.tokenize(text) for text in self.texts]

        self.bm25 = BM25Okapi(tokenized)

    def tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def search(self, query, k=5):

        tokenized_query = self.tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        # 🔥 Normalize scores
        max_score = max(scores) if max(scores) > 0 else 1
        normalized_scores = [s / max_score for s in scores]

        # 🔥 Get top-k indices
        ranked_indices = [
            i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        ][:k]

        results = []

        for i in ranked_indices:
            results.append({
                "text": self.docs[i]["text"],
                "source": self.docs[i]["source"],
                "score": normalized_scores[i]
            })

        return results