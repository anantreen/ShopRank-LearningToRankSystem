"""Optional bi-encoder + exact FAISS inner-product retrieval."""


class DenseRetriever:
    def __init__(self, documents, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Install semantic extras: pip install -e '.[semantic]'") from error
        self.faiss = faiss
        self.model = SentenceTransformer(model_name)
        vectors = self.model.encode(documents, normalize_embeddings=True, convert_to_numpy=True)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors.astype("float32"))

    def search(self, query, k=100):
        vector = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, indices = self.index.search(vector.astype("float32"), k)
        return list(zip(indices[0].tolist(), scores[0].tolist()))

