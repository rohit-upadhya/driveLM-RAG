import faiss
import numpy as np


class FaissDB:
    """Class to do the vector db operations"""

    def __init__(self):
        pass

    def build_index(
        self,
        vectors,
        ids,
    ):
        vectors = vectors.astype("float32")
        vector_dimension = vectors.shape[1]
        faiss.normalize_L2(vectors)
        cpu_index = faiss.IndexFlatIP(vector_dimension)
        self.index = faiss.IndexIDMap(cpu_index)
        self.index.add_with_ids(vectors, np.array(ids, dtype=np.int64))

    def perform_search(self, query, k=5):

        query = query.astype("float32")
        faiss.normalize_L2(query)
        _, ann = self.index.search(query, k=k)
        return ann[0]

    def perform_search_with_scores(self, query, k=10):

        query = query.astype("float32")
        faiss.normalize_L2(query)
        scores, ann = self.index.search(query, k=k)
        return scores[0], ann[0]
