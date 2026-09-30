from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SimilarityCalculator:

    def __init__(self):
        print("Loading Embedding Model...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Embedding Model Loaded!")

    def calculate(self, responses):

        embeddings = self.model.encode(
            responses,
            convert_to_numpy=True
        )

        similarity_matrix = cosine_similarity(embeddings)

        return similarity_matrix

    def average_similarity(self, similarity_matrix):

        n = similarity_matrix.shape[0]

        values = []

        for i in range(n):
            for j in range(i + 1, n):
                values.append(similarity_matrix[i][j])

        return float(np.mean(values))