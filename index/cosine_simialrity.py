import numpy as np


def cosine_similarity(query, vectors):
    """
    Calculate the cosine similarity between a query vector and a list of vectors.

    Args:
    - query (np.ndarray): The query vector.
    - vectors (list of np.ndarray): The list of vectors to compare against the query.

    Returns:
    - list: A list of cosine similarity scores.
    """
    # Normalize the query vector
    query_norm = query / np.linalg.norm(query)

    # Normalize each vector in the list
    vectors_norm = np.array([v / np.linalg.norm(v) for v in vectors])

    # Dot product of query vector with each of the normalized vectors
    cosine_similarities = np.dot(vectors_norm, query_norm)

    return cosine_similarities


if __name__ == "__main__":
    query_vector = np.array([1, 2, 3])
    list_of_vectors = [
        np.array([1, 0, 3]),
        np.array([1, 2, 2]),
        np.array([3, 2, 1])
    ]
    similarities = cosine_similarity(query_vector, list_of_vectors)
    print(similarities)