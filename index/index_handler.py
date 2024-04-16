from documents import document_handler
from documents.document_handler import DocumentHandler
from index.cosine_simialrity import cosine_similarity
import numpy as np


class IndexHandler:
    def __init__(self, document_handler_module: DocumentHandler = document_handler):
        self.document_handler_module = document_handler_module
        self.texts, self.embeddings = self.document_handler_module.get_texts_and_embeddings()
        self.embedding_dim = self.document_handler_module.embedding_fn.get_embedding_size()

    def add_text(self, text: str):
        print(f"Adding text <{text}> to index.")
        embedding = self.document_handler_module.add_doc(text)
        # This also saves the document in the cache

        # And we also add it in the list to be able to retrieve it
        self.texts.append(text)
        self.embeddings.append(embedding)
        print(f"Index contains {self.embeddings} samples.")

    def search(self, query_text, num_results=3):
        # We don't want to add the text to our cache, we just want to get its embedding in order to search the index
        query_embedding = self.document_handler_module.embedding_fn(query_text)

        if query_embedding.size != self.embedding_dim:
            raise ValueError("Mismatch between embedding sizes")

        similarities = cosine_similarity(
            query=query_embedding,
            vectors=self.embeddings
        )

        # Select the top k similarities indices
        top_k_indices = np.argsort(similarities)[-num_results:][::-1]
        similar_texts = [self.texts[ind] for ind in top_k_indices]
        results = {
            text: float(similarity) for text, similarity in zip(similar_texts, similarities)
        }
        return results


if __name__ == '__main__':
    index = IndexHandler()
    index.add_text("Random stuff here")
    index.add_text("This is a sentence")
    index.add_text("Bla Bla bla")

    for query in ["This is a proposition", "Random stuff here", "Random stuff here!!"]:
        print(f"\nComputing search for <{query}>:")
        print(index.search(query))



