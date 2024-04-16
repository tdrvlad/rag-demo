from documents import document_handler
from documents.document_handler import DocumentHandler
import faiss
import numpy as np
from typing import List


class IndexHandler:
    def __init__(self, document_handler_module: DocumentHandler = document_handler):
        self.document_handler_module = document_handler_module
        self.texts, self.embeddings = self.document_handler_module.get_texts_and_embeddings()

        # Build the Faiss Index
        embedding_dim = self.document_handler_module.embedding_fn.get_embedding_size()
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)

        if len(self.texts):
            self.add_embeddings_to_index(self.embeddings)

    def convert_embeddings_to_array(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        FAISS expects numpy arrays instead of lists.
        """
        embeddings_array = np.array(embeddings).astype(np.float32)
        return embeddings_array

    def add_embeddings_to_index(self, embeddings: List[np.ndarray]):
        """
        Adds a list of embeddings to the Index
        :param embeddings:
        :return:
        """
        # Convert the list of embeddings to an array (as FAISS expects)
        embeddings_array = self.convert_embeddings_to_array(embeddings)

        # # Normalized embeddings should work better than normal
        # normalized_embeddings_array = faiss.normalize_L2(embeddings_array)

        self.faiss_index.add(embeddings_array)
        print(f"Added {len(embeddings)} samples. Index now contains {self.faiss_index.ntotal} samples.")

    def add_text(self, text: str):
        print(f"Adding text <{text}> to index.")
        embedding = self.document_handler_module.add_doc(text)
        # This also saves the document in the cache

        # Now we add the new text and embedding to the index
        self.add_embeddings_to_index([embedding])

        # And we also add it in the list to be able to retrieve it
        self.texts.append(text)
        self.embeddings.append(embedding)
        print(f"Index contains {self.faiss_index.ntotal} samples.")

    def search(self, query_text, num_results=3):
        # We don't want to add the text to our cache, we just want to get its embedding in order to search the index
        query_embedding = self.document_handler_module.embedding_fn(query_text)

        # 1. FAISS is optimized to do multiple searches in parallel, therefore we make our query into a list
        query_embeddings_batch = [query_embedding]
        query_embeddings_batch_array = self.convert_embeddings_to_array(query_embeddings_batch)

        # We perform the search
        distances_batch, indices_batch = self.faiss_index.search(query_embeddings_batch_array, k=num_results)

        # 2. Because we converted our query into a list, we now extract the first element (the reverse of 1.)
        distances = distances_batch[0]
        indices = indices_batch[0]

        # Process the result to a readable shape
        found_texts = [self.texts[ind] for ind in indices]
        results = {
            text: float(dist) for text, dist in zip(found_texts, distances)
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



