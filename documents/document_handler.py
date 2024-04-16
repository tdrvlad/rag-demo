import os
from embedder import embedder_module
from embedder.embedder import Embedder
import json
from typing import List, Tuple
from uuid import uuid4
import numpy as np


DEFAULT_CACHE_DIR = ".data/documents"


class DocumentHandler:
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR, embedding_fn: Embedder = embedder_module):
        self.embedding_fn = embedding_fn
        # We save embedding separately based on the model we used
        self.docs_cache_dir = os.path.join(cache_dir, embedding_fn.model_id)
        os.makedirs(self.docs_cache_dir, exist_ok=True)
        self.docs = self.load_docs()

    def load_docs(self) -> List[dict]:
        """
        Searches all JSON files in self.docs_cache_dir and loads them as a list of dictionaries
        :return: list-of-docs-dict
        """
        json_files = [f for f in os.listdir(self.docs_cache_dir) if f.endswith('.json')]
        docs = []
        for file_name in json_files:
            file_path = os.path.join(self.docs_cache_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    data['embedding'] = np.array(data["embedding"])
                    # The embedding is saved as a list because arrays are not JSON serializable
                    docs.append(data)
                    print(f"Loaded {file_name}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {file_name}")
            except Exception as e:
                print(f"An error occurred with {file_name}: {e}")
        print(f"Loaded {len(docs)} existing documents.")
        return docs

    def save_doc(self, doc_dict: dict):
        self.docs.append(doc_dict)
        with open(doc_dict['file_path'], 'w', encoding='utf-8') as file:
            json.dump(doc_dict, file, ensure_ascii=False, indent=4)
        return True

    def add_doc(self, text: str) -> np.ndarray:
        """
        Add document to the cache. Returns the embedding of the text for further usage.
        :param text: string if the text
        :return: the embedding vector
        """

        # Check if document already exists
        texts, embeddings = self.get_texts_and_embeddings()
        if text in texts:
            print('Document already added.')
            ind = texts.index(text)
            return embeddings[ind]

        doc_id = str(uuid4())
        file_path = os.path.join(self.docs_cache_dir, f"{doc_id}.json")
        embedding = self.embedding_fn(text)
        embedding_list = [float(e) for e in embedding]
        # List conversion of the numpy array is necessary in order to save it as a JSON

        doc_dict = {
            'file_path': file_path,
            'id': doc_id,
            'text': text,
            'embedding': embedding_list
        }
        self.save_doc(doc_dict)
        return embedding

    def get_docs(self):
        """
        Returns the loaded documents
        :return:  list-of-docs-dict
        """
        return self.docs

    def get_texts_and_embeddings(self) -> Tuple[List[str], List[np.ndarray]]:
        """
        Returns the cached texts and embeddings as lists.
        :return:
        """
        texts = []
        embeddings = []
        for doc in self.docs:
            texts.append(doc["text"])
            embeddings.append(doc["embedding"])
        return texts, embeddings

    def delete_doc(self, text):
        """
        Removes a document from the cache.
        :param text: The text of the document to be removed
        """
        to_remove = []
        for doc_dict in self.docs:
            if doc_dict['text'] == text:
                os.remove(doc_dict['file_path'])
        for item in to_remove:
            self.docs.remove(item)


if __name__ == "__main__":
    doc_handler = DocumentHandler()
    doc_handler.add_doc("This is a sentence")

