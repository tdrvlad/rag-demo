from sentence_transformers import SentenceTransformer
import numpy as np
DEFAULT_EMBEDDER = "paraphrase-MiniLM-L6-v2"


class Embedder:
    """
    Module that converts text to semantic embeddings
    """
    def __init__(self, model_id: str = DEFAULT_EMBEDDER):
        """
        Initialize the module.
        :param model_id: id of model on Huggingface
        """
        self.encoder_model = SentenceTransformer(model_id)

    def __call__(self, text: str) -> np.ndarray:
        """
        Encode a short text into a vector embedding
        :param text: The input text to be encoded
        :return: The embedding of the text as a numpy array of size (embedding_size)
        """
        embedding = self.encoder_model.encode(
            sentences=[text],
            convert_to_numpy=True
        )
        embedding = embedding[0] # This is because the encode rmodel is called for a list of texts
        return embedding

    def get_embedding_size(self) -> int:
        """
        Returns the size (dimension) of the embedding vectors
        :return:
        """
        dummy_text = "this is a sentence"
        dummy_embedding = self.__call__(dummy_text)
        return dummy_embedding.size


def test():
    embedder = Embedder()
    text = "This is a sentecne"
    embedding = embedder(text)
    print(embedding.size)


if __name__ == '__main__':
    test()
