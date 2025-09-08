"""Fake Embedding class for testing purposes."""

from typing import List
import hashlib

from langchain_core.embeddings import Embeddings

fake_texts = ["foo", "bar", "baz"]


class FakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings.
        Embeddings encode each text as its index."""
        temp = [[float(1.0)] * 9 + [float(i)] for i in range(len(texts))]
        print(temp)
        return temp

    def embed_query(self, text: str) -> List[float]:
        """Return constant query embeddings.
        Embeddings are identical to embed_documents(texts)[0].
        Distance to each text will be that text's index,
        as it was passed to embed_documents."""
        return [float(1.0)] * 9 + [float(0.0)]


class ConsistentFakeEmbeddings(FakeEmbeddings):
    """Fake embeddings which remember all the texts seen so far to return consistent
    vectors for the same texts."""

    def __init__(self, dimensionality: int = 10) -> None:
        self.known_texts: List[str] = []
        self.dimensionality = dimensionality

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return consistent embeddings for each text seen so far."""
        out_vectors = []
        for text in texts:
            if text not in self.known_texts:
                self.known_texts.append(text)
            vector = [float(1.0)] * (self.dimensionality - 1) + [
                float(self.known_texts.index(text))
            ]
            out_vectors.append(vector)
        return out_vectors

    def embed_query(self, text: str) -> List[float]:
        """Return consistent embeddings for the text, if seen before, or a constant
        one if the text is unknown."""
        return (self.embed_documents([text]))[0]
    
class StableHashEmbeddings(Embeddings):
    """Embeddings which return stable hash-based vectors for the same texts."""

    @staticmethod
    def _encode(text: str) -> List[float]:
        digest = hashlib.md5(text.encode("utf-8")).digest()
        raw = [b for b in digest[:10]]
        total = sum(raw)
        return [float(v)/float(total) for v in raw]
    
    def embed_documents(self, texts):
        return [self._encode(text) for text in texts]
    
    def embed_query(self, text):
        return self._encode(text)
    
