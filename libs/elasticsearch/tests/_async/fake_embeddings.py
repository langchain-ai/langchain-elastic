"""Fake Embedding class for testing purposes."""

import hashlib
from typing import List

from langchain_core.embeddings import Embeddings

fake_texts = ["foo", "bar", "baz"]


class AsyncFakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings.
        Embeddings encode each text as its index."""
        return [[float(1.0)] * 9 + [float(i)] for i in range(len(texts))]

    async def aembed_query(self, text: str) -> List[float]:
        """Return constant query embeddings.
        Embeddings are identical to embed_documents(texts)[0].
        Distance to each text will be that text's index,
        as it was passed to embed_documents."""
        return [float(1.0)] * 9 + [float(0.0)]


class AsyncConsistentFakeEmbeddings(AsyncFakeEmbeddings):
    """Fake embeddings which remember all the texts seen so far to return consistent
    vectors for the same texts."""

    def __init__(self, dimensionality: int = 10) -> None:
        self.known_texts: List[str] = []
        self.dimensionality = dimensionality

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
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

    async def aembed_query(self, text: str) -> List[float]:
        """Return consistent embeddings for the text, if seen before, or a constant
        one if the text is unknown."""
        return (await self.aembed_documents([text]))[0]
    
class AsyncStableHashEmbeddings(Embeddings):
    """Embeddings which return stable hash-based vectors for the same texts."""

    @staticmethod
    def _encode(text: str) -> List[float]:
        digest = hashlib.md5(text.encode("utf-8")).digest()
        raw = [b for b in digest[:10]]
        total = sum(raw)
        return [float(v)/float(total) for v in raw]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return stable hash-based embeddings for each text."""
        return [self._encode(text) for text in texts]

    async def aembed_query(self, text: str) -> List[float]:
        """Return stable hash-based embeddings for the text."""
        return self._encode(text)