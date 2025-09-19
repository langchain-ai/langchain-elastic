"""Fake Embedding class for testing purposes."""

import hashlib
from typing import List

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
    """Deterministic hash-based embeddings for robust testing. (sync version)

    Why:
    - Elasticsearch 8.14+ indexes dense vectors with int8_hnsw by default.
      Quantization (int8) + HNSW ANN can slightly disturb scores/ranking
      especially when vectors are nearly identical.
    - Tests need deterministic separation so small quantization/ANN
      effects do not flip top-1 results or break strict assertions.

    What:
    - Produce a 16-dim vector from md5(text), convert to integers, then L1-normalize 
      so values sum to 1.0. Round to 2 decimal places for precision stability.
      This gives stable, well-separated but deterministic vectors which will work 
      across ES versions.
    """

    @staticmethod
    def _encode(text: str) -> List[float]:
        digest = hashlib.md5(text.encode("utf-8")).digest()
        total = sum(digest)
        # Round to 2 decimal places to avoid precision issues
        return [round(float(v) / float(total), 2) for v in digest]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._encode(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._encode(text)
