"""Fake Embedding class for testing purposes."""

from typing import List

from ._async.fake_embeddings import (
    AsyncConsistentFakeEmbeddings as _AsyncConsistentFakeEmbeddings,
)
from ._async.fake_embeddings import (
    AsyncFakeEmbeddings as _AsyncFakeEmbeddings,
)
from ._sync.fake_embeddings import (  # noqa: F401
    ConsistentFakeEmbeddings,
    FakeEmbeddings,
)


# langchain defines embed_documents and embed_query as abstract in its base class
# so we have to add dummy methods for them, even though we only use the async versions
class AsyncFakeEmbeddings(_AsyncFakeEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("This class is asynchronous, use aembed_documents()")

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError("This class is asynchronous, use aembed_query()")


class AsyncConsistentFakeEmbeddings(_AsyncConsistentFakeEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("This class is asynchronous, use aembed_documents()")

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError("This class is asynchronous, use aembed_query()")
