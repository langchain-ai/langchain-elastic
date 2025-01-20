from typing import List

from langchain_core.embeddings import Embeddings  # noqa: F401

from langchain_elasticsearch._async.embeddings import (
    AsyncElasticsearchEmbeddings as _AsyncElasticsearchEmbeddings,
)
from langchain_elasticsearch._async.embeddings import (
    AsyncEmbeddingService as _AsyncEmbeddingService,
)
from langchain_elasticsearch._async.embeddings import (
    AsyncEmbeddingServiceAdapter as _AsyncEmbeddingServiceAdapter,
)
from langchain_elasticsearch._sync.embeddings import (
    ElasticsearchEmbeddings as _ElasticsearchEmbeddings,
)
from langchain_elasticsearch._sync.embeddings import (
    EmbeddingService as _EmbeddingService,
)
from langchain_elasticsearch._sync.embeddings import (
    EmbeddingServiceAdapter as _EmbeddingServiceAdapter,
)


# langchain defines some sync methods as abstract in its base class
# so we have to add dummy methods for them, even though we only use the async versions
class AsyncElasticsearchEmbeddings(_AsyncElasticsearchEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("This class is asynchronous, use aembed_documents()")

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError("This class is asynchronous, use aembed_query()")


# these are only defined here so that they are picked up by Langchain's docs generator
class ElasticsearchEmbeddings(_ElasticsearchEmbeddings):
    pass


class EmbeddingService(_EmbeddingService):
    pass


class EmbeddingServiceAdapter(_EmbeddingServiceAdapter):
    pass


class AsyncEmbeddingService(_AsyncEmbeddingService):
    pass


class AsyncEmbeddingServiceAdapter(_AsyncEmbeddingServiceAdapter):
    pass
