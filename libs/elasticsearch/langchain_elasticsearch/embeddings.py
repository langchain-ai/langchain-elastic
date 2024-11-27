from typing import List

from langchain_elasticsearch._async.embeddings import (
    AsyncElasticsearchEmbeddings as _AsyncElasticsearchEmbeddings,
)
from langchain_elasticsearch._async.embeddings import (  # noqa: F401
    AsyncEmbeddingService,
    AsyncEmbeddingServiceAdapter,
    Embeddings,
)
from langchain_elasticsearch._sync.embeddings import (  # noqa: F401
    ElasticsearchEmbeddings,
    EmbeddingService,
    EmbeddingServiceAdapter,
)


# langchain defines some sync methods as abstract in its base class
# so we have to add dummy methods for them, even though we only use the async versions
class AsyncElasticsearchEmbeddings(_AsyncElasticsearchEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("This class is asynchronous, use aembed_documents()")

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError("This class is asynchronous, use aembed_query()")
