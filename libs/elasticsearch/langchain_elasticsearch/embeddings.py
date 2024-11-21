from typing import List

from langchain_elasticsearch._async.embeddings import (
    AsyncElasticsearchEmbeddings as _AsyncElasticsearchEmbeddings,
    AsyncEmbeddingServiceAdapter,
    AsyncEmbeddingService,
    Embeddings,
)
from langchain_elasticsearch._sync.embeddings import (
    ElasticsearchEmbeddings,
    EmbeddingServiceAdapter,
    EmbeddingService,
)


# langchain defines some sync methods as abstract in its base class
# so we have to add dummy methods for them, even though we only use the async versions
class AsyncElasticsearchEmbeddings(_AsyncElasticsearchEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplemented("This class is asynchronous, use aembed_documents()")

    def embed_query(self, text: str) -> List[float]:
        raise NotImplemented("This class is asynchronous, use aembed_query()")
