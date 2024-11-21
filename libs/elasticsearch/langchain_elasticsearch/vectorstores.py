from typing import Any, Optional

from langchain_elasticsearch._async.vectorstores import (
    AsyncBM25Strategy,
    AsyncDenseVectorScriptScoreStrategy,
    AsyncDenseVectorStrategy,
    AsyncRetrievalStrategy,
    AsyncSparseVectorStrategy,
    DistanceMetric,
    Document,
    Embeddings,
)
from langchain_elasticsearch._async.vectorstores import (
    AsyncElasticsearchStore as _AsyncElasticsearchStore,
)
from langchain_elasticsearch._sync.vectorstores import (
    BM25Strategy,
    DenseVectorScriptScoreStrategy,
    DenseVectorStrategy,
    ElasticsearchStore,
    RetrievalStrategy,
    SparseVectorStrategy,
)

# deprecated strategy classes
from langchain_elasticsearch._utilities import (
    ApproxRetrievalStrategy,
    BaseRetrievalStrategy,
    BM25RetrievalStrategy,
    DistanceStrategy,
    ExactRetrievalStrategy,
    SparseRetrievalStrategy,
)


# langchain defines some sync methods as abstract in its base class
# so we have to add dummy methods for them, even though we only use the async versions
class AsyncElasticsearchStore(_AsyncElasticsearchStore):
    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> "AsyncElasticsearchStore":
        raise NotImplemented("This class is asynchronous, use afrom_texts()")

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        raise NotImplemented("This class is asynchronous, use asimilarity_search()")
