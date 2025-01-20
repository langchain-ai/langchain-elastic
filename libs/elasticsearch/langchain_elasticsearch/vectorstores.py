from typing import Any, Optional

from langchain_elasticsearch._async.vectorstores import (
    AsyncBM25Strategy as _AsyncBM25Strategy,
)
from langchain_elasticsearch._async.vectorstores import (
    AsyncDenseVectorScriptScoreStrategy as _AsyncDenseVectorScriptScoreStrategy,
)
from langchain_elasticsearch._async.vectorstores import (
    AsyncDenseVectorStrategy as _AsyncDenseVectorStrategy,
)
from langchain_elasticsearch._async.vectorstores import (
    AsyncElasticsearchStore as _AsyncElasticsearchStore,
)
from langchain_elasticsearch._async.vectorstores import (
    AsyncRetrievalStrategy as _AsyncRetrievalStrategy,
)
from langchain_elasticsearch._async.vectorstores import (
    AsyncSparseVectorStrategy as _AsyncSparseVectorStrategy,
)
from langchain_elasticsearch._async.vectorstores import (
    DistanceMetric,  # noqa: F401
    Document,
    Embeddings,
)
from langchain_elasticsearch._sync.vectorstores import (
    BM25Strategy as _BM25Strategy,
)
from langchain_elasticsearch._sync.vectorstores import (
    DenseVectorScriptScoreStrategy as _DenseVectorScriptScoreStrategy,
)
from langchain_elasticsearch._sync.vectorstores import (
    DenseVectorStrategy as _DenseVectorStrategy,
)
from langchain_elasticsearch._sync.vectorstores import (
    ElasticsearchStore as _ElasticsearchStore,
)
from langchain_elasticsearch._sync.vectorstores import (
    RetrievalStrategy as _RetrievalStrategy,
)
from langchain_elasticsearch._sync.vectorstores import (
    SparseVectorStrategy as _SparseVectorStrategy,
)

# deprecated strategy classes
from langchain_elasticsearch._utilities import (  # noqa: F401
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
        raise NotImplementedError("This class is asynchronous, use afrom_texts()")

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        raise NotImplementedError(
            "This class is asynchronous, use asimilarity_search()"
        )


# these are only defined here so that they are picked up by Langchain's docs generator
class ElasticsearchStore(_ElasticsearchStore):
    pass


class BM25Strategy(_BM25Strategy):
    pass


class DenseVectorScriptScoreStrategy(_DenseVectorScriptScoreStrategy):
    pass


class DenseVectorStrategy(_DenseVectorStrategy):
    pass


class RetrievalStrategy(_RetrievalStrategy):
    pass


class SparseVectorStrategy(_SparseVectorStrategy):
    pass


class AsyncBM25Strategy(_AsyncBM25Strategy):
    pass


class AsyncDenseVectorScriptScoreStrategy(_AsyncDenseVectorScriptScoreStrategy):
    pass


class AsyncDenseVectorStrategy(_AsyncDenseVectorStrategy):
    pass


class AsyncRetrievalStrategy(_AsyncRetrievalStrategy):
    pass


class AsyncSparseVectorStrategy(_AsyncSparseVectorStrategy):
    pass
