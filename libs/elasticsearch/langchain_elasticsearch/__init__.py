from elasticsearch.helpers.vectorstore import (
    BM25Strategy,
    DenseVectorScriptScoreStrategy,
    DenseVectorStrategy,
    DistanceMetric,
    RetrievalStrategy,
    SparseVectorStrategy,
)

from langchain_elasticsearch.cache import (
    ElasticsearchCache,
    ElasticsearchEmbeddingsCache,
)
from langchain_elasticsearch.chat_history import ElasticsearchChatMessageHistory
from langchain_elasticsearch.embeddings import ElasticsearchEmbeddings
from langchain_elasticsearch.retrievers import ElasticsearchRetriever
from langchain_elasticsearch.vectorstores import (
    ApproxRetrievalStrategy,
    BM25RetrievalStrategy,
    ElasticsearchStore,
    ExactRetrievalStrategy,
    SparseRetrievalStrategy,
)

__all__ = [
    "ElasticsearchCache",
    "ElasticsearchChatMessageHistory",
    "ElasticsearchEmbeddings",
    "ElasticsearchEmbeddingsCache",
    "ElasticsearchRetriever",
    "ElasticsearchStore",
    # retrieval strategies
    "BM25Strategy",
    "DenseVectorScriptScoreStrategy",
    "DenseVectorStrategy",
    "DistanceMetric",
    "RetrievalStrategy",
    "SparseVectorStrategy",
    # deprecated retrieval strategies
    "ApproxRetrievalStrategy",
    "BM25RetrievalStrategy",
    "ExactRetrievalStrategy",
    "SparseRetrievalStrategy",
]
