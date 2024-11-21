from langchain_elasticsearch import __all__

EXPECTED_ALL = sorted([
    "ElasticsearchCache",
    "ElasticsearchChatMessageHistory",
    "ElasticsearchEmbeddings",
    "ElasticsearchEmbeddingsCache",
    "ElasticsearchRetriever",
    "ElasticsearchStore",
    "AsyncElasticsearchCache",
    "AsyncElasticsearchChatMessageHistory",
    "AsyncElasticsearchEmbeddings",
    "AsyncElasticsearchEmbeddingsCache",
    "AsyncElasticsearchRetriever",
    "AsyncElasticsearchStore",
    # retrieval strategies
    "BM25Strategy",
    "DenseVectorScriptScoreStrategy",
    "DenseVectorStrategy",
    "DistanceMetric",
    "RetrievalStrategy",
    "SparseVectorStrategy",
    "AsyncBM25Strategy",
    "AsyncDenseVectorScriptScoreStrategy",
    "AsyncDenseVectorStrategy",
    "AsyncRetrievalStrategy",
    "AsyncSparseVectorStrategy",
    # deprecated retrieval strategies
    "ApproxRetrievalStrategy",
    "BM25RetrievalStrategy",
    "ExactRetrievalStrategy",
    "SparseRetrievalStrategy",
])


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
