from langchain_elasticsearch import __all__

EXPECTED_ALL = [
    "ElasticsearchCache",
    "ElasticsearchCacheBackedEmbeddings",
    "ElasticsearchChatMessageHistory",
    "ElasticsearchEmbeddings",
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


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
