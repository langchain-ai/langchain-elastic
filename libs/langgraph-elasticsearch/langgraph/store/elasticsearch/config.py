from typing import Optional
from langgraph.store.base import IndexConfig
from langchain_elasticsearch._utilities import DistanceStrategy
from elasticsearch.helpers.vectorstore import (
    BM25Strategy,
    RetrievalStrategy,
)

DEFAULT_INDEX_NAME = "laggraph"

class ElasticsearchIndexConfig(IndexConfig):
    store_index_name: str = f"{DEFAULT_INDEX_NAME}-store"
    vector_index_name: str = f"{DEFAULT_INDEX_NAME}-vectorstore"
    strategy: RetrievalStrategy | None = BM25Strategy()
    distance_strategy: Optional[DistanceStrategy] = DistanceStrategy.COSINE,