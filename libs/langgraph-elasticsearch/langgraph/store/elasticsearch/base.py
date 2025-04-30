import asyncio
import logging
from typing import (
    Any, 
    Dict, 
    Generic, 
    Iterable,
    Tuple, 
    TypeVar, 
    Union
)
from elasticsearch import Elasticsearch, AsyncElasticsearch
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables.base import RunnableLike
from langchain_core.runnables.config import run_in_executor
from langchain_elasticsearch.client import create_elasticsearch_client
from langchain_elasticsearch.vectorstores import ElasticsearchStore as ElasticsearchVectorStore
from langgraph.store.base import (
    BaseStore, 
    Result, 
    Op, 
    PutOp, 
    GetOp, 
    SearchOp, 
    ListNamespacesOp, 
    ensure_embeddings, 
    tokenize_path, 
)
from langgraph.store.elasticsearch.config import ElasticsearchIndexConfig
from langgraph.store.elasticsearch.queries import (
    QueryGet, 
    QueryListNamespaces,
    ElasticQueryPut,
    VectorQueryPut,
    ElasticQuerySearch,
    VectorQuerySearch,
)

T = TypeVar("T", bound=Union[Elasticsearch, AsyncElasticsearch])

logger = logging.getLogger(__name__)

class BaseElasticsearchMemoryStore(Generic[T]):
    """
    Base class for Elasticsearch memory store operations.

    Attributes:
        operation_handlers (Dict[Op, RunnableLike]): Handlers for different operations.
    """

    def __init__(self,
                 es_connection: T | None = None,
                 index_config: ElasticsearchIndexConfig | None = None,):
        """
        Initialize the Elasticsearch memory store.

        Args:
            es_connection (T | None): Elasticsearch connection instance.
            index_config (ElasticsearchIndexConfig | None): Configuration for Elasticsearch index.
        """
        if not es_connection:
            raise ValueError("No Elasticsearch connection provided")

        index_config, es_connection, vector_store = self._ensure_index_config(es_connection, index_config)
        self.operation_handlers = self._initialize_operation_handlers(index_config, es_connection, vector_store)
        
    def _initialize_operation_handlers(self, 
                                       index_config: ElasticsearchIndexConfig, 
                                       es_connection: 
                                       T, vector_store: ElasticsearchVectorStore | None) -> Dict[Op, RunnableLike]:
        """
        Initialize handlers for different operations.

        Args:
            index_config (ElasticsearchIndexConfig): Configuration for Elasticsearch index.
            es_connection (T): Elasticsearch connection instance.
            vector_store (ElasticsearchVectorStore | None): Vector store instance.

        Returns:
            Dict[Op, RunnableLike]: Handlers for different operations.
        """
        return {
            PutOp: (
                RunnableParallel({
                    "elastic": ElasticQueryPut(index_config, es_connection),
                    "vector": VectorQueryPut(index_config, vector_store)
                }) if vector_store 
                   else ElasticQueryPut(index_config, es_connection)
            ),
            GetOp: QueryGet(index_config, es_connection),
            SearchOp: (
                RunnableParallel({
                    "elastic": ElasticQuerySearch(index_config, es_connection),
                    "vector": VectorQuerySearch(index_config, vector_store)
                }) if vector_store
                   else ElasticQuerySearch(index_config, es_connection)
            ),
            ListNamespacesOp: QueryListNamespaces(index_config, es_connection),
        }

    def _ensure_index_config(self,
                             es_connection: T,
                             index_config: ElasticsearchIndexConfig | None) -> Tuple[ElasticsearchIndexConfig | None,
                                                                                     T,
                                                                                     ElasticsearchVectorStore | None]:
        """
        Ensure the index configuration is set up correctly.

        Args:
            es_connection (T): Elasticsearch connection instance.
            index_config (ElasticsearchIndexConfig | None): Configuration for Elasticsearch index.

        Returns:
            Tuple[ElasticsearchIndexConfig | None, T, ElasticsearchVectorStore | None]: Configured index, connection, and vector store.
        """
        config: ElasticsearchIndexConfig | None = index_config.copy() if index_config else ElasticsearchIndexConfig()
        vector_store: ElasticsearchVectorStore | None = None

        embedd = ensure_embeddings(config.get("embed")) if config.get("embed") else None

        if embedd:
            vector_store = ElasticsearchVectorStore(
                es_connection=es_connection,
                index_name=config.get("vector_index_name"),
                embedding=embedd,
                strategy=config.get("strategy"),
                distance_strategy=config.get("distance_strategy"),
            )
        config["__tokenized_fields"] = [
            (p, tokenize_path(p)) if p != "$" else (p, p)
            for p in (config.get("fields") or ["$"])
            ]

        return config, es_connection, vector_store

    def process(self, ops: Iterable[Op]) -> list[Result]:
        """
        Process a list of operations and return the results.

        Args:
            ops (Iterable[Op]): A list of operations to be processed.

        Returns:
            list[Result]: A list of results from the processed operations.
        """
        results = []
        for op_type, op_instance in self.operation_handlers.items():
            filtered_ops = [op for op in ops if isinstance(op, op_type)]
            if filtered_ops:
                processed = op_instance.invoke(filtered_ops)
                if isinstance(processed, dict):
                    flattened_output = list(processed.values())
                    results.extend(flattened_output)
                elif processed:
                    results.extend(processed)
        return results

    async def aprocess(self, ops: Iterable[Op]) -> list[Result]:
        """
        Asynchronously process a list of operations and return the results.

        Args:
            ops (Iterable[Op]): A list of operations to be processed.

        Returns:
            list[Result]: A list of results from the processed operations.
        """
        tasks = [
                op_instance.ainvoke([op for op in ops if isinstance(op, op_type)])
                for op_type, op_instance in self.operation_handlers.items()
                if any(isinstance(op, op_type) for op in ops)
        ]
        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]

class ElasticsearchMemoryStore(BaseStore, BaseElasticsearchMemoryStore[Elasticsearch]):
    """
    Elasticsearch memory store for synchronous operations.
    """

    def __init__(self, 
                 es_connection: Elasticsearch | None = None,
                 es_url: str | None = None,
                 es_cloud_id: str | None = None,
                 es_user: str | None = None,
                 es_api_key: str | None = None,
                 es_password: str | None = None,
                 es_params: Dict[str, Any] | None = None,
                 index_config: ElasticsearchIndexConfig | None = None,
                ):
        """
        Initialize the Elasticsearch memory store.

        Args:
            es_connection (Elasticsearch | None): Elasticsearch connection instance.
            es_url (str | None): URL for Elasticsearch.
            es_cloud_id (str | None): Cloud ID for Elasticsearch.
            es_user (str | None): Username for Elasticsearch.
            es_api_key (str | None): API key for Elasticsearch.
            es_password (str | None): Password for Elasticsearch.
            es_params (Dict[str, Any] | None): Additional parameters for Elasticsearch.
            index_config (ElasticsearchIndexConfig | None): Configuration for Elasticsearch index.
        """
        if not es_connection:
            es_connection = create_elasticsearch_client(
                url=es_url,
                cloud_id=es_cloud_id,
                api_key=es_api_key,
                username=es_user,
                password=es_password,
                params=es_params,
            )
        super().__init__(es_connection=es_connection,
            index_config=index_config)

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """
        Process a batch of operations synchronously.

        Args:
            ops (Iterable[Op]): A list of operations to be processed.

        Returns:
            list[Result]: A list of results from the processed operations.
        """
        return self.process(ops)

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """
        Process a batch of operations asynchronously.

        Args:
            ops (Iterable[Op]): A list of operations to be processed.

        Returns:
            list[Result]: A list of results from the processed operations.
        """
        logging.warning("For asynchronous operations, please use the appropriate class: AsyncElasticsearchStore.")
        return await run_in_executor(self.aprocess, ops)
