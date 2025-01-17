import asyncio
from abc import abstractmethod
from datetime import datetime, timezone
import logging
from typing import (
    Any, 
    Awaitable, 
    Callable,
    Dict, 
    Iterable, 
    List, 
    Optional, 
    Tuple,
    TypeVar, 
    Union
)
from elasticsearch import Elasticsearch, AsyncElasticsearch
from langchain_core.structured_query import (
    Comparison, 
    Operation, 
    Operator,
    StructuredQuery
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_elasticsearch.vectorstores import ElasticsearchStore
from langgraph.translator import ElasticsearchTranslator, ExtendedStructuredQuery
from langgraph.store.base import Result, Op
from langgraph.store.elasticsearch.config import ElasticsearchIndexConfig

OpInput = TypeVar("OpInput", bound=Op)
OpResult = TypeVar("OpResult", bound=Result)

DELIMITER_NAMESPACE = "/"

logger = logging.getLogger(__name__)

def text_to_namespace(namespace: Union[str, List[str]]) -> Tuple[str, ...]:
    """Convert a text namespace to a tuple of strings.
    
    Args:
        namespace (Union[str, List[str]]): The namespace as a string or list of strings.
    
    Returns:
        Tuple[str, ...]: The namespace as a tuple of strings.
    """
    if isinstance(namespace, list):
        return tuple(namespace)
    return tuple(namespace.split(DELIMITER_NAMESPACE))
    # if isinstance(namespace, str):
    #     return tuple(namespace.split(DELIMITER_NAMESPACE))
    # elif isinstance(namespace, (list, tuple)):
    #     # Flatten para garantir que não haja tuples aninhados
    #     flat_namespace = []
    #     for item in namespace:
    #         if isinstance(item, (list, tuple)):
    #             flat_namespace.extend(item)
    #         else:
    #             flat_namespace.append(item)
    #     return tuple(flat_namespace)
    raise TypeError("Namespace deve ser uma string, lista ou tuple.")

def namespace_to_text(namespace: Tuple[str, ...], handle_wildcards: bool = False) -> str:
    """Convert a tuple of strings to a text namespace.
    
    Args:
        namespace (Tuple[str, ...]): The namespace as a tuple of strings.
        handle_wildcards (bool, optional): Whether to handle wildcards. Defaults to False.
    
    Returns:
        str: The namespace as a string.
    """
    if handle_wildcards:
        namespace = tuple("%" if val == "*" else val for val in namespace)
    return DELIMITER_NAMESPACE.join(namespace)

class Query(Runnable[OpInput, OpResult]):
    """Base class for queries.
    
    Attributes:
        index_config (ElasticsearchIndexConfig): The index configuration.
    """
    index_config: ElasticsearchIndexConfig

    def __init__(self, index_config: ElasticsearchIndexConfig):
        """Initialize the Query with the given index configuration.
        
        Args:
            index_config (ElasticsearchIndexConfig): The index configuration.
        """
        self.index_config = index_config

    def _create_item(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: Any,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        **kwargs: Any,
    ) -> OpResult:
        """Create an item with the given parameters.
        
        Args:
            namespace (Tuple[str, ...]): The namespace of the item.
            key (str): The key of the item.
            value (Any): The value of the item.
            created_at (Optional[datetime], optional): The creation time of the item. Defaults to None.
            updated_at (Optional[datetime], optional): The update time of the item. Defaults to None.
            **kwargs (Any): Additional keyword arguments.
        
        Returns:
            OpResult: The created item.
        """
        result = self.OutputType(
            namespace=namespace,
            key=key,
            value=value,
            created_at=created_at or datetime.now(tz=timezone.utc),
            updated_at=updated_at or datetime.now(tz=timezone.utc),
            **kwargs,
        )
        return result

    def _execute_operations(
        self,
        ops: Iterable[OpInput],
        sync_func: Callable[[OpInput], List[OpResult]],
    ) -> List[OpResult]:
        """Execute operations synchronously.
        
        Args:
            ops (Iterable[OpInput]): The operations to execute.
            sync_func (Callable[[OpInput], List[OpResult]]): The synchronous function to execute.
        
        Returns:
            List[OpResult]: The list of results.
        """
        return [item for op in ops for item in (sync_func(op) or [])]
    
    async def _aexecute_operations(
        self,
        ops: Iterable[OpInput],
        async_func: Callable[[OpInput], Awaitable[List[OpResult]]],
    ) -> List[OpResult]:
        """Execute operations asynchronously.
        
        Args:
            ops (Iterable[OpInput]): The operations to execute.
            async_func (Callable[[OpInput], Awaitable[List[OpResult]]]): The asynchronous function to execute.
        
        Returns:
            List[OpResult]: The list of results.
        """
        results = await asyncio.gather(*(async_func(op) for op in ops))
        return [item for sublist in results for item in sublist]

    @abstractmethod
    def invoke(
        self, input: OpInput, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> OpResult:
        """Invoke the query synchronously.
        
        Args:
            input (OpInput): The input operation.
            config (Optional[RunnableConfig], optional): The runnable configuration. Defaults to None.
            **kwargs (Any): Additional keyword arguments.
        
        Returns:
            OpResult: The result of the query.
        """
        ...

    @abstractmethod
    async def ainvoke(
        self, input: OpInput, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> OpResult:
        """Invoke the query asynchronously.
        
        Args:
            input (OpInput): The input operation.
            config (Optional[RunnableConfig], optional): The runnable configuration. Defaults to None.
            **kwargs (Any): Additional keyword arguments.
        
        Returns:
            OpResult: The result of the query.
        """
        ...

class ElasticQuery(Query[OpInput, OpResult]):
    """Class for Elasticsearch queries.
    
    Attributes:
        es_connection (Optional[Union[Elasticsearch, AsyncElasticsearch]]): The Elasticsearch connection.
        index_config (ElasticsearchIndexConfig): The index configuration.
        store_index_name (str): The name of the store index.
        translator (ElasticsearchTranslator): The Elasticsearch translator.
    """
    es_connection: Optional[Union[Elasticsearch, AsyncElasticsearch]]
    index_config: ElasticsearchIndexConfig
    store_index_name: str
    translator: ElasticsearchTranslator

    def __init__(
        self, 
        index_config: ElasticsearchIndexConfig,
        es_connection: Optional[Union[Elasticsearch, AsyncElasticsearch]] = None,
    ):
        """Initialize the ElasticQuery with the given index configuration and Elasticsearch connection.
        
        Args:
            index_config (ElasticsearchIndexConfig): The index configuration.
            es_connection (Optional[Union[Elasticsearch, AsyncElasticsearch]], optional): The Elasticsearch connection. Defaults to None.
        """
        self.index_config = index_config
        self.es_connection = es_connection
        self.store_index_name = index_config.get("store_index_name")
        self.translator = ElasticsearchTranslator()

        self.es_connection.indices.create(
            index=self.store_index_name,
            body=self.es_schema,
            ignore=400,
        )

    def es_to_items(self, response: Dict[str, Any]) -> List[OpResult]:
        """Convert Elasticsearch response to a list of items.
        
        Args:
            response (Dict[str, Any]): The Elasticsearch response.
        
        Returns:
            List[OpResult]: The list of items.
        """
        if not response:
            return []

        print('RESPONSE', response)
        hits = response.get("hits", {}).get("hits", [])
        if not hits and (source := response.get("_source")):
            hits = [{"_source": source}]

        now = datetime.now(tz=timezone.utc).isoformat()
        return [
            self._create_item(
                namespace=text_to_namespace(source.get("namespace")),
                key=source.get("key"),
                value=source.get("value"),
                created_at=datetime.fromisoformat(source.get("created_at", now)),
                updated_at=datetime.fromisoformat(source.get("updated_at", now)),
                # score=1.0,
            ) for hit in hits if (source := hit.get("_source"))
        ]
    
    def es_to_tuple(self, response: Dict[str, Any]) -> Tuple[str, ...]:
        """Convert Elasticsearch response to a tuple.
        
        Args:
            response (Dict[str, Any]): The Elasticsearch response.
        
        Returns:
            Tuple[str, ...]: The tuple of namespaces.
        """
        if not response or not (hits := response.get("hits", {}).get("hits", [])):
            return ()

        return tuple(text_to_namespace(hit["_source"]["namespace"]) for hit in hits)

    def es_body_upsert(self, op: OpInput) -> Dict[str, Any]:
        """Create the body for an upsert operation.
        
        Args:
            op (OpInput): The input operation.
        
        Returns:
            Dict[str, Any]: The body for the upsert operation.
        """
        updated_at = datetime.now(tz=timezone.utc)
        return {
            "doc": {
                "value": op.value,
                'updated_at': updated_at
            },
            "upsert": self.es_body(op)
        }

    def es_body(self, op: OpInput) -> Dict[str, Any]:
        """Create the body for an Elasticsearch document.
        
        Args:
            op (OpInput): The input operation.
        
        Returns:
            Dict[str, Any]: The body for the Elasticsearch document.
        """
        created_at = datetime.now(tz=timezone.utc)
        return {
            "namespace": namespace_to_text(op.namespace),
            "key": op.key,
            "value": op.value,
            'created_at': created_at,
            'updated_at': created_at
        }

    
    @property
    def es_schema(self) -> Dict[str, Any]:
        """Return the Elasticsearch index mapping.
        
        Returns:
            Dict[str, Any]: The Elasticsearch index mapping.
        """
        return {
            "properties": {
                "namespace": {"type": "keyword"},
                "key": {"type": "keyword"},
                "value": {
                    "type": "object",
                    "enabled": True,
                },
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"},
            }
        }

    def build_query(self, 
                    filters: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], 
                    property: Optional[str] = None,
                    **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Build the filter for an Elasticsearch query.
        
        Args:
            filters (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]): The filters for the query.
            property (Optional[str], optional): The property to filter on. Defaults to None.
            **kwargs (Any): Additional keyword arguments.
        
        Returns:
            Optional[Dict[str, Any]]: The built query.
        """
        if isinstance(filters, dict):
            filters = [filters]

        comparisons = []
        for filter_item in (filters or []):
            for key, value in (filter_item or {}).items():
                if isinstance(value, str):
                    value = {"$eq": value}
                if isinstance(value, dict):
                    for comparator_str, comp_value in value.items():
                        comparator = self.translator.str_to_comparator(comparator_str)
                        comparison = Comparison(
                            comparator=comparator,
                            attribute=key if not property else f"{property}.{key}",
                            value=comp_value
                        )
                        comparisons.append(comparison)
        struc_query = ExtendedStructuredQuery(query="", filter=Operation(Operator.AND, comparisons), **kwargs)
        _, query = self.translator.visit_structured_query(struc_query)
        return query

class VectorQuery(Query[OpInput, OpResult]):
    """Class for vector store queries.
    
    Attributes:
        vector_store (Optional[ElasticsearchStore]): The vector store.
        index_config (ElasticsearchIndexConfig): The index configuration.
        vector_index_name (str): The name of the vector index.
    """
    vector_store: Optional[ElasticsearchStore]
    index_config: ElasticsearchIndexConfig
    vector_index_name: str

    def __init__(
        self, 
        index_config: ElasticsearchIndexConfig,
        vector_store: Optional[ElasticsearchStore] = None
    ):
        """Initialize the VectorQuery with the given index configuration and vector store.
        
        Args:
            index_config (ElasticsearchIndexConfig): The index configuration.
            vector_store (Optional[ElasticsearchStore], optional): The vector store. Defaults to None.
        """
        self.index_config = index_config
        self.vector_store = vector_store
        self.vector_index_name = index_config.get("vector_index_name")
