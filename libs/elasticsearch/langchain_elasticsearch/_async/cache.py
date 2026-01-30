import base64
import hashlib
import logging
from datetime import datetime
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from elasticsearch import AsyncElasticsearch, exceptions, helpers
from elasticsearch.helpers import BulkIndexError
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.load import dumps, loads
from langchain_core.stores import ByteStore

from langchain_elasticsearch.client import create_async_elasticsearch_client

if TYPE_CHECKING:
    from elasticsearch import AsyncElasticsearch

logger = logging.getLogger(__name__)


async def _manage_cache_index(
    es_client: AsyncElasticsearch, index_name: str, mapping: Dict[str, Any]
) -> bool:
    """Write or update an index or alias according to the default mapping"""
    if await es_client.indices.exists_alias(name=index_name):
        await es_client.indices.put_mapping(index=index_name, body=mapping["mappings"])
        return True

    elif not await es_client.indices.exists(index=index_name):
        logger.debug(f"Creating new Elasticsearch index: {index_name}")
        await es_client.indices.create(index=index_name, body=mapping)
        return False

    return False


class AsyncElasticsearchCache(BaseCache):
    """`Elasticsearch` LLM cache.

    Caches LLM responses in Elasticsearch to avoid repeated calls for identical
    prompts.

    Setup:
        Install `langchain_elasticsearch` and start Elasticsearch locally using
        the start-local script.

        ```bash
        pip install -qU langchain_elasticsearch
        curl -fsSL https://elastic.co/start-local | sh
        ```

        This will create an `elastic-start-local` folder. To start Elasticsearch
        and Kibana:
        ```bash
        cd elastic-start-local
        ./start.sh
        ```

        Elasticsearch will be available at `http://localhost:9200`. The password
        for the `elastic` user and API key are stored in the `.env` file in the
        `elastic-start-local` folder.

    Instantiate:
        ```python
        from langchain_elasticsearch import ElasticsearchCache

        cache = ElasticsearchCache(
            index_name="llm-cache",
            es_url="http://localhost:9200"
        )
        ```

    Instantiate with API key:
        ```python
        from langchain_elasticsearch import ElasticsearchCache

        cache = ElasticsearchCache(
            index_name="llm-cache",
            es_url="http://localhost:9200",
            es_api_key="your-api-key"
        )
        ```

    Instantiate from cloud:
        ```python
        from langchain_elasticsearch import ElasticsearchCache

        cache = ElasticsearchCache(
            index_name="llm-cache",
            es_cloud_id="<cloud_id>",
            es_api_key="your-api-key"
        )
        ```

    Instantiate from existing connection:
        ```python
        from langchain_elasticsearch import ElasticsearchCache
        from elasticsearch import Elasticsearch

        client = Elasticsearch("http://localhost:9200")
        cache = ElasticsearchCache(
            index_name="llm-cache",
            client=client
        )
        ```

    Use with LangChain:
        ```python
        from langchain.globals import set_llm_cache
        from langchain_elasticsearch import ElasticsearchCache

        set_llm_cache(ElasticsearchCache(
            index_name="llm-cache",
            es_url="http://localhost:9200"
        ))
        ```

    For synchronous applications, use the `ElasticsearchCache` class.
    For asynchronous applications, use the `AsyncElasticsearchCache` class.
    """  # noqa: E501

    def __init__(
        self,
        index_name: str,
        *,
        store_input: bool = True,
        store_input_params: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        client: Optional[AsyncElasticsearch] = None,
        es_url: Optional[str] = None,
        es_cloud_id: Optional[str] = None,
        es_user: Optional[str] = None,
        es_api_key: Optional[str] = None,
        es_password: Optional[str] = None,
    ):
        """Initialize the Elasticsearch cache store.

        Args:
            index_name (str): The name of the index or alias to use for the
                cache. If it doesn't exist, an index is created according to
                the default mapping.
            store_input (bool): Whether to store the LLM input (prompt) in the
                cache. Default True.
            store_input_params (bool): Whether to store the LLM parameters in
                the cache. Default True.
            metadata (dict, optional): Additional metadata to store in the
                cache for filtering. Must be JSON serializable.
            client (AsyncElasticsearch, optional): Pre-existing Elasticsearch
                connection. Either provide this OR credentials.
            es_url (str, optional): URL of the Elasticsearch instance.
            es_cloud_id (str, optional): Cloud ID of the Elasticsearch instance.
            es_user (str, optional): Username for Elasticsearch.
            es_api_key (str, optional): API key for Elasticsearch.
            es_password (str, optional): Password for Elasticsearch.
        """
        self._index_name = index_name
        self._store_input = store_input
        self._store_input_params = store_input_params
        self._metadata = metadata

        # Accept either client OR credentials (one required)
        if client is not None:
            self._es_client = client
        elif es_url is not None or es_cloud_id is not None:
            self._es_client = create_async_elasticsearch_client(
                url=es_url,
                cloud_id=es_cloud_id,
                api_key=es_api_key,
                username=es_user,
                password=es_password,
                user_agent="langchain-py-c",
            )
        else:
            raise ValueError(
                "Either 'client' or credentials (es_url, es_cloud_id, etc.) "
                "must be provided."
            )

        self._is_alias: Optional[bool] = None

    async def is_alias(self) -> bool:
        if self._is_alias is None:
            self._is_alias = await _manage_cache_index(
                self._es_client,
                self._index_name,
                self.mapping,
            )
        return self._is_alias  # type: ignore[return-value]

    @cached_property
    def mapping(self) -> Dict[str, Any]:
        """Get the default mapping for the index."""
        return {
            "mappings": {
                "properties": {
                    "llm_output": {"type": "text", "index": False},
                    "llm_params": {"type": "text", "index": False},
                    "llm_input": {"type": "text", "index": False},
                    "metadata": {"type": "object"},
                    "timestamp": {"type": "date"},
                }
            }
        }

    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        """Generate a key for the cache store."""
        return hashlib.md5((prompt + llm_string).encode()).hexdigest()

    async def alookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        cache_key = self._key(prompt, llm_string)
        if await self.is_alias():
            # get the latest record according to its writing date, in order to
            # address cases where multiple indices have a doc with the same id
            result = await self._es_client.search(
                index=self._index_name,
                body={
                    "query": {"term": {"_id": cache_key}},
                    "sort": {"timestamp": {"order": "asc"}},
                },
                source_includes=["llm_output"],
            )
            if result["hits"]["total"]["value"] > 0:
                record = result["hits"]["hits"][0]
            else:
                return None
        else:
            try:
                record = await self._es_client.get(
                    index=self._index_name, id=cache_key, source=["llm_output"]
                )
            except exceptions.NotFoundError:
                return None
        return [loads(item) for item in record["_source"]["llm_output"]]

    def build_document(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> Dict[str, Any]:
        """Build the Elasticsearch document for storing a single LLM interaction"""
        body: Dict[str, Any] = {
            "llm_output": [dumps(item) for item in return_val],
            "timestamp": datetime.now().isoformat(),
        }
        if self._store_input_params:
            body["llm_params"] = llm_string
        if self._metadata is not None:
            body["metadata"] = self._metadata
        if self._store_input:
            body["llm_input"] = prompt
        return body

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        """Update based on prompt and llm_string."""
        body = self.build_document(prompt, llm_string, return_val)
        await self._es_client.index(
            index=self._index_name,
            id=self._key(prompt, llm_string),
            body=body,
            require_alias=await self.is_alias(),
            refresh=True,
        )

    async def aclear(self, **kwargs: Any) -> None:
        """Clear cache."""
        await self._es_client.delete_by_query(
            index=self._index_name,
            body={"query": {"match_all": {}}},
            refresh=True,
            wait_for_completion=True,
        )


class AsyncElasticsearchEmbeddingsCache(ByteStore):
    """`Elasticsearch` embeddings cache.

    Caches embeddings in Elasticsearch to avoid repeated embedding computations.

    Setup:
        Install `langchain_elasticsearch` and start Elasticsearch locally using
        the start-local script.

        ```bash
        pip install -qU langchain_elasticsearch
        curl -fsSL https://elastic.co/start-local | sh
        ```

        This will create an `elastic-start-local` folder. To start Elasticsearch
        and Kibana:
        ```bash
        cd elastic-start-local
        ./start.sh
        ```

        Elasticsearch will be available at `http://localhost:9200`. The password
        for the `elastic` user and API key are stored in the `.env` file in the
        `elastic-start-local` folder.

    Instantiate:
        ```python
        from langchain_elasticsearch import ElasticsearchEmbeddingsCache

        cache = ElasticsearchEmbeddingsCache(
            index_name="embeddings-cache",
            es_url="http://localhost:9200"
        )
        ```

        **Instantiate with API key:**
            ```python
            from langchain_elasticsearch import ElasticsearchEmbeddingsCache

            cache = ElasticsearchEmbeddingsCache(
                index_name="embeddings-cache",
                es_url="http://localhost:9200",
                es_api_key="your-api-key"
            )
            ```

        **Instantiate from cloud:**
            ```python
            from langchain_elasticsearch import ElasticsearchEmbeddingsCache

            cache = ElasticsearchEmbeddingsCache(
                index_name="embeddings-cache",
                es_cloud_id="<cloud_id>",
                es_api_key="your-api-key"
            )
            ```

        **Instantiate from existing connection:**
            ```python
            from langchain_elasticsearch import ElasticsearchEmbeddingsCache
            from elasticsearch import Elasticsearch

            client = Elasticsearch("http://localhost:9200")
            cache = ElasticsearchEmbeddingsCache(
                index_name="embeddings-cache",
                client=client
            )
            ```

    Use with CacheBackedEmbeddings:
        ```python
        from langchain.embeddings import CacheBackedEmbeddings
        from langchain_openai import OpenAIEmbeddings
        from langchain_elasticsearch import ElasticsearchEmbeddingsCache

        underlying_embeddings = OpenAIEmbeddings()
        cache = ElasticsearchEmbeddingsCache(
            index_name="embeddings-cache",
            es_url="http://localhost:9200"
        )
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings,
            cache,
            namespace=underlying_embeddings.model
        )
        ```

    For synchronous applications, use the `ElasticsearchEmbeddingsCache` class.
    For asynchronous applications, use the `AsyncElasticsearchEmbeddingsCache`
    class.
    """  # noqa: E501

    def __init__(
        self,
        index_name: str,
        *,
        store_input: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        maximum_duplicates_allowed: int = 1,
        client: Optional[AsyncElasticsearch] = None,
        es_url: Optional[str] = None,
        es_cloud_id: Optional[str] = None,
        es_user: Optional[str] = None,
        es_api_key: Optional[str] = None,
        es_password: Optional[str] = None,
    ):
        """Initialize the Elasticsearch embeddings cache store.

        Args:
            index_name (str): The name of the index or alias to use for the
                cache. If it doesn't exist, an index is created according to
                the default mapping.
            store_input (bool): Whether to store the input text in the cache.
                Default is True.
            metadata (dict, optional): Additional metadata to store in the
                cache for filtering. Must be JSON serializable.
            namespace (str, optional): A namespace to organize the cache.
            maximum_duplicates_allowed (int): Maximum duplicate keys permitted
                when using aliases across multiple indices. Default is 1.
            client (AsyncElasticsearch, optional): Pre-existing Elasticsearch
                connection. Either provide this OR credentials.
            es_url (str, optional): URL of the Elasticsearch instance.
            es_cloud_id (str, optional): Cloud ID of the Elasticsearch instance.
            es_user (str, optional): Username for Elasticsearch.
            es_api_key (str, optional): API key for Elasticsearch.
            es_password (str, optional): Password for Elasticsearch.
        """
        self._namespace = namespace
        self._maximum_duplicates_allowed = maximum_duplicates_allowed
        self._index_name = index_name
        self._store_input = store_input
        self._metadata = metadata

        # Accept either client OR credentials (one required)
        if client is not None:
            self._es_client = client
        elif es_url is not None or es_cloud_id is not None:
            self._es_client = create_async_elasticsearch_client(
                url=es_url,
                cloud_id=es_cloud_id,
                api_key=es_api_key,
                username=es_user,
                password=es_password,
                user_agent="langchain-py-ec",
            )
        else:
            raise ValueError(
                "Either 'client' or credentials (es_url, es_cloud_id, etc.) "
                "must be provided."
            )

        self._is_alias: Optional[bool] = None

    async def is_alias(self) -> bool:
        if self._is_alias is None:
            self._is_alias = await _manage_cache_index(
                self._es_client,
                self._index_name,
                self.mapping,
            )
        return self._is_alias  # type: ignore[return-value]

    @staticmethod
    def encode_vector(data: bytes) -> str:
        """Encode the vector data as bytes to as a base64 string."""
        return base64.b64encode(data).decode("utf-8")

    @staticmethod
    def decode_vector(data: str) -> bytes:
        """Decode the base64 string to vector data as bytes."""
        return base64.b64decode(data)

    @cached_property
    def mapping(self) -> Dict[str, Any]:
        """Get the default mapping for the index."""
        return {
            "mappings": {
                "properties": {
                    "text_input": {"type": "text", "index": False},
                    "vector_dump": {
                        "type": "binary",
                        "doc_values": False,
                    },
                    "metadata": {"type": "object"},
                    "timestamp": {"type": "date"},
                    "namespace": {"type": "keyword"},
                }
            }
        }

    def _key(self, input_text: str) -> str:
        """Generate a key for the store."""
        return hashlib.md5(((self._namespace or "") + input_text).encode()).hexdigest()

    @classmethod
    def _deduplicate_hits(cls, hits: List[dict]) -> Dict[str, bytes]:
        """
        Collapse the results from a search query with multiple indices
        returning only the latest version of the documents
        """
        map_ids = {}
        for hit in sorted(
            hits,
            key=lambda x: datetime.fromisoformat(x["_source"]["timestamp"]),
            reverse=True,
        ):
            vector_id: str = hit["_id"]
            if vector_id not in map_ids:
                map_ids[vector_id] = cls.decode_vector(hit["_source"]["vector_dump"])

        return map_ids

    async def amget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        """Get the values associated with the given keys."""
        if not any(keys):
            return []

        cache_keys = [self._key(k) for k in keys]
        if await self.is_alias():
            try:
                results = await self._es_client.search(
                    index=self._index_name,
                    body={
                        "query": {"ids": {"values": cache_keys}},
                        "size": len(cache_keys) * self._maximum_duplicates_allowed,
                    },
                    source_includes=["vector_dump", "timestamp"],
                )

            except exceptions.BadRequestError as e:
                if "window too large" in (
                    e.body.get("error", {}).get("root_cause", [{}])[0].get("reason", "")
                ):
                    logger.warning(
                        "Exceeded the maximum window size, "
                        "Reduce the duplicates manually or lower "
                        "`maximum_duplicate_allowed.`"
                    )
                    raise e

            total_hits = results["hits"]["total"]["value"]
            if self._maximum_duplicates_allowed > 1 and total_hits > len(cache_keys):
                logger.warning(
                    f"Deduplicating, found {total_hits} hits for {len(cache_keys)} keys"
                )
                map_ids = self._deduplicate_hits(results["hits"]["hits"])
            else:
                map_ids = {
                    r["_id"]: self.decode_vector(r["_source"]["vector_dump"])
                    for r in results["hits"]["hits"]
                }

            return [map_ids.get(k) for k in cache_keys]

        else:
            records = await self._es_client.mget(
                index=self._index_name, ids=cache_keys, source_includes=["vector_dump"]
            )
            return [
                self.decode_vector(r["_source"]["vector_dump"]) if r["found"] else None
                for r in records["docs"]
            ]

    def build_document(self, text_input: str, vector: bytes) -> Dict[str, Any]:
        """Build the Elasticsearch document for storing a single embedding"""
        body: Dict[str, Any] = {
            "vector_dump": self.encode_vector(vector),
            "timestamp": datetime.now().isoformat(),
        }
        if self._metadata is not None:
            body["metadata"] = self._metadata
        if self._store_input:
            body["text_input"] = text_input
        if self._namespace:
            body["namespace"] = self._namespace
        return body

    async def _bulk(self, actions: Iterable[Dict[str, Any]]) -> None:
        try:
            await helpers.async_bulk(
                client=self._es_client,
                actions=actions,
                index=self._index_name,
                require_alias=await self.is_alias(),
                refresh=True,
            )
        except BulkIndexError as e:
            first_error = e.errors[0].get("index", {}).get("error", {})
            logger.error(f"First bulk error reason: {first_error.get('reason')}")
            raise e

    async def amset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        """Set the values for the given keys."""
        actions = (
            {
                "_op_type": "index",
                "_id": self._key(key),
                "_source": self.build_document(key, vector),
            }
            for key, vector in key_value_pairs
        )
        await self._bulk(actions)

    async def amdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their associated values."""
        actions = ({"_op_type": "delete", "_id": self._key(key)} for key in keys)
        await self._bulk(actions)

    async def ayield_keys(self, *, prefix: Optional[str] = None) -> AsyncIterator[str]:  # type: ignore[override]
        """Get an iterator over keys that match the given prefix."""
        # TODO This method is not currently used by CacheBackedEmbeddings,
        #  we can leave it blank. It could be implemented with ES "index_prefixes",
        #  but they are limited and expensive.
        raise NotImplementedError()
