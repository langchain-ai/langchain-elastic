import hashlib
import logging
from datetime import datetime
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

from elasticsearch import NotFoundError, helpers
from elasticsearch.helpers import BulkIndexError
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.load import dumps, loads
from langchain_core.stores import BaseStore

from langchain_elasticsearch._utilities import ElasticsearchCacheIndexer

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


class ElasticsearchCache(BaseCache, ElasticsearchCacheIndexer):
    """An Elasticsearch cache integration for LLMs."""

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

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        cache_key = self._key(prompt, llm_string)
        if self._is_alias:
            # get the latest record according to its writing date, in order to
            # address cases where multiple indices have a doc with the same id
            result = self._es_client.search(
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
                record = self._es_client.get(
                    index=self._index_name, id=cache_key, source=["llm_output"]
                )
            except NotFoundError:
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

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update based on prompt and llm_string."""
        body = self.build_document(prompt, llm_string, return_val)
        self._es_client.index(
            index=self._index_name,
            id=self._key(prompt, llm_string),
            body=body,
            require_alias=self._is_alias,
            refresh=True,
        )

    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        self._es_client.delete_by_query(
            index=self._index_name,
            body={"query": {"match_all": {}}},
            refresh=True,
            wait_for_completion=True,
        )


class ElasticsearchStoreEmbeddings(
    BaseStore[str, List[float]], ElasticsearchCacheIndexer
):
    @cached_property
    def mapping(self) -> Dict[str, Any]:
        """Get the default mapping for the index."""
        return {
            "mappings": {
                "properties": {
                    "llm_input": {"type": "text", "index": False},
                    "vector_dump": {
                        "type": "float",
                        "index": False,
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

    def mget(self, keys: Sequence[str]) -> List[Optional[List[float]]]:
        """Get the values associated with the given keys."""
        if not any(keys):
            return []
        cache_keys = [self._key(k) for k in keys]
        if self._is_alias:
            results = self._es_client.search(
                index=self._index_name,
                body={
                    "query": {"ids": {"values": cache_keys}},
                    "size": len(cache_keys),
                },
                source_includes=["vector_dump"],
            )
            map_ids = {
                r["_id"]: r["_source"]["vector_dump"] for r in results["hits"]["hits"]
            }
            return [map_ids.get(k) for k in cache_keys]
        else:
            records = self._es_client.mget(
                index=self._index_name, ids=cache_keys, source_includes=["vector_dump"]
            )
            return [
                r["_source"]["vector_dump"] if r["found"] else None
                for r in records["docs"]
            ]

    def build_document(self, llm_input: str, vector: List[float]) -> Dict[str, Any]:
        """Build the Elasticsearch document for storing a single embedding"""
        body: Dict[str, Any] = {"vector_dump": vector}
        if self._metadata is not None:
            body["metadata"] = self._metadata
        if self._store_input:
            body["llm_input"] = llm_input
        if self._store_input:
            body["timestamp"] = datetime.now().isoformat()
        if self._namespace:
            body["namespace"] = self._namespace
        return body

    def _bulk(self, actions: Iterable[Dict[str, Any]]):
        try:
            helpers.bulk(
                client=self._es_client,
                actions=actions,
                index=self._index_name,
                require_alias=self._is_alias,
                refresh=True,
            )
        except BulkIndexError as e:
            first_error = e.errors[0].get("index", {}).get("error", {})
            logger.error(f"First bulk error reason: {first_error.get('reason')}")
            raise e

    def mset(self, key_value_pairs: Sequence[Tuple[str, List[float]]]) -> None:
        """Set the values for the given keys."""
        actions = (
            {
                "_op_type": "index",
                "_id": self._key(key),
                "_source": self.build_document(key, vector),
            }
            for key, vector in key_value_pairs
        )
        self._bulk(actions)
        return

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their associated values."""
        actions = ({"_op_type": "delete", "_id": self._key(key)} for key in keys)
        self._bulk(actions)
        return

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        """Get an iterator over keys that match the given prefix."""
        # TODO This method is not currently used by CacheBackedEmbeddings, we can leave it blank.
        #      It could be implemented with ES "index_prefixes", but they are limited and expensive.
        raise NotImplementedError()
