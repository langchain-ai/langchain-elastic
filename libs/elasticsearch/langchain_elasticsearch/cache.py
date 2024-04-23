import hashlib
import logging
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, Optional

import elasticsearch
from elasticsearch import NotFoundError
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.load import dumps, loads

from langchain_elasticsearch.client import create_elasticsearch_client

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


class ElasticsearchCache(BaseCache):
    """An Elasticsearch cache integration for LLMs."""

    def __init__(
        self,
        index_name: str,
        store_input: bool = True,
        store_input_params: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        es_connection: Optional["Elasticsearch"] = None,
        es_url: Optional[str] = None,
        es_cloud_id: Optional[str] = None,
        es_user: Optional[str] = None,
        es_api_key: Optional[str] = None,
        es_password: Optional[str] = None,
        es_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Elasticsearch cache store by specifying the index/alias
        to use and determining which additional information (like input, timestamp,
        input parameters, and any other metadata) should be stored in the cache.

        Args:
            index_name (str): The name of the index or the alias to use for the cache.
                If they do not exist an index is created,
                according to the default mapping defined by the `mapping` property.
            store_input (bool): Whether to store the LLM input in the cache, i.e.,
                the input prompt. Default to True.
            store_input_params (bool): Whether to store the input parameters in the
                cache, i.e., the LLM parameters used to generate the LLM response.
                Default to True.
            metadata (Optional[dict]): Additional metadata to store in the cache,
                for filtering purposes. This must be JSON serializable in an
                Elasticsearch document. Default to None.
            es_connection: Optional pre-existing Elasticsearch connection.
            es_url: URL of the Elasticsearch instance to connect to.
            es_cloud_id: Cloud ID of the Elasticsearch instance to connect to.
            es_user: Username to use when connecting to Elasticsearch.
            es_password: Password to use when connecting to Elasticsearch.
            es_api_key: API key to use when connecting to Elasticsearch.
            es_params: Other parameters for the Elasticsearch client.
        """
        if es_connection is not None:
            self._es_client = es_connection
            if not self._es_client.ping():
                raise elasticsearch.exceptions.ConnectionError(
                    "Elasticsearch cluster is not available,"
                    " not able to set up the cache"
                )
        elif es_url is not None or es_cloud_id is not None:
            try:
                self._es_client = create_elasticsearch_client(
                    url=es_url,
                    cloud_id=es_cloud_id,
                    api_key=es_api_key,
                    username=es_user,
                    password=es_password,
                    params=es_params,
                )
            except Exception as err:
                logger.error(f"Error connecting to Elasticsearch: {err}")
                raise err
        else:
            raise ValueError(
                """Either provide a pre-existing Elasticsearch connection, \
                or valid credentials for creating a new connection."""
            )
        self._index_name = index_name
        self._store_input = store_input
        self._store_input_params = store_input_params
        self._metadata = metadata
        self._manage_index()

    def _manage_index(self) -> None:
        """Write or update an index or alias according to the default mapping"""
        self._is_alias = False
        if self._es_client.indices.exists_alias(name=self._index_name):
            self._is_alias = True
        elif not self._es_client.indices.exists(index=self._index_name):
            logger.debug(f"Creating new Elasticsearch index: {self._index_name}")
            self._es_client.indices.create(index=self._index_name, body=self.mapping)
            return
        self._es_client.indices.put_mapping(
            index=self._index_name, body=self.mapping["mappings"]
        )

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
