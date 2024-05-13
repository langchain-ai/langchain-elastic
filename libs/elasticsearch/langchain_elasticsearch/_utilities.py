import logging
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

from elasticsearch import (
    BadRequestError,
    ConflictError,
    Elasticsearch,
    NotFoundError,
    exceptions,
)
from langchain_core import __version__ as langchain_version

from langchain_elasticsearch.client import create_elasticsearch_client

logger = logging.getLogger(__name__)


class ElasticsearchCacheIndexer:
    """Mixin for Elasticsearch clients"""

    def __init__(
        self,
        index_name: str,
        store_input: bool = True,
        store_input_params: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
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
            namespace: Optional namespace to use for the cache. Default to None.
                utilized only by CacheBackedEmbeddings.
            es_connection: Optional pre-existing Elasticsearch connection.
            es_url: URL of the Elasticsearch instance to connect to.
            es_cloud_id: Cloud ID of the Elasticsearch instance to connect to.
            es_user: Username to use when connecting to Elasticsearch.
            es_password: Password to use when connecting to Elasticsearch.
            es_api_key: API key to use when connecting to Elasticsearch.
            es_params: Other parameters for the Elasticsearch client.
        """

        self._index_name = index_name
        self._store_input = store_input
        self._store_input_params = store_input_params
        self._metadata = metadata
        self._namespace = namespace
        self._setup_connection(
            es_connection=es_connection,
            es_url=es_url,
            es_cloud_id=es_cloud_id,
            es_user=es_user,
            es_api_key=es_api_key,
            es_password=es_password,
            es_params=es_params,
        )
        self._manage_index()

    def _setup_connection(
        self,
        *,
        es_connection: Optional["Elasticsearch"] = None,
        es_url: Optional[str] = None,
        es_cloud_id: Optional[str] = None,
        es_user: Optional[str] = None,
        es_api_key: Optional[str] = None,
        es_password: Optional[str] = None,
        es_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if es_connection is not None:
            self._es_client = es_connection
            if not self._es_client.ping():
                raise exceptions.ConnectionError(
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

    @property
    @abstractmethod
    def mapping(self) -> dict[str, Any]:
        """Get the default mapping for the index."""
        return {}


class DistanceStrategy(str, Enum):
    """Enumerator of the Distance strategies for calculating distances
    between vectors."""

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    DOT_PRODUCT = "DOT_PRODUCT"
    JACCARD = "JACCARD"
    COSINE = "COSINE"


def user_agent(prefix: str) -> str:
    return f"{prefix}/{langchain_version}"


def with_user_agent_header(client: Elasticsearch, header_prefix: str) -> Elasticsearch:
    headers = dict(client._headers)
    headers.update({"user-agent": f"{user_agent(header_prefix)}"})
    return client.options(headers=headers)


def model_must_be_deployed(client: Elasticsearch, model_id: str) -> None:
    try:
        dummy = {"x": "y"}
        client.ml.infer_trained_model(model_id=model_id, docs=[dummy])
    except NotFoundError as err:
        raise err
    except ConflictError as err:
        raise NotFoundError(
            f"model '{model_id}' not found, please deploy it first",
            meta=err.meta,
            body=err.body,
        ) from err
    except BadRequestError:
        # This error is expected because we do not know the expected document
        # shape and just use a dummy doc above.
        pass
