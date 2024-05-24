import logging
from enum import Enum
from typing import Any, Dict, Optional

from elasticsearch import Elasticsearch, exceptions
from langchain_core import __version__ as langchain_version

from langchain_elasticsearch.client import create_elasticsearch_client

logger = logging.getLogger(__name__)


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
    except exceptions.NotFoundError as err:
        raise err
    except exceptions.ConflictError as err:
        raise exceptions.NotFoundError(
            f"model '{model_id}' not found, please deploy it first",
            meta=err.meta,
            body=err.body,
        ) from err
    except exceptions.BadRequestError:
        # This error is expected because we do not know the expected document
        # shape and just use a dummy doc above.
        pass


def setup_connection(
    es_connection: Optional[Elasticsearch] = None,
    es_url: Optional[str] = None,
    es_cloud_id: Optional[str] = None,
    es_user: Optional[str] = None,
    es_api_key: Optional[str] = None,
    es_password: Optional[str] = None,
    es_params: Optional[Dict[str, Any]] = None,
) -> Elasticsearch:
    if es_connection is not None:
        _es_client = es_connection
        if not _es_client.ping():
            raise exceptions.ConnectionError("Elasticsearch cluster is not available")
    elif es_url is not None or es_cloud_id is not None:
        try:
            _es_client = create_elasticsearch_client(
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

    return _es_client


def manage_cache_index(
    es_client: Elasticsearch, index_name: str, mapping: Dict[str, Any]
) -> bool:
    """Write or update an index or alias according to the default mapping"""
    if es_client.indices.exists_alias(name=index_name):
        es_client.indices.put_mapping(index=index_name, body=mapping["mappings"])
        return True

    elif not es_client.indices.exists(index=index_name):
        logger.debug(f"Creating new Elasticsearch index: {index_name}")
        es_client.indices.create(index=index_name, body=mapping)
        return False

    return False
