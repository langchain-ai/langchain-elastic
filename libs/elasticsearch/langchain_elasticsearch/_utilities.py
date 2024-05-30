import logging
from enum import Enum

from elasticsearch import Elasticsearch, exceptions
from langchain_core import __version__ as langchain_version

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
