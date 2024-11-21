import os
from typing import Any, Dict, List, Optional

from elastic_transport import Transport

from elasticsearch import (BadRequestError, ConflictError, Elasticsearch,
                           NotFoundError)


def read_env() -> Dict:
    url = os.environ.get("ES_URL", "http://localhost:9200")
    cloud_id = os.environ.get("ES_CLOUD_ID")
    api_key = os.environ.get("ES_API_KEY")

    if cloud_id:
        return {"es_cloud_id": cloud_id, "es_api_key": api_key}
    return {"es_url": url}


class RequestSavingTransport(Transport):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.requests: List[Dict] = []

    def perform_request(self, *args, **kwargs):  # type: ignore
        self.requests.append(kwargs)
        return super().perform_request(*args, **kwargs)


def create_es_client(
    es_params: Optional[Dict[str, str]] = None,
    es_kwargs: Dict = {},
) -> Elasticsearch:
    if es_params is None:
        es_params = read_env()
    if not es_kwargs:
        es_kwargs = {}

    if "es_cloud_id" in es_params:
        return Elasticsearch(
            cloud_id=es_params["es_cloud_id"],
            api_key=es_params["es_api_key"],
            **es_kwargs,
        )

    return Elasticsearch(hosts=[es_params["es_url"]], **es_kwargs)


def requests_saving_es_client() -> Elasticsearch:
    return create_es_client(es_kwargs={"transport_class": RequestSavingTransport})


def clear_test_indices(es: Elasticsearch) -> None:
    index_names_response = es.indices.get(index="_all")
    index_names = index_names_response.keys()
    for index_name in index_names:
        if index_name.startswith("test_"):
            es.indices.delete(index=index_name)
    es.indices.refresh(index="_all")


def model_is_deployed(client: Elasticsearch, model_id: str) -> bool:
    try:
        dummy = {"x": "y"}
        client.ml.infer_trained_model(model_id=model_id, docs=[dummy])
        return True
    except NotFoundError:
        return False
    except ConflictError:
        return False
    except BadRequestError:
        # This error is expected because we do not know the expected document
        # shape and just use a dummy doc above.
        return True
