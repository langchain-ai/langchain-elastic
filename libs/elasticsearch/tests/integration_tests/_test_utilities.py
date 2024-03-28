import os
from typing import Any, Dict, List, Optional

from elastic_transport import Transport
from elasticsearch import Elasticsearch


def read_env() -> Dict:
    url = os.environ.get("ES_URL", "http://localhost:9200")
    cloud_id = os.environ.get("ES_CLOUD_ID")
    api_key = os.environ.get("ES_API_KEY")

    if cloud_id:
        return {"es_cloud_id": cloud_id, "es_api_key": api_key}
    return {"es_url": url}


def create_es_client(
    es_params: Optional[Dict[str, str]] = None, es_kwargs: Dict = {}
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


def clear_test_indices(es: Elasticsearch) -> None:
    index_names = es.indices.get(index="_all").keys()
    for index_name in index_names:
        if index_name.startswith("test_"):
            es.indices.delete(index=index_name)
    es.indices.refresh(index="_all")


def requests_saving_es_client() -> Elasticsearch:
    class CustomTransport(Transport):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.requests: List[Dict] = []

        def perform_request(self, *args, **kwargs):  # type: ignore
            self.requests.append(kwargs)
            return super().perform_request(*args, **kwargs)

    return create_es_client(es_kwargs=dict(transport_class=CustomTransport))
