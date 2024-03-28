import os
import time
from typing import Any, Dict, List, Optional

from elastic_transport import Transport
from elasticsearch import Elasticsearch, NotFoundError

from langchain_elasticsearch._utilities import check_if_model_deployed


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

    es_url = os.environ.get("ES_URL", "http://localhost:9200")
    cloud_id = os.environ.get("ES_CLOUD_ID")
    api_key = os.environ.get("ES_API_KEY")

    if cloud_id:
        # Running this integration test with Elastic Cloud
        # Required for in-stack inference testing (ELSER + model_id)
        es = Elasticsearch(
            cloud_id=cloud_id,
            api_key=api_key,
            transport_class=CustomTransport,
        )
    else:
        # Running this integration test with local docker instance
        es = Elasticsearch(hosts=[es_url], transport_class=CustomTransport)

    return es


def deploy_model(
    es_client: Elasticsearch,
    model_id: str = ".elser_model_2",
    field: str = "text_field",
    model_type: Optional[str] = None,
    inference_config: Optional[Dict] = None,
) -> None:
    try:
        check_if_model_deployed(es_client, model_id)
    except NotFoundError:
        # download the model
        es_client.ml.put_trained_model(
            model_id=model_id,
            input={"field_names": [field]},
            model_type=model_type,
            inference_config=inference_config,
        )

        # wait until ready
        while True:
            status = es_client.ml.get_trained_models(
                model_id=model_id, include="definition_status"
            )
            if status["trained_model_configs"][0]["fully_defined"]:
                # model is ready
                break
            time.sleep(1)

        # deploy the model
        es_client.ml.start_trained_model_deployment(model_id=model_id)
