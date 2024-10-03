from typing import Any, Dict, Optional

from elasticsearch import AsyncElasticsearch, Elasticsearch


def _create_client_params(
    url: Optional[str] = None,
    cloud_id: Optional[str] = None,
    api_key: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if url and cloud_id:
        raise ValueError(
            "Both es_url and cloud_id are defined. Please provide only one."
        )

    connection_params: Dict[str, Any] = {}

    if url:
        connection_params["hosts"] = [url]
    elif cloud_id:
        connection_params["cloud_id"] = cloud_id
    else:
        raise ValueError("Please provide either elasticsearch_url or cloud_id.")

    if api_key:
        connection_params["api_key"] = api_key
    elif username and password:
        connection_params["basic_auth"] = (username, password)

    if params is not None:
        connection_params.update(params)

    return connection_params


def create_elasticsearch_client(
    url: Optional[str] = None,
    cloud_id: Optional[str] = None,
    api_key: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Elasticsearch:
    connection_params = _create_client_params(
        url, cloud_id, api_key, username, password, params
    )

    es_client = Elasticsearch(**connection_params)

    es_client.info()  # test connection

    return es_client


def create_elasticsearch_async_client(
    url: Optional[str] = None,
    cloud_id: Optional[str] = None,
    api_key: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> AsyncElasticsearch:
    connection_params = _create_client_params(
        url, cloud_id, api_key, username, password, params
    )

    es_client = AsyncElasticsearch(**connection_params)

    es_client.info()  # test connection

    return es_client
