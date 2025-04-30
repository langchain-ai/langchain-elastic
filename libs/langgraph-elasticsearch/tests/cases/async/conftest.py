from typing import Generator
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, Mock
from elasticsearch import AsyncElasticsearch, Elasticsearch
from elasticsearch.client import IndicesClient as AsyncIndicesClient, IndicesClient
import pytest

from langgraph.store.elasticsearch.aio import AsyncElasticsearchMemoryStore
from langgraph.store.elasticsearch.base import ElasticsearchMemoryStore
from langgraph.store.elasticsearch.config import ElasticsearchIndexConfig
from elastic_transport import ConnectionError

@pytest.fixture
def factory_es_client(request):
    if request.node.get_closest_marker("fail"):
        return request.getfixturevalue("async_es_client_fail")
    return request.getfixturevalue("async_es_client")

@pytest.fixture
def async_es_client() -> Generator[MagicMock, None, None]:
    client_mock = MagicMock(spec=AsyncElasticsearch)
    client_mock.return_value.hosts = ["http://localhost:9200"]
    client_mock.return_value.indices = MagicMock(spec=AsyncIndicesClient)
    # coroutines need to be mocked explicitly
    client_mock.return_value.indices.exists_alias = AsyncMock()
    client_mock.return_value.indices.put_mapping = AsyncMock()
    client_mock.return_value.indices.exists = AsyncMock()
    client_mock.return_value.indices.create = AsyncMock()
    yield client_mock()

@pytest.fixture
def async_es_client_fail() -> Generator[MagicMock, None, None]:
    client_mock = MagicMock(spec=AsyncElasticsearch)
    def raise_connection_error(*args, **kwargs):
        raise ConnectionError("Failed to connect to Elasticsearch")
    client_mock.return_value.update.side_effect = raise_connection_error
    yield client_mock()

@pytest.fixture
def index_config():
    return ElasticsearchIndexConfig(
        store_index_name="langgraph-store",
    )

@pytest.fixture
def store(factory_es_client, index_config):
    return AsyncElasticsearchMemoryStore(
        es_connection=factory_es_client, 
        index_config=index_config
    )