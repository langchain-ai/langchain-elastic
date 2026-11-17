import pytest
from typing import Generator
from unittest.mock import MagicMock, Mock
from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.embeddings.fake import FakeEmbeddings

from langgraph.store.elasticsearch.base import ElasticsearchMemoryStore
from langgraph.store.elasticsearch.config import ElasticsearchIndexConfig
from elastic_transport import ConnectionError
from langchain_elasticsearch._utilities import DistanceStrategy
from langchain_elasticsearch import DenseVectorStrategy, DistanceMetric
from testcontainers.elasticsearch import ElasticSearchContainer

@pytest.fixture
def factory_es_client(request):
    if request.node.get_closest_marker("fail"):
        return request.getfixturevalue("sync_es_client_fail")
    if request.node.get_closest_marker("integration"):
        return request.getfixturevalue("sync_es_client")
    if request.node.get_closest_marker("mock"):
        return request.getfixturevalue("sync_es_client")




@pytest.fixture
def sync_es_client(request) -> Generator[MagicMock, None, None]:
    if request.node.get_closest_marker("integration"):
        client = Elasticsearch(
            basic_auth=("elastic","elastic"),
            hosts=["http://localhost:9200"],
            verify_certs=False
        )
        yield client
    else:
        client_mock = MagicMock(spec=Elasticsearch)
        client_mock.return_value.hosts = ["http://localhost:9200"]
        client_mock.return_value.indices = MagicMock(spec=IndicesClient)
        client_mock.return_value.indices.exists_alias = Mock()
        client_mock.return_value.indices.put_mapping = Mock()
        client_mock.return_value.indices.exists = Mock()
        client_mock.return_value.indices.create = Mock()
        yield client_mock()

        # with ElasticSearchContainer('elasticsearch:8.9.2') as container:
        #     client = Elasticsearch([container.get_url()])
        #     yield client

@pytest.fixture
def sync_es_client_fail() -> Generator[MagicMock, None, None]:
    client_mock = MagicMock(spec=Elasticsearch)
    def raise_connection_error(*args, **kwargs):
        raise ConnectionError("Failed to connect to Elasticsearch")
    client_mock.return_value.update.side_effect = raise_connection_error
    yield client_mock()

@pytest.fixture
def index_config(request):
    if request.node.get_closest_marker("embed"):
        return ElasticsearchIndexConfig(
            store_index_name="langgraph-store",
            vector_index_name="langgraph-vectorstore",
            dims=3072,
            distance_strategy=DistanceStrategy.COSINE,
            strategy=DenseVectorStrategy(
                distance=DistanceMetric.COSINE,
            ),
            embed=FakeEmbeddings(
                size=50,
            ) if not request.node.get_closest_marker("fail") else None,
            fields=["data"],
        )

    return ElasticsearchIndexConfig(
        store_index_name="langgraph-store",
    )

@pytest.fixture
def store(factory_es_client, index_config):
    return ElasticsearchMemoryStore(
        es_connection=factory_es_client, 
        index_config=index_config
    )

@pytest.fixture
def async_store(factory_es_client, index_config):
    return ElasticsearchMemoryStore(
        es_connection=factory_es_client, 
        index_config=index_config
    )












# Mocks
@pytest.fixture
def mock_get(sync_es_client_fx):
    sync_es_client_fx.get.return_value = {
        "_index": "langgraph-store",
        "_id": "test/foo/key1",
        "_source": {
            "namespace": "test/foo",
            "key": "key1",
            "value": {
                "data": "value1"
            },
            "data": "value1",
            "created_at": "2021-10-01T00:00:00Z",
            "updated_at": "2021-10-01T00:00:00Z"
        }
    }

@pytest.fixture
def mock_search(sync_es_client_fx):
    sync_es_client_fx.search.return_value = {
        "_index": "langgraph-store",
        "_id": "test/foo/key1",
        "_source": {
            "namespace": "test/foo",
            "key": "key1",
            "value": {
                "data": "value1"
            },
            "data": "value1",
            "created_at": "2021-10-01T00:00:00Z",
            "updated_at": "2021-10-01T00:00:00Z"
        }
    }

@pytest.fixture
def mock_listname(sync_es_client_fx):
    sync_es_client_fx.search.return_value = {
        'hits': {
            'hits': [
                {
                    '_index': 'langgraph-store',
                    '_id': 'test/foo/key1',
                    '_source': {
                        'namespace': 'test/foo',
                    }
                }
            ]
        }
    }