from typing import Generator
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
from elasticsearch import Elasticsearch, AsyncElasticsearch
from elasticsearch._async.client import IndicesClient as AsyncIndicesClient
from elasticsearch._sync.client import IndicesClient
from langchain_community.chat_models.fake import FakeMessagesListChatModel
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from langchain_elasticsearch import (
    ElasticsearchCache,
    ElasticsearchEmbeddingsCache,
    AsyncElasticsearchCache,
    AsyncElasticsearchEmbeddingsCache,
)


@pytest.fixture
def es_client_fx() -> Generator[MagicMock, None, None]:
    client_mock = MagicMock(spec=Elasticsearch)
    client_mock.return_value.indices = MagicMock(spec=IndicesClient)
    yield client_mock()


@pytest.fixture
def async_es_client_fx() -> Generator[MagicMock, None, None]:
    client_mock = MagicMock(spec=AsyncElasticsearch)
    client_mock.return_value.indices = MagicMock(spec=AsyncIndicesClient)
    # coroutines need to be mocked explicitly
    client_mock.return_value.indices.exists_alias = AsyncMock()
    client_mock.return_value.indices.put_mapping = AsyncMock()
    client_mock.return_value.indices.exists = AsyncMock()
    client_mock.return_value.indices.create = AsyncMock()
    yield client_mock()


@pytest.fixture
def es_embeddings_cache_fx(
    es_client_fx: MagicMock,
) -> Generator[ElasticsearchEmbeddingsCache, None, None]:
    with mock.patch(
        "langchain_elasticsearch._sync.cache.create_elasticsearch_client",
        return_value=es_client_fx,
    ):
        yield ElasticsearchEmbeddingsCache(
            es_url="http://localhost:9200",
            index_name="test_index",
            store_input=True,
            namespace="test",
            metadata={"project": "test_project"},
        )


@pytest.fixture
def async_es_embeddings_cache_fx(
    async_es_client_fx: MagicMock,
) -> Generator[AsyncElasticsearchEmbeddingsCache, None, None]:
    with mock.patch(
        "langchain_elasticsearch._async.cache.create_async_elasticsearch_client",
        return_value=async_es_client_fx,
    ):
        yield AsyncElasticsearchEmbeddingsCache(
            es_url="http://localhost:9200",
            index_name="test_index",
            store_input=True,
            namespace="test",
            metadata={"project": "test_project"},
        )


@pytest.fixture
def es_cache_fx(
    es_client_fx: MagicMock,
) -> Generator[AsyncElasticsearchCache, None, None]:
    with mock.patch(
        "langchain_elasticsearch._sync.cache.create_elasticsearch_client",
        return_value=es_client_fx,
    ):
        yield ElasticsearchCache(
            es_url="http://localhost:30096",
            index_name="test_index",
            store_input=True,
            store_input_params=True,
            metadata={"project": "test_project"},
        )


@pytest.fixture
def async_es_cache_fx(
    async_es_client_fx: MagicMock,
) -> Generator[AsyncElasticsearchCache, None, None]:
    with mock.patch(
        "langchain_elasticsearch._async.cache.create_async_elasticsearch_client",
        return_value=async_es_client_fx,
    ):
        yield AsyncElasticsearchCache(
            es_url="http://localhost:30096",
            index_name="test_index",
            store_input=True,
            store_input_params=True,
            metadata={"project": "test_project"},
        )


@pytest.fixture
def fake_chat_fx() -> Generator[BaseChatModel, None, None]:
    yield FakeMessagesListChatModel(
        cache=True, responses=[AIMessage(content="test output")]
    )
