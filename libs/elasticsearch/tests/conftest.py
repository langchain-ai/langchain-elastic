from typing import Generator
from unittest.mock import MagicMock

import pytest
from elasticsearch import Elasticsearch
from elasticsearch._sync.client import IndicesClient
from langchain_community.chat_models.fake import FakeMessagesListChatModel
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from langchain_elasticsearch import ElasticsearchCache, ElasticsearchEmbeddingsCache


@pytest.fixture
def es_client_fx() -> Generator[MagicMock, None, None]:
    client_mock = MagicMock(spec=Elasticsearch)
    client_mock.indices = MagicMock(spec=IndicesClient)
    yield client_mock()


@pytest.fixture
def es_cache_store_fx(
    es_client_fx: MagicMock,
) -> Generator[ElasticsearchEmbeddingsCache, None, None]:
    yield ElasticsearchEmbeddingsCache(
        es_connection=es_client_fx,
        index_name="test_index",
        store_input=True,
        store_input_params=True,
        namespace="test",
        metadata={"project": "test_project"},
    )


@pytest.fixture
def es_cache_fx(es_client_fx: MagicMock) -> Generator[ElasticsearchCache, None, None]:
    yield ElasticsearchCache(
        es_connection=es_client_fx,
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
