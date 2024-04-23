from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from _pytest.fixtures import FixtureRequest
from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig
from elasticsearch import NotFoundError, exceptions
from langchain_core.load import dumps
from langchain_core.outputs import Generation

from langchain_elasticsearch import ElasticsearchCache


def test_initialization(es_client_fx: MagicMock) -> None:
    es_client_fx.ping.return_value = False
    with pytest.raises(exceptions.ConnectionError):
        ElasticsearchCache(es_connection=es_client_fx, index_name="test_index")
    es_client_fx.ping.return_value = True
    es_client_fx.indices.exists_alias.return_value = True
    cache = ElasticsearchCache(es_connection=es_client_fx, index_name="test_index")
    es_client_fx.indices.exists_alias.assert_called_with(name="test_index")
    assert cache._is_alias
    es_client_fx.indices.put_mapping.assert_called_with(
        index="test_index", body=cache.mapping["mappings"]
    )
    es_client_fx.indices.exists_alias.return_value = False
    es_client_fx.indices.exists.return_value = False
    cache = ElasticsearchCache(es_connection=es_client_fx, index_name="test_index")
    assert not cache._is_alias
    es_client_fx.indices.create.assert_called_with(
        index="test_index", body=cache.mapping
    )


def test_mapping(es_cache_fx: ElasticsearchCache, request: FixtureRequest) -> None:
    mapping = request.getfixturevalue("es_cache_fx").mapping
    assert mapping.get("mappings")
    assert mapping["mappings"].get("properties")


def test_key_generation(es_cache_fx: ElasticsearchCache) -> None:
    key1 = es_cache_fx._key("test_prompt", "test_llm_string")
    assert key1 and isinstance(key1, str)
    key2 = es_cache_fx._key("test_prompt", "test_llm_string1")
    assert key2 and key1 != key2
    key3 = es_cache_fx._key("test_prompt1", "test_llm_string")
    assert key3 and key1 != key3


def test_clear(es_client_fx: MagicMock, es_cache_fx: ElasticsearchCache) -> None:
    es_cache_fx.clear()
    es_client_fx.delete_by_query.assert_called_once_with(
        index="test_index",
        body={"query": {"match_all": {}}},
        refresh=True,
        wait_for_completion=True,
    )


def test_build_document(es_cache_fx: ElasticsearchCache) -> None:
    doc = es_cache_fx.build_document(
        "test_prompt", "test_llm_string", [Generation(text="test_prompt")]
    )
    assert doc["llm_input"] == "test_prompt"
    assert doc["llm_params"] == "test_llm_string"
    assert isinstance(doc["llm_output"], list)
    assert all(isinstance(gen, str) for gen in doc["llm_output"])
    assert datetime.fromisoformat(str(doc["timestamp"]))
    assert doc["metadata"] == es_cache_fx._metadata


def test_update(es_client_fx: MagicMock, es_cache_fx: ElasticsearchCache) -> None:
    es_cache_fx.update("test_prompt", "test_llm_string", [Generation(text="test")])
    timestamp = es_client_fx.index.call_args.kwargs["body"]["timestamp"]
    doc = es_cache_fx.build_document(
        "test_prompt", "test_llm_string", [Generation(text="test")]
    )
    doc["timestamp"] = timestamp
    es_client_fx.index.assert_called_once_with(
        index=es_cache_fx._index_name,
        id=es_cache_fx._key("test_prompt", "test_llm_string"),
        body=doc,
        require_alias=es_cache_fx._is_alias,
        refresh=True,
    )


def test_lookup(es_client_fx: MagicMock, es_cache_fx: ElasticsearchCache) -> None:
    cache_key = es_cache_fx._key("test_prompt", "test_llm_string")
    doc: Dict[str, Any] = {
        "_source": {
            "llm_output": [dumps(Generation(text="test"))],
            "timestamp": "2024-03-07T13:25:36.410756",
        }
    }
    es_cache_fx._is_alias = False
    es_client_fx.get.side_effect = NotFoundError(
        "not found",
        ApiResponseMeta(404, "0", HttpHeaders(), 0, NodeConfig("http", "xxx", 80)),
        "",
    )
    assert es_cache_fx.lookup("test_prompt", "test_llm_string") is None
    es_client_fx.get.assert_called_once_with(
        index="test_index", id=cache_key, source=["llm_output"]
    )
    es_client_fx.get.side_effect = None
    es_client_fx.get.return_value = doc
    assert es_cache_fx.lookup("test_prompt", "test_llm_string") == [
        Generation(text="test")
    ]
    es_cache_fx._is_alias = True
    es_client_fx.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}
    assert es_cache_fx.lookup("test_prompt", "test_llm_string") is None
    es_client_fx.search.assert_called_once_with(
        index="test_index",
        body={
            "query": {"term": {"_id": cache_key}},
            "sort": {"timestamp": {"order": "asc"}},
        },
        source_includes=["llm_output"],
    )
    doc2 = {
        "_source": {
            "llm_output": [dumps(Generation(text="test2"))],
            "timestamp": "2024-03-08T13:25:36.410756",
        },
    }
    es_client_fx.search.return_value = {
        "hits": {"total": {"value": 2}, "hits": [doc2, doc]}
    }
    assert es_cache_fx.lookup("test_prompt", "test_llm_string") == [
        Generation(text="test2")
    ]
