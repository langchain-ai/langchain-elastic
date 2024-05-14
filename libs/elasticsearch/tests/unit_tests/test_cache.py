from datetime import datetime
from typing import Any, Dict
from unittest.mock import ANY, MagicMock, patch

import pytest
from _pytest.fixtures import FixtureRequest
from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig
from elasticsearch import NotFoundError, exceptions
from langchain_core.load import dumps
from langchain_core.outputs import Generation

from langchain_elasticsearch import ElasticsearchCache, ElasticsearchEmbeddingsCache


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


def test_key_generation_cache_store(
    es_cache_store_fx: ElasticsearchEmbeddingsCache,
) -> None:
    key1 = es_cache_store_fx._key("test_text")
    assert key1 and isinstance(key1, str)
    key2 = es_cache_store_fx._key("test_text2")
    assert key2 and key1 != key2
    es_cache_store_fx._namespace = "other"
    key3 = es_cache_store_fx._key("test_text")
    assert key3 and key1 != key3
    es_cache_store_fx._namespace = None
    key4 = es_cache_store_fx._key("test_text")
    assert key4 and key1 != key4 and key3 != key4


def test_build_document_cache_store(
    es_cache_store_fx: ElasticsearchEmbeddingsCache,
) -> None:
    doc = es_cache_store_fx.build_document("test_text", [1.5, 2, 3.6])
    assert doc["llm_input"] == "test_text"
    assert doc["vector_dump"] == [1.5, 2, 3.6]
    assert datetime.fromisoformat(str(doc["timestamp"]))
    assert doc["metadata"] == es_cache_store_fx._metadata


def test_mget_cache_store(
    es_client_fx: MagicMock, es_cache_store_fx: ElasticsearchEmbeddingsCache
) -> None:
    cache_keys = [
        es_cache_store_fx._key("test_text1"),
        es_cache_store_fx._key("test_text2"),
        es_cache_store_fx._key("test_text3"),
    ]
    docs = {
        "docs": [
            {"_index": "test_index", "_id": cache_keys[0], "found": False},
            {
                "_index": "test_index",
                "_id": cache_keys[1],
                "found": True,
                "_source": {"vector_dump": [1.5, 2, 3.6]},
            },
            {
                "_index": "test_index",
                "_id": cache_keys[2],
                "found": True,
                "_source": {"vector_dump": [5, 6, 7.1]},
            },
        ]
    }
    es_cache_store_fx._is_alias = False
    es_client_fx.mget.return_value = docs
    assert es_cache_store_fx.mget([]) == []
    assert es_cache_store_fx.mget(["test_text1", "test_text2", "test_text3"]) == [
        None,
        [1.5, 2, 3.6],
        [5, 6, 7.1],
    ]
    es_client_fx.mget.assert_called_with(
        index="test_index", ids=cache_keys, source_includes=["vector_dump"]
    )
    es_cache_store_fx._is_alias = True
    es_client_fx.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}
    assert es_cache_store_fx.mget([]) == []
    assert es_cache_store_fx.mget(["test_text1", "test_text2", "test_text3"]) == [
        None,
        None,
        None,
    ]
    es_client_fx.search.assert_called_with(
        index="test_index",
        body={
            "query": {"ids": {"values": cache_keys}},
            "size": 3,
            "sort": {"timestamp": {"order": "asc"}},
        },
        source_includes=["vector_dump"],
    )
    resp = {
        "hits": {"total": {"value": 3}, "hits": [d for d in docs["docs"] if d["found"]]}
    }
    es_client_fx.search.return_value = resp
    assert es_cache_store_fx.mget(["test_text1", "test_text2", "test_text3"]) == [
        None,
        [1.5, 2, 3.6],
        [5, 6, 7.1],
    ]


def _del_timestamp(doc: Dict[str, Any]) -> Dict[str, Any]:
    del doc["_source"]["timestamp"]
    return doc


def test_mset_cache_store(es_cache_store_fx: ElasticsearchEmbeddingsCache) -> None:
    input = [("test_text1", [1.5, 2, 3.6]), ("test_text2", [5, 6, 7.1])]
    actions = [
        {
            "_op_type": "index",
            "_id": es_cache_store_fx._key(k),
            "_source": es_cache_store_fx.build_document(k, v),
        }
        for k, v in input
    ]
    es_cache_store_fx._is_alias = False
    with patch("elasticsearch.helpers.bulk") as bulk_mock:
        es_cache_store_fx.mset([])
        bulk_mock.assert_called_once()
        es_cache_store_fx.mset(input)
        bulk_mock.assert_called_with(
            client=es_cache_store_fx._es_client,
            actions=ANY,
            index="test_index",
            require_alias=False,
            refresh=True,
        )
        assert [_del_timestamp(d) for d in bulk_mock.call_args.kwargs["actions"]] == [
            _del_timestamp(d) for d in actions
        ]


def test_mdelete_cache_store(es_cache_store_fx: ElasticsearchEmbeddingsCache) -> None:
    input = ["test_text1", "test_text2"]
    actions = [{"_op_type": "delete", "_id": es_cache_store_fx._key(k)} for k in input]
    es_cache_store_fx._is_alias = False
    with patch("elasticsearch.helpers.bulk") as bulk_mock:
        es_cache_store_fx.mdelete([])
        bulk_mock.assert_called_once()
        es_cache_store_fx.mdelete(input)
        bulk_mock.assert_called_with(
            client=es_cache_store_fx._es_client,
            actions=ANY,
            index="test_index",
            require_alias=False,
            refresh=True,
        )
        assert list(bulk_mock.call_args.kwargs["actions"]) == actions
