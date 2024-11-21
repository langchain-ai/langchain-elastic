from datetime import datetime
from typing import Any, Dict
from unittest import mock
from unittest.mock import ANY, MagicMock, patch

from _pytest.fixtures import FixtureRequest
from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig
from elasticsearch import NotFoundError
from langchain.embeddings.cache import _value_serializer
from langchain_core.load import dumps
from langchain_core.outputs import Generation
import pytest

from langchain_elasticsearch import (
    AsyncElasticsearchCache,
    AsyncElasticsearchEmbeddingsCache,
)


def serialize_encode_vector(vector: Any) -> str:
    return AsyncElasticsearchEmbeddingsCache.encode_vector(_value_serializer(vector))


@pytest.mark.asyncio
async def test_initialization_llm_cache(async_es_client_fx: MagicMock) -> None:
    async_es_client_fx.ping.return_value = True
    async_es_client_fx.indices.exists_alias.return_value = True
    with mock.patch(
        "langchain_elasticsearch._sync.cache.create_elasticsearch_client",
        return_value=async_es_client_fx,
    ):
        with mock.patch(
            "langchain_elasticsearch._async.cache.create_async_elasticsearch_client",
            return_value=async_es_client_fx,
        ):
            cache = AsyncElasticsearchCache(
                es_url="http://localhost:9200", index_name="test_index"
            )
            assert await cache.is_alias()
            async_es_client_fx.indices.exists_alias.assert_awaited_with(
                name="test_index"
            )
            async_es_client_fx.indices.put_mapping.assert_awaited_with(
                index="test_index", body=cache.mapping["mappings"]
            )
            async_es_client_fx.indices.exists_alias.return_value = False
            async_es_client_fx.indices.exists.return_value = False
            cache = AsyncElasticsearchCache(
                es_url="http://localhost:9200", index_name="test_index"
            )
            assert not (await cache.is_alias())
            async_es_client_fx.indices.create.assert_awaited_with(
                index="test_index", body=cache.mapping
            )


def test_mapping_llm_cache(
    async_es_cache_fx: AsyncElasticsearchCache, request: FixtureRequest
) -> None:
    mapping = request.getfixturevalue("es_cache_fx").mapping
    assert mapping.get("mappings")
    assert mapping["mappings"].get("properties")


def test_key_generation_llm_cache(es_cache_fx: AsyncElasticsearchCache) -> None:
    key1 = es_cache_fx._key("test_prompt", "test_llm_string")
    assert key1 and isinstance(key1, str)
    key2 = es_cache_fx._key("test_prompt", "test_llm_string1")
    assert key2 and key1 != key2
    key3 = es_cache_fx._key("test_prompt1", "test_llm_string")
    assert key3 and key1 != key3


def test_clear_llm_cache(
    es_client_fx: MagicMock, es_cache_fx: AsyncElasticsearchCache
) -> None:
    es_cache_fx.clear()
    es_client_fx.delete_by_query.assert_called_once_with(
        index="test_index",
        body={"query": {"match_all": {}}},
        refresh=True,
        wait_for_completion=True,
    )


def test_build_document_llm_cache(es_cache_fx: AsyncElasticsearchCache) -> None:
    doc = es_cache_fx.build_document(
        "test_prompt", "test_llm_string", [Generation(text="test_prompt")]
    )
    assert doc["llm_input"] == "test_prompt"
    assert doc["llm_params"] == "test_llm_string"
    assert isinstance(doc["llm_output"], list)
    assert all(isinstance(gen, str) for gen in doc["llm_output"])
    assert datetime.fromisoformat(str(doc["timestamp"]))
    assert doc["metadata"] == es_cache_fx._metadata


def test_update_llm_cache(
    es_client_fx: MagicMock, es_cache_fx: AsyncElasticsearchCache
) -> None:
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


def test_lookup_llm_cache(
    es_client_fx: MagicMock, es_cache_fx: AsyncElasticsearchCache
) -> None:
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
    es_embeddings_cache_fx: AsyncElasticsearchEmbeddingsCache,
) -> None:
    key1 = es_embeddings_cache_fx._key("test_text")
    assert key1 and isinstance(key1, str)
    key2 = es_embeddings_cache_fx._key("test_text2")
    assert key2 and key1 != key2
    es_embeddings_cache_fx._namespace = "other"
    key3 = es_embeddings_cache_fx._key("test_text")
    assert key3 and key1 != key3
    es_embeddings_cache_fx._namespace = None
    key4 = es_embeddings_cache_fx._key("test_text")
    assert key4 and key1 != key4 and key3 != key4


def test_build_document_cache_store(
    es_embeddings_cache_fx: AsyncElasticsearchEmbeddingsCache,
) -> None:
    doc = es_embeddings_cache_fx.build_document(
        "test_text", _value_serializer([1.5, 2, 3.6])
    )
    assert doc["text_input"] == "test_text"
    assert doc["vector_dump"] == serialize_encode_vector([1.5, 2, 3.6])
    assert datetime.fromisoformat(str(doc["timestamp"]))
    assert doc["metadata"] == es_embeddings_cache_fx._metadata


def test_mget_cache_store(
    es_client_fx: MagicMock, es_embeddings_cache_fx: AsyncElasticsearchEmbeddingsCache
) -> None:
    cache_keys = [
        es_embeddings_cache_fx._key("test_text1"),
        es_embeddings_cache_fx._key("test_text2"),
        es_embeddings_cache_fx._key("test_text3"),
    ]
    docs = {
        "docs": [
            {"_index": "test_index", "_id": cache_keys[0], "found": False},
            {
                "_index": "test_index",
                "_id": cache_keys[1],
                "found": True,
                "_source": {"vector_dump": serialize_encode_vector([1.5, 2, 3.6])},
            },
            {
                "_index": "test_index",
                "_id": cache_keys[2],
                "found": True,
                "_source": {"vector_dump": serialize_encode_vector([5, 6, 7.1])},
            },
        ]
    }
    es_embeddings_cache_fx._is_alias = False
    es_client_fx.mget.return_value = docs
    assert es_embeddings_cache_fx.mget([]) == []
    assert es_embeddings_cache_fx.mget(["test_text1", "test_text2", "test_text3"]) == [
        None,
        _value_serializer([1.5, 2, 3.6]),
        _value_serializer([5, 6, 7.1]),
    ]
    es_client_fx.mget.assert_called_with(
        index="test_index", ids=cache_keys, source_includes=["vector_dump"]
    )
    es_embeddings_cache_fx._is_alias = True
    es_client_fx.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}
    assert es_embeddings_cache_fx.mget([]) == []
    assert es_embeddings_cache_fx.mget(["test_text1", "test_text2", "test_text3"]) == [
        None,
        None,
        None,
    ]
    es_client_fx.search.assert_called_with(
        index="test_index",
        body={
            "query": {"ids": {"values": cache_keys}},
            "size": 3,
        },
        source_includes=["vector_dump", "timestamp"],
    )
    resp = {
        "hits": {"total": {"value": 3}, "hits": [d for d in docs["docs"] if d["found"]]}
    }
    es_client_fx.search.return_value = resp
    assert es_embeddings_cache_fx.mget(["test_text1", "test_text2", "test_text3"]) == [
        None,
        _value_serializer([1.5, 2, 3.6]),
        _value_serializer([5, 6, 7.1]),
    ]


def test_deduplicate_hits(
    es_embeddings_cache_fx: AsyncElasticsearchEmbeddingsCache,
) -> None:
    hits = [
        {
            "_id": "1",
            "_source": {
                "timestamp": "2022-01-01T00:00:00",
                "vector_dump": serialize_encode_vector([1, 2, 3]),
            },
        },
        {
            "_id": "1",
            "_source": {
                "timestamp": "2022-01-02T00:00:00",
                "vector_dump": serialize_encode_vector([4, 5, 6]),
            },
        },
        {
            "_id": "2",
            "_source": {
                "timestamp": "2022-01-01T00:00:00",
                "vector_dump": serialize_encode_vector([7, 8, 9]),
            },
        },
    ]

    result = es_embeddings_cache_fx._deduplicate_hits(hits)

    assert len(result) == 2
    assert result["1"] == _value_serializer([4, 5, 6])
    assert result["2"] == _value_serializer([7, 8, 9])


def test_mget_duplicate_keys_cache_store(
    es_client_fx: MagicMock, es_embeddings_cache_fx: AsyncElasticsearchEmbeddingsCache
) -> None:
    cache_keys = [
        es_embeddings_cache_fx._key("test_text1"),
        es_embeddings_cache_fx._key("test_text2"),
    ]

    resp = {
        "hits": {
            "total": {"value": 3},
            "hits": [
                {
                    "_index": "test_index",
                    "_id": cache_keys[1],
                    "found": True,
                    "_source": {
                        "vector_dump": serialize_encode_vector([1.5, 2, 3.6]),
                        "timestamp": "2024-03-07T13:25:36.410756",
                    },
                },
                {
                    "_index": "test_index",
                    "_id": cache_keys[0],
                    "found": True,
                    "_source": {
                        "vector_dump": serialize_encode_vector([1, 6, 7.1]),
                        "timestamp": "2024-03-07T13:25:46.410756",
                    },
                },
                {
                    "_index": "test_index",
                    "_id": cache_keys[0],
                    "found": True,
                    "_source": {
                        "vector_dump": serialize_encode_vector([2, 6, 7.1]),
                        "timestamp": "2024-03-07T13:27:46.410756",
                    },
                },
            ],
        }
    }

    es_embeddings_cache_fx._is_alias = True
    es_client_fx.search.return_value = resp
    assert es_embeddings_cache_fx.mget(["test_text1", "test_text2"]) == [
        _value_serializer([2, 6, 7.1]),
        _value_serializer([1.5, 2, 3.6]),
    ]
    es_client_fx.search.assert_called_with(
        index="test_index",
        body={
            "query": {"ids": {"values": cache_keys}},
            "size": len(cache_keys),
        },
        source_includes=["vector_dump", "timestamp"],
    )


def _del_timestamp(doc: Dict[str, Any]) -> Dict[str, Any]:
    del doc["_source"]["timestamp"]
    return doc


def test_mset_cache_store(
    es_embeddings_cache_fx: AsyncElasticsearchEmbeddingsCache,
) -> None:
    input = [
        ("test_text1", _value_serializer([1.5, 2, 3.6])),
        ("test_text2", _value_serializer([5, 6, 7.1])),
    ]
    actions = [
        {
            "_op_type": "index",
            "_id": es_embeddings_cache_fx._key(k),
            "_source": es_embeddings_cache_fx.build_document(k, v),
        }
        for k, v in input
    ]
    es_embeddings_cache_fx._is_alias = False
    with patch("elasticsearch.helpers.bulk") as bulk_mock:
        es_embeddings_cache_fx.mset([])
        bulk_mock.assert_called_once()
        es_embeddings_cache_fx.mset(input)
        bulk_mock.assert_called_with(
            client=es_embeddings_cache_fx._es_client,
            actions=ANY,
            index="test_index",
            require_alias=False,
            refresh=True,
        )
        assert [_del_timestamp(d) for d in bulk_mock.call_args.kwargs["actions"]] == [
            _del_timestamp(d) for d in actions
        ]


def test_mdelete_cache_store(
    es_embeddings_cache_fx: AsyncElasticsearchEmbeddingsCache,
) -> None:
    input = ["test_text1", "test_text2"]
    actions = [
        {"_op_type": "delete", "_id": es_embeddings_cache_fx._key(k)} for k in input
    ]
    es_embeddings_cache_fx._is_alias = False
    with patch("elasticsearch.helpers.bulk") as bulk_mock:
        es_embeddings_cache_fx.mdelete([])
        bulk_mock.assert_called_once()
        es_embeddings_cache_fx.mdelete(input)
        bulk_mock.assert_called_with(
            client=es_embeddings_cache_fx._es_client,
            actions=ANY,
            index="test_index",
            require_alias=False,
            refresh=True,
        )
        assert list(bulk_mock.call_args.kwargs["actions"]) == actions
