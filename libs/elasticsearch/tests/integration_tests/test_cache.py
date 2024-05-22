from typing import Dict, Generator, Union
from unittest.mock import MagicMock

import pytest
from elasticsearch.helpers import BulkIndexError
from langchain.embeddings import CacheBackedEmbeddings
from langchain.globals import set_llm_cache
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models import BaseChatModel

from langchain_elasticsearch import ElasticsearchCache, ElasticsearchEmbeddingsCache
from tests.integration_tests._test_utilities import (
    clear_test_indices,
    create_es_client,
    read_env,
)


@pytest.fixture
def es_env_fx() -> Union[dict, Generator[dict, None, None]]:
    params = read_env()
    es = create_es_client(params)
    es.options(ignore_status=404).indices.delete(index="test_index1")
    es.options(ignore_status=404).indices.delete(index="test_index2")
    es.indices.create(index="test_index1")
    es.indices.create(index="test_index2")
    es.indices.put_alias(index="test_index1", name="test_alias")
    es.indices.put_alias(index="test_index2", name="test_alias", is_write_index=True)
    yield params
    es.options(ignore_status=404).indices.delete_alias(
        index="test_index1,test_index2,test_index3", name="test_alias"
    )
    clear_test_indices(es)
    return None


def test_index_llm_cache(es_env_fx: Dict, fake_chat_fx: BaseChatModel) -> None:
    cache = ElasticsearchCache(
        **es_env_fx, index_name="test_index1", metadata={"project": "test"}
    )
    es_client = cache._es_client
    set_llm_cache(cache)
    fake_chat_fx.invoke("test")
    assert es_client.count(index="test_index1")["count"] == 1
    fake_chat_fx.invoke("test")
    assert es_client.count(index="test_index1")["count"] == 1
    record = es_client.search(index="test_index1")["hits"]["hits"][0]["_source"]
    assert "test output" in record.get("llm_output", [""])[0]
    assert record.get("llm_input")
    assert record.get("timestamp")
    assert record.get("llm_params")
    assert record.get("metadata") == {"project": "test"}
    cache2 = ElasticsearchCache(
        **es_env_fx,
        index_name="test_index1",
        metadata={"project": "test"},
        store_input=False,
        store_input_params=False,
    )
    set_llm_cache(cache2)
    fake_chat_fx.invoke("test")
    assert es_client.count(index="test_index1")["count"] == 1
    fake_chat_fx.invoke("test2")
    assert es_client.count(index="test_index1")["count"] == 2
    fake_chat_fx.invoke("test2")
    records = [
        record["_source"]
        for record in es_client.search(index="test_index1")["hits"]["hits"]
    ]
    assert all("test output" in record.get("llm_output", [""])[0] for record in records)
    assert not all(record.get("llm_input", "") for record in records)
    assert all(record.get("timestamp", "") for record in records)
    assert not all(record.get("llm_params", "") for record in records)
    assert all(record.get("metadata") == {"project": "test"} for record in records)


def test_alias_llm_cache(es_env_fx: Dict, fake_chat_fx: BaseChatModel) -> None:
    cache = ElasticsearchCache(
        **es_env_fx, index_name="test_alias", metadata={"project": "test"}
    )
    es_client = cache._es_client
    set_llm_cache(cache)
    fake_chat_fx.invoke("test")
    assert es_client.count(index="test_index2")["count"] == 1
    fake_chat_fx.invoke("test2")
    assert es_client.count(index="test_index2")["count"] == 2
    es_client.indices.put_alias(
        index="test_index2", name="test_alias", is_write_index=False
    )
    es_client.indices.put_alias(
        index="test_index1", name="test_alias", is_write_index=True
    )
    fake_chat_fx.invoke("test3")
    assert es_client.count(index="test_index1")["count"] == 1
    fake_chat_fx.invoke("test2")
    assert es_client.count(index="test_index1")["count"] == 1
    es_client.indices.delete_alias(index="test_index2", name="test_alias")
    # we cache the response for prompt "test2" on both test_index1 and test_index2
    fake_chat_fx.invoke("test2")
    assert es_client.count(index="test_index1")["count"] == 2
    es_client.indices.put_alias(index="test_index2", name="test_alias")
    # we just test the latter scenario is working
    assert fake_chat_fx.invoke("test2")


def test_clear_llm_cache(es_env_fx: Dict, fake_chat_fx: BaseChatModel) -> None:
    cache = ElasticsearchCache(
        **es_env_fx, index_name="test_alias", metadata={"project": "test"}
    )
    es_client = cache._es_client
    set_llm_cache(cache)
    fake_chat_fx.invoke("test")
    fake_chat_fx.invoke("test2")
    es_client.indices.put_alias(
        index="test_index2", name="test_alias", is_write_index=False
    )
    es_client.indices.put_alias(
        index="test_index1", name="test_alias", is_write_index=True
    )
    fake_chat_fx.invoke("test3")
    assert es_client.count(index="test_alias")["count"] == 3
    cache.clear()
    assert es_client.count(index="test_alias")["count"] == 0


def test_hit_and_miss_cache_store(
    es_cache_store_fx: ElasticsearchEmbeddingsCache,
) -> None:
    store_mock = MagicMock(es_cache_store_fx)
    underlying_embeddings = FakeEmbeddings(size=3)
    cached_embeddings = CacheBackedEmbeddings(underlying_embeddings, store_mock)
    store_mock.mget.return_value = [None, None]
    assert all(cached_embeddings.embed_documents(["test_text1", "test_text2"]))
    store_mock.mget.assert_called_once()
    store_mock.mset.assert_called_once()
    store_mock.reset_mock()
    store_mock.mget.return_value = [None, [1.5, 2, 3.6]]
    assert all(cached_embeddings.embed_documents(["test_text1", "test_text2"]))
    store_mock.mget.assert_called_once()
    store_mock.mset.assert_called_once()
    assert len(store_mock.mset.call_args.args) == 1
    assert store_mock.mset.call_args.args[0][0][0] == "test_text1"
    store_mock.reset_mock()
    store_mock.mget.return_value = [[1.5, 2.3, 3], [1.5, 2, 3.6]]
    assert all(cached_embeddings.embed_documents(["test_text1", "test_text2"]))
    store_mock.mget.assert_called_once()
    store_mock.mset.assert_not_called()


def test_mdelete_cache_store(es_env_fx: Dict) -> None:
    store = ElasticsearchEmbeddingsCache(
        **es_env_fx, index_name="test_alias", metadata={"project": "test"}
    )

    recors = ["my little tests", "my little tests2", "my little tests3"]
    store.mset([(recors[0], [1, 2, 3]), (recors[1], [1, 2, 3]), (recors[2], [1, 2, 3])])

    assert store._es_client.count(index="test_alias")["count"] == 3

    store.mdelete(recors[:2])
    assert store._es_client.count(index="test_alias")["count"] == 1

    store.mdelete(recors[2:])
    assert store._es_client.count(index="test_alias")["count"] == 0

    with pytest.raises(BulkIndexError):
        store.mdelete(recors)


def test_mset_cache_store(es_env_fx: Dict) -> None:
    store = ElasticsearchEmbeddingsCache(
        **es_env_fx, index_name="test_alias", metadata={"project": "test"}
    )

    records = ["my little tests", "my little tests2", "my little tests3"]

    store.mset([(records[0], [1, 2, 3])])
    assert store._es_client.count(index="test_alias")["count"] == 1
    store.mset([(records[0], [1, 2, 3])])
    assert store._es_client.count(index="test_alias")["count"] == 1
    store.mset([(records[1], [1, 2, 3]), (records[2], [1, 2, 3])])
    assert store._es_client.count(index="test_alias")["count"] == 3


def test_mget_cache_store(es_env_fx: Dict) -> None:
    store_no_alias = ElasticsearchEmbeddingsCache(
        **es_env_fx,
        index_name="test_index3",
        metadata={"project": "test"},
        namespace="test",
    )

    records = ["my little tests", "my little tests2", "my little tests3"]
    docs = [(r, [0.1, 2, i]) for i, r in enumerate(records)]

    store_no_alias.mset(docs)
    assert store_no_alias._es_client.count(index="test_index3")["count"] == 3

    cached_records = store_no_alias.mget([d[0] for d in docs])
    assert all(cached_records)
    assert all([r == d[1] for r, d in zip(cached_records, docs)])

    store_alias = ElasticsearchEmbeddingsCache(
        **es_env_fx,
        index_name="test_alias",
        metadata={"project": "test"},
        namespace="test",
        maximum_duplicate_allowed=1,
    )

    store_alias.mset(docs)
    assert store_alias._es_client.count(index="test_alias")["count"] == 3

    cached_records = store_alias.mget([d[0] for d in docs])
    assert all(cached_records)
    assert all([r == d[1] for r, d in zip(cached_records, docs)])


def test_mget_cache_store_multiple_keys(es_env_fx: Dict) -> None:
    store_alias = ElasticsearchEmbeddingsCache(
        **es_env_fx,
        index_name="test_alias",
        metadata={"project": "test"},
        namespace="test",
        maximum_duplicate_allowed=2,
    )

    es_client = store_alias._es_client

    records = ["my little tests", "my little tests2", "my little tests3"]
    docs = [(r, [0.1, 2, i]) for i, r in enumerate(records)]

    store_alias.mset(docs)
    assert es_client.count(index="test_alias")["count"] == 3

    store_no_alias = ElasticsearchEmbeddingsCache(
        **es_env_fx,
        index_name="test_index3",
        metadata={"project": "test"},
        namespace="test",
        maximum_duplicate_allowed=1,
    )

    new_records = records + ["my little tests4", "my little tests5"]
    new_docs = [(r, [0.1, 2, i + 100]) for i, r in enumerate(new_records)]
    store_no_alias.mset(new_docs)
    assert es_client.count(index="test_index3")["count"] == 5

    es_client.indices.update_aliases(
        actions=[
            {
                "add": {
                    "index": "test_index3",
                    "alias": "test_alias",
                    "is_write_index": True,
                }
            },
            {
                "add": {
                    "index": "test_index2",
                    "alias": "test_alias",
                    "is_write_index": False,
                }
            },
        ]
    )

    cached_records = store_alias.mget([d[0] for d in new_docs])
    assert all(cached_records)
    assert len(cached_records) == 5
    assert es_client.count(index="test_alias")["count"] == 8
    assert cached_records[:3] != [d[1] for d in docs]
    assert cached_records == [d[1] for d in new_docs]
    assert all([r == d[1] for r, d in zip(cached_records, new_docs)])


def test_build_document_cache_store(es_env_fx: Dict) -> None:
    store = ElasticsearchEmbeddingsCache(
        **es_env_fx,
        index_name="test_alias",
        metadata={"project": "test"},
        namespace="test",
    )

    store.mset([("my little tests", [0.1, 2, 3])])
    record = store._es_client.search(index="test_alias")["hits"]["hits"][0]["_source"]
    assert record.get("metadata") == {"project": "test"}
    assert record.get("namespace") == "test"
    assert record.get("timestamp")
    assert record.get("llm_input") == "my little tests"
    assert record.get("vector_dump") == [0.1, 2, 3]
