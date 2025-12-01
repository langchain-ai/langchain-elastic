from typing import Dict, Generator, Union

import pytest
from elasticsearch.helpers import BulkIndexError
from langchain_classic.embeddings.cache import _value_serializer
from langchain_core.globals import set_llm_cache
from langchain_core.language_models import BaseChatModel

from langchain_elasticsearch import (
    ElasticsearchCache,
    ElasticsearchEmbeddingsCache,
)

from ._test_utilities import clear_test_indices, create_es_client, read_env


@pytest.fixture
def es_env_fx() -> Union[dict, Generator]:
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
        index="test_index1,test_index2", name="test_alias"
    )
    clear_test_indices(es)
    es.close()


@pytest.mark.sync
def test_index_llm_cache(es_env_fx: Dict, fake_chat_fx: BaseChatModel) -> None:
    cache = ElasticsearchCache(
        **es_env_fx, index_name="test_index1", metadata={"project": "test"}
    )
    es_client = cache._es_client
    set_llm_cache(cache)
    fake_chat_fx.invoke("test")
    assert (es_client.count(index="test_index1"))["count"] == 1
    fake_chat_fx.invoke("test")
    assert (es_client.count(index="test_index1"))["count"] == 1
    record = (es_client.search(index="test_index1"))["hits"]["hits"][0]["_source"]
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
    assert (es_client.count(index="test_index1"))["count"] == 1
    fake_chat_fx.invoke("test2")
    assert (es_client.count(index="test_index1"))["count"] == 2
    fake_chat_fx.invoke("test2")
    records = [
        record["_source"]
        for record in (es_client.search(index="test_index1"))["hits"]["hits"]
    ]
    assert all("test output" in record.get("llm_output", [""])[0] for record in records)
    assert not all(record.get("llm_input", "") for record in records)
    assert all(record.get("timestamp", "") for record in records)
    assert not all(record.get("llm_params", "") for record in records)
    assert all(record.get("metadata") == {"project": "test"} for record in records)


@pytest.mark.sync
def test_alias_llm_cache(es_env_fx: Dict, fake_chat_fx: BaseChatModel) -> None:
    cache = ElasticsearchCache(
        **es_env_fx, index_name="test_alias", metadata={"project": "test"}
    )
    es_client = cache._es_client
    set_llm_cache(cache)
    fake_chat_fx.invoke("test")
    assert (es_client.count(index="test_index2"))["count"] == 1
    fake_chat_fx.invoke("test2")
    assert (es_client.count(index="test_index2"))["count"] == 2
    es_client.indices.put_alias(
        index="test_index2", name="test_alias", is_write_index=False
    )
    es_client.indices.put_alias(
        index="test_index1", name="test_alias", is_write_index=True
    )
    fake_chat_fx.invoke("test3")
    assert (es_client.count(index="test_index1"))["count"] == 1
    fake_chat_fx.invoke("test2")
    assert (es_client.count(index="test_index1"))["count"] == 1
    es_client.indices.delete_alias(index="test_index2", name="test_alias")
    # we cache the response for prompt "test2" on both test_index1 and test_index2
    fake_chat_fx.invoke("test2")
    assert (es_client.count(index="test_index1"))["count"] == 2
    es_client.indices.put_alias(index="test_index2", name="test_alias")
    # we just test the latter scenario is working
    assert fake_chat_fx.invoke("test2")


@pytest.mark.sync
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
    assert (es_client.count(index="test_alias"))["count"] == 3
    cache.clear()
    assert (es_client.count(index="test_alias"))["count"] == 0


@pytest.mark.sync
def test_mdelete_cache_store(es_env_fx: Dict) -> None:
    store = ElasticsearchEmbeddingsCache(
        **es_env_fx, index_name="test_alias", metadata={"project": "test"}
    )

    recors = ["my little tests", "my little tests2", "my little tests3"]
    store.mset(
        [
            (recors[0], _value_serializer([1, 2, 3])),
            (recors[1], _value_serializer([1, 2, 3])),
            (recors[2], _value_serializer([1, 2, 3])),
        ]
    )

    assert (store._es_client.count(index="test_alias"))["count"] == 3

    store.mdelete(recors[:2])
    assert (store._es_client.count(index="test_alias"))["count"] == 1

    store.mdelete(recors[2:])
    assert (store._es_client.count(index="test_alias"))["count"] == 0

    with pytest.raises(BulkIndexError):
        store.mdelete(recors)


@pytest.mark.sync
def test_mset_cache_store(es_env_fx: Dict) -> None:
    store = ElasticsearchEmbeddingsCache(
        **es_env_fx, index_name="test_alias", metadata={"project": "test"}
    )

    records = ["my little tests", "my little tests2", "my little tests3"]

    store.mset([(records[0], _value_serializer([1, 2, 3]))])
    assert (store._es_client.count(index="test_alias"))["count"] == 1
    store.mset([(records[0], _value_serializer([1, 2, 3]))])
    assert (store._es_client.count(index="test_alias"))["count"] == 1
    store.mset(
        [
            (records[1], _value_serializer([1, 2, 3])),
            (records[2], _value_serializer([1, 2, 3])),
        ]
    )
    assert (store._es_client.count(index="test_alias"))["count"] == 3


@pytest.mark.sync
def test_mget_cache_store(es_env_fx: Dict) -> None:
    store_no_alias = ElasticsearchEmbeddingsCache(
        **es_env_fx,
        index_name="test_index3",
        metadata={"project": "test"},
        namespace="test",
    )

    records = ["my little tests", "my little tests2", "my little tests3"]
    docs = [(r, _value_serializer([0.1, 2, i])) for i, r in enumerate(records)]

    store_no_alias.mset(docs)
    assert (store_no_alias._es_client.count(index="test_index3"))["count"] == 3

    cached_records = store_no_alias.mget([d[0] for d in docs])
    assert all(cached_records)
    assert all([r == d[1] for r, d in zip(cached_records, docs)])

    store_alias = ElasticsearchEmbeddingsCache(
        **es_env_fx,
        index_name="test_alias",
        metadata={"project": "test"},
        namespace="test",
        maximum_duplicates_allowed=1,
    )

    store_alias.mset(docs)
    assert (store_alias._es_client.count(index="test_alias"))["count"] == 3

    cached_records = store_alias.mget([d[0] for d in docs])
    assert all(cached_records)
    assert all([r == d[1] for r, d in zip(cached_records, docs)])


@pytest.mark.sync
def test_mget_cache_store_multiple_keys(es_env_fx: Dict) -> None:
    """verify the logic of deduplication of keys in the cache store"""

    store_alias = ElasticsearchEmbeddingsCache(
        **es_env_fx,
        index_name="test_alias",
        metadata={"project": "test"},
        namespace="test",
        maximum_duplicates_allowed=2,
    )

    es_client = store_alias._es_client

    records = ["my little tests", "my little tests2", "my little tests3"]
    docs = [(r, _value_serializer([0.1, 2, i])) for i, r in enumerate(records)]

    store_alias.mset(docs)
    assert (es_client.count(index="test_alias"))["count"] == 3

    store_no_alias = ElasticsearchEmbeddingsCache(
        **es_env_fx,
        index_name="test_index3",
        metadata={"project": "test"},
        namespace="test",
        maximum_duplicates_allowed=1,
    )

    new_records = records + ["my little tests4", "my little tests5"]
    new_docs = [
        (r, _value_serializer([0.1, 2, i + 100])) for i, r in enumerate(new_records)
    ]

    # store the same 3 previous records and 2 more in a fresh index
    store_no_alias.mset(new_docs)
    assert (es_client.count(index="test_index3"))["count"] == 5

    # update the alias to point to the new index and verify the cache
    es_client.indices.update_aliases(
        actions=[
            {
                "add": {
                    "index": "test_index3",
                    "alias": "test_alias",
                }
            }
        ]
    )

    # the alias now point to two indices that contains multiple records
    # of the same keys, the cache store should return the latest records.
    cached_records = store_alias.mget([d[0] for d in new_docs])
    assert all(cached_records)
    assert len(cached_records) == 5
    assert (es_client.count(index="test_alias"))["count"] == 8
    assert cached_records[:3] != [
        d[1] for d in docs
    ], "the first 3 records should be updated"
    assert cached_records == [
        d[1] for d in new_docs
    ], "new records should be returned and the updated ones"
    assert all([r == d[1] for r, d in zip(cached_records, new_docs)])
    es_client.options(ignore_status=404).indices.delete_alias(
        index="test_index3", name="test_alias"
    )


@pytest.mark.sync
def test_build_document_cache_store(es_env_fx: Dict) -> None:
    store = ElasticsearchEmbeddingsCache(
        **es_env_fx,
        index_name="test_alias",
        metadata={"project": "test"},
        namespace="test",
    )

    store.mset([("my little tests", _value_serializer([0.1, 2, 3]))])
    record = (store._es_client.search(index="test_alias"))["hits"]["hits"][0]["_source"]

    assert record.get("metadata") == {"project": "test"}
    assert record.get("namespace") == "test"
    assert record.get("timestamp")
    assert record.get("text_input") == "my little tests"
    assert record.get("vector_dump") == ElasticsearchEmbeddingsCache.encode_vector(
        _value_serializer([0.1, 2, 3])
    )
