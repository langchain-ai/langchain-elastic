import json
from typing import Any, AsyncGenerator, Dict, List, Union

import pytest
from elasticsearch.helpers import BulkIndexError
from langchain_core.globals import set_llm_cache
from langchain_core.language_models import BaseChatModel

from langchain_elasticsearch import (
    AsyncElasticsearchCache,
    AsyncElasticsearchEmbeddingsCache,
)

from ._test_utilities import clear_test_indices, create_es_client, read_env


def _value_serializer(value: List[float]) -> bytes:
    """Serialize embedding values to bytes (replaces private langchain function)."""
    return json.dumps(value).encode()


@pytest.fixture(autouse=True)
async def _close_async_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[None, None]:
    """Ensure cache clients close cleanly to avoid aiohttp warnings."""
    created_clients: List = []

    original_cache_init = AsyncElasticsearchCache.__init__
    original_store_init = AsyncElasticsearchEmbeddingsCache.__init__

    def wrapped_cache_init(self, *args: Any, **kwargs: Any) -> None:
        original_cache_init(self, *args, **kwargs)
        created_clients.append(self._es_client)

    def wrapped_store_init(self, *args: Any, **kwargs: Any) -> None:
        original_store_init(self, *args, **kwargs)
        created_clients.append(self._es_client)

    monkeypatch.setattr(AsyncElasticsearchCache, "__init__", wrapped_cache_init)
    monkeypatch.setattr(
        AsyncElasticsearchEmbeddingsCache, "__init__", wrapped_store_init
    )
    try:
        yield
    finally:
        for client in created_clients:
            close = getattr(client, "close", None)
            if close:
                try:
                    await close()
                except Exception:
                    pass
        monkeypatch.setattr(AsyncElasticsearchCache, "__init__", original_cache_init)
        monkeypatch.setattr(
            AsyncElasticsearchEmbeddingsCache, "__init__", original_store_init
        )


@pytest.fixture
async def es_env_fx() -> Union[dict, AsyncGenerator]:
    params = read_env()
    es = create_es_client(params)
    await es.options(ignore_status=404).indices.delete(index="test_index1")
    await es.options(ignore_status=404).indices.delete(index="test_index2")
    await es.indices.create(index="test_index1")
    await es.indices.create(index="test_index2")
    await es.indices.put_alias(index="test_index1", name="test_alias")
    await es.indices.put_alias(
        index="test_index2", name="test_alias", is_write_index=True
    )
    yield params
    await es.options(ignore_status=404).indices.delete_alias(
        index="test_index1,test_index2", name="test_alias"
    )
    await clear_test_indices(es)
    await es.close()


@pytest.mark.asyncio
async def test_index_llm_cache(es_env_fx: Dict, fake_chat_fx: BaseChatModel) -> None:
    cache = AsyncElasticsearchCache(
        **es_env_fx, index_name="test_index1", metadata={"project": "test"}
    )
    es_client = cache._es_client
    set_llm_cache(cache)
    await fake_chat_fx.ainvoke("test")
    assert (await es_client.count(index="test_index1"))["count"] == 1
    await fake_chat_fx.ainvoke("test")
    assert (await es_client.count(index="test_index1"))["count"] == 1
    record = (await es_client.search(index="test_index1"))["hits"]["hits"][0]["_source"]
    assert "test output" in record.get("llm_output", [""])[0]
    assert record.get("llm_input")
    assert record.get("timestamp")
    assert record.get("llm_params")
    assert record.get("metadata") == {"project": "test"}
    cache2 = AsyncElasticsearchCache(
        **es_env_fx,
        index_name="test_index1",
        metadata={"project": "test"},
        store_input=False,
        store_input_params=False,
    )
    set_llm_cache(cache2)
    await fake_chat_fx.ainvoke("test")
    assert (await es_client.count(index="test_index1"))["count"] == 1
    await fake_chat_fx.ainvoke("test2")
    assert (await es_client.count(index="test_index1"))["count"] == 2
    await fake_chat_fx.ainvoke("test2")
    records = [
        record["_source"]
        for record in (await es_client.search(index="test_index1"))["hits"]["hits"]
    ]
    assert all("test output" in record.get("llm_output", [""])[0] for record in records)
    assert not all(record.get("llm_input", "") for record in records)
    assert all(record.get("timestamp", "") for record in records)
    assert not all(record.get("llm_params", "") for record in records)
    assert all(record.get("metadata") == {"project": "test"} for record in records)


@pytest.mark.asyncio
async def test_alias_llm_cache(es_env_fx: Dict, fake_chat_fx: BaseChatModel) -> None:
    cache = AsyncElasticsearchCache(
        **es_env_fx, index_name="test_alias", metadata={"project": "test"}
    )
    es_client = cache._es_client
    set_llm_cache(cache)
    await fake_chat_fx.ainvoke("test")
    assert (await es_client.count(index="test_index2"))["count"] == 1
    await fake_chat_fx.ainvoke("test2")
    assert (await es_client.count(index="test_index2"))["count"] == 2
    await es_client.indices.put_alias(
        index="test_index2", name="test_alias", is_write_index=False
    )
    await es_client.indices.put_alias(
        index="test_index1", name="test_alias", is_write_index=True
    )
    await fake_chat_fx.ainvoke("test3")
    assert (await es_client.count(index="test_index1"))["count"] == 1
    await fake_chat_fx.ainvoke("test2")
    assert (await es_client.count(index="test_index1"))["count"] == 1
    await es_client.indices.delete_alias(index="test_index2", name="test_alias")
    # we cache the response for prompt "test2" on both test_index1 and test_index2
    await fake_chat_fx.ainvoke("test2")
    assert (await es_client.count(index="test_index1"))["count"] == 2
    await es_client.indices.put_alias(index="test_index2", name="test_alias")
    # we just test the latter scenario is working
    assert await fake_chat_fx.ainvoke("test2")


@pytest.mark.asyncio
async def test_clear_llm_cache(es_env_fx: Dict, fake_chat_fx: BaseChatModel) -> None:
    cache = AsyncElasticsearchCache(
        **es_env_fx, index_name="test_alias", metadata={"project": "test"}
    )
    es_client = cache._es_client
    set_llm_cache(cache)
    await fake_chat_fx.ainvoke("test")
    await fake_chat_fx.ainvoke("test2")
    await es_client.indices.put_alias(
        index="test_index2", name="test_alias", is_write_index=False
    )
    await es_client.indices.put_alias(
        index="test_index1", name="test_alias", is_write_index=True
    )
    await fake_chat_fx.ainvoke("test3")
    assert (await es_client.count(index="test_alias"))["count"] == 3
    await cache.aclear()
    assert (await es_client.count(index="test_alias"))["count"] == 0


@pytest.mark.asyncio
async def test_mdelete_cache_store(es_env_fx: Dict) -> None:
    store = AsyncElasticsearchEmbeddingsCache(
        **es_env_fx, index_name="test_alias", metadata={"project": "test"}
    )

    recors = ["my little tests", "my little tests2", "my little tests3"]
    await store.amset(
        [
            (recors[0], _value_serializer([1, 2, 3])),
            (recors[1], _value_serializer([1, 2, 3])),
            (recors[2], _value_serializer([1, 2, 3])),
        ]
    )

    assert (await store._es_client.count(index="test_alias"))["count"] == 3

    await store.amdelete(recors[:2])
    assert (await store._es_client.count(index="test_alias"))["count"] == 1

    await store.amdelete(recors[2:])
    assert (await store._es_client.count(index="test_alias"))["count"] == 0

    with pytest.raises(BulkIndexError):
        await store.amdelete(recors)


@pytest.mark.asyncio
async def test_mset_cache_store(es_env_fx: Dict) -> None:
    store = AsyncElasticsearchEmbeddingsCache(
        **es_env_fx, index_name="test_alias", metadata={"project": "test"}
    )

    records = ["my little tests", "my little tests2", "my little tests3"]

    await store.amset([(records[0], _value_serializer([1, 2, 3]))])
    assert (await store._es_client.count(index="test_alias"))["count"] == 1
    await store.amset([(records[0], _value_serializer([1, 2, 3]))])
    assert (await store._es_client.count(index="test_alias"))["count"] == 1
    await store.amset(
        [
            (records[1], _value_serializer([1, 2, 3])),
            (records[2], _value_serializer([1, 2, 3])),
        ]
    )
    assert (await store._es_client.count(index="test_alias"))["count"] == 3


@pytest.mark.asyncio
async def test_mget_cache_store(es_env_fx: Dict) -> None:
    store_no_alias = AsyncElasticsearchEmbeddingsCache(
        **es_env_fx,
        index_name="test_index3",
        metadata={"project": "test"},
        namespace="test",
    )

    records = ["my little tests", "my little tests2", "my little tests3"]
    docs = [(r, _value_serializer([0.1, 2, i])) for i, r in enumerate(records)]

    await store_no_alias.amset(docs)
    assert (await store_no_alias._es_client.count(index="test_index3"))["count"] == 3

    cached_records = await store_no_alias.amget([d[0] for d in docs])
    assert all(cached_records)
    assert all([r == d[1] for r, d in zip(cached_records, docs)])

    store_alias = AsyncElasticsearchEmbeddingsCache(
        **es_env_fx,
        index_name="test_alias",
        metadata={"project": "test"},
        namespace="test",
        maximum_duplicates_allowed=1,
    )

    await store_alias.amset(docs)
    assert (await store_alias._es_client.count(index="test_alias"))["count"] == 3

    cached_records = await store_alias.amget([d[0] for d in docs])
    assert all(cached_records)
    assert all([r == d[1] for r, d in zip(cached_records, docs)])


@pytest.mark.asyncio
async def test_mget_cache_store_multiple_keys(es_env_fx: Dict) -> None:
    """verify the logic of deduplication of keys in the cache store"""

    store_alias = AsyncElasticsearchEmbeddingsCache(
        **es_env_fx,
        index_name="test_alias",
        metadata={"project": "test"},
        namespace="test",
        maximum_duplicates_allowed=2,
    )

    es_client = store_alias._es_client

    records = ["my little tests", "my little tests2", "my little tests3"]
    docs = [(r, _value_serializer([0.1, 2, i])) for i, r in enumerate(records)]

    await store_alias.amset(docs)
    assert (await es_client.count(index="test_alias"))["count"] == 3

    store_no_alias = AsyncElasticsearchEmbeddingsCache(
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
    await store_no_alias.amset(new_docs)
    assert (await es_client.count(index="test_index3"))["count"] == 5

    # update the alias to point to the new index and verify the cache
    await es_client.indices.update_aliases(
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
    cached_records = await store_alias.amget([d[0] for d in new_docs])
    assert all(cached_records)
    assert len(cached_records) == 5
    assert (await es_client.count(index="test_alias"))["count"] == 8
    assert cached_records[:3] != [
        d[1] for d in docs
    ], "the first 3 records should be updated"
    assert cached_records == [
        d[1] for d in new_docs
    ], "new records should be returned and the updated ones"
    assert all([r == d[1] for r, d in zip(cached_records, new_docs)])
    await es_client.options(ignore_status=404).indices.delete_alias(
        index="test_index3", name="test_alias"
    )


@pytest.mark.asyncio
async def test_build_document_cache_store(es_env_fx: Dict) -> None:
    store = AsyncElasticsearchEmbeddingsCache(
        **es_env_fx,
        index_name="test_alias",
        metadata={"project": "test"},
        namespace="test",
    )

    await store.amset([("my little tests", _value_serializer([0.1, 2, 3]))])
    record = (await store._es_client.search(index="test_alias"))["hits"]["hits"][0][
        "_source"
    ]

    assert record.get("metadata") == {"project": "test"}
    assert record.get("namespace") == "test"
    assert record.get("timestamp")
    assert record.get("text_input") == "my little tests"
    assert record.get("vector_dump") == AsyncElasticsearchEmbeddingsCache.encode_vector(
        _value_serializer([0.1, 2, 3])
    )
