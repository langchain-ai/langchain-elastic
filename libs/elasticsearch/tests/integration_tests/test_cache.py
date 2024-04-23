from typing import Dict, Generator, Union

import pytest
from langchain.globals import set_llm_cache
from langchain_core.language_models import BaseChatModel

from langchain_elasticsearch import ElasticsearchCache
from tests.integration_tests._test_utilities import (
    clear_test_indices,
    create_es_client,
    read_env,
)


@pytest.fixture
def es_env_fx() -> Union[dict, Generator[dict, None, None]]:
    params = read_env()
    es = create_es_client(params)
    es.indices.create(index='test_index1')
    es.indices.create(index='test_index2')
    es.indices.put_alias(index='test_index1', name='test_alias')
    es.indices.put_alias(index='test_index2', name='test_alias', is_write_index=True)
    yield params
    es.indices.delete_alias(index='test_index1,test_index2', name='test_alias')
    clear_test_indices(es)
    return None


def test_index(es_env_fx: Dict, fake_chat_fx: BaseChatModel) -> None:
    cache = ElasticsearchCache(
        **es_env_fx,
        es_index="test_index1",
        metadata={"project": "test"}
    )
    es_client = cache._es_client
    set_llm_cache(cache)
    fake_chat_fx.invoke("test")
    assert es_client.count(index="test_index1")['count'] == 1
    fake_chat_fx.invoke("test")
    assert es_client.count(index="test_index1")['count'] == 1
    record = es_client.search(index="test_index1")['hits']['hits'][0]['_source']
    assert "test output" in record.get('llm_output', [''])[0]
    assert record.get('llm_input')
    assert record.get('timestamp')
    assert record.get('llm_params')
    assert record.get('metadata') == {"project": "test"}
    cache2 = ElasticsearchCache(
        **es_env_fx,
        es_index="test_index1",
        metadata={"project": "test"},
        store_input=False,
        store_input_params=False
    )
    set_llm_cache(cache2)
    fake_chat_fx.invoke("test")
    assert es_client.count(index="test_index1")['count'] == 1
    fake_chat_fx.invoke("test2")
    assert es_client.count(index="test_index1")['count'] == 2
    fake_chat_fx.invoke("test2")
    records = [record["_source"]
               for record in es_client.search(index="test_index1")['hits']['hits']]
    assert all("test output" in record.get('llm_output', [''])[0] for record in records)
    assert not all(record.get('llm_input', '') for record in records)
    assert all(record.get('timestamp', '') for record in records)
    assert not all(record.get('llm_params', '') for record in records)
    assert all(record.get('metadata') == {"project": "test"} for record in records)


def test_alias(es_env_fx: Dict, fake_chat_fx: BaseChatModel) -> None:
    cache = ElasticsearchCache(
        **es_env_fx,
        es_index="test_alias",
        metadata={"project": "test"}
    )
    es_client = cache._es_client
    set_llm_cache(cache)
    fake_chat_fx.invoke("test")
    assert es_client.count(index="test_index2")['count'] == 1
    fake_chat_fx.invoke("test2")
    assert es_client.count(index="test_index2")['count'] == 2
    es_client.indices.put_alias(
        index='test_index2', name='test_alias', is_write_index=False)
    es_client.indices.put_alias(
        index='test_index1', name='test_alias', is_write_index=True)
    fake_chat_fx.invoke("test3")
    assert es_client.count(index="test_index1")['count'] == 1
    fake_chat_fx.invoke("test2")
    assert es_client.count(index="test_index1")['count'] == 1
    es_client.indices.delete_alias(index='test_index2', name='test_alias')
    # we cache the response for prompt "test2" on both test_index1 and test_index2
    fake_chat_fx.invoke("test2")
    assert es_client.count(index="test_index1")['count'] == 2
    es_client.indices.put_alias(index='test_index2', name='test_alias')
    # we just test the latter scenario is working
    assert fake_chat_fx.invoke("test2")


def test_clear(es_env_fx: Dict, fake_chat_fx: BaseChatModel) -> None:
    cache = ElasticsearchCache(
        **es_env_fx,
        es_index="test_alias",
        metadata={"project": "test"}
    )
    es_client = cache._es_client
    set_llm_cache(cache)
    fake_chat_fx.invoke("test")
    fake_chat_fx.invoke("test2")
    es_client.indices.put_alias(
        index='test_index2', name='test_alias', is_write_index=False)
    es_client.indices.put_alias(
        index='test_index1', name='test_alias', is_write_index=True)
    fake_chat_fx.invoke("test3")
    assert es_client.count(index="test_alias")['count'] == 3
    cache.clear()
    assert es_client.count(index="test_alias")['count'] == 0
