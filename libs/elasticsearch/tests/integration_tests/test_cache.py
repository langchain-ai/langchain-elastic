from unittest.mock import MagicMock

from langchain.globals import set_llm_cache
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

from langchain_elasticsearch.cache import ElasticsearchCache


def test_hit_and_miss(
    es_cache_fx: ElasticsearchCache, fake_chat_fx: BaseChatModel
) -> None:
    cache_mock = MagicMock(es_cache_fx)
    set_llm_cache(cache_mock)
    cache_mock.lookup.return_value = None
    fake_chat_fx.invoke("test")
    cache_mock.lookup.assert_called_once()
    cache_mock.update.assert_called_once()
    cache_mock.reset_mock()
    cache_mock.lookup.return_value = [
        ChatGeneration(message=AIMessage(content="test output"))
    ]
    fake_chat_fx.invoke("test")
    cache_mock.lookup.assert_called_once()
    cache_mock.update.assert_not_called()
