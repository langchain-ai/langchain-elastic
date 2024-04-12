import json
import uuid
from typing import Iterator

import pytest
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import message_to_dict

from langchain_elasticsearch.chat_history import ElasticsearchChatMessageHistory

from ._test_utilities import (
    clear_test_indices,
    create_es_client,
    read_env,
)

"""
cd tests/integration_tests
docker-compose up elasticsearch

By default runs against local docker instance of Elasticsearch.
To run against Elastic Cloud, set the following environment variables:
- ES_CLOUD_ID
- ES_USERNAME
- ES_PASSWORD
"""


class TestElasticsearch:
    @pytest.fixture
    def elasticsearch_connection(self) -> Iterator[dict]:
        params = read_env()
        es = create_es_client(params)

        yield params

        clear_test_indices(es)

    @pytest.fixture(scope="function")
    def index_name(self) -> str:
        """Return the index name."""
        return f"test_{uuid.uuid4().hex}"

    def test_memory_with_message_store(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test the memory with a message store."""
        # setup Elasticsearch as a message store
        message_history = ElasticsearchChatMessageHistory(
            **elasticsearch_connection, index=index_name, session_id="test-session"
        )

        memory = ConversationBufferMemory(
            memory_key="baz", chat_memory=message_history, return_messages=True
        )

        # add some messages
        memory.chat_memory.add_ai_message("This is me, the AI")
        memory.chat_memory.add_user_message("This is me, the human")

        # get the message history from the memory store and turn it into a json
        messages = memory.chat_memory.messages
        messages_json = json.dumps([message_to_dict(msg) for msg in messages])

        assert "This is me, the AI" in messages_json
        assert "This is me, the human" in messages_json

        # remove the record from Elasticsearch, so the next test run won't pick it up
        memory.chat_memory.clear()

        assert memory.chat_memory.messages == []
