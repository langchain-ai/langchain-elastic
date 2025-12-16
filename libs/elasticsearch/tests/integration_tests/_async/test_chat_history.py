import uuid
from typing import AsyncIterator

import pytest
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, message_to_dict

from langchain_elasticsearch.chat_history import AsyncElasticsearchChatMessageHistory

from ._test_utilities import clear_test_indices, create_es_client, read_env

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
    async def elasticsearch_connection(self) -> AsyncIterator[dict]:
        params = read_env()
        es = create_es_client(params)

        yield params

        await clear_test_indices(es)
        await es.close()

    @pytest.fixture(scope="function")
    def index_name(self) -> str:
        """Return the index name."""
        return f"test_{uuid.uuid4().hex}"

    async def test_memory_with_message_store(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test the memory with a message store."""
        # setup Elasticsearch as a message store
        message_history = AsyncElasticsearchChatMessageHistory(
            **elasticsearch_connection, index=index_name, session_id="test-session"
        )

        memory = ConversationBufferMemory(
            memory_key="baz", chat_memory=message_history, return_messages=True
        )

        # add some messages
        await memory.chat_memory.aadd_messages(
            [
                AIMessage("This is me, the AI (1)"),
                HumanMessage("This is me, the human (1)"),
                AIMessage("This is me, the AI (2)"),
                HumanMessage("This is me, the human (2)"),
                AIMessage("This is me, the AI (3)"),
                HumanMessage("This is me, the human (3)"),
                AIMessage("This is me, the AI (4)"),
                HumanMessage("This is me, the human (4)"),
                AIMessage("This is me, the AI (5)"),
                HumanMessage("This is me, the human (5)"),
                AIMessage("This is me, the AI (6)"),
                HumanMessage("This is me, the human (6)"),
                AIMessage("This is me, the AI (7)"),
                HumanMessage("This is me, the human (7)"),
            ]
        )

        # get the message history from the memory store and turn it into a json
        messages = [
            message_to_dict(msg) for msg in (await memory.chat_memory.aget_messages())
        ]

        assert len(messages) == 14
        for i in range(7):
            assert messages[i * 2]["data"]["content"] == f"This is me, the AI ({i+1})"
            assert (
                messages[i * 2 + 1]["data"]["content"]
                == f"This is me, the human ({i+1})"
            )

        # remove the record from Elasticsearch, so the next test run won't pick it up
        await memory.chat_memory.aclear()

        assert await memory.chat_memory.aget_messages() == []
