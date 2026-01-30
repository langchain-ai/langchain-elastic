import json
import logging
from time import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

from langchain_elasticsearch._utilities import async_with_user_agent_header
from langchain_elasticsearch.client import create_async_elasticsearch_client

if TYPE_CHECKING:
    from elasticsearch import AsyncElasticsearch

logger = logging.getLogger(__name__)


class AsyncElasticsearchChatMessageHistory(BaseChatMessageHistory):
    """`Elasticsearch` chat message history.

    Stores chat message history in Elasticsearch for persistence across sessions.

    Setup:
        Install `langchain_elasticsearch` and start Elasticsearch locally using
        the start-local script.

        ```bash
        pip install -qU langchain_elasticsearch
        curl -fsSL https://elastic.co/start-local | sh
        ```

        This will create an `elastic-start-local` folder. To start Elasticsearch
        and Kibana:
        ```bash
        cd elastic-start-local
        ./start.sh
        ```

        Elasticsearch will be available at `http://localhost:9200`. The password
        for the `elastic` user and API key are stored in the `.env` file in the
        `elastic-start-local` folder.

    Instantiate:
        ```python
        from langchain_elasticsearch import ElasticsearchChatMessageHistory

        history = ElasticsearchChatMessageHistory(
            index="chat-history",
            session_id="user-123",
            es_url="http://localhost:9200"
        )
        ```

        **Instantiate with API key (URL):**
            ```python
            from langchain_elasticsearch import ElasticsearchChatMessageHistory

            history = ElasticsearchChatMessageHistory(
                index="chat-history",
                session_id="user-123",
                es_url="http://localhost:9200",
                es_api_key="your-api-key"
            )
            ```

        **Instantiate with username/password (URL):**
            ```python
            from langchain_elasticsearch import ElasticsearchChatMessageHistory

            history = ElasticsearchChatMessageHistory(
                index="chat-history",
                session_id="user-123",
                es_url="http://localhost:9200",
                es_user="elastic",
                es_password="password"
            )
        ```

        If you want to use a cloud hosted Elasticsearch instance, you can pass in the
        es_cloud_id argument instead of the es_url argument.

        **Instantiate from cloud (with API key):**
            ```python
            from langchain_elasticsearch import ElasticsearchChatMessageHistory

            history = ElasticsearchChatMessageHistory(
                index="chat-history",
                session_id="user-123",
                es_cloud_id="<cloud_id>",
                es_api_key="your-api-key"
            )
            ```

        You can also connect to an existing Elasticsearch instance by passing in a
        pre-existing Elasticsearch connection via the client argument.

        **Instantiate from existing connection:**
            ```python
            from langchain_elasticsearch import ElasticsearchChatMessageHistory
            from elasticsearch import Elasticsearch

            client = Elasticsearch("http://localhost:9200")
            history = ElasticsearchChatMessageHistory(
                index="chat-history",
                session_id="user-123",
                client=client
            )
            ```

    Add messages:
        ```python
        from langchain_core.messages import HumanMessage, AIMessage

        history.add_message(HumanMessage(content="Hello!"))
        history.add_message(AIMessage(content="Hi there! How can I help?"))
        ```

    Get messages:
        ```python
        messages = history.messages
        for msg in messages:
            print(f"{msg.type}: {msg.content}")
        ```

    Clear history:
        ```python
        history.clear()
        ```

    For synchronous applications, use the `ElasticsearchChatMessageHistory` class.
    For asynchronous applications, use the `AsyncElasticsearchChatMessageHistory`
    class.
    """  # noqa: E501

    def __init__(
        self,
        index: str,
        session_id: str,
        *,
        ensure_ascii: Optional[bool] = True,
        client: Optional["AsyncElasticsearch"] = None,
        es_url: Optional[str] = None,
        es_cloud_id: Optional[str] = None,
        es_user: Optional[str] = None,
        es_api_key: Optional[str] = None,
        es_password: Optional[str] = None,
    ):
        """Initialize the ElasticsearchChatMessageHistory instance.

        Args:
            index (str): Name of the Elasticsearch index to use for storing
                messages.
            session_id (str): Arbitrary key that is used to store the messages
                of a single chat session.
            ensure_ascii (bool, optional): Used to escape ASCII symbols in
                json.dumps. Defaults to True.
            client (AsyncElasticsearch, optional): Pre-existing Elasticsearch
                connection. Either provide this OR credentials.
            es_url (str, optional): URL of the Elasticsearch instance to
                connect to.
            es_cloud_id (str, optional): Cloud ID of the Elasticsearch instance.
            es_user (str, optional): Username to use when connecting to
                Elasticsearch.
            es_api_key (str, optional): API key to use when connecting to
                Elasticsearch.
            es_password (str, optional): Password to use when connecting to
                Elasticsearch.
        """
        self.index: str = index
        self.session_id: str = session_id
        self.ensure_ascii = ensure_ascii

        # Accept either client OR credentials (one required)
        if client is not None:
            es_connection = client
        elif es_url is not None or es_cloud_id is not None:
            es_connection = create_async_elasticsearch_client(
                url=es_url,
                username=es_user,
                password=es_password,
                cloud_id=es_cloud_id,
                api_key=es_api_key,
            )
        else:
            raise ValueError(
                "Either 'client' or credentials (es_url, es_cloud_id, etc.) "
                "must be provided."
            )

        self.client = async_with_user_agent_header(es_connection, "langchain-py-ms")
        self.created = False

    async def create_if_missing(self) -> None:
        if not self.created:
            if await self.client.indices.exists(index=self.index):
                logger.debug(
                    (
                        f"Chat history index {self.index} already exists, "
                        "skipping creation."
                    )
                )
            else:
                logger.debug(f"Creating index {self.index} for storing chat history.")

                await self.client.indices.create(
                    index=self.index,
                    mappings={
                        "properties": {
                            "session_id": {"type": "keyword"},
                            "created_at": {"type": "date"},
                            "history": {"type": "text"},
                        }
                    },
                )
            self.created = True

    async def aget_messages(self) -> List[BaseMessage]:  # type: ignore[override]
        """Retrieve the messages from Elasticsearch"""
        from elasticsearch import ApiError

        await self.create_if_missing()

        search_after: Dict[str, Any] = {}
        items = []
        while True:
            try:
                result = await self.client.search(
                    index=self.index,
                    query={"term": {"session_id": self.session_id}},
                    sort="created_at:asc",
                    size=100,
                    **search_after,
                )
            except ApiError as err:
                logger.error(f"Could not retrieve messages from Elasticsearch: {err}")
                raise err

            if result and len(result["hits"]["hits"]) > 0:
                items += [
                    json.loads(document["_source"]["history"])
                    for document in result["hits"]["hits"]
                ]
                search_after = {"search_after": result["hits"]["hits"][-1]["sort"]}
            else:
                break

        return messages_from_dict(items)

    async def aadd_message(self, message: BaseMessage) -> None:
        """Add messages to the chat session in Elasticsearch"""
        try:
            from elasticsearch import ApiError

            await self.create_if_missing()
            await self.client.index(
                index=self.index,
                document={
                    "session_id": self.session_id,
                    "created_at": round(time() * 1000),
                    "history": json.dumps(
                        message_to_dict(message),
                        ensure_ascii=bool(self.ensure_ascii),
                    ),
                },
                refresh=True,
            )
        except ApiError as err:
            logger.error(f"Could not add message to Elasticsearch: {err}")
            raise err

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        for message in messages:
            await self.aadd_message(message)

    async def aclear(self) -> None:
        """Clear session memory in Elasticsearch"""
        try:
            from elasticsearch import ApiError

            await self.create_if_missing()
            await self.client.delete_by_query(
                index=self.index,
                query={"term": {"session_id": self.session_id}},
                refresh=True,
            )
        except ApiError as err:
            logger.error(f"Could not clear session memory in Elasticsearch: {err}")
            raise err
