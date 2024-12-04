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
    """Chat message history that stores history in Elasticsearch.

    Args:
        es_url: URL of the Elasticsearch instance to connect to.
        es_cloud_id: Cloud ID of the Elasticsearch instance to connect to.
        es_user: Username to use when connecting to Elasticsearch.
        es_password: Password to use when connecting to Elasticsearch.
        es_api_key: API key to use when connecting to Elasticsearch.
        es_connection: Optional pre-existing Elasticsearch connection.
        esnsure_ascii: Used to escape ASCII symbols in json.dumps. Defaults to True.
        index: Name of the index to use.
        session_id: Arbitrary key that is used to store the messages
            of a single chat session.

    For synchronous applications, use the `ElasticsearchChatMessageHistory` class.
    For asyhchronous applications, use the `AsyncElasticsearchChatMessageHistory` class.
    """

    def __init__(
        self,
        index: str,
        session_id: str,
        *,
        es_connection: Optional["AsyncElasticsearch"] = None,
        es_url: Optional[str] = None,
        es_cloud_id: Optional[str] = None,
        es_user: Optional[str] = None,
        es_api_key: Optional[str] = None,
        es_password: Optional[str] = None,
        esnsure_ascii: Optional[bool] = True,
    ):
        self.index: str = index
        self.session_id: str = session_id
        self.ensure_ascii = esnsure_ascii

        # Initialize Elasticsearch client from passed client arg or connection info
        if es_connection is not None:
            self.client = es_connection
        elif es_url is not None or es_cloud_id is not None:
            try:
                self.client = create_async_elasticsearch_client(
                    url=es_url,
                    username=es_user,
                    password=es_password,
                    cloud_id=es_cloud_id,
                    api_key=es_api_key,
                )
            except Exception as err:
                logger.error(f"Error connecting to Elasticsearch: {err}")
                raise err
        else:
            raise ValueError(
                """Either provide a pre-existing Elasticsearch connection, \
                or valid credentials for creating a new connection."""
            )

        self.client = async_with_user_agent_header(self.client, "langchain-py-ms")
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
