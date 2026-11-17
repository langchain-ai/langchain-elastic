import logging
from typing import Any, Awaitable, Dict, Iterable
from elasticsearch import AsyncElasticsearch
from langgraph.store.base import Result, Op
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.elasticsearch.base import BaseElasticsearchMemoryStore, ElasticsearchIndexConfig
from langchain_elasticsearch.client import create_async_elasticsearch_client

logger = logging.getLogger(__name__)

class AsyncElasticsearchMemoryStore(BaseElasticsearchMemoryStore[AsyncElasticsearch], AsyncBatchedBaseStore):
    def __init__(self, 
                 es_connection: AsyncElasticsearch | None = None,
                 es_url: str | None = None,
                 es_cloud_id: str | None = None,
                 es_user: str | None = None,
                 es_api_key: str | None = None,
                 es_password: str | None = None,
                 es_params: Dict[str, Any] | None = None,
                 index_config: ElasticsearchIndexConfig | None = None,
                ):
        if not es_connection:
            es_connection = create_async_elasticsearch_client(
                url=es_url,
                cloud_id=es_cloud_id,
                api_key=es_api_key,
                username=es_user,
                password=es_password,
                params=es_params,
            )

        super(BaseElasticsearchMemoryStore[AsyncElasticsearch], self).__init__(es_connection=es_connection,
                         index_config=index_config)

    async def abatch(self, ops: Iterable[Op]) -> Awaitable[list[Result]]:
        return await self.aprocess(ops) 