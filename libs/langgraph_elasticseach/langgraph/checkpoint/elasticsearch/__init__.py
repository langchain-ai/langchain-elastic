import asyncio
from typing import Any, Dict, Generic, Optional
from elasticsearch import Elasticsearch
from langchain_elasticsearch.client import create_elasticsearch_client
from langgraph.checkpoint.elasticsearch.base import BaseElasticsearchSaver
from langgraph.checkpoint.serde.base import SerializerProtocol, maybe_add_typed_methods
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointTuple, get_checkpoint_id
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    get_checkpoint_id,
    CheckpointTuple,
    PendingWrite
)

from langgraph.checkpoint.elasticsearch.configurable import Configurable, get_configurable
from langgraph.util import syncify

class ElasticsearchSaver(BaseElasticsearchSaver[Elasticsearch]):
    def __init__(self,
                 es_connection: Elasticsearch | None = None,
                 es_url: str | None = None,
                 es_cloud_id: str | None = None,
                 es_user: str | None = None,
                 es_api_key: str | None = None,
                 es_password: str | None = None,
                 es_params: Dict[str, Any] | None = None,
                 *,
                serde: Optional[SerializerProtocol] = None,):
        if not es_connection:
            es_connection = create_elasticsearch_client(
                url=es_url,
                cloud_id=es_cloud_id,
                api_key=es_api_key,
                username=es_user,
                password=es_password,
                params=es_params,
            )
        if not es_connection:
            raise ValueError("No Elasticsearch connection provided.")
        
        try:
            es_connection.info()
        except Exception as e:
            raise ValueError(f"Failed to connect to Elasticsearch: {e}")

        super().__init__(es_connection=es_connection, serde=serde)

    def _search_checkpoint(self, configurable: Configurable) -> Dict[str, Any]:
        result_checkpoint = self.es_connection.search(
            index=self.index_checkpoints,
            body=self._build_query_checkpoint(configurable),
            size=1,
        )

        return self._extract_hits(result_checkpoint)
    
    def _search_writes(self, configurable: Configurable) -> Dict[str, Any]:
        result_writes = self.es_connection.search(
            index=self.index_writes,
            body=self._build_query_writes(configurable),
        )

        return self._extract_hits(result_writes)
    
    def _search_parent(self, configurable: Configurable, parent_checkpoint_id) -> Dict[str, Any]:
        if not parent_checkpoint_id:
            return []

        result_parent = self.es_connection.search(
            index=self.index_writes,
            body=self._build_query_parent(configurable, parent_checkpoint_id),
        )

        return self._extract_hits(result_parent)

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(super().aget_tuple(config))
    
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        raise NotImplementedError("Not supported in ElasticsearchSaver, use ElasticsearchSaver.get_tuple instead.")

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return syncify(self.aput, config, checkpoint, metadata, new_versions)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        raise NotImplementedError("Not supported in ElasticsearchSaver, use ElasticsearchSaver.put instead.")




    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (List[Tuple[str, Any]]): List of writes to store.
            task_id (str): Identifier for the task creating the writes.
            task_path (str): Path of the task creating the writes.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError