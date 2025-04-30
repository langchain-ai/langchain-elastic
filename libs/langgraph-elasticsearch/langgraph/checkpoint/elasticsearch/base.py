from abc import abstractmethod
import asyncio
from typing import Any, AsyncIterator, Dict, Generic, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union
from elasticsearch import AsyncElasticsearch, Elasticsearch
from langchain_core.runnables import ConfigurableFieldSpec, RunnableConfig
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
from langgraph.checkpoint.serde.base import SerializerProtocol, maybe_add_typed_methods
from langgraph.checkpoint.serde.types import (
    ERROR,
    INTERRUPT,
    RESUME,
    SCHEDULED,
    ChannelProtocol,
    SendProtocol,
)

from langgraph.checkpoint.elasticsearch.configurable import Configurable, get_configurable
T = TypeVar("T", bound=Union[Elasticsearch, AsyncElasticsearch])

class BaseElasticsearchSaver(BaseCheckpointSaver[str], Generic[T]):
    es_connection: T
    index_checkpoints: str = "langgraph-checkpoints"
    index_writes: str = "langgraph-writes"
    index_blobs: str = "langgraph-blobs"

    def __init__(
        self,
        es_connection: T,
        *,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        self.es_connection = es_connection

    def _extract_hits(result: Dict[str, Any]) -> list[Dict[str, Any]]:
        hits = result.get("hits", {}).get("hits", [])
        if not hits and (source := result.get("_source")):
            hits = [{"_source": source}]
        return hits

    def _build_clauses(configurable: Configurable) -> list[Dict[str, Dict[str, str]]]:
        clauses = [
            {"term": {"thread_id": configurable.thread_id}},
            {"term": {"checkpoint_ns": configurable.checkpoint_ns}},
        ]
        if configurable.checkpoint_id:
            clauses.append({"term": {"checkpoint_id": configurable.checkpoint_id}})
        return clauses
    
    def _build_query_checkpoint(self, configurable: Configurable) -> list[Dict[str, Dict[str, str]]]:
         return {
                    "query": {"bool": {"must": self._build_clauses(configurable)}},
                    "sort": {"checkpoint_id": {"order": "desc"}},
                }
    
    def _build_query_writes(self, configurable: Configurable) -> list[Dict[str, Dict[str, str]]]:
        return {
                "query": {"bool": {"must": self._build_clauses(configurable)}},
                "sort": [
                    {"task_id": {"order": "asc"}},
                    {"idx": {"order": "asc"}},
                ],
        }
    
    def _build_query_parent(self, configurable: Configurable, parent_checkpoint_id: str) -> list[Dict[str, Dict[str, str]]]:
        return {"query": {
                    "bool": {
                        "must": [
                            {"term": {"thread_id": configurable.thread_id}},
                            {"term": {"checkpoint_ns": configurable.checkpoint_ns}},
                            {"term": {"parent_checkpoint_id": parent_checkpoint_id}},
                        ],
                        "filter": [
                            {"term": {"task_type": "TASKS"}}  # Filtro adicional baseado na lógica original
                        ]
                    }
                },
                "sort": [
                    {"field_3": {"order": "asc"}},  # Substitua por w[3]
                    {"field_0": {"order": "asc"}},  # Substitua por w[0]
                    {"field_4": {"order": "asc"}},  # Substitua por w[4]
                ],
                "size": 100  # Ajuste o tamanho conforme necessário
            }

    def _body_upsert(self, 
                       configurable: Configurable,
                       checkpoint: Checkpoint,
                       metadata: CheckpointMetadata, ) -> Dict[str, Any]:
    
        return {
            "doc": {
                "checkpoint": self.serde.dumps_typed(checkpoint),
                "metadata": self.serde.dumps_typed(metadata),
            },
            "upsert": {
                "thread_id": configurable.thread_id,
                "checkpoint_ns": configurable.checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
                "checkpoint": self.serde.dumps_typed(checkpoint),
                "metadata": self.serde.dumps_typed(metadata),
                "parent_checkpoint_id": configurable.checkpoint_id,
            }
        }

    def _load_checkpoint(self, saved, sends) -> Checkpoint:
        (
            type,
            checkpoint,
        ) = saved
        return {
            **self.serde.loads_typed((type, checkpoint)),
            "pending_sends": [self.serde.loads_typed(s[2]) for s in sends],
        }
    
    def _load_pending_writes(self, writes) -> List[PendingWrite]:
        return [
                (id, c, self.serde.loads_typed(v)) for id, c, v, _ in writes
            ]

    def _load_metadata(self, saved) -> CheckpointMetadata:
        (metadata) = saved
        return self.serde.loads_typed(metadata)
    
    def _load_parent_config(self, configurable: Configurable, parent_checkpoint_id) -> RunnableConfig:
                return {
                    "configurable": {
                        "thread_id": configurable.thread_id,
                        "checkpoint_ns": configurable.checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                } if parent_checkpoint_id else None,

    @abstractmethod
    def _search_checkpoint(self, configurable: Configurable) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _search_writes(self, configurable: Configurable) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _search_parent(self, configurable: Configurable, parent_checkpoint_id) -> Dict[str, Any]:
        raise NotImplementedError

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        configurable = get_configurable(config)

        if saved := await self._search_checkpoint(configurable):
            checkpoint_id, parent_checkpoint_id = saved
            
            configurable = get_configurable({
                "configurable": {
                    "thread_id": configurable.thread_id,
                    "checkpoint_ns": configurable.checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }) if not get_checkpoint_id(config) else get_configurable(config)

            sends = await self._search_parent(configurable, parent_checkpoint_id)

            writes = await self._search_writes(configurable)

            config["configurable"] = configurable.to_dict()

            return CheckpointTuple(
                config=config,
                checkpoint=self._load_checkpoint(saved, sends),
                metadata=self._load_metadata(saved),
                pending_writes=self._load_pending_writes(writes),
                parent_config=self._load_parent_config(configurable, parent_checkpoint_id),
            )
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        configurable = get_configurable(config)

        await self.es_connection.update(
            index=self.index_checkpoints,
            id=f"{configurable.thread_id}-{configurable.checkpoint_ns}-{checkpoint['id']}",
            body=self._body_upsert(configurable, checkpoint, metadata),
        )

        config["configurable"]["checkpoint_id"] = checkpoint["id"]

        return config













    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints that match the given criteria.

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria.
            before (Optional[RunnableConfig]): List checkpoints created before this configuration.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Returns:
            Iterator[CheckpointTuple]: Iterator of matching checkpoint tuples.

        Raises:
            NotImplementedError: Implement this method in your custom checkpoint saver.
        """
        raise NotImplementedError
    

    
    # async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
    #     """Asynchronously fetch a checkpoint tuple using the given configuration.

    #     Args:
    #         config (RunnableConfig): Configuration specifying which checkpoint to retrieve.

    #     Returns:
    #         Optional[CheckpointTuple]: The requested checkpoint tuple, or None if not found.

    #     Raises:
    #         NotImplementedError: Implement this method in your custom checkpoint saver.
    #     """
    #     raise NotImplementedError
    
    # async def alist(
    #     self,
    #     config: Optional[RunnableConfig],
    #     *,
    #     filter: Optional[Dict[str, Any]] = None,
    #     before: Optional[RunnableConfig] = None,
    #     limit: Optional[int] = None,
    # ) -> AsyncIterator[CheckpointTuple]:
    #     """Asynchronously list checkpoints that match the given criteria.

    #     Args:
    #         config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
    #         filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
    #         before (Optional[RunnableConfig]): List checkpoints created before this configuration.
    #         limit (Optional[int]): Maximum number of checkpoints to return.

    #     Returns:
    #         AsyncIterator[CheckpointTuple]: Async iterator of matching checkpoint tuples.

    #     Raises:
    #         NotImplementedError: Implement this method in your custom checkpoint saver.
    #     """
    #     raise NotImplementedError
    #     yield

    # async def aput(
    #     self,
    #     config: RunnableConfig,
    #     checkpoint: Checkpoint,
    #     metadata: CheckpointMetadata,
    #     new_versions: ChannelVersions,
    # ) -> RunnableConfig:
    #     """Asynchronously store a checkpoint with its configuration and metadata.

    #     Args:
    #         config (RunnableConfig): Configuration for the checkpoint.
    #         checkpoint (Checkpoint): The checkpoint to store.
    #         metadata (CheckpointMetadata): Additional metadata for the checkpoint.
    #         new_versions (ChannelVersions): New channel versions as of this write.

    #     Returns:
    #         RunnableConfig: Updated configuration after storing the checkpoint.

    #     Raises:
    #         NotImplementedError: Implement this method in your custom checkpoint saver.
    #     """
    #     raise NotImplementedError
    
    # async def aput_writes(
    #     self,
    #     config: RunnableConfig,
    #     writes: Sequence[Tuple[str, Any]],
    #     task_id: str,
    #     task_path: str = "",
    # ) -> None:
    #     """Asynchronously store intermediate writes linked to a checkpoint.

    #     Args:
    #         config (RunnableConfig): Configuration of the related checkpoint.
    #         writes (List[Tuple[str, Any]]): List of writes to store.
    #         task_id (str): Identifier for the task creating the writes.
    #         task_path (str): Path of the task creating the writes.

    #     Raises:
    #         NotImplementedError: Implement this method in your custom checkpoint saver.
    #     """
    #     raise NotImplementedError
    
    def get_next_version(self, current: Optional[str], channel: ChannelProtocol) -> str:
        """Generate the next version ID for a channel.

        Default is to use integer versions, incrementing by 1. If you override, you can use str/int/float versions,
        as long as they are monotonically increasing.

        Args:
            current (Optional[V]): The current version identifier (int, float, or str).
            channel (BaseChannel): The channel being versioned.

        Returns:
            V: The next version identifier, which must be increasing.
        """
        if isinstance(current, str):
            raise NotImplementedError
        elif current is None:
            return 1
        else:
            return current + 1
        




