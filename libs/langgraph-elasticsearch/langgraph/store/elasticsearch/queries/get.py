import logging
from typing import Any, Iterable, Optional
from langchain_core.runnables.config import run_in_executor
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import GetOp, Item
from langgraph.store.elasticsearch.queries.base import ElasticQuery, namespace_to_text

logger = logging.getLogger(__name__)

class QueryGet(ElasticQuery[GetOp, Item]):
    """
    Query class for handling 'Get' operations in Elasticsearch.
    """

    def get_by_id(self, op: GetOp) -> list[Item]:
        """
        Retrieve an item by its ID from Elasticsearch.

        Args:
            op (GetOp): The operation containing the namespace and key.

        Returns:
            list[Item]: The retrieved items.
        """
        response = self.es_connection.get(
            index=self.store_index_name,
            id=namespace_to_text(op.namespace + (op.key,))
        )
        return self.es_to_items(response)

    async def aget_by_id(self, op: GetOp) -> list[Item]:
        """
        Asynchronously retrieve an item by its ID from Elasticsearch.

        Args:
            op (GetOp): The operation containing the namespace and key.

        Returns:
            list[Item]: The retrieved items.
        """
        return await run_in_executor(None, self.get_by_id, op)

    def invoke(
        self, input: Iterable[GetOp], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> list[Item]:
        """
        Invoke the get operation synchronously for a batch of inputs.

        Args:
            input (Iterable[GetOp]): The batch of get operations.
            config (Optional[RunnableConfig]): Optional configuration for the operation.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            list[Item]: The retrieved items.
        """
        return self._execute_operations(input, sync_func=self.get_by_id)
    
    async def ainvoke(
        self, input: Iterable[GetOp], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> list[Item]:
        """
        Invoke the get operation asynchronously for a batch of inputs.

        Args:
            input (Iterable[GetOp]): The batch of get operations.
            config (Optional[RunnableConfig]): Optional configuration for the operation.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            list[Item]: The retrieved items.
        """
        return await self._aexecute_operations(input, sync_func=self.aget_by_id)