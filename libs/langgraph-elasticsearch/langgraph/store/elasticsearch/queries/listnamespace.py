import logging
from typing import (
    Any,
    Iterable,
    Optional, 
)
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import (
    ListNamespacesOp, 
    Item, 
    Result,
)
from langgraph.store.elasticsearch.queries.base import ElasticQuery, namespace_to_text

logger = logging.getLogger(__name__)

class QueryListNamespaces(ElasticQuery[ListNamespacesOp, list[Result]]):
    def _query(self, op: ListNamespacesOp) -> dict:
        """Build the Elasticsearch query for listing namespaces."""
        return self.build_query(
            filters=[
                {
                    "namespace": {
                        "$contain": f"{namespace_to_text(match.path)}*" 
                            if match.match_type == "prefix" 
                            else f"*{namespace_to_text(match.path)}"
                    }
                }
                for match in op.match_conditions
            ],
            aggs={
                "group_by_namespace": {
                    "terms": {
                        "field": "namespace.keyword"
                    }
                }
            },
            _source=["namespace"]
        )

    def execute_list_namespaces(self, op: ListNamespacesOp) -> list[Result]:
        query_conditions = self._query(op)
        documents = self.es_connection.search(
            index=self.store_index_name,
            body=query_conditions,
        )
        return self.es_to_tuple(documents)

    async def aexecute_list_namespaces(self, op: ListNamespacesOp) -> list[Result]:
        query_conditions = self._query(op)
        documents = await self.es_connection.search(
            index=self.store_index_name,
            body=query_conditions,
        )
        return self.es_to_tuple(documents)

    def invoke(
        self, input: Iterable[ListNamespacesOp], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> list[Result]:
        return self._execute_operations(input, sync_func=self.execute_list_namespaces)
    
    async def ainvoke(
        self, input: Iterable[ListNamespacesOp], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> list[Result]:
        return await self._execute_operations(input, sync_func=self.aexecute_list_namespaces)