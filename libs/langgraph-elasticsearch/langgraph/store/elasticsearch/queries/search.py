import logging
from typing import Any, Dict, Iterable, List, Optional
from langchain_core.runnables import run_in_executor, RunnableConfig
from langchain_core.documents import Document
from langgraph.store.base import SearchOp, SearchItem
from langgraph.store.elasticsearch.queries.base import ElasticQuery, VectorQuery, text_to_namespace, namespace_to_text
from langgraph.util import validate_before_execution

logger = logging.getLogger(__name__)

class ElasticQuerySearch(ElasticQuery[SearchOp, SearchItem]):
    """Class for executing search operations using Elasticsearch."""

    def execute_search(self, op: SearchOp) -> List[SearchItem]:
        """Execute a synchronous search operation.

        Args:
            op (SearchOp): The search operation to execute.

        Returns:
            List[SearchItem]: The search results.
        """
        args = {
            "from": op.offset,
            "sort": [{"updated_at": {"order": "desc"}}],
        }
        print('cheguei aqui')
        print(self.store_index_name)
        print(op.namespace_prefix)
        print(op.filter)
        print(op.limit)
        print(args)

        response = self.es_connection.search(
            index=self.store_index_name,
            body=self.build_query(op.namespace_prefix, op.filter, limit=op.limit, property="value", **args)
        )
        return self.es_to_items(response)

    async def aexecute_search(self, op: SearchOp) -> list[SearchItem]:
        """Execute an asynchronous search operation.

        Args:
            op (SearchOp): The search operation to execute.

        Returns:
            list[SearchItem]: The search results.
        """
        return await run_in_executor(None, self.execute_search, op)

    def invoke(
        self, input: Iterable[SearchOp], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> list[SearchItem]:
        """Invoke the search operation synchronously.

        Args:
            input (Iterable[SearchOp]): The search operations to execute.
            config (Optional[RunnableConfig], optional): Configuration for the runnable. Defaults to None.

        Returns:
            list[SearchItem]: The search results.
        """
        return self._execute_operations(input, sync_func=self.execute_search)
    
    async def ainvoke(
        self, input: Iterable[SearchOp], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> list[SearchItem]:
        """Invoke the search operation asynchronously.

        Args:
            input (Iterable[SearchOp]): The search operations to execute.
            config (Optional[RunnableConfig], optional): Configuration for the runnable. Defaults to None.

        Returns:
            list[SearchItem]: The search results.
        """
        return await self._execute_operations(input, sync_func=self.aexecute_search)
class VectorQuerySearch(VectorQuery[SearchOp, SearchItem]):
    """Class for executing vector search operations."""

    def _document_to_search_item(self, documents: list[tuple[Document, float]]) -> list[SearchItem]:
        """Convert documents to search items.

        Args:
            documents (list[tuple[Document, float]]): The documents to convert.

        Returns:
            list[SearchItem]: The converted search items.
        """
        return [
            self._create_item(
                namespace=text_to_namespace(doc.metadata["namespace"]),
                key=doc.id,
                value=doc.page_content,
                created_at=doc.metadata["created_at"],
                updated_at=doc.metadata["updated_at"],
                # created_at=doc.model_extra["created_at"],
                # updated_at=doc.model_extra["updated_at"],
                score=score,
            )
            for doc, score in documents
        ]

    def _query(self, op: SearchOp) -> Dict[str, Any]:
        """Generate the query dictionary for the search operation.

        Args:
            op (SearchOp): The search operation.

        Returns:
            dict: The query dictionary.
        """
        query = {
            "query": op.query,
            "k": op.limit,
        }
        if op.namespace_prefix:
            query.update({
                "from": op.offset,
                "filter": {
                    "prefix": {
                        "metadata.namespace.keyword": namespace_to_text(op.namespace_prefix)
                    }
                }
            })
        return query

    @validate_before_execution
    def execute_search(self, op: SearchOp) -> list[SearchItem]:
        """Execute a synchronous vector search operation.

        Args:
            op (SearchOp): The search operation to execute.

        Returns:
            list[SearchItem]: The search results.
        """
        if not self.vector_store or not op.query:
            return op
        query = self._query(op)
        documents = self.vector_store.similarity_search_with_relevance_scores(**query)
        return self._document_to_search_item(documents)

    @validate_before_execution
    async def aexecute_search(self, op: SearchOp) -> list[SearchItem]:
        """Execute an asynchronous vector search operation.

        Args:
            op (SearchOp): The search operation to execute.

        Returns:
            list[SearchItem]: The search results.
        """
        if not self.vector_store or not op.query:
            return op
        query = self._query(op)
        documents = await self.vector_store.asimilarity_search_with_relevance_scores(**query)
        return self._document_to_search_item(documents)

    def invoke(
        self, input: Iterable[SearchOp], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> list[SearchItem]:
        """Invoke the vector search operation synchronously.

        Args:
            input (Iterable[SearchOp]): The search operations to execute.
            config (Optional[RunnableConfig], optional): Configuration for the runnable. Defaults to None.

        Returns:
            list[SearchItem]: The search results.
        """
        return self._execute_operations(input, sync_func=self.execute_search)
    
    async def ainvoke(
        self, input: Iterable[SearchOp], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> list[SearchItem]:
        """Invoke the vector search operation asynchronously.

        Args:
            input (Iterable[SearchOp]): The search operations to execute.
            config (Optional[RunnableConfig], optional): Configuration for the runnable. Defaults to None.

        Returns:
            list[SearchItem]: The search results.
        """
        return await self._execute_operations(input, sync_func=self.aexecute_search)