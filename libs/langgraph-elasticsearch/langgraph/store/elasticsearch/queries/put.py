from datetime import datetime, timezone
import logging
from typing import Any, Iterable, Optional
from langchain_core.documents import Document
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.config import run_in_executor
from langgraph.store.base import PutOp, tokenize_path, get_text_at_path
from langgraph.store.elasticsearch.queries.base import ElasticQuery, VectorQuery, namespace_to_text

logger = logging.getLogger(__name__)

class ElasticQueryPut(ElasticQuery[Iterable[PutOp], None]):
    def put_texts(self, op: PutOp) -> None:
        doc_id = namespace_to_text(op.namespace + (op.key,))
        if not op.value:
            self.es_connection.delete(
                index=self.store_index_name,
                id=doc_id,
                ignore=[404]
            )
            return
    
        self.es_connection.update(
            index=self.store_index_name,
            id=doc_id,
            body=self.es_body_upsert(op),
        )

    async def aput_texts(self, op: PutOp) -> None:
        await run_in_executor(None, self.put_texts, op)
    
    def invoke(
        self, input: Iterable[PutOp], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> None:
        self._execute_operations(input, sync_func=self.put_texts)
    
    async def ainvoke(
        self, input: Iterable[PutOp], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> None:
        await self._aexecute_operations(input, sync_func=self.aput_texts)
    
class VectorQueryPut(VectorQuery[Iterable[PutOp], None]):
    def put_texts(self, op: PutOp) -> None:
        if not op.value:
            self.vector_store.delete(ids=[namespace_to_text(op.namespace + (op.key,))])
            return []

        paths = self.index_config["__tokenized_fields"] if op.index is None else [(ix, tokenize_path(ix)) for ix in op.index]
        created_at = datetime.now(tz=timezone.utc)
        documents = [
            Document(
                id=namespace_to_text(op.namespace + (op.key, path)),
                page_content=text,
                metadata={
                    "namespace": namespace_to_text(op.namespace),
                    "key": op.key,
                    "path": path,
                    "created_at": created_at,
                    "updated_at": created_at,
                },
            )
            for path, tokenized_path in paths
            for text in get_text_at_path(op.value, tokenized_path)
        ]
        self.vector_store.add_documents(
                            documents=documents,
                            create_index_if_not_exists=True,)

    async def aput_texts(self, op: PutOp) -> None:
        await run_in_executor(None, self.put_texts, op)

    def invoke(
        self, input: Iterable[PutOp], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> None:
        self._execute_operations(input, sync_func=self.put_texts)
    
    async def ainvoke(
        self, input: Iterable[PutOp], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> None:
        await self._aexecute_operations(input, sync_func=self.aput_texts)
