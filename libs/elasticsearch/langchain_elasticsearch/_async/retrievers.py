import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union, cast

from elasticsearch import AsyncElasticsearch
from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_elasticsearch._utilities import async_with_user_agent_header
from langchain_elasticsearch.client import create_async_elasticsearch_client

logger = logging.getLogger(__name__)


class AsyncElasticsearchRetriever(BaseRetriever):
    """Elasticsearch retriever.

    Args:
        index_name: The name of the index to query. Can also be a list of names.
        body_func: Function to create an Elasticsearch DSL query body from a search
            string. The returned query body must fit what you would normally send in a
            POST request to the _search endpoint. If applicable, it also includes
            parameters like the `size` parameter etc.
        content_field: The document field name that contains the page content. If
            multiple indices are queried, specify a dict {index_name: field_name} here.
        document_mapper: Function to map Elasticsearch hits to LangChain Documents.
        client: Optional pre-existing Elasticsearch client connection.
            Alternatively, provide credentials (es_url, es_cloud_id, etc.).
        es_url: URL of the Elasticsearch instance to connect to.
        es_cloud_id: Cloud ID of the Elasticsearch instance to connect to.
        es_user: Username to use when connecting to Elasticsearch.
        es_api_key: API key to use when connecting to Elasticsearch.
        es_password: Password to use when connecting to Elasticsearch.

    For synchronous applications, use the `ElasticsearchRetriever` class.
    For asynchronous applications, use the `AsyncElasticsearchRetriever` class.
    """

    client: AsyncElasticsearch
    index_name: Union[str, Sequence[str]]
    body_func: Callable[[str], Dict]
    content_field: Optional[Union[str, Mapping[str, str]]] = None
    document_mapper: Optional[Callable[[Mapping], Document]] = None

    def __init__(
        self,
        index_name: Union[str, Sequence[str]],
        body_func: Callable[[str], Dict],
        *,
        content_field: Optional[Union[str, Mapping[str, str]]] = None,
        document_mapper: Optional[Callable[[Mapping], Document]] = None,
        client: Optional[AsyncElasticsearch] = None,
        es_url: Optional[str] = None,
        es_cloud_id: Optional[str] = None,
        es_user: Optional[str] = None,
        es_api_key: Optional[str] = None,
        es_password: Optional[str] = None,
    ) -> None:
        # Create client from credentials if needed (BEFORE super().__init__)
        if client is not None:
            es_connection = client
        elif es_url is not None or es_cloud_id is not None:
            es_connection = create_async_elasticsearch_client(
                url=es_url,
                cloud_id=es_cloud_id,
                api_key=es_api_key,
                username=es_user,
                password=es_password,
            )
        else:
            raise ValueError(
                "Either 'client' or credentials (es_url, es_cloud_id, etc.) must be provided."
            )

        # Apply user agent
        es_connection = async_with_user_agent_header(es_connection, "langchain-py-r")

        # Pass ALL Pydantic fields to super().__init__() so Pydantic sets them
        super().__init__(
            client=es_connection,
            index_name=index_name,
            body_func=body_func,
            content_field=content_field,
            document_mapper=document_mapper,
        )

        # Now Pydantic has set everything, do validation
        if self.content_field is None and self.document_mapper is None:
            raise ValueError("One of content_field or document_mapper must be defined.")
        if self.content_field is not None and self.document_mapper is not None:
            raise ValueError(
                "Both content_field and document_mapper are defined. "
                "Please provide only one."
            )

        if not self.document_mapper:
            if isinstance(self.content_field, str):
                self.document_mapper = self._single_field_mapper
            elif isinstance(self.content_field, Mapping):
                self.document_mapper = self._multi_field_mapper
            else:
                raise ValueError(
                    "unknown type for content_field, expected string or dict."
                )

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        if not self.client or not self.document_mapper:
            raise ValueError("faulty configuration")  # should not happen

        body = self.body_func(query)
        results = await self.client.search(index=self.index_name, body=body)
        return [self.document_mapper(hit) for hit in results["hits"]["hits"]]

    def _single_field_mapper(self, hit: Mapping[str, Any]) -> Document:
        content = hit["_source"].pop(self.content_field)
        return Document(page_content=content, metadata=hit)

    def _multi_field_mapper(self, hit: Mapping[str, Any]) -> Document:
        self.content_field = cast(Mapping, self.content_field)
        field = self.content_field[hit["_index"]]
        content = hit["_source"].pop(field)
        return Document(page_content=content, metadata=hit)
