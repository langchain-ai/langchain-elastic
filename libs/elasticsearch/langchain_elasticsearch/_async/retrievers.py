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
    """`Elasticsearch` retriever.

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

    Key init args:
        - `index_name` (Union[str, Sequence[str]]):
            The name of the index to query. Can also be a list of names.
        - `body_func` (Callable[[str], Dict]):
            Function that creates an Elasticsearch DSL query body
            from a search string.
            The returned query body must fit what you would normally send in a POST
            request to the _search endpoint.
            If applicable, it also includes parameters
            like the `size` parameter etc.

    Instantiate:
        ```python
        from langchain_elasticsearch import ElasticsearchRetriever

        def body_func(query: str) -> dict:
            return {"query": {"match": {"text": {"query": query}}}}

        retriever = ElasticsearchRetriever(
            index_name="langchain-demo",
            body_func=body_func,
            content_field="text",
            es_url="http://localhost:9200",
        )
        ```

        **Instantiate with API key (URL):**
            ```python
            from langchain_elasticsearch import ElasticsearchRetriever

            def body_func(query: str) -> dict:
                return {"query": {"match": {"text": {"query": query}}}}

            retriever = ElasticsearchRetriever(
                index_name="langchain-demo",
                body_func=body_func,
                content_field="text",
                es_url="http://localhost:9200",
                es_api_key="your-api-key"
            )
            ```

        **Instantiate with username/password (URL):**
            ```python
            from langchain_elasticsearch import ElasticsearchRetriever

            def body_func(query: str) -> dict:
                return {"query": {"match": {"text": {"query": query}}}}

            retriever = ElasticsearchRetriever(
                index_name="langchain-demo",
                body_func=body_func,
                content_field="text",
                es_url="http://localhost:9200",
                es_user="elastic",
                es_password="password"
            )
            ```

        If you want to use a cloud hosted Elasticsearch instance, you can pass in the
        es_cloud_id argument instead of the es_url argument.

        **Instantiate from cloud (with username/password):**
            ```python
            from langchain_elasticsearch import ElasticsearchRetriever

            def body_func(query: str) -> dict:
                return {"query": {"match": {"text": {"query": query}}}}

            retriever = ElasticsearchRetriever(
                index_name="langchain-demo",
                body_func=body_func,
                content_field="text",
                es_cloud_id="<cloud_id>",
                es_user="elastic",
                es_password="<password>"
            )
            ```

        **Instantiate from cloud (with API key):**
            ```python
            from langchain_elasticsearch import ElasticsearchRetriever

            def body_func(query: str) -> dict:
                return {"query": {"match": {"text": {"query": query}}}}

            retriever = ElasticsearchRetriever(
                index_name="langchain-demo",
                body_func=body_func,
                content_field="text",
                es_cloud_id="<cloud_id>",
                es_api_key="your-api-key"
            )
            ```

        You can also connect to an existing Elasticsearch instance by passing in a
        pre-existing Elasticsearch connection via the client argument.

        **Instantiate from existing connection:**
            ```python
            from langchain_elasticsearch import ElasticsearchRetriever
            from elasticsearch import Elasticsearch

            def body_func(query: str) -> dict:
                return {"query": {"match": {"text": {"query": query}}}}

            client = Elasticsearch("http://localhost:9200")
            retriever = ElasticsearchRetriever(
                index_name="langchain-demo",
                body_func=body_func,
                content_field="text",
                client=client
            )
            ```

    Retrieve documents:
        Note: Use `invoke()` or `ainvoke()` instead of the deprecated
        `get_relevant_documents()` or `aget_relevant_documents()` methods.

        First, index some documents:
        ```python
        from elasticsearch import Elasticsearch

        client = Elasticsearch("http://localhost:9200")

        # Index sample documents
        client.index(
            index="some-index",
            document={"text": "The quick brown fox jumps over the lazy dog"},
            id="1",
            refresh=True
        )
        client.index(
            index="some-index",
            document={"text": "Python is a popular programming language"},
            id="2",
            refresh=True
        )
        client.index(
            index="some-index",
            document={"text": "Elasticsearch is a search engine"},
            id="3",
            refresh=True
        )
        ```

        Then retrieve documents:
        ```python
        from langchain_elasticsearch import ElasticsearchRetriever

        def body_func(query: str) -> dict:
            return {"query": {"match": {"text": {"query": query}}}}

        retriever = ElasticsearchRetriever(
            index_name="some-index",
            body_func=body_func,
            content_field="text",
            es_url="http://localhost:9200"
        )

        # Retrieve documents
        documents = retriever.invoke("Python")
        for doc in documents:
            print(f"* {doc.page_content}")
        ```
        ```python
        * Python is a popular programming language
        ```



    Use custom document mapper:
        ```python
        from langchain_elasticsearch import ElasticsearchRetriever
        from langchain_core.documents import Document
        from elasticsearch import Elasticsearch
        from typing import Mapping, Any

        def body_func(query: str) -> dict:
            return {"query": {"match": {"custom_field": {"query": query}}}}

        def custom_mapper(hit: Mapping[str, Any]) -> Document:
            # Custom logic to extract content and metadata
            return Document(
                page_content=hit["_source"]["custom_field"],
                metadata={"score": hit["_score"]}
            )

        client = Elasticsearch("http://localhost:9200")
        retriever = ElasticsearchRetriever(
            index_name="langchain-demo",
            body_func=body_func,
            document_mapper=custom_mapper,
            client=client
        )
        ```

    Use with multiple indices:
        ```python
        from langchain_elasticsearch import ElasticsearchRetriever
        from elasticsearch import Elasticsearch

        def body_func(query: str) -> dict:
            return {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text_field_1", "text_field_2"]
                    }
                }
            }

        client = Elasticsearch("http://localhost:9200")
        retriever = ElasticsearchRetriever(
            index_name=["index1", "index2"],
            body_func=body_func,
            content_field={
                "index1": "text_field_1",
                "index2": "text_field_2"
            },
            client=client
        )
        ```

    Use as LangChain retriever in chains:
        Note: Before running this example, ensure you have indexed documents
        in your Elasticsearch index. The retriever will search this index
        for relevant documents to use as context.

        ```python
        from langchain_elasticsearch import ElasticsearchRetriever
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama import ChatOllama

        # ElasticsearchRetriever is already a BaseRetriever
        retriever = ElasticsearchRetriever(
            index_name="some-index",
            body_func=lambda q: {"query": {"match": {"text": {"query": q}}}},
            content_field="text",
            es_url="http://localhost:9200"
        )

        llm = ChatOllama(model="llama3", temperature=0)

        # Create a chain that retrieves documents and then generates a response
        def format_docs(docs):
            # Format documents for the prompt
            return "\\n\\n".join(doc.page_content for doc in docs)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\\n\\n"
            "Context: {context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )

        result = chain.invoke("what is the answer to this question?")
        ```

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
        """Initialize the AsyncElasticsearchRetriever instance.

        Args:
            index_name (Union[str, Sequence[str]]): The name of the index to query.
                Can also be a list of names.
            body_func (Callable[[str], Dict]): Function that creates an Elasticsearch
                DSL query body from a search string. The returned query body must fit
                what you would normally send in a POST request to the _search
                endpoint. If applicable, it also includes parameters like the size
                parameter.
            content_field (Optional[Union[str, Mapping[str, str]]]): The document field
                name that contains the page content. If multiple indices are queried,
                specify a dict {index_name: field_name} here.
            document_mapper (Optional[Callable[[Mapping], Document]]): Function that
                maps Elasticsearch hits to LangChain Documents. If not provided, it
                will be automatically created based on content_field.
            client (AsyncElasticsearch, optional): Pre-existing Elasticsearch
                connection. Either provide this OR credentials.
            es_url (str, optional): URL of the Elasticsearch instance to connect to.
            es_cloud_id (str, optional): Cloud ID of the Elasticsearch instance to
                connect to.
            es_user (str, optional): Username to use when connecting to Elasticsearch.
            es_api_key (str, optional): API key to use when connecting to
                Elasticsearch.
            es_password (str, optional): Password to use when connecting to
                Elasticsearch.
        """
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
                "Either 'client' or credentials (es_url, es_cloud_id, etc.) "
                "must be provided."
            )

        # Apply user agent
        es_connection = async_with_user_agent_header(es_connection, "langchain-py-r")

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
