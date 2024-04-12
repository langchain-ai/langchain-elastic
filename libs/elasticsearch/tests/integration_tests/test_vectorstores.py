"""Test ElasticsearchStore functionality."""

import logging
import uuid
from typing import Dict, Iterator

import pytest
from langchain_core.documents import Document

from langchain_elasticsearch.vectorstores import ElasticsearchStore

from ..fake_embeddings import (
    ConsistentFakeEmbeddings,
    FakeEmbeddings,
)
from ._test_utilities import (
    clear_test_indices,
    create_es_client,
    read_env,
)

logging.basicConfig(level=logging.DEBUG)

"""
cd tests/integration_tests
docker-compose up elasticsearch

By default runs against local docker instance of Elasticsearch.
To run against Elastic Cloud, set the following environment variables:
- ES_CLOUD_ID
- ES_API_KEY
"""


class TestElasticsearch:
    @pytest.fixture
    def es_params(self) -> Iterator[dict]:
        params = read_env()
        es = create_es_client(params)

        yield params

        clear_test_indices(es)
        es.close()

    @pytest.fixture(scope="function")
    def index_name(self) -> str:
        """Return the index name."""
        return f"test_{uuid.uuid4().hex}"

    def test_from_texts_similarity_search_with_doc_builder(
        self, es_params: dict, index_name: str
    ) -> None:
        texts = ["foo", "foo", "foo"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            metadatas=metadatas,
            **es_params,
            index_name=index_name,
        )

        def custom_document_builder(_: Dict) -> Document:
            return Document(
                page_content="Mock content!",
                metadata={
                    "page_number": -1,
                    "original_filename": "Mock filename!",
                },
            )

        output = docsearch.similarity_search(
            query="foo", k=1, doc_builder=custom_document_builder
        )
        assert output[0].page_content == "Mock content!"
        assert output[0].metadata["page_number"] == -1
        assert output[0].metadata["original_filename"] == "Mock filename!"

        docsearch.close()

    def test_search_with_relevance_threshold(
        self, es_params: dict, index_name: str
    ) -> None:
        """Test to make sure the relevance threshold is respected."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        embeddings = ConsistentFakeEmbeddings()

        docsearch = ElasticsearchStore.from_texts(
            index_name=index_name,
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            **es_params,
        )

        # Find a good threshold for testing
        query_string = "foo"
        top3 = docsearch.similarity_search_with_relevance_scores(
            query=query_string, k=3
        )
        similarity_of_second_ranked = top3[1][1]
        assert len(top3) == 3

        # Test threshold
        retriever = docsearch.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": similarity_of_second_ranked},
        )
        output = retriever.get_relevant_documents(query=query_string)

        assert output == [
            top3[0][0],
            top3[1][0],
            # third ranked is out
        ]

        docsearch.close()

    def test_search_by_vector_with_relevance_threshold(
        self, es_params: dict, index_name: str
    ) -> None:
        """Test to make sure the relevance threshold is respected."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        embeddings = ConsistentFakeEmbeddings()

        docsearch = ElasticsearchStore.from_texts(
            index_name=index_name,
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            **es_params,
        )

        # Find a good threshold for testing
        query_string = "foo"
        embedded_query = embeddings.embed_query(query_string)
        top3 = docsearch.similarity_search_by_vector_with_relevance_scores(
            embedding=embedded_query, k=3
        )
        similarity_of_second_ranked = top3[1][1]
        assert len(top3) == 3

        # Test threshold
        retriever = docsearch.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": similarity_of_second_ranked},
        )
        output = retriever.get_relevant_documents(query=query_string)

        assert output == [
            top3[0][0],
            top3[1][0],
            # third ranked is out
        ]

        docsearch.close()
