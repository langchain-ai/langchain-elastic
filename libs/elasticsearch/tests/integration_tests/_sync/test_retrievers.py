"""Test ElasticsearchRetriever functionality."""

import os
import re
import uuid
from typing import Any, Dict

import pytest
from elasticsearch import Elasticsearch
from langchain_core.documents import Document

from langchain_elasticsearch.retrievers import ElasticsearchRetriever

from ._test_utilities import requests_saving_es_client

"""
cd tests/integration_tests
docker-compose up elasticsearch

By default runs against local docker instance of Elasticsearch.
To run against Elastic Cloud, set the following environment variables:
- ES_CLOUD_ID
- ES_API_KEY
"""


def index_test_data(es_client: Elasticsearch, index_name: str, field_name: str) -> None:
    docs = [(1, "foo bar"), (2, "bar"), (3, "foo"), (4, "baz"), (5, "foo baz")]
    for identifier, text in docs:
        es_client.index(
            index=index_name,
            document={field_name: text, "another_field": 1},
            id=str(identifier),
            refresh=True,
        )


class TestElasticsearchRetriever:
    @pytest.fixture(scope="function")
    def es_client(self) -> Any:
        client = requests_saving_es_client()
        yield client
        client.close()

    @pytest.fixture(scope="function")
    def index_name(self) -> str:
        """Return the index name."""
        return f"test_{uuid.uuid4().hex}"

    @pytest.mark.sync
    def test_user_agent_header(self, es_client: Elasticsearch, index_name: str) -> None:
        """Test that the user agent header is set correctly."""

        retriever = ElasticsearchRetriever(
            index_name=index_name,
            body_func=lambda _: {"query": {"match_all": {}}},
            content_field="text",
            es_client=es_client,
        )

        assert retriever.es_client
        user_agent = retriever.es_client._headers["User-Agent"]
        assert (
            re.match(r"^langchain-py-r/\d+\.\d+\.\d+(?:rc\d+)?$", user_agent)
            is not None
        ), f"The string '{user_agent}' does not match the expected pattern."

        index_test_data(es_client, index_name, "text")
        retriever.get_relevant_documents("foo")

        search_request = es_client.transport.requests[-1]  # type: ignore[attr-defined]
        user_agent = search_request["headers"]["User-Agent"]
        assert (
            re.match(r"^langchain-py-r/\d+\.\d+\.\d+(?:rc\d+)?$", user_agent)
            is not None
        ), f"The string '{user_agent}' does not match the expected pattern."

    @pytest.mark.sync
    def test_init_url(self, index_name: str) -> None:
        """Test end-to-end indexing and search."""

        text_field = "text"

        def body_func(query: str) -> Dict:
            return {"query": {"match": {text_field: {"query": query}}}}

        es_url = os.environ.get("ES_URL", "http://localhost:9200")
        cloud_id = os.environ.get("ES_CLOUD_ID")
        api_key = os.environ.get("ES_API_KEY")

        config = (
            {"cloud_id": cloud_id, "api_key": api_key} if cloud_id else {"url": es_url}
        )

        retriever = ElasticsearchRetriever.from_es_params(
            index_name=index_name,
            body_func=body_func,
            content_field=text_field,
            **config,  # type: ignore[arg-type]
        )

        index_test_data(retriever.es_client, index_name, text_field)
        result = retriever.get_relevant_documents("foo")

        assert {r.page_content for r in result} == {"foo", "foo bar", "foo baz"}
        assert {r.metadata["_id"] for r in result} == {"3", "1", "5"}
        for r in result:
            assert set(r.metadata.keys()) == {"_index", "_id", "_score", "_source"}
            assert text_field not in r.metadata["_source"]
            assert "another_field" in r.metadata["_source"]

    @pytest.mark.sync
    def test_init_client(self, es_client: Elasticsearch, index_name: str) -> None:
        """Test end-to-end indexing and search."""

        text_field = "text"

        def body_func(query: str) -> Dict:
            return {"query": {"match": {text_field: {"query": query}}}}

        retriever = ElasticsearchRetriever(
            index_name=index_name,
            body_func=body_func,
            content_field=text_field,
            es_client=es_client,
        )

        index_test_data(es_client, index_name, text_field)
        result = retriever.get_relevant_documents("foo")

        assert {r.page_content for r in result} == {"foo", "foo bar", "foo baz"}
        assert {r.metadata["_id"] for r in result} == {"3", "1", "5"}
        for r in result:
            assert set(r.metadata.keys()) == {"_index", "_id", "_score", "_source"}
            assert text_field not in r.metadata["_source"]
            assert "another_field" in r.metadata["_source"]

    @pytest.mark.sync
    def test_multiple_index_and_content_fields(
        self, es_client: Elasticsearch, index_name: str
    ) -> None:
        """Test multiple content fields"""
        index_name_1 = f"{index_name}_1"
        index_name_2 = f"{index_name}_2"
        text_field_1 = "text_1"
        text_field_2 = "text_2"

        def body_func(query: str) -> Dict:
            return {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": [text_field_1, text_field_2],
                    }
                }
            }

        retriever = ElasticsearchRetriever(
            index_name=[index_name_1, index_name_2],
            content_field={index_name_1: text_field_1, index_name_2: text_field_2},
            body_func=body_func,
            es_client=es_client,
        )

        index_test_data(es_client, index_name_1, text_field_1)
        index_test_data(es_client, index_name_2, text_field_2)
        result = retriever.get_relevant_documents("foo")

        # matches from both indices
        assert sorted([(r.page_content, r.metadata["_index"]) for r in result]) == [
            ("foo", index_name_1),
            ("foo", index_name_2),
            ("foo bar", index_name_1),
            ("foo bar", index_name_2),
            ("foo baz", index_name_1),
            ("foo baz", index_name_2),
        ]

    @pytest.mark.sync
    def test_custom_mapper(self, es_client: Elasticsearch, index_name: str) -> None:
        """Test custom document maper"""

        text_field = "text"
        meta = {"some_field": 12}

        def body_func(query: str) -> Dict:
            return {"query": {"match": {text_field: {"query": query}}}}

        def id_as_content(hit: Dict) -> Document:
            return Document(page_content=hit["_id"], metadata=meta)

        retriever = ElasticsearchRetriever(
            index_name=index_name,
            body_func=body_func,
            document_mapper=id_as_content,
            es_client=es_client,
        )

        index_test_data(es_client, index_name, text_field)
        result = retriever.get_relevant_documents("foo")

        assert [r.page_content for r in result] == ["3", "1", "5"]
        assert [r.metadata for r in result] == [meta, meta, meta]

    @pytest.mark.sync
    def test_fail_content_field_and_mapper(self, es_client: Elasticsearch) -> None:
        """Raise exception if both content_field and document_mapper are specified."""

        with pytest.raises(ValueError):
            ElasticsearchRetriever(
                content_field="text",
                document_mapper=lambda x: x,
                index_name="foo",
                body_func=lambda x: x,
                es_client=es_client,
            )

    @pytest.mark.sync
    def test_fail_neither_content_field_nor_mapper(
        self, es_client: Elasticsearch
    ) -> None:
        """Raise exception if neither content_field nor document_mapper are specified"""

        with pytest.raises(ValueError):
            ElasticsearchRetriever(
                index_name="foo",
                body_func=lambda x: x,
                es_client=es_client,
            )
