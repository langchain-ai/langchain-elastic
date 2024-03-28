"""Test elasticsearch_embeddings embeddings."""

import os

from elasticsearch import Elasticsearch

from langchain_elasticsearch.embeddings import ElasticsearchEmbeddings

from ._test_utilities import deploy_model

ES_CLIENT = Elasticsearch(hosts=[os.environ.get("ES_URL", "http://localhost:9200")])
MODEL_ID = ".elser_model_2"


class TestEmbeddings:
    @classmethod
    def setup_class(cls) -> None:
        deploy_model(ES_CLIENT, MODEL_ID)

    def test_elasticsearch_embedding_documents(self) -> None:
        """Test Elasticsearch embedding documents."""
        documents = ["foo bar", "bar foo", "foo"]
        embedding = ElasticsearchEmbeddings(ES_CLIENT.ml, MODEL_ID)
        output = embedding.embed_documents(documents)
        assert len(output) == 3
        assert "foo" in output[0]
        assert "##bar" in output[0]
        assert "bar" in output[1]
        assert "foo" in output[1]
        assert "foo" in output[2]

    def test_elasticsearch_embedding_query(self) -> None:
        """Test Elasticsearch embedding query."""
        document = "foo bar"
        embedding = ElasticsearchEmbeddings(ES_CLIENT.ml, MODEL_ID)
        output = embedding.embed_query(document)
        assert "foo" in output
        assert "##bar" in output
