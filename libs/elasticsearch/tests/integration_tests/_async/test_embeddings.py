"""Test elasticsearch_embeddings embeddings."""

import os

import pytest
from elasticsearch import AsyncElasticsearch

from langchain_elasticsearch.embeddings import AsyncElasticsearchEmbeddings

from ._test_utilities import model_is_deployed

# deployed with
# https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-text-emb-vector-search-example.html
MODEL_ID = os.getenv("MODEL_ID", "sentence-transformers__msmarco-minilm-l-12-v3")
NUM_DIMENSIONS = int(os.getenv("NUM_DIMENTIONS", "384"))

ES_URL = os.environ.get("ES_URL", "http://localhost:9200")


@pytest.mark.asyncio
async def test_elasticsearch_embedding_documents() -> None:
    """Test Elasticsearch embedding documents."""
    client = AsyncElasticsearch(hosts=[ES_URL])
    if not (await model_is_deployed(client, MODEL_ID)):
        await client.close()
        pytest.skip(reason=f"{MODEL_ID} model is not deployed in ML Node, skipping test")

    documents = ["foo bar", "bar foo", "foo"]
    embedding = AsyncElasticsearchEmbeddings.from_es_connection(MODEL_ID, client)
    output = await embedding.aembed_documents(documents)
    await client.close()
    assert len(output) == 3
    assert len(output[0]) == NUM_DIMENSIONS
    assert len(output[1]) == NUM_DIMENSIONS
    assert len(output[2]) == NUM_DIMENSIONS


@pytest.mark.asyncio
async def test_elasticsearch_embedding_query() -> None:
    """Test Elasticsearch embedding query."""
    client = AsyncElasticsearch(hosts=[ES_URL])
    if not (await model_is_deployed(client, MODEL_ID)):
        await client.close()
        pytest.skip(reason=f"{MODEL_ID} model is not deployed in ML Node, skipping test")

    document = "foo bar"
    embedding = AsyncElasticsearchEmbeddings.from_es_connection(MODEL_ID, client)
    output = await embedding.aembed_query(document)
    await client.close()
    assert len(output) == NUM_DIMENSIONS
