"""Test ElasticsearchStore functionality."""

import logging
import re
import uuid
from typing import Any, Dict, Generator, List, Union

import numpy as np
import pytest
from elasticsearch import NotFoundError
from elasticsearch.helpers import BulkIndexError
from langchain_core.documents import Document

from langchain_elasticsearch._utilities import model_is_deployed
from langchain_elasticsearch.vectorstores import ElasticsearchStore

from ..fake_embeddings import (
    ConsistentFakeEmbeddings,
    FakeEmbeddings,
)
from ._test_utilities import (
    clear_test_indices,
    create_es_client,
    read_env,
    requests_saving_es_client,
)

logging.basicConfig(level=logging.DEBUG)

"""
cd tests/integration_tests
docker-compose up elasticsearch

By default runs against local docker instance of Elasticsearch.
To run against Elastic Cloud, set the following environment variables:
- ES_CLOUD_ID
- ES_API_KEY

Some of the tests require the following models to be deployed in the ML Node:
- elser (can be downloaded and deployed through Kibana and trained models UI)
- sentence-transformers__all-minilm-l6-v2 (can be deployed through the API,
  loaded via eland)

These tests that require the models to be deployed are skipped by default. 
Enable them by adding the model name to the modelsDeployed list below.
"""

ELSER_MODEL_ID = ".elser_model_2"
TRANSFORMER_MODEL_ID = "sentence-transformers__all-minilm-l6-v2"


class TestElasticsearch:
    @pytest.fixture(scope="class", autouse=True)
    def elasticsearch_connection(self) -> Union[dict, Generator[dict, None, None]]:
        params = read_env()
        es = create_es_client(params)

        yield params

        # clear indices
        clear_test_indices(es)

        # clear all test pipelines
        try:
            response = es.ingest.get_pipeline(id="test_*,*_sparse_embedding")

            for pipeline_id, _ in response.items():
                try:
                    es.ingest.delete_pipeline(id=pipeline_id)
                    print(f"Deleted pipeline: {pipeline_id}")  # noqa: T201
                except Exception as e:
                    print(f"Pipeline error: {e}")  # noqa: T201
        except Exception:
            pass

        return None

    @pytest.fixture(scope="function")
    def es_client(self) -> Any:
        return requests_saving_es_client()

    @pytest.fixture(scope="function")
    def index_name(self) -> str:
        """Return the index name."""
        return f"test_{uuid.uuid4().hex}"

    def test_similarity_search_without_metadata(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and search without metadata."""

        def assert_query(query_body: dict, query: str) -> dict:
            assert query_body == {
                "knn": {
                    "field": "vector",
                    "filter": [],
                    "k": 1,
                    "num_candidates": 50,
                    "query_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                }
            }
            return query_body

        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
        )
        output = docsearch.similarity_search("foo", k=1, custom_query=assert_query)
        assert output == [Document(page_content="foo")]

    async def test_similarity_search_without_metadata_async(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and search without metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
        )
        output = await docsearch.asimilarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]

    def test_add_embeddings(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """
        Test add_embeddings, which accepts pre-built embeddings instead of
         using inference for the texts.
        This allows you to separate the embeddings text and the page_content
         for better proximity between user's question and embedded text.
        For example, your embedding text can be a question, whereas page_content
         is the answer.
        """
        embeddings = ConsistentFakeEmbeddings()
        text_input = ["foo1", "foo2", "foo3"]
        metadatas = [{"page": i} for i in range(len(text_input))]

        """In real use case, embedding_input can be questions for each text"""
        embedding_input = ["foo2", "foo3", "foo1"]
        embedding_vectors = embeddings.embed_documents(embedding_input)

        docsearch = ElasticsearchStore._create_cls_from_kwargs(
            embeddings,
            **elasticsearch_connection,
            index_name=index_name,
        )
        docsearch.add_embeddings(list(zip(text_input, embedding_vectors)), metadatas)
        output = docsearch.similarity_search("foo1", k=1)
        assert output == [Document(page_content="foo3", metadata={"page": 2})]

    def test_similarity_search_with_metadata(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            ConsistentFakeEmbeddings(),
            metadatas=metadatas,
            **elasticsearch_connection,
            index_name=index_name,
        )

        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo", metadata={"page": 0})]

        output = docsearch.similarity_search("bar", k=1)
        assert output == [Document(page_content="bar", metadata={"page": 1})]

    def test_similarity_search_with_filter(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "foo", "foo"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            metadatas=metadatas,
            **elasticsearch_connection,
            index_name=index_name,
        )

        def assert_query(query_body: dict, query: str) -> dict:
            assert query_body == {
                "knn": {
                    "field": "vector",
                    "filter": [{"term": {"metadata.page": "1"}}],
                    "k": 3,
                    "num_candidates": 50,
                    "query_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                }
            }
            return query_body

        output = docsearch.similarity_search(
            query="foo",
            k=3,
            filter=[{"term": {"metadata.page": "1"}}],
            custom_query=assert_query,
        )
        assert output == [Document(page_content="foo", metadata={"page": 1})]

    def test_similarity_search_with_doc_builder(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        texts = ["foo", "foo", "foo"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            metadatas=metadatas,
            **elasticsearch_connection,
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

    def test_similarity_search_exact_search(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
            strategy=ElasticsearchStore.ExactRetrievalStrategy(),
        )

        expected_query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",  # noqa: E501
                        "params": {
                            "query_vector": [
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                0.0,
                            ]
                        },
                    },
                }
            }
        }

        def assert_query(query_body: dict, query: str) -> dict:
            assert query_body == expected_query
            return query_body

        output = docsearch.similarity_search("foo", k=1, custom_query=assert_query)
        assert output == [Document(page_content="foo")]

    def test_similarity_search_exact_search_with_filter(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
            metadatas=metadatas,
            strategy=ElasticsearchStore.ExactRetrievalStrategy(),
        )

        def assert_query(query_body: dict, query: str) -> dict:
            expected_query = {
                "query": {
                    "script_score": {
                        "query": {"bool": {"filter": [{"term": {"metadata.page": 0}}]}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",  # noqa: E501
                            "params": {
                                "query_vector": [
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    0.0,
                                ]
                            },
                        },
                    }
                }
            }
            assert query_body == expected_query
            return query_body

        output = docsearch.similarity_search(
            "foo",
            k=1,
            custom_query=assert_query,
            filter=[{"term": {"metadata.page": 0}}],
        )
        assert output == [Document(page_content="foo", metadata={"page": 0})]

    def test_similarity_search_exact_search_distance_dot_product(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
            strategy=ElasticsearchStore.ExactRetrievalStrategy(),
            distance_strategy="DOT_PRODUCT",
        )

        def assert_query(query_body: dict, query: str) -> dict:
            assert query_body == {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": """
            double value = dotProduct(params.query_vector, 'vector');
            return sigmoid(1, Math.E, -value);
            """,
                            "params": {
                                "query_vector": [
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    0.0,
                                ]
                            },
                        },
                    }
                }
            }
            return query_body

        output = docsearch.similarity_search("foo", k=1, custom_query=assert_query)
        assert output == [Document(page_content="foo")]

    def test_similarity_search_exact_search_unknown_distance_strategy(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with unknown distance strategy."""

        with pytest.raises(KeyError):
            texts = ["foo", "bar", "baz"]
            ElasticsearchStore.from_texts(
                texts,
                FakeEmbeddings(),
                **elasticsearch_connection,
                index_name=index_name,
                strategy=ElasticsearchStore.ExactRetrievalStrategy(),
                distance_strategy="NOT_A_STRATEGY",
            )

    def test_max_marginal_relevance_search(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test max marginal relevance search."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
            strategy=ElasticsearchStore.ExactRetrievalStrategy(),
        )

        mmr_output = docsearch.max_marginal_relevance_search(texts[0], k=3, fetch_k=3)
        sim_output = docsearch.similarity_search(texts[0], k=3)
        assert mmr_output == sim_output

        mmr_output = docsearch.max_marginal_relevance_search(texts[0], k=2, fetch_k=3)
        assert len(mmr_output) == 2
        assert mmr_output[0].page_content == texts[0]
        assert mmr_output[1].page_content == texts[1]

        mmr_output = docsearch.max_marginal_relevance_search(
            texts[0],
            k=2,
            fetch_k=3,
            lambda_mult=0.1,  # more diversity
        )
        assert len(mmr_output) == 2
        assert mmr_output[0].page_content == texts[0]
        assert mmr_output[1].page_content == texts[2]

        # if fetch_k < k, then the output will be less than k
        mmr_output = docsearch.max_marginal_relevance_search(texts[0], k=3, fetch_k=2)
        assert len(mmr_output) == 2

    def test_similarity_search_approx_with_hybrid_search(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True),
        )

        def assert_query(query_body: dict, query: str) -> dict:
            assert query_body == {
                "knn": {
                    "field": "vector",
                    "filter": [],
                    "k": 1,
                    "num_candidates": 50,
                    "query_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                },
                "query": {
                    "bool": {
                        "filter": [],
                        "must": [{"match": {"text": {"query": "foo"}}}],
                    }
                },
                "rank": {"rrf": {}},
            }
            return query_body

        output = docsearch.similarity_search("foo", k=1, custom_query=assert_query)
        assert output == [Document(page_content="foo")]

    def test_similarity_search_approx_by_vector(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        embeddings = ConsistentFakeEmbeddings()
        docsearch = ElasticsearchStore.from_texts(
            texts,
            embedding=embeddings,
            **elasticsearch_connection,
            index_name=index_name,
        )
        query_vector = embeddings.embed_query("foo")

        def assert_query(query_body: dict, query: str) -> dict:
            assert query_body == {
                "knn": {
                    "field": "vector",
                    "filter": [],
                    "k": 1,
                    "num_candidates": 50,
                    "query_vector": query_vector,
                },
            }
            return query_body

        # accept ndarray as query vector
        output = docsearch.similarity_search_by_vector_with_relevance_scores(
            np.array(query_vector),  # type: ignore
            k=1,
            custom_query=assert_query,
        )
        assert output == [(Document(page_content="foo"), 1.0)]

    def test_similarity_search_approx_with_hybrid_search_rrf(
        self, es_client: Any, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end construction and rrf hybrid search with metadata."""
        from functools import partial
        from typing import Optional

        # 1. check query_body is okay
        rrf_test_cases: List[Optional[Union[dict, bool]]] = [
            True,
            False,
            {"rank_constant": 1, "window_size": 5},
        ]
        for rrf_test_case in rrf_test_cases:
            texts = ["foo", "bar", "baz"]
            docsearch = ElasticsearchStore.from_texts(
                texts,
                FakeEmbeddings(),
                **elasticsearch_connection,
                index_name=index_name,
                strategy=ElasticsearchStore.ApproxRetrievalStrategy(
                    hybrid=True, rrf=rrf_test_case
                ),
            )

            def assert_query(
                query_body: dict,
                query: str,
                rrf: Optional[Union[dict, bool]] = True,
            ) -> dict:
                cmp_query_body = {
                    "knn": {
                        "field": "vector",
                        "filter": [],
                        "k": 3,
                        "num_candidates": 50,
                        "query_vector": [
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.0,
                        ],
                    },
                    "query": {
                        "bool": {
                            "filter": [],
                            "must": [{"match": {"text": {"query": "foo"}}}],
                        }
                    },
                }

                if isinstance(rrf, dict):
                    cmp_query_body["rank"] = {"rrf": rrf}
                elif isinstance(rrf, bool) and rrf is True:
                    cmp_query_body["rank"] = {"rrf": {}}

                assert query_body == cmp_query_body

                return query_body

            ## without fetch_k parameter
            output = docsearch.similarity_search(
                "foo", k=3, custom_query=partial(assert_query, rrf=rrf_test_case)
            )

        # 2. check query result is okay
        es_output = es_client.search(
            index=index_name,
            query={
                "bool": {
                    "filter": [],
                    "must": [{"match": {"text": {"query": "foo"}}}],
                }
            },
            knn={
                "field": "vector",
                "filter": [],
                "k": 3,
                "num_candidates": 50,
                "query_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            },
            size=3,
            rank={"rrf": {"rank_constant": 1, "window_size": 5}},
        )

        assert [o.page_content for o in output] == [
            e["_source"]["text"] for e in es_output["hits"]["hits"]
        ]

        # 3. check rrf default option is okay
        docsearch = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True),
        )

        ## with fetch_k parameter
        output = docsearch.similarity_search(
            "foo", k=3, fetch_k=50, custom_query=assert_query
        )

    def test_similarity_search_approx_with_custom_query_fn(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """test that custom query function is called
        with the query string and query body"""

        def my_custom_query(query_body: dict, query: str) -> dict:
            assert query == "foo"
            assert query_body == {
                "knn": {
                    "field": "vector",
                    "filter": [],
                    "k": 1,
                    "num_candidates": 50,
                    "query_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                }
            }
            return {"query": {"match": {"text": {"query": "bar"}}}}

        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts, FakeEmbeddings(), **elasticsearch_connection, index_name=index_name
        )
        output = docsearch.similarity_search("foo", k=1, custom_query=my_custom_query)
        assert output == [Document(page_content="bar")]

    @pytest.mark.skipif(
        not model_is_deployed(create_es_client(), TRANSFORMER_MODEL_ID),
        reason=f"{TRANSFORMER_MODEL_ID} model not deployed in ML Node, "
        "skipping test",
    )
    def test_similarity_search_with_approx_infer_instack(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """test end to end with approx retrieval strategy and inference in-stack"""
        docsearch = ElasticsearchStore(
            index_name=index_name,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(
                query_model_id="sentence-transformers__all-minilm-l6-v2"
            ),
            query_field="text_field",
            vector_query_field="vector_query_field.predicted_value",
            **elasticsearch_connection,
        )

        # setting up the pipeline for inference
        docsearch.client.ingest.put_pipeline(
            id="test_pipeline",
            processors=[
                {
                    "inference": {
                        "model_id": TRANSFORMER_MODEL_ID,
                        "field_map": {"query_field": "text_field"},
                        "target_field": "vector_query_field",
                    }
                }
            ],
        )

        # creating a new index with the pipeline,
        # not relying on langchain to create the index
        docsearch.client.indices.create(
            index=index_name,
            mappings={
                "properties": {
                    "text_field": {"type": "text"},
                    "vector_query_field": {
                        "properties": {
                            "predicted_value": {
                                "type": "dense_vector",
                                "dims": 384,
                                "index": True,
                                "similarity": "l2_norm",
                            }
                        }
                    },
                }
            },
            settings={"index": {"default_pipeline": "test_pipeline"}},
        )

        # adding documents to the index
        texts = ["foo", "bar", "baz"]

        for i, text in enumerate(texts):
            docsearch.client.create(
                index=index_name,
                id=str(i),
                document={"text_field": text, "metadata": {}},
            )

        docsearch.client.indices.refresh(index=index_name)

        def assert_query(query_body: dict, query: str) -> dict:
            assert query_body == {
                "knn": {
                    "filter": [],
                    "field": "vector_query_field.predicted_value",
                    "k": 1,
                    "num_candidates": 50,
                    "query_vector_builder": {
                        "text_embedding": {
                            "model_id": TRANSFORMER_MODEL_ID,
                            "model_text": "foo",
                        }
                    },
                }
            }
            return query_body

        output = docsearch.similarity_search("foo", k=1, custom_query=assert_query)
        assert output == [Document(page_content="foo")]

        output = docsearch.similarity_search("bar", k=1)
        assert output == [Document(page_content="bar")]

    @pytest.mark.skipif(
        not model_is_deployed(create_es_client(), ELSER_MODEL_ID),
        reason=f"{ELSER_MODEL_ID} model not deployed in ML Node, skipping test",
    )
    def test_similarity_search_with_sparse_infer_instack(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """test end to end with sparse retrieval strategy and inference in-stack"""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            **elasticsearch_connection,
            index_name=index_name,
            strategy=ElasticsearchStore.SparseVectorRetrievalStrategy(ELSER_MODEL_ID),
        )
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]

    def test_deployed_model_check_fails_approx(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """test that exceptions are raised if a specified model is not deployed"""
        with pytest.raises(NotFoundError):
            ElasticsearchStore.from_texts(
                texts=["foo", "bar", "baz"],
                embedding=ConsistentFakeEmbeddings(10),
                **elasticsearch_connection,
                index_name=index_name,
                strategy=ElasticsearchStore.ApproxRetrievalStrategy(
                    query_model_id="non-existing model ID",
                ),
            )

    def test_deployed_model_check_fails_sparse(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """test that exceptions are raised if a specified model is not deployed"""
        with pytest.raises(NotFoundError):
            ElasticsearchStore.from_texts(
                texts=["foo", "bar", "baz"],
                **elasticsearch_connection,
                index_name=index_name,
                strategy=ElasticsearchStore.SparseVectorRetrievalStrategy(
                    model_id="non-existing model ID"
                ),
            )

    def test_elasticsearch_with_relevance_score(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test to make sure the relevance score is scaled to 0-1."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        embeddings = FakeEmbeddings()

        docsearch = ElasticsearchStore.from_texts(
            index_name=index_name,
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            **elasticsearch_connection,
        )

        embedded_query = embeddings.embed_query("foo")
        output = docsearch.similarity_search_by_vector_with_relevance_scores(
            embedding=embedded_query, k=1
        )
        assert output == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]

    def test_similarity_search_bm25_search(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to end using the BM25 retrieval strategy."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            None,
            **elasticsearch_connection,
            index_name=index_name,
            strategy=ElasticsearchStore.BM25RetrievalStrategy(),
        )

        def assert_query(query_body: dict, query: str) -> dict:
            assert query_body == {
                "query": {
                    "bool": {
                        "must": [{"match": {"text": {"query": "foo"}}}],
                        "filter": [],
                    }
                }
            }
            return query_body

        output = docsearch.similarity_search("foo", k=1, custom_query=assert_query)
        assert output == [Document(page_content="foo")]

    def test_similarity_search_bm25_search_with_filter(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test end to using the BM25 retrieval strategy with metadata."""
        texts = ["foo", "foo", "foo"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            None,
            **elasticsearch_connection,
            index_name=index_name,
            metadatas=metadatas,
            strategy=ElasticsearchStore.BM25RetrievalStrategy(),
        )

        def assert_query(query_body: dict, query: str) -> dict:
            assert query_body == {
                "query": {
                    "bool": {
                        "must": [{"match": {"text": {"query": "foo"}}}],
                        "filter": [{"term": {"metadata.page": 1}}],
                    }
                }
            }
            return query_body

        output = docsearch.similarity_search(
            "foo",
            k=3,
            custom_query=assert_query,
            filter=[{"term": {"metadata.page": 1}}],
        )
        assert output == [Document(page_content="foo", metadata={"page": 1})]

    def test_elasticsearch_with_relevance_threshold(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test to make sure the relevance threshold is respected."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        embeddings = FakeEmbeddings()

        docsearch = ElasticsearchStore.from_texts(
            index_name=index_name,
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            **elasticsearch_connection,
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

    def test_elasticsearch_delete_ids(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """Test delete methods from vector store."""
        texts = ["foo", "bar", "baz", "gni"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = ElasticsearchStore(
            embedding=ConsistentFakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
        )

        ids = docsearch.add_texts(texts, metadatas)
        output = docsearch.similarity_search("foo", k=10)
        assert len(output) == 4

        docsearch.delete(ids[1:3])
        output = docsearch.similarity_search("foo", k=10)
        assert len(output) == 2

        docsearch.delete(["not-existing"])
        output = docsearch.similarity_search("foo", k=10)
        assert len(output) == 2

        docsearch.delete([ids[0]])
        output = docsearch.similarity_search("foo", k=10)
        assert len(output) == 1

        docsearch.delete([ids[3]])
        output = docsearch.similarity_search("gni", k=10)
        assert len(output) == 0

    def test_elasticsearch_indexing_exception_error(
        self,
        elasticsearch_connection: dict,
        index_name: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test bulk exception logging is giving better hints."""

        docsearch = ElasticsearchStore(
            embedding=ConsistentFakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
        )

        docsearch.client.indices.create(
            index=index_name,
            mappings={"properties": {}},
            settings={"index": {"default_pipeline": "not-existing-pipeline"}},
        )

        texts = ["foo"]

        with pytest.raises(BulkIndexError):
            docsearch.add_texts(texts)

        error_reason = "pipeline with id [not-existing-pipeline] does not exist"
        log_message = f"First error reason: {error_reason}"

        assert log_message in caplog.text

    def test_elasticsearch_with_user_agent(
        self, es_client: Any, index_name: str
    ) -> None:
        """Test to make sure the user-agent is set correctly."""

        texts = ["foo", "bob", "baz"]
        ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            es_connection=es_client,
            index_name=index_name,
        )

        user_agent = es_client.transport.requests[0]["headers"]["User-Agent"]
        assert (
            re.match(r"^langchain-py-vs/\d+\.\d+\.\d+$", user_agent) is not None
        ), f"The string '{user_agent}' does not match the expected pattern."

    def test_elasticsearch_with_internal_user_agent(
        self, elasticsearch_connection: Dict, index_name: str
    ) -> None:
        """Test to make sure the user-agent is set correctly."""

        texts = ["foo"]
        store = ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            **elasticsearch_connection,
            index_name=index_name,
        )

        user_agent = store.client._headers["User-Agent"]
        assert (
            re.match(r"^langchain-py-vs/\d+\.\d+\.\d+$", user_agent) is not None
        ), f"The string '{user_agent}' does not match the expected pattern."

    def test_bulk_args(self, es_client: Any, index_name: str) -> None:
        """Test to make sure the bulk arguments work as expected."""

        texts = ["foo", "bob", "baz"]
        ElasticsearchStore.from_texts(
            texts,
            FakeEmbeddings(),
            es_connection=es_client,
            index_name=index_name,
            bulk_kwargs={"chunk_size": 1},
        )

        # 1 for index exist, 1 for index create, 3 for index docs
        assert len(es_client.transport.requests) == 5  # type: ignore
