"""Test AsyncElasticsearchStore functionality."""

import logging
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union

import pytest
from elasticsearch import NotFoundError
from langchain_core.documents import Document

from langchain_elasticsearch.vectorstores import ElasticsearchStore
from tests.fake_embeddings import ConsistentFakeEmbeddings

from ._test_utilities import clear_test_indices, create_es_client, read_env

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

    @pytest.mark.sync
    def test_from_texts_similarity_search_with_doc_builder(
        self, es_params: dict, index_name: str
    ) -> None:
        texts = ["foo", "foo", "foo"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            ConsistentFakeEmbeddings(),
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

    @pytest.mark.sync
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

    @pytest.mark.sync
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

    # Also tested in elasticsearch.helpers.vectorstore

    @pytest.mark.sync
    def test_similarity_search_without_metadata(
        self, es_params: dict, index_name: str
    ) -> None:
        """Test end to end construction and search without metadata."""

        def assert_query(
            query_body: Dict[str, Any], query: Optional[str]
        ) -> Dict[str, Any]:
            assert query_body == {
                "knn": {
                    "field": "vector",
                    "filter": [],
                    "k": 1,
                    "num_candidates": 50,
                    "query_vector": [
                        0.06,
                        0.07,
                        0.01,
                        0.08,
                        0.03,
                        0.07,
                        0.09,
                        0.03,
                        0.09,
                        0.09,
                        0.04,
                        0.03,
                        0.08,
                        0.07,
                        0.06,
                        0.08,
                    ],
                }
            }
            return query_body

        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            ConsistentFakeEmbeddings(),
            **es_params,
            index_name=index_name,
        )
        output = docsearch.similarity_search("foo", k=1, custom_query=assert_query)
        assert output == [Document(page_content="foo")]

    @pytest.mark.sync
    def test_add_embeddings(self, es_params: dict, index_name: str) -> None:
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

        docsearch = ElasticsearchStore(
            embedding=embeddings,
            **es_params,
            index_name=index_name,
        )
        docsearch.add_embeddings(list(zip(text_input, embedding_vectors)), metadatas)
        output = docsearch.similarity_search("foo1", k=1)
        assert output == [Document(page_content="foo3", metadata={"page": 2})]

    @pytest.mark.sync
    def test_similarity_search_with_metadata(
        self, es_params: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            ConsistentFakeEmbeddings(),
            metadatas=metadatas,
            **es_params,
            index_name=index_name,
        )

        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo", metadata={"page": 0})]

        output = docsearch.similarity_search("bar", k=1)
        assert output == [Document(page_content="bar", metadata={"page": 1})]

    @pytest.mark.sync
    def test_similarity_search_with_filter(
        self, es_params: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "foo", "foo"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            ConsistentFakeEmbeddings(),
            metadatas=metadatas,
            **es_params,
            index_name=index_name,
        )

        def assert_query(
            query_body: Dict[str, Any], query: Optional[str]
        ) -> Dict[str, Any]:
            assert query_body == {
                "knn": {
                    "field": "vector",
                    "filter": [{"term": {"metadata.page": "1"}}],
                    "k": 3,
                    "num_candidates": 50,
                    "query_vector": [
                        0.06,
                        0.07,
                        0.01,
                        0.08,
                        0.03,
                        0.07,
                        0.09,
                        0.03,
                        0.09,
                        0.09,
                        0.04,
                        0.03,
                        0.08,
                        0.07,
                        0.06,
                        0.08,
                    ],
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

    @pytest.mark.sync
    def test_similarity_search_with_doc_builder(
        self, es_params: dict, index_name: str
    ) -> None:
        texts = ["foo", "foo", "foo"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            ConsistentFakeEmbeddings(),
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

    @pytest.mark.sync
    def test_similarity_search_exact_search(
        self, es_params: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            ConsistentFakeEmbeddings(),
            **es_params,
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
                                0.06,
                                0.07,
                                0.01,
                                0.08,
                                0.03,
                                0.07,
                                0.09,
                                0.03,
                                0.09,
                                0.09,
                                0.04,
                                0.03,
                                0.08,
                                0.07,
                                0.06,
                                0.08,
                            ]
                        },
                    },
                }
            }
        }

        def assert_query(
            query_body: Dict[str, Any], query: Optional[str]
        ) -> Dict[str, Any]:
            assert query_body == expected_query
            return query_body

        output = docsearch.similarity_search("foo", k=1, custom_query=assert_query)
        assert output == [Document(page_content="foo")]

    @pytest.mark.sync
    def test_similarity_search_exact_search_with_filter(
        self, es_params: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            ConsistentFakeEmbeddings(),
            **es_params,
            index_name=index_name,
            metadatas=metadatas,
            strategy=ElasticsearchStore.ExactRetrievalStrategy(),
        )

        def assert_query(
            query_body: Dict[str, Any], query: Optional[str]
        ) -> Dict[str, Any]:
            expected_query = {
                "query": {
                    "script_score": {
                        "query": {"bool": {"filter": [{"term": {"metadata.page": 0}}]}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",  # noqa: E501
                            "params": {
                                "query_vector": [
                                    0.06,
                                    0.07,
                                    0.01,
                                    0.08,
                                    0.03,
                                    0.07,
                                    0.09,
                                    0.03,
                                    0.09,
                                    0.09,
                                    0.04,
                                    0.03,
                                    0.08,
                                    0.07,
                                    0.06,
                                    0.08,
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

    @pytest.mark.sync
    def test_similarity_search_exact_search_distance_dot_product(
        self, es_params: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            ConsistentFakeEmbeddings(),
            **es_params,
            index_name=index_name,
            strategy=ElasticsearchStore.ExactRetrievalStrategy(),
            distance_strategy="DOT_PRODUCT",
        )

        def assert_query(
            query_body: Dict[str, Any], query: Optional[str]
        ) -> Dict[str, Any]:
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
                                    0.06,
                                    0.07,
                                    0.01,
                                    0.08,
                                    0.03,
                                    0.07,
                                    0.09,
                                    0.03,
                                    0.09,
                                    0.09,
                                    0.04,
                                    0.03,
                                    0.08,
                                    0.07,
                                    0.06,
                                    0.08,
                                ]
                            },
                        },
                    }
                }
            }
            return query_body

        output = docsearch.similarity_search("foo", k=1, custom_query=assert_query)
        assert output == [Document(page_content="foo")]

    @pytest.mark.sync
    def test_similarity_search_exact_search_unknown_distance_strategy(
        self, es_params: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with unknown distance strategy."""

        with pytest.raises(KeyError):
            texts = ["foo", "bar", "baz"]
            ElasticsearchStore.from_texts(
                texts,
                ConsistentFakeEmbeddings(),
                **es_params,
                index_name=index_name,
                strategy=ElasticsearchStore.ExactRetrievalStrategy(),
                distance_strategy="NOT_A_STRATEGY",
            )

    @pytest.mark.sync
    def test_max_marginal_relevance_search(
        self, es_params: dict, index_name: str
    ) -> None:
        """Test max marginal relevance search."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            ConsistentFakeEmbeddings(),
            **es_params,
            index_name=index_name,
            strategy=ElasticsearchStore.ExactRetrievalStrategy(),
        )

        mmr_output = docsearch.max_marginal_relevance_search(texts[0], k=3, fetch_k=3)
        sim_output = docsearch.similarity_search(texts[0], k=3)
        assert mmr_output == sim_output

        mmr_output = docsearch.max_marginal_relevance_search(texts[0], k=2, fetch_k=3)
        assert len(mmr_output) == 2
        assert mmr_output[0].page_content == texts[0]
        # baz is more similar to foo when using ConsistentFakeEmbeddings
        assert mmr_output[1].page_content == texts[2]

        mmr_output = docsearch.max_marginal_relevance_search(
            texts[0],
            k=2,
            fetch_k=3,
            lambda_mult=0.1,  # more diversity
        )
        assert len(mmr_output) == 2
        assert mmr_output[0].page_content == texts[0]
        # with more diversity, prefer bar over baz
        assert mmr_output[1].page_content == texts[1]

        # if fetch_k < k, then the output will be less than k
        mmr_output = docsearch.max_marginal_relevance_search(texts[0], k=3, fetch_k=2)
        assert len(mmr_output) == 2

    @pytest.mark.sync
    def test_similarity_search_approx_with_hybrid_search(
        self, es_params: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            ConsistentFakeEmbeddings(),
            **es_params,
            index_name=index_name,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True),
        )

        def assert_query(
            query_body: Dict[str, Any], query: Optional[str]
        ) -> Dict[str, Any]:
            assert query_body == {
                "knn": {
                    "field": "vector",
                    "filter": [],
                    "k": 1,
                    "num_candidates": 50,
                    "query_vector": [
                        0.06,
                        0.07,
                        0.01,
                        0.08,
                        0.03,
                        0.07,
                        0.09,
                        0.03,
                        0.09,
                        0.09,
                        0.04,
                        0.03,
                        0.08,
                        0.07,
                        0.06,
                        0.08,
                    ],
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

    @pytest.mark.sync
    def test_similarity_search_approx_by_vector(
        self, es_params: dict, index_name: str
    ) -> None:
        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        embeddings = ConsistentFakeEmbeddings()
        docsearch = ElasticsearchStore.from_texts(
            texts,
            embedding=embeddings,
            **es_params,
            index_name=index_name,
        )
        query_vector = embeddings.embed_query("foo")

        def assert_query(
            query_body: Dict[str, Any], query: Optional[str]
        ) -> Dict[str, Any]:
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
            query_vector,
            k=1,
            custom_query=assert_query,
        )
        doc, score = output[0]

        assert doc == Document(page_content="foo")
        assert score == pytest.approx(1.0, rel=0.05)

    @pytest.mark.sync
    def test_similarity_search_approx_with_hybrid_search_rrf(
        self, es_params: dict, index_name: str
    ) -> None:
        """Test end to end construction and rrf hybrid search with metadata."""
        from functools import partial

        # 1. check query_body is okay
        rrf_test_cases: List[Optional[Union[dict, bool]]] = [
            True,
            False,
            {"rank_constant": 1, "rank_window_size": 5},
        ]
        for rrf_test_case in rrf_test_cases:
            texts = ["foo", "bar", "baz"]
            docsearch = ElasticsearchStore.from_texts(
                texts,
                ConsistentFakeEmbeddings(),
                **es_params,
                index_name=index_name,
                strategy=ElasticsearchStore.ApproxRetrievalStrategy(
                    hybrid=True, rrf=rrf_test_case
                ),
            )

            def assert_query(
                query_body: Dict[str, Any],
                query: Optional[str],
                rrf: Optional[Union[dict, bool]] = True,
            ) -> dict:
                cmp_query_body = {
                    "knn": {
                        "field": "vector",
                        "filter": [],
                        "k": 3,
                        "num_candidates": 50,
                        "query_vector": [
                            0.06,
                            0.07,
                            0.01,
                            0.08,
                            0.03,
                            0.07,
                            0.09,
                            0.03,
                            0.09,
                            0.09,
                            0.04,
                            0.03,
                            0.08,
                            0.07,
                            0.06,
                            0.08,
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
        es_output = docsearch.client.search(
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
                "query_vector": [
                    0.06,
                    0.07,
                    0.01,
                    0.08,
                    0.03,
                    0.07,
                    0.09,
                    0.03,
                    0.09,
                    0.09,
                    0.04,
                    0.03,
                    0.08,
                    0.07,
                    0.06,
                    0.08,
                ],
            },
            size=3,
            rank={"rrf": {"rank_constant": 1, "rank_window_size": 5}},
        )

        assert [o.page_content for o in output] == [
            e["_source"]["text"] for e in es_output["hits"]["hits"]
        ]

        # 3. check rrf default option is okay
        docsearch = ElasticsearchStore.from_texts(
            texts,
            ConsistentFakeEmbeddings(),
            **es_params,
            index_name=index_name,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True),
        )

        ## with fetch_k parameter
        output = docsearch.similarity_search(
            "foo", k=3, fetch_k=50, custom_query=assert_query
        )

    @pytest.mark.sync
    def test_similarity_search_approx_with_custom_query_fn(
        self, es_params: dict, index_name: str
    ) -> None:
        """test that custom query function is called
        with the query string and query body"""

        def my_custom_query(
            query_body: Dict[str, Any], query: Optional[str]
        ) -> Dict[str, Any]:
            assert query == "foo"
            assert query_body == {
                "knn": {
                    "field": "vector",
                    "filter": [],
                    "k": 1,
                    "num_candidates": 50,
                    "query_vector": [
                        0.06,
                        0.07,
                        0.01,
                        0.08,
                        0.03,
                        0.07,
                        0.09,
                        0.03,
                        0.09,
                        0.09,
                        0.04,
                        0.03,
                        0.08,
                        0.07,
                        0.06,
                        0.08,
                    ],
                }
            }
            return {"query": {"match": {"text": {"query": "bar"}}}}

        """Test end to end construction and search with metadata."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts, ConsistentFakeEmbeddings(), **es_params, index_name=index_name
        )
        output = docsearch.similarity_search("foo", k=1, custom_query=my_custom_query)
        assert output == [Document(page_content="bar")]

    @pytest.mark.sync
    def test_deployed_model_check_fails_approx(
        self, es_params: dict, index_name: str
    ) -> None:
        """test that exceptions are raised if a specified model is not deployed"""
        with pytest.raises(NotFoundError):
            ElasticsearchStore.from_texts(
                texts=["foo", "bar", "baz"],
                embedding=ConsistentFakeEmbeddings(),
                **es_params,
                index_name=index_name,
                strategy=ElasticsearchStore.ApproxRetrievalStrategy(
                    query_model_id="non-existing model ID",
                ),
            )

    @pytest.mark.sync
    def test_deployed_model_check_fails_sparse(
        self, es_params: dict, index_name: str
    ) -> None:
        """test that exceptions are raised if a specified model is not deployed"""
        with pytest.raises(NotFoundError):
            ElasticsearchStore.from_texts(
                texts=["foo", "bar", "baz"],
                **es_params,
                index_name=index_name,
                strategy=ElasticsearchStore.SparseVectorRetrievalStrategy(
                    model_id="non-existing model ID"
                ),
            )

    @pytest.mark.sync
    def test_elasticsearch_with_relevance_score(
        self, es_params: dict, index_name: str
    ) -> None:
        """Test to make sure the relevance score is scaled to 0-1."""
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

        embedded_query = embeddings.embed_query("foo")
        output = docsearch.similarity_search_by_vector_with_relevance_scores(
            embedding=embedded_query, k=1
        )
        doc, score = output[0]

        assert doc == Document(page_content="foo", metadata={"page": "0"})
        assert score == pytest.approx(1.0, rel=0.05)

    @pytest.mark.sync
    def test_similarity_search_bm25_search(
        self, es_params: dict, index_name: str
    ) -> None:
        """Test end to end using the BM25 retrieval strategy."""
        texts = ["foo", "bar", "baz"]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            None,
            **es_params,
            index_name=index_name,
            strategy=ElasticsearchStore.BM25RetrievalStrategy(),
        )

        def assert_query(
            query_body: Dict[str, Any], query: Optional[str]
        ) -> Dict[str, Any]:
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

    @pytest.mark.sync
    def test_similarity_search_bm25_search_with_filter(
        self, es_params: dict, index_name: str
    ) -> None:
        """Test end to using the BM25 retrieval strategy with metadata."""
        texts = ["foo", "foo", "foo"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = ElasticsearchStore.from_texts(
            texts,
            None,
            **es_params,
            index_name=index_name,
            metadatas=metadatas,
            strategy=ElasticsearchStore.BM25RetrievalStrategy(),
        )

        def assert_query(
            query_body: Dict[str, Any], query: Optional[str]
        ) -> Dict[str, Any]:
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

    @pytest.mark.sync
    def test_elasticsearch_with_relevance_threshold(
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

    @pytest.mark.sync
    def test_elasticsearch_delete_ids(self, es_params: dict, index_name: str) -> None:
        """Test delete methods from vector store."""
        texts = ["foo", "bar", "baz", "gni"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = ElasticsearchStore(
            embedding=ConsistentFakeEmbeddings(),
            **es_params,
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
