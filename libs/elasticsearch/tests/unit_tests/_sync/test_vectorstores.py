"""Test Elasticsearch functionality."""

import re
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from elasticsearch import Elasticsearch
from langchain_elasticsearch._sync.vectorstores import _convert_retrieval_strategy
from langchain_elasticsearch._utilities import _hits_to_docs_scores
from langchain_elasticsearch.embeddings import Embeddings, EmbeddingServiceAdapter
from langchain_elasticsearch.vectorstores import (
    ApproxRetrievalStrategy,
    BM25RetrievalStrategy,
    BM25Strategy,
    DenseVectorScriptScoreStrategy,
    DenseVectorStrategy,
    DistanceMetric,
    DistanceStrategy,
    ElasticsearchStore,
    ExactRetrievalStrategy,
    SparseRetrievalStrategy,
    SparseVectorStrategy,
)

from ...fake_embeddings import ConsistentFakeEmbeddings


class TestHitsToDocsScores:
    def test_basic(self) -> None:
        content_field = "content"
        hits = [
            {
                "_score": 11,
                "_source": {content_field: "abc", "metadata": {"meta1": "one"}},
            },
            {
                "_score": 22,
                "_source": {content_field: "def", "metadata": {"meta2": "two"}},
            },
        ]
        expected = [
            (Document("abc", metadata={"meta1": "one"}), 11),
            (Document("def", metadata={"meta2": "two"}), 22),
        ]
        actual = _hits_to_docs_scores(hits, content_field)
        assert actual == expected

    def test_custom_builder(self) -> None:
        content_field = "content"
        hits = [
            {
                "_score": 11,
                "_source": {content_field: "abc", "metadata": {"meta1": "one"}},
            },
            {
                "_score": 22,
                "_source": {content_field: "def", "metadata": {"meta2": "two"}},
            },
        ]

        def custom_builder(hit: Dict) -> Document:
            return Document("static", metadata={"score": hit["_score"]})

        expected = [
            (Document("static", metadata={"score": 11}), 11),
            (Document("static", metadata={"score": 22}), 22),
        ]
        actual = _hits_to_docs_scores(hits, content_field, doc_builder=custom_builder)
        assert actual == expected

    def test_fields(self) -> None:
        content_field = "content"
        extra_field = "extra"
        hits = [
            {
                "_score": 11,
                "_source": {
                    content_field: "abc",
                    extra_field: "extra1",
                    "ignore_me": "please",
                },
            },
            {"_score": 22, "_source": {content_field: "def", extra_field: "extra2"}},
        ]
        expected = [
            (Document("abc", metadata={extra_field: "extra1"}), 11),
            (Document("def", metadata={extra_field: "extra2"}), 22),
        ]
        actual = _hits_to_docs_scores(hits, content_field, fields=[extra_field])
        assert actual == expected

    def test_missing_content_field(self) -> None:
        content_field = "content"
        hits = [
            {
                "_score": 11,
                "_source": {content_field: "abc", "metadata": {"meta1": "one"}},
            },
            {
                "_score": 22,
                "_source": {content_field: "def", "metadata": {"meta2": "two"}},
            },
        ]
        expected = [
            (Document("", metadata={"meta1": "one"}), 11),
            (Document("", metadata={"meta2": "two"}), 22),
        ]
        actual = _hits_to_docs_scores(hits, "missing_content_field")
        assert actual == expected

    def test_missing_metadata_field(self) -> None:
        content_field = "content"
        hits = [
            {"_score": 11, "_source": {content_field: "abc"}},  # missing metadata
        ]
        expected = [
            (Document("abc", metadata={}), 11),  # empty metadata
        ]
        actual = _hits_to_docs_scores(hits, content_field)
        assert actual == expected

    def test_doc_field_to_metadata(self) -> None:
        content_field = "content"
        other_field = "other"
        hits = [
            {
                "_score": 11,
                "_source": {
                    content_field: "abc",
                    other_field: "foo",
                    "metadata": {"meta1": "one"},
                },
            },
            {
                "_score": 22,
                "_source": {
                    content_field: "def",
                    other_field: "bar",
                    "metadata": {"meta2": "two"},
                },
            },
        ]
        expected = [
            (Document("abc", metadata={"meta1": "one", other_field: "foo"}), 11),
            (Document("def", metadata={"meta2": "two", other_field: "bar"}), 22),
        ]
        actual = _hits_to_docs_scores(
            hits, content_field=content_field, fields=[other_field]
        )
        assert actual == expected


class TestConvertStrategy:
    def test_dense_approx(self) -> None:
        actual = _convert_retrieval_strategy(
            ApproxRetrievalStrategy(query_model_id="my model", hybrid=True, rrf=False),
            distance=DistanceStrategy.DOT_PRODUCT,
        )
        assert isinstance(actual, DenseVectorStrategy)
        assert actual.distance == DistanceMetric.DOT_PRODUCT
        assert actual.model_id == "my model"
        assert actual.hybrid is True
        assert actual.rrf is False

    def test_dense_exact(self) -> None:
        actual = _convert_retrieval_strategy(
            ExactRetrievalStrategy(), distance=DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        assert isinstance(actual, DenseVectorScriptScoreStrategy)
        assert actual.distance == DistanceMetric.EUCLIDEAN_DISTANCE

    def test_sparse(self) -> None:
        actual = _convert_retrieval_strategy(
            SparseRetrievalStrategy(model_id="my model ID")
        )
        assert isinstance(actual, SparseVectorStrategy)
        assert actual.model_id == "my model ID"

    def test_bm25(self) -> None:
        actual = _convert_retrieval_strategy(BM25RetrievalStrategy(k1=1.7, b=5.4))
        assert isinstance(actual, BM25Strategy)
        assert actual.k1 == 1.7
        assert actual.b == 5.4


class TestVectorStore:
    @pytest.fixture
    def embeddings(self) -> Embeddings:
        return ConsistentFakeEmbeddings()

    @pytest.fixture
    def store(self) -> Generator[ElasticsearchStore, None, None]:
        client = Elasticsearch(hosts=["http://dummy:9200"])  # never connected to
        store = ElasticsearchStore(index_name="test_index", es_connection=client)
        try:
            yield store
        finally:
            store.close()

    @pytest.fixture
    def hybrid_store(
        self, embeddings: Embeddings
    ) -> Generator[ElasticsearchStore, None, None]:
        client = Elasticsearch(hosts=["http://dummy:9200"])  # never connected to
        store = ElasticsearchStore(
            index_name="test_index",
            embedding=embeddings,
            strategy=ApproxRetrievalStrategy(hybrid=True),
            es_connection=client,
        )
        try:
            yield store
        finally:
            store.close()

    @pytest.fixture
    def static_hits(self) -> List[Dict[str, Any]]:
        default_content_field = "text"
        return [
            {"_score": 1, "_source": {default_content_field: "test", "metadata": {}}}
        ]

    @staticmethod
    def dummy_custom_query(query_body: dict, query: Optional[str]) -> Dict[str, Any]:
        return {"dummy": "query"}

    def test_agent_header(self, store: ElasticsearchStore) -> None:
        agent = store.client._headers["User-Agent"]
        assert (
            re.match(r"^langchain-py-vs/\d+\.\d+\.\d+(?:rc\d+)?(?:\.dev\d+)?$", agent)
            is not None
        ), f"The string '{agent}' does not match the expected pattern."

    @pytest.mark.sync
    def test_similarity_search(
        self, store: ElasticsearchStore, static_hits: List[Dict]
    ) -> None:
        store._store.search = Mock(return_value=static_hits)  # type: ignore[assignment]
        actual1 = store.similarity_search(
            query="test",
            k=7,
            fetch_k=34,
            filter=[{"f": 1}],
            custom_query=self.dummy_custom_query,
        )
        assert actual1 == [Document("test")]
        store._store.search.assert_called_with(
            query="test",
            k=7,
            num_candidates=34,
            filter=[{"f": 1}],
            custom_query=self.dummy_custom_query,
        )

        store._store.search = Mock(return_value=static_hits)  # type: ignore[assignment]

        actual2 = store.similarity_search_with_score(
            query="test",
            k=7,
            fetch_k=34,
            filter=[{"f": 1}],
            custom_query=self.dummy_custom_query,
        )
        assert actual2 == [(Document("test"), 1)]
        store._store.search.assert_called_with(
            query="test",
            k=7,
            filter=[{"f": 1}],
            custom_query=self.dummy_custom_query,
        )

    @pytest.mark.sync
    def test_similarity_search_by_vector_with_relevance_scores(
        self, store: ElasticsearchStore, static_hits: List[Dict]
    ) -> None:
        store._store.search = Mock(return_value=static_hits)  # type: ignore[assignment]
        actual = store.similarity_search_by_vector_with_relevance_scores(
            embedding=[1, 2, 3],
            k=7,
            fetch_k=34,
            filter=[{"f": 1}],
            custom_query=self.dummy_custom_query,
        )
        assert actual == [(Document("test"), 1)]
        store._store.search.assert_called_with(
            query=None,
            query_vector=[1, 2, 3],
            k=7,
            filter=[{"f": 1}],
            custom_query=self.dummy_custom_query,
        )

    @pytest.mark.sync
    def test_delete(self, store: ElasticsearchStore) -> None:
        store._store.delete = Mock(return_value=True)  # type: ignore[assignment]
        actual = store.delete(
            ids=["10", "20"],
            refresh_indices=True,
        )
        assert actual is True
        store._store.delete.assert_called_with(
            ids=["10", "20"],
            refresh_indices=True,
        )

    @pytest.mark.sync
    def test_add_texts(self, store: ElasticsearchStore) -> None:
        store._store.add_texts = Mock(return_value=["10", "20"])  # type: ignore[assignment]
        actual = store.add_texts(
            texts=["t1", "t2"],
        )
        assert actual == ["10", "20"]
        store._store.add_texts.assert_called_with(
            texts=["t1", "t2"],
            metadatas=None,
            ids=None,
            refresh_indices=True,
            create_index_if_not_exists=True,
            bulk_kwargs=None,
        )

        store._store.add_texts = Mock(return_value=["10", "20"])  # type: ignore[assignment]
        store.add_texts(
            texts=["t1", "t2"],
            metadatas=[{1: 2}, {3: 4}],
            ids=["10", "20"],
            refresh_indices=False,
            create_index_if_not_exists=False,
            bulk_kwargs={"x": "y"},
        )
        store._store.add_texts.assert_called_with(
            texts=["t1", "t2"],
            metadatas=[{1: 2}, {3: 4}],
            ids=["10", "20"],
            refresh_indices=False,
            create_index_if_not_exists=False,
            bulk_kwargs={"x": "y"},
        )

    @pytest.mark.sync
    def test_add_embeddings(self, store: ElasticsearchStore) -> None:
        store._store.add_texts = Mock(return_value=["10", "20"])  # type: ignore[assignment]
        actual = store.add_embeddings(
            text_embeddings=[("t1", [1, 2, 3]), ("t2", [4, 5, 6])],
        )
        assert actual == ["10", "20"]
        store._store.add_texts.assert_called_with(
            texts=["t1", "t2"],
            metadatas=None,
            vectors=[[1, 2, 3], [4, 5, 6]],
            ids=None,
            refresh_indices=True,
            create_index_if_not_exists=True,
            bulk_kwargs=None,
        )

        store._store.add_texts = Mock(return_value=["10", "20"])  # type: ignore[assignment]
        store.add_embeddings(
            text_embeddings=[("t1", [1, 2, 3]), ("t2", [4, 5, 6])],
            metadatas=[{1: 2}, {3: 4}],
            ids=["10", "20"],
            refresh_indices=False,
            create_index_if_not_exists=False,
            bulk_kwargs={"x": "y"},
        )
        store._store.add_texts.assert_called_with(
            texts=["t1", "t2"],
            metadatas=[{1: 2}, {3: 4}],
            vectors=[[1, 2, 3], [4, 5, 6]],
            ids=["10", "20"],
            refresh_indices=False,
            create_index_if_not_exists=False,
            bulk_kwargs={"x": "y"},
        )

    @pytest.mark.sync
    def test_max_marginal_relevance_search(
        self,
        hybrid_store: ElasticsearchStore,
        embeddings: Embeddings,
        static_hits: List[Dict],
    ) -> None:
        hybrid_store._store.max_marginal_relevance_search = Mock(  # type: ignore[assignment]
            return_value=static_hits
        )
        actual = hybrid_store.max_marginal_relevance_search(
            query="qqq",
            k=8,
            fetch_k=19,
            lambda_mult=0.3,
        )
        assert actual == [Document("test")]
        hybrid_store._store.max_marginal_relevance_search.assert_called_with(
            embedding_service=EmbeddingServiceAdapter(embeddings),
            query="qqq",
            vector_field="vector",
            k=8,
            num_candidates=19,
            lambda_mult=0.3,
            fields=None,
            custom_query=None,
        )

    @pytest.mark.sync
    def test_elasticsearch_hybrid_scores_guard(
        self, hybrid_store: ElasticsearchStore
    ) -> None:
        """Ensure an error is raised when search with score in hybrid mode
        because in this case Elasticsearch does not return any score.
        """
        with pytest.raises(ValueError):
            hybrid_store.similarity_search_with_score("foo")

        with pytest.raises(ValueError):
            hybrid_store.similarity_search_by_vector_with_relevance_scores([1, 2, 3])
