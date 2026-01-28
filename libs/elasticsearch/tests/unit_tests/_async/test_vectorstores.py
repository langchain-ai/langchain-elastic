"""Test Elasticsearch functionality."""

import inspect
import re
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers.vectorstore import VectorStore as EVectorStore
from langchain_core.documents import Document

from langchain_elasticsearch._async.vectorstores import (
    ApproxRetrievalStrategy,
    AsyncBM25Strategy,
    AsyncDenseVectorScriptScoreStrategy,
    AsyncDenseVectorStrategy,
    AsyncSparseVectorStrategy,
    _convert_retrieval_strategy,
)
from langchain_elasticsearch._utilities import _hits_to_docs_scores
from langchain_elasticsearch.embeddings import AsyncEmbeddingServiceAdapter, Embeddings
from langchain_elasticsearch.vectorstores import (
    AsyncElasticsearchStore,
    BM25RetrievalStrategy,
    DistanceMetric,
    DistanceStrategy,
    ExactRetrievalStrategy,
    SparseRetrievalStrategy,
)

from ...fake_embeddings import AsyncConsistentFakeEmbeddings


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


@pytest.mark.filterwarnings(
    "ignore::langchain_core._api.deprecation.LangChainPendingDeprecationWarning"
)
class TestConvertStrategy:
    """Test conversion of deprecated retrieval strategies to new ones."""

    def test_dense_approx(self) -> None:
        actual = _convert_retrieval_strategy(
            ApproxRetrievalStrategy(query_model_id="my model", hybrid=True, rrf=False),
            distance=DistanceStrategy.DOT_PRODUCT,
        )
        assert isinstance(actual, AsyncDenseVectorStrategy)
        assert actual.distance == DistanceMetric.DOT_PRODUCT
        assert actual.model_id == "my model"
        assert actual.hybrid is True
        assert actual.rrf is False

    def test_dense_exact(self) -> None:
        actual = _convert_retrieval_strategy(
            ExactRetrievalStrategy(), distance=DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        assert isinstance(actual, AsyncDenseVectorScriptScoreStrategy)
        assert actual.distance == DistanceMetric.EUCLIDEAN_DISTANCE

    def test_sparse(self) -> None:
        actual = _convert_retrieval_strategy(
            SparseRetrievalStrategy(model_id="my model ID")
        )
        assert isinstance(actual, AsyncSparseVectorStrategy)
        assert actual.model_id == "my model ID"

    def test_bm25(self) -> None:
        actual = _convert_retrieval_strategy(BM25RetrievalStrategy(k1=1.7, b=5.4))
        assert isinstance(actual, AsyncBM25Strategy)
        assert actual.k1 == 1.7
        assert actual.b == 5.4


class TestVectorStore:
    @pytest.fixture
    def embeddings(self) -> Embeddings:
        return AsyncConsistentFakeEmbeddings()

    @pytest.fixture
    async def store(self) -> AsyncGenerator:
        client = AsyncElasticsearch(hosts=["http://dummy:9200"])  # never connected to
        store = AsyncElasticsearchStore(index_name="test_index", client=client)
        try:
            yield store
        finally:
            await store.aclose()

    @pytest.fixture
    async def hybrid_store(self, embeddings: Embeddings) -> AsyncGenerator:
        client = AsyncElasticsearch(hosts=["http://dummy:9200"])  # never connected to
        store = AsyncElasticsearchStore(
            index_name="test_index",
            embedding=embeddings,
            strategy=ApproxRetrievalStrategy(hybrid=True),
            client=client,
        )
        try:
            yield store
        finally:
            await store.aclose()

    @pytest.fixture
    def static_hits(self) -> List[Dict[str, Any]]:
        default_content_field = "text"
        return [
            {"_score": 1, "_source": {default_content_field: "test", "metadata": {}}}
        ]

    @staticmethod
    def dummy_custom_query(query_body: dict, query: Optional[str]) -> Dict[str, Any]:
        return {"dummy": "query"}

    def test_agent_header(self, store: AsyncElasticsearchStore) -> None:
        agent = store.client._headers["User-Agent"]
        assert (
            re.match(r"^langchain-py-vs/\d+\.\d+\.\d+(?:rc\d+)?$", agent) is not None
        ), f"The string '{agent}' does not match the expected pattern."

    @pytest.mark.asyncio
    async def test_similarity_search(
        self, store: AsyncElasticsearchStore, static_hits: List[Dict]
    ) -> None:
        store._store.search = AsyncMock(return_value=static_hits)  # type: ignore[assignment]
        actual1 = await store.asimilarity_search(
            query="test",
            k=7,
            fetch_k=34,
            filter=[{"f": 1}],
            custom_query=self.dummy_custom_query,
        )
        assert actual1 == [Document("test")]
        store._store.search.assert_awaited_with(
            query="test",
            k=7,
            num_candidates=34,
            filter=[{"f": 1}],
            custom_query=self.dummy_custom_query,
        )

        store._store.search = AsyncMock(return_value=static_hits)  # type: ignore[assignment]

        actual2 = await store.asimilarity_search_with_score(
            query="test",
            k=7,
            fetch_k=34,
            filter=[{"f": 1}],
            custom_query=self.dummy_custom_query,
        )
        assert actual2 == [(Document("test"), 1)]
        store._store.search.assert_awaited_with(
            query="test",
            k=7,
            num_candidates=34,
            filter=[{"f": 1}],
            custom_query=self.dummy_custom_query,
        )

    @pytest.mark.asyncio
    async def test_similarity_search_by_vector_with_relevance_scores(
        self, store: AsyncElasticsearchStore, static_hits: List[Dict]
    ) -> None:
        store._store.search = AsyncMock(return_value=static_hits)  # type: ignore[assignment]
        actual = await store.asimilarity_search_by_vector_with_relevance_scores(
            embedding=[1, 2, 3],
            k=7,
            fetch_k=34,
            filter=[{"f": 1}],
            custom_query=self.dummy_custom_query,
        )
        assert actual == [(Document("test"), 1)]
        store._store.search.assert_awaited_with(
            query=None,
            query_vector=[1, 2, 3],
            k=7,
            num_candidates=34,
            filter=[{"f": 1}],
            custom_query=self.dummy_custom_query,
        )

    @pytest.mark.asyncio
    async def test_delete(self, store: AsyncElasticsearchStore) -> None:
        store._store.delete = AsyncMock(return_value=True)  # type: ignore[assignment]
        actual = await store.adelete(
            ids=["10", "20"],
            refresh_indices=True,
        )
        assert actual is True
        store._store.delete.assert_awaited_with(
            ids=["10", "20"],
            refresh_indices=True,
        )

    @pytest.mark.asyncio
    async def test_add_texts(self, store: AsyncElasticsearchStore) -> None:
        store._store.add_texts = AsyncMock(return_value=["10", "20"])  # type: ignore[assignment]
        actual = await store.aadd_texts(
            texts=["t1", "t2"],
        )
        assert actual == ["10", "20"]
        store._store.add_texts.assert_awaited_with(
            texts=["t1", "t2"],
            metadatas=None,
            ids=None,
            refresh_indices=True,
            create_index_if_not_exists=True,
            bulk_kwargs=None,
        )

        store._store.add_texts = AsyncMock(return_value=["10", "20"])  # type: ignore[assignment]
        await store.aadd_texts(
            texts=["t1", "t2"],
            metadatas=[{1: 2}, {3: 4}],
            ids=["10", "20"],
            refresh_indices=False,
            create_index_if_not_exists=False,
            bulk_kwargs={"x": "y"},
        )
        store._store.add_texts.assert_awaited_with(
            texts=["t1", "t2"],
            metadatas=[{1: 2}, {3: 4}],
            ids=["10", "20"],
            refresh_indices=False,
            create_index_if_not_exists=False,
            bulk_kwargs={"x": "y"},
        )

    @pytest.mark.asyncio
    async def test_add_embeddings(self, store: AsyncElasticsearchStore) -> None:
        store._store.add_texts = AsyncMock(return_value=["10", "20"])  # type: ignore[assignment]
        actual = await store.aadd_embeddings(
            text_embeddings=[("t1", [1, 2, 3]), ("t2", [4, 5, 6])],
        )
        assert actual == ["10", "20"]
        store._store.add_texts.assert_awaited_with(
            texts=["t1", "t2"],
            metadatas=None,
            vectors=[[1, 2, 3], [4, 5, 6]],
            ids=None,
            refresh_indices=True,
            create_index_if_not_exists=True,
            bulk_kwargs=None,
        )

        store._store.add_texts = AsyncMock(return_value=["10", "20"])  # type: ignore[assignment]
        await store.aadd_embeddings(
            text_embeddings=[("t1", [1, 2, 3]), ("t2", [4, 5, 6])],
            metadatas=[{1: 2}, {3: 4}],
            ids=["10", "20"],
            refresh_indices=False,
            create_index_if_not_exists=False,
            bulk_kwargs={"x": "y"},
        )
        store._store.add_texts.assert_awaited_with(
            texts=["t1", "t2"],
            metadatas=[{1: 2}, {3: 4}],
            vectors=[[1, 2, 3], [4, 5, 6]],
            ids=["10", "20"],
            refresh_indices=False,
            create_index_if_not_exists=False,
            bulk_kwargs={"x": "y"},
        )

    @pytest.mark.asyncio
    async def test_max_marginal_relevance_search(
        self,
        hybrid_store: AsyncElasticsearchStore,
        embeddings: Embeddings,
        static_hits: List[Dict],
    ) -> None:
        hybrid_store._store.max_marginal_relevance_search = AsyncMock(  # type: ignore[assignment]
            return_value=static_hits
        )
        actual = await hybrid_store.amax_marginal_relevance_search(
            query="qqq",
            k=8,
            fetch_k=19,
            lambda_mult=0.3,
        )
        assert actual == [Document("test")]
        hybrid_store._store.max_marginal_relevance_search.assert_awaited_with(
            embedding_service=AsyncEmbeddingServiceAdapter(embeddings),
            query="qqq",
            vector_field="vector",
            k=8,
            num_candidates=19,
            lambda_mult=0.3,
            fields=None,
            custom_query=None,
        )

    @pytest.mark.asyncio
    async def test_elasticsearch_hybrid_scores_guard(
        self, hybrid_store: AsyncElasticsearchStore
    ) -> None:
        """Ensure an error is raised when search with score in hybrid mode
        because in this case Elasticsearch does not return any score.
        """
        with pytest.raises(ValueError):
            await hybrid_store.asimilarity_search_with_score("foo")

        with pytest.raises(ValueError):
            await hybrid_store.asimilarity_search_by_vector_with_relevance_scores(
                [1, 2, 3]
            )

    @pytest.mark.asyncio
    async def test_parameter_forwarding_to_evectorstore(self) -> None:
        """Test to catch missing EVectorStore parameters.

        This test compares the EVectorStore constructor signature against what
        ElasticsearchStore actually forwards. If EVectorStore adds new
        parameters, this test will fail and alert us to update ElasticsearchStore.
        """

        client = AsyncElasticsearch(hosts=["http://dummy:9200"])

        # Get EVectorStore constructor signature
        evectorstore_sig = inspect.signature(EVectorStore.__init__)
        # Remove self from the parameters set
        evectorstore_params = set(evectorstore_sig.parameters.keys()) - {"self"}

        # Use variable so unasync can transform the path
        # Break up the string so unasync can transform _async to _sync
        async_or_sync_module = "_async"
        evectorstore_path = (
            f"langchain_elasticsearch.{async_or_sync_module}.vectorstores.EVectorStore"
        )
        with patch(evectorstore_path) as mock_evectorstore:
            # Mock the close method
            mock_evectorstore.return_value.close = AsyncMock()

            store = AsyncElasticsearchStore(
                index_name="test_index",
                client=client,
                num_dimensions=1536,
            )

            # Get what parameters were actually passed to EVectorStore
            mock_evectorstore.assert_called_once()
            call_args = mock_evectorstore.call_args
            forwarded_params = set(call_args.kwargs.keys())

            # Check for missing parameters
            missing_params = evectorstore_params - forwarded_params
            if missing_params:
                pytest.fail(
                    f"ElasticsearchStore is missing these EVectorStore parameters:"
                    f"{missing_params}. Please add them to ElasticsearchStore "
                    f"and forward them to EVectorStore."
                )

            # Check for unexpected parameters
            unexpected_params = forwarded_params - evectorstore_params
            if unexpected_params:
                pytest.fail(
                    f"ElasticsearchStore is forwarding unexpected parameters to "
                    f"EVectorStore: {unexpected_params}. These parameters don't exist "
                    f"in EVectorStore.__init__."
                )

            await store.aclose()

    @pytest.mark.asyncio
    async def test_parameter_forwarding_defaults(self) -> None:
        """Test that default parameter values are properly forwarded to
        EVectorStore."""

        client = AsyncElasticsearch(hosts=["http://dummy:9200"])

        # Use variable so unasync can transform the path
        # Break up the string so unasync can transform _async to _sync
        async_or_sync_module = "_async"
        evectorstore_path = (
            f"langchain_elasticsearch.{async_or_sync_module}.vectorstores.EVectorStore"
        )
        with patch(evectorstore_path) as mock_evectorstore:
            # Mock the close method
            mock_evectorstore.return_value.close = AsyncMock()

            # Test with minimal parameters (should use defaults)
            store = AsyncElasticsearchStore(index_name="test_index", client=client)

            # Verify EVectorStore was called with default values
            mock_evectorstore.assert_called_once()
            call_args = mock_evectorstore.call_args

            # Check default values
            assert call_args.kwargs["index"] == "test_index"
            assert call_args.kwargs["client"] == client
            assert call_args.kwargs["vector_field"] == "vector"  # default
            assert call_args.kwargs["text_field"] == "text"  # default
            assert call_args.kwargs["num_dimensions"] is None  # default

            await store.aclose()
