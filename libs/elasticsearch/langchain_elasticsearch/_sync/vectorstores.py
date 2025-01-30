import logging
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

from elasticsearch import Elasticsearch
from elasticsearch.helpers.vectorstore import (
    BM25Strategy,
    DenseVectorScriptScoreStrategy,
    DenseVectorStrategy,
    DistanceMetric,
    RetrievalStrategy,
    SparseVectorStrategy,
)
from elasticsearch.helpers.vectorstore import VectorStore as EVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_elasticsearch._utilities import (
    ApproxRetrievalStrategy,
    BaseRetrievalStrategy,
    BM25RetrievalStrategy,
    DistanceStrategy,
    ExactRetrievalStrategy,
    SparseRetrievalStrategy,
    _hits_to_docs_scores,
    user_agent,
)
from langchain_elasticsearch.client import create_elasticsearch_client
from langchain_elasticsearch.embeddings import EmbeddingServiceAdapter

logger = logging.getLogger(__name__)


def _convert_retrieval_strategy(
    langchain_strategy: BaseRetrievalStrategy,
    distance: Optional[DistanceStrategy] = None,
) -> RetrievalStrategy:
    if isinstance(langchain_strategy, ApproxRetrievalStrategy):
        if distance is None:
            raise ValueError(
                "ApproxRetrievalStrategy requires a distance strategy to be provided."
            )
        return DenseVectorStrategy(
            distance=DistanceMetric[distance],
            model_id=langchain_strategy.query_model_id,
            hybrid=(
                False
                if langchain_strategy.hybrid is None
                else langchain_strategy.hybrid
            ),
            rrf=False if langchain_strategy.rrf is None else langchain_strategy.rrf,
        )
    elif isinstance(langchain_strategy, ExactRetrievalStrategy):
        if distance is None:
            raise ValueError(
                "ExactRetrievalStrategy requires a distance strategy to be provided."
            )
        return DenseVectorScriptScoreStrategy(distance=DistanceMetric[distance])
    elif isinstance(langchain_strategy, SparseRetrievalStrategy):
        return SparseVectorStrategy(langchain_strategy.model_id)
    elif isinstance(langchain_strategy, BM25RetrievalStrategy):
        return BM25Strategy(k1=langchain_strategy.k1, b=langchain_strategy.b)
    else:
        raise TypeError(
            f"Strategy {langchain_strategy} not supported. To provide a "
            f"custom strategy, please subclass {RetrievalStrategy}."
        )


class ElasticsearchStore(VectorStore):
    """`Elasticsearch` vector store.

     Setup:
        Install ``langchain_elasticsearch`` and running the Elasticsearch docker container.

        .. code-block:: bash

            pip install -qU langchain_elasticsearch
            docker run -p 9200:9200 \
              -e "discovery.type=single-node" \
              -e "xpack.security.enabled=false" \
              -e "xpack.security.http.ssl.enabled=false" \
              docker.elastic.co/elasticsearch/elasticsearch:8.12.1

    Key init args — indexing params:
        index_name: str
            Name of the index to create.
        embedding: Embeddings
            Embedding function to use.
        custom_index_settings: Optional[Dict[str, Any]]
            A dictionary of custom settings for the index.
            This can include configurations like the number of shards, number of replicas,
            analysis settings, and other index-specific settings. If not provided, default
            settings will be used. Note that if the same setting is provided by both the user
            and the strategy, will raise an error.

    Key init args — client params:
        es_connection: Optional[Elasticsearch]
            Pre-existing Elasticsearch connection.
        es_url: Optional[str]
            URL of the Elasticsearch instance to connect to.
        es_cloud_id: Optional[str]
            Cloud ID of the Elasticsearch instance to connect to.
        es_user: Optional[str]
            Username to use when connecting to Elasticsearch.
        es_password: Optional[str]
            Password to use when connecting to Elasticsearch.
        es_api_key: Optional[str]
            API key to use when connecting to Elasticsearch.
        es_params: Optional[Dict[str, Any]]
            Additional parameters for the Elasticsearch client.

    Instantiate:
        .. code-block:: python

            from langchain_elasticsearch import ElasticsearchStore
            from langchain_openai import OpenAIEmbeddings

            vector_store = ElasticsearchStore(
                index_name="langchain-demo",
                embedding=OpenAIEmbeddings(),
                es_url="http://localhost:9200",
            )

    If you want to use a cloud hosted Elasticsearch instance, you can pass in the
    cloud_id argument instead of the es_url argument.

    Instantiate from cloud:
        .. code-block:: python

            from langchain_elasticsearch.vectorstores import ElasticsearchStore
            from langchain_openai import OpenAIEmbeddings

            store = ElasticsearchStore(
                embedding=OpenAIEmbeddings(),
                index_name="langchain-demo",
                es_cloud_id="<cloud_id>"
                es_user="elastic",
                es_password="<password>"
            )

    You can also connect to an existing Elasticsearch instance by passing in a
    pre-existing Elasticsearch connection via the es_connection argument.

    Instantiate from existing connection:
        .. code-block:: python

            from langchain_elasticsearch.vectorstores import ElasticsearchStore
            from langchain_openai import OpenAIEmbeddings

            from elasticsearch import Elasticsearch

            es_connection = Elasticsearch("http://localhost:9200")

            store = ElasticsearchStore(
                embedding=OpenAIEmbeddings(),
                index_name="langchain-demo",
                es_connection=es_connection
            )

    Add Documents:
        .. code-block:: python

            from langchain_core.documents import Document

            document_1 = Document(page_content="foo", metadata={"baz": "bar"})
            document_2 = Document(page_content="thud", metadata={"bar": "baz"})
            document_3 = Document(page_content="i will be deleted :(")

            documents = [document_1, document_2, document_3]
            ids = ["1", "2", "3"]
            vector_store.add_documents(documents=documents, ids=ids)

    Delete Documents:
        .. code-block:: python

            vector_store.delete(ids=["3"])

    Search:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * thud [{'bar': 'baz'}]

    Search with filter:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1,filter=[{"term": {"metadata.bar.keyword": "baz"}}])
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * thud [{'bar': 'baz'}]

    Search with score:
        .. code-block:: python

            results = vector_store.similarity_search_with_score(query="qux",k=1)
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * [SIM=0.916092] foo [{'baz': 'bar'}]

    Async:
        .. code-block:: python

            from langchain_elasticsearch import AsyncElasticsearchStore

            vector_store = AsyncElasticsearchStore(...)

            # add documents
            await vector_store.aadd_documents(documents=documents, ids=ids)

            # delete documents
            await vector_store.adelete(ids=["3"])

            # search
            results = vector_store.asimilarity_search(query="thud",k=1)

            # search with score
            results = await vector_store.asimilarity_search_with_score(query="qux",k=1)
            for doc,score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * [SIM=0.916092] foo [{'baz': 'bar'}]

    Use as Retriever:

        .. code-block:: bash

            pip install "elasticsearch[vectorstore_mmr]"

        .. code-block:: python

            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
            )
            retriever.invoke("thud")

        .. code-block:: python

            [Document(metadata={'bar': 'baz'}, page_content='thud')]

    **Advanced Uses:**

    ElasticsearchStore by default uses the ApproxRetrievalStrategy, which uses the
    HNSW algorithm to perform approximate nearest neighbor search. This is the
    fastest and most memory efficient algorithm.

    If you want to use the Brute force / Exact strategy for searching vectors, you
    can pass in the ExactRetrievalStrategy to the ElasticsearchStore constructor.

    Use ExactRetrievalStrategy:
        .. code-block:: python

            from langchain_elasticsearch.vectorstores import ElasticsearchStore
            from langchain_openai import OpenAIEmbeddings

            store = ElasticsearchStore(
                embedding=OpenAIEmbeddings(),
                index_name="langchain-demo",
                es_url="http://localhost:9200",
                strategy=ElasticsearchStore.ExactRetrievalStrategy()
            )

    Both strategies require that you know the similarity metric you want to use
    when creating the index. The default is cosine similarity, but you can also
    use dot product or euclidean distance.

    Use dot product similarity:
        .. code-block:: python

            from langchain_elasticsearch.vectorstores import ElasticsearchStore
            from langchain_openai import OpenAIEmbeddings
            from langchain_community.vectorstores.utils import DistanceStrategy

            store = ElasticsearchStore(
                "langchain-demo",
                embedding=OpenAIEmbeddings(),
                es_url="http://localhost:9200",
                distance_strategy="DOT_PRODUCT"
            )

    """  # noqa: E501

    def __init__(
        self,
        index_name: str,
        *,
        embedding: Optional[Embeddings] = None,
        es_connection: Optional[Elasticsearch] = None,
        es_url: Optional[str] = None,
        es_cloud_id: Optional[str] = None,
        es_user: Optional[str] = None,
        es_api_key: Optional[str] = None,
        es_password: Optional[str] = None,
        vector_query_field: str = "vector",
        query_field: str = "text",
        distance_strategy: Optional[
            Literal[
                DistanceStrategy.COSINE,
                DistanceStrategy.DOT_PRODUCT,
                DistanceStrategy.EUCLIDEAN_DISTANCE,
                DistanceStrategy.MAX_INNER_PRODUCT,
            ]
        ] = None,
        strategy: Union[
            BaseRetrievalStrategy, RetrievalStrategy
        ] = ApproxRetrievalStrategy(),
        es_params: Optional[Dict[str, Any]] = None,
        custom_index_settings: Optional[Dict[str, Any]] = None,
    ):
        if isinstance(strategy, BaseRetrievalStrategy):
            strategy = _convert_retrieval_strategy(
                strategy, distance=distance_strategy or DistanceStrategy.COSINE
            )

        embedding_service = None
        if embedding:
            embedding_service = EmbeddingServiceAdapter(embedding)

        if not es_connection:
            es_connection = create_elasticsearch_client(
                url=es_url,
                cloud_id=es_cloud_id,
                api_key=es_api_key,
                username=es_user,
                password=es_password,
                params=es_params,
            )

        self._store = EVectorStore(
            client=es_connection,
            index=index_name,
            retrieval_strategy=strategy,
            embedding_service=embedding_service,
            text_field=query_field,
            vector_field=vector_query_field,
            user_agent=user_agent("langchain-py-vs"),
            custom_index_settings=custom_index_settings,
        )

        self.embedding = embedding
        self.client = self._store.client
        self._embedding_service = embedding_service
        self.query_field = query_field
        self.vector_query_field = vector_query_field

    def close(self) -> None:
        self._store.close()

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding

    @staticmethod
    def connect_to_elasticsearch(
        *,
        es_url: Optional[str] = None,
        cloud_id: Optional[str] = None,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        es_params: Optional[Dict[str, Any]] = None,
    ) -> Elasticsearch:
        return create_elasticsearch_client(
            url=es_url,
            cloud_id=cloud_id,
            api_key=api_key,
            username=username,
            password=password,
            params=es_params,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 50,
        filter: Optional[List[dict]] = None,
        fields: Optional[List[str]] = None,
        *,
        custom_query: Optional[
            Callable[[Dict[str, Any], Optional[str]], Dict[str, Any]]
        ] = None,
        doc_builder: Optional[Callable[[Dict], Document]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return Elasticsearch documents most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to knn num_candidates.
            filter: Array of Elasticsearch filter clauses to apply to the query.

        Returns:
            List of Documents most similar to the query,
            in descending order of similarity.
        """
        hits = self._store.search(
            query=query,
            k=k,
            num_candidates=fetch_k,
            filter=filter,
            fields=fields,
            custom_query=custom_query,
        )
        docs = _hits_to_docs_scores(
            hits=hits,
            content_field=self.query_field,
            doc_builder=doc_builder,
        )
        return [doc for doc, _score in docs]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        fields: Optional[List[str]] = None,
        *,
        custom_query: Optional[
            Callable[[Dict[str, Any], Optional[str]], Dict[str, Any]]
        ] = None,
        doc_builder: Optional[Callable[[Dict], Document]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            query (str): Text to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            fields: Other fields to get from elasticsearch source. These fields
                will be added to the document metadata.

        Returns:
            List[Document]: A list of Documents selected by maximal marginal relevance.
        """
        if self._embedding_service is None:
            raise ValueError(
                "maximal marginal relevance search requires an embedding service."
            )

        hits = self._store.max_marginal_relevance_search(
            embedding_service=self._embedding_service,
            query=query,
            vector_field=self.vector_query_field,
            k=k,
            num_candidates=fetch_k,
            lambda_mult=lambda_mult,
            fields=fields,
            custom_query=custom_query,
        )

        docs_scores = _hits_to_docs_scores(
            hits=hits,
            content_field=self.query_field,
            fields=fields,
            doc_builder=doc_builder,
        )

        return [doc for doc, _score in docs_scores]

    @staticmethod
    def _identity_fn(score: float) -> float:
        return score

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.

        Vectorstores should define their own selection based method of relevance.
        """
        # All scores from Elasticsearch are already normalized similarities:
        # https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html#dense-vector-params
        return self._identity_fn

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[List[dict]] = None,
        fields: Optional[List[str]] = None,
        *,
        custom_query: Optional[
            Callable[[Dict[str, Any], Optional[str]], Dict[str, Any]]
        ] = None,
        doc_builder: Optional[Callable[[Dict], Document]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return Elasticsearch documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Array of Elasticsearch filter clauses to apply to the query.

        Returns:
            List of Documents most similar to the query and score for each
        """
        if (
            isinstance(self._store.retrieval_strategy, DenseVectorStrategy)
            and self._store.retrieval_strategy.hybrid
        ):
            raise ValueError("scores are currently not supported in hybrid mode")

        hits = self._store.search(
            query=query,
            k=k,
            filter=filter,
            fields=fields,
            custom_query=custom_query
        )
        return _hits_to_docs_scores(
            hits=hits,
            content_field=self.query_field,
            doc_builder=doc_builder,
        )

    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[List[Dict]] = None,
        fields: Optional[List[str]] = None,
        *,
        custom_query: Optional[
            Callable[[Dict[str, Any], Optional[str]], Dict[str, Any]]
        ] = None,
        doc_builder: Optional[Callable[[Dict], Document]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return Elasticsearch documents most similar to query, along with scores.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Array of Elasticsearch filter clauses to apply to the query.

        Returns:
            List of Documents most similar to the embedding and score for each
        """
        if (
            isinstance(self._store.retrieval_strategy, DenseVectorStrategy)
            and self._store.retrieval_strategy.hybrid
        ):
            raise ValueError("scores are currently not supported in hybrid mode")

        hits = self._store.search(
            query=None,
            query_vector=embedding,
            k=k,
            filter=filter,
            fields=fields,
            custom_query=custom_query,
        )
        return _hits_to_docs_scores(
            hits=hits,
            content_field=self.query_field,
            doc_builder=doc_builder,
        )

    def delete(
        self,
        ids: Optional[List[str]] = None,
        refresh_indices: Optional[bool] = True,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete documents from the Elasticsearch index.

        Args:
            ids: List of ids of documents to delete.
            refresh_indices: Whether to refresh the index
                            after deleting documents. Defaults to True.
        """
        if ids is None:
            raise ValueError("please specify some IDs")

        return self._store.delete(ids=ids, refresh_indices=refresh_indices or False)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        refresh_indices: bool = True,
        create_index_if_not_exists: bool = True,
        bulk_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the store.

        Args:
            texts: Iterable of strings to add to the store.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            refresh_indices: Whether to refresh the Elasticsearch indices
                            after adding the texts.
            create_index_if_not_exists: Whether to create the Elasticsearch
                                        index if it doesn't already exist.
            *bulk_kwargs: Additional arguments to pass to Elasticsearch bulk.
                - chunk_size: Optional. Number of texts to add to the
                    index at a time. Defaults to 500.

        Returns:
            List of ids from adding the texts into the store.
        """
        return self._store.add_texts(
            texts=list(texts),
            metadatas=metadatas,
            ids=ids,
            refresh_indices=refresh_indices,
            create_index_if_not_exists=create_index_if_not_exists,
            bulk_kwargs=bulk_kwargs,
        )

    def add_embeddings(
        self,
        text_embeddings: Iterable[Tuple[str, List[float]]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        refresh_indices: bool = True,
        create_index_if_not_exists: bool = True,
        bulk_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add the given texts and embeddings to the store.

        Args:
            text_embeddings: Iterable pairs of string and embedding to
                add to the store.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.
            refresh_indices: Whether to refresh the Elasticsearch indices
                            after adding the texts.
            create_index_if_not_exists: Whether to create the Elasticsearch
                                        index if it doesn't already exist.
            *bulk_kwargs: Additional arguments to pass to Elasticsearch bulk.
                - chunk_size: Optional. Number of texts to add to the
                    index at a time. Defaults to 500.

        Returns:
            List of ids from adding the texts into the store.
        """
        texts, embeddings = zip(*text_embeddings)
        return self._store.add_texts(
            texts=list(texts),
            metadatas=metadatas,
            vectors=list(embeddings),
            ids=ids,
            refresh_indices=refresh_indices,
            create_index_if_not_exists=create_index_if_not_exists,
            bulk_kwargs=bulk_kwargs,
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        bulk_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> "ElasticsearchStore":
        """Construct ElasticsearchStore wrapper from raw documents.

        Example:
            .. code-block:: python

                from langchain_elasticsearch.vectorstores import ElasticsearchStore
                from langchain_openai import OpenAIEmbeddings

                db = ElasticsearchStore.from_texts(
                    texts,
                    // embeddings optional if using
                    // a strategy that doesn't require inference
                    embeddings,
                    index_name="langchain-demo",
                    es_url="http://localhost:9200"
                )

        Args:
            texts: List of texts to add to the Elasticsearch index.
            embedding: Embedding function to use to embed the texts.
            metadatas: Optional list of metadatas associated with the texts.
            index_name: Name of the Elasticsearch index to create.
            es_url: URL of the Elasticsearch instance to connect to.
            cloud_id: Cloud ID of the Elasticsearch instance to connect to.
            es_user: Username to use when connecting to Elasticsearch.
            es_password: Password to use when connecting to Elasticsearch.
            es_api_key: API key to use when connecting to Elasticsearch.
            es_connection: Optional pre-existing Elasticsearch connection.
            vector_query_field: Optional. Name of the field to
                                store the embedding vectors in.
            query_field: Optional. Name of the field to store the texts in.
            distance_strategy: Optional. Name of the distance
                                strategy to use. Defaults to "COSINE".
                                can be one of "COSINE",
                                "EUCLIDEAN_DISTANCE", "DOT_PRODUCT",
                                "MAX_INNER_PRODUCT".
            bulk_kwargs: Optional. Additional arguments to pass to
                        Elasticsearch bulk.
        """

        index_name = kwargs.get("index_name")
        if index_name is None:
            raise ValueError("Please provide an index_name.")

        elasticsearchStore = cls(embedding=embedding, **kwargs)

        # Encode the provided texts and add them to the newly created index.
        elasticsearchStore.add_texts(
            texts=texts, metadatas=metadatas, bulk_kwargs=bulk_kwargs
        )

        return elasticsearchStore

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        bulk_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> "ElasticsearchStore":
        """Construct ElasticsearchStore wrapper from documents.

        Example:
            .. code-block:: python

                from langchain_elasticsearch.vectorstores import ElasticsearchStore
                from langchain_openai import OpenAIEmbeddings

                db = ElasticsearchStore.from_documents(
                    texts,
                    embeddings,
                    index_name="langchain-demo",
                    es_url="http://localhost:9200"
                )

        Args:
            texts: List of texts to add to the Elasticsearch index.
            embedding: Embedding function to use to embed the texts.
                      Do not provide if using a strategy
                      that doesn't require inference.
            metadatas: Optional list of metadatas associated with the texts.
            index_name: Name of the Elasticsearch index to create.
            es_url: URL of the Elasticsearch instance to connect to.
            cloud_id: Cloud ID of the Elasticsearch instance to connect to.
            es_user: Username to use when connecting to Elasticsearch.
            es_password: Password to use when connecting to Elasticsearch.
            es_api_key: API key to use when connecting to Elasticsearch.
            es_connection: Optional pre-existing Elasticsearch connection.
            vector_query_field: Optional. Name of the field
                                to store the embedding vectors in.
            query_field: Optional. Name of the field to store the texts in.
            bulk_kwargs: Optional. Additional arguments to pass to
                        Elasticsearch bulk.
        """

        index_name = kwargs.get("index_name")
        if index_name is None:
            raise ValueError("Please provide an index_name.")

        elasticsearchStore = cls(embedding=embedding, **kwargs)

        # Encode the provided texts and add them to the newly created index.
        elasticsearchStore.add_documents(documents, bulk_kwargs=bulk_kwargs)

        return elasticsearchStore

    @staticmethod
    def ExactRetrievalStrategy() -> "ExactRetrievalStrategy":
        """Used to perform brute force / exact
        nearest neighbor search via script_score."""
        return ExactRetrievalStrategy()

    @staticmethod
    def ApproxRetrievalStrategy(
        query_model_id: Optional[str] = None,
        hybrid: Optional[bool] = False,
        rrf: Optional[Union[dict, bool]] = True,
    ) -> "ApproxRetrievalStrategy":
        """Used to perform approximate nearest neighbor search
        using the HNSW algorithm.

        At build index time, this strategy will create a
        dense vector field in the index and store the
        embedding vectors in the index.

        At query time, the text will either be embedded using the
        provided embedding function or the query_model_id
        will be used to embed the text using the model
        deployed to Elasticsearch.

        if query_model_id is used, do not provide an embedding function.

        Args:
            query_model_id: Optional. ID of the model to use to
                            embed the query text within the stack. Requires
                            embedding model to be deployed to Elasticsearch.
            hybrid: Optional. If True, will perform a hybrid search
                    using both the knn query and a text query.
                    Defaults to False.
            rrf: Optional. rrf is Reciprocal Rank Fusion.
                 When `hybrid` is True,
                    and `rrf` is True, then rrf: {}.
                    and `rrf` is False, then rrf is omitted.
                    and isinstance(rrf, dict) is True, then pass in the dict values.
                 rrf could be passed for adjusting 'rank_constant' and 'window_size'.
        """
        return ApproxRetrievalStrategy(
            query_model_id=query_model_id, hybrid=hybrid, rrf=rrf
        )

    @staticmethod
    def SparseVectorRetrievalStrategy(
        model_id: Optional[str] = None,
    ) -> "SparseRetrievalStrategy":
        """Used to perform sparse vector search via text_expansion.
        Used for when you want to use ELSER model to perform document search.

        At build index time, this strategy will create a pipeline that
        will embed the text using the ELSER model and store the
        resulting tokens in the index.

        At query time, the text will be embedded using the ELSER
        model and the resulting tokens will be used to
        perform a text_expansion query.

        Args:
            model_id: Optional. Default is ".elser_model_1".
                    ID of the model to use to embed the query text
                    within the stack. Requires embedding model to be
                    deployed to Elasticsearch.
        """
        return SparseRetrievalStrategy(model_id=model_id)

    @staticmethod
    def BM25RetrievalStrategy(
        k1: Union[float, None] = None, b: Union[float, None] = None
    ) -> "BM25RetrievalStrategy":
        """Used to apply BM25 without vector search.

        Args:
            k1: Optional. This corresponds to the BM25 parameter, k1. Default is None,
                which uses the default setting of Elasticsearch.
            b: Optional. This corresponds to the BM25 parameter, b. Default is None,
               which uses the default setting of Elasticsearch.
        """
        return BM25RetrievalStrategy(k1=k1, b=b)
