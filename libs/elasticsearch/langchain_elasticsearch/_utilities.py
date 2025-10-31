import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from elasticsearch import AsyncElasticsearch, Elasticsearch, exceptions
from langchain_core import __version__ as langchain_version
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DistanceStrategy(str, Enum):
    """Enumerator of the Distance strategies for calculating distances
    between vectors."""

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    DOT_PRODUCT = "DOT_PRODUCT"
    JACCARD = "JACCARD"
    COSINE = "COSINE"


def user_agent(prefix: str) -> str:
    return f"{prefix}/{langchain_version}"


def with_user_agent_header(client: Elasticsearch, header_prefix: str) -> Elasticsearch:
    headers = dict(client._headers)
    headers.update({"user-agent": f"{user_agent(header_prefix)}"})
    return client.options(headers=headers)


def async_with_user_agent_header(
    client: AsyncElasticsearch, header_prefix: str
) -> AsyncElasticsearch:
    headers = dict(client._headers)
    headers.update({"user-agent": f"{user_agent(header_prefix)}"})
    return client.options(headers=headers)


def model_must_be_deployed(client: Elasticsearch, model_id: str) -> None:
    try:
        dummy = {"x": "y"}
        client.ml.infer_trained_model(model_id=model_id, docs=[dummy])
    except exceptions.NotFoundError as err:
        raise err
    except exceptions.ConflictError as err:
        raise exceptions.NotFoundError(
            f"model '{model_id}' not found, please deploy it first",
            meta=err.meta,
            body=err.body,
        ) from err
    except exceptions.BadRequestError:
        # This error is expected because we do not know the expected document
        # shape and just use a dummy doc above.
        pass


def _hits_to_docs_scores(
    hits: List[Dict[str, Any]],
    content_field: str,
    fields: Optional[List[str]] = None,
    doc_builder: Optional[Callable[[Dict], Document]] = None,
) -> List[Tuple[Document, float]]:
    if fields is None:
        fields = []

    documents = []

    def default_doc_builder(hit: Dict, fields: List[str]) -> Document:
        doc = Document(
            page_content=hit["_source"].get(content_field, ""),
            metadata=hit["_source"].get("metadata", {}),
        )
        for field_key in fields:
            doc.metadata[field_key] = hit["_source"].get(field_key, None)
        return doc

    doc_builder = doc_builder or default_doc_builder

    for hit in hits:
        for field in fields:
            if "metadata" not in hit["_source"]:
                hit["_source"]["metadata"] = {}
            if field in hit["_source"] and field not in [
                "metadata",
                content_field,
            ]:
                hit["_source"]["metadata"][field] = hit["_source"][field]

        doc = doc_builder(hit, fields)
        documents.append((doc, hit["_score"]))

    return documents


@deprecated("0.2.0", alternative="RetrievalStrategy", pending=True)
class BaseRetrievalStrategy(ABC):
    """Base class for `Elasticsearch` retrieval strategies."""

    @abstractmethod
    def query(
        self,
        query_vector: Union[List[float], None],
        query: Union[str, None],
        *,
        k: int,
        fetch_k: int,
        vector_query_field: str,
        text_field: str,
        filter: List[dict],
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        """
        Executes when a search is performed on the store.

        Args:
            query_vector: The query vector,
                          or None if not using vector-based query.
            query: The text query, or None if not using text-based query.
            k: The total number of results to retrieve.
            fetch_k: The number of results to fetch initially.
            vector_query_field: The field containing the vector
                                representations in the index.
            text_field: The field containing the text data in the index.
            filter: List of filter clauses to apply to the query.
            similarity: The similarity strategy to use, or None if not using one.

        Returns:
            Dict: The Elasticsearch query body.
        """

    @abstractmethod
    def index(
        self,
        dims_length: Union[int, None],
        vector_query_field: str,
        text_field: str,
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        """
        Executes when the index is created.

        Args:
            dims_length: Numeric length of the embedding vectors,
                        or None if not using vector-based query.
            vector_query_field: The field containing the vector
                                representations in the index.
            text_field: The field containing the text data in the index.
            similarity: The similarity strategy to use,
                        or None if not using one.

        Returns:
            Dict: The Elasticsearch settings and mappings for the strategy.
        """

    def before_index_setup(
        self, client: "Elasticsearch", text_field: str, vector_query_field: str
    ) -> None:
        """
        Executes before the index is created. Used for setting up
        any required Elasticsearch resources like a pipeline.

        Args:
            client: The Elasticsearch client.
            text_field: The field containing the text data in the index.
            vector_query_field: The field containing the vector
                                representations in the index.
        """

    def require_inference(self) -> bool:
        """
        Returns whether or not the strategy requires inference
        to be performed on the text before it is added to the index.

        Returns:
            bool: Whether or not the strategy requires inference
            to be performed on the text before it is added to the index.
        """
        return True


@deprecated("0.2.0", alternative="DenseVectorStrategy", pending=True)
class ApproxRetrievalStrategy(BaseRetrievalStrategy):
    """Approximate retrieval strategy using the `HNSW` algorithm."""

    def __init__(
        self,
        query_model_id: Optional[str] = None,
        hybrid: Optional[bool] = False,
        rrf: Optional[Union[dict, bool]] = True,
    ):
        self.query_model_id = query_model_id
        self.hybrid = hybrid

        # RRF has two optional parameters
        # 'rank_constant', 'window_size'
        # https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html
        self.rrf = rrf

    def query(
        self,
        query_vector: Union[List[float], None],
        query: Union[str, None],
        k: int,
        fetch_k: int,
        vector_query_field: str,
        text_field: str,
        filter: List[dict],
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        knn = {
            "filter": filter,
            "field": vector_query_field,
            "k": k,
            "num_candidates": fetch_k,
        }

        # Embedding provided via the embedding function
        if query_vector is not None and not self.query_model_id:
            knn["query_vector"] = list(query_vector)

        # Case 2: Used when model has been deployed to
        # Elasticsearch and can infer the query vector from the query text
        elif query and self.query_model_id:
            knn["query_vector_builder"] = {
                "text_embedding": {
                    "model_id": self.query_model_id,  # use 'model_id' argument
                    "model_text": query,  # use 'query' argument
                }
            }

        else:
            raise ValueError(
                "You must provide an embedding function or a"
                " query_model_id to perform a similarity search."
            )

        # If hybrid, add a query to the knn query
        # RRF is used to even the score from the knn query and text query
        # RRF has two optional parameters: {'rank_constant':int, 'window_size':int}
        # https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html
        if self.hybrid:
            query_body = {
                "knn": knn,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    text_field: {
                                        "query": query,
                                    }
                                }
                            }
                        ],
                        "filter": filter,
                    }
                },
            }

            if isinstance(self.rrf, dict):
                query_body["rank"] = {"rrf": self.rrf}
            elif isinstance(self.rrf, bool) and self.rrf is True:
                query_body["rank"] = {"rrf": {}}

            return query_body
        else:
            return {"knn": knn}

    def before_index_setup(
        self, client: "Elasticsearch", text_field: str, vector_query_field: str
    ) -> None:
        if self.query_model_id:
            model_must_be_deployed(client, self.query_model_id)

    def index(
        self,
        dims_length: Union[int, None],
        vector_query_field: str,
        text_field: str,
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        """Create the mapping for the Elasticsearch index."""

        if similarity is DistanceStrategy.COSINE:
            similarityAlgo = "cosine"
        elif similarity is DistanceStrategy.EUCLIDEAN_DISTANCE:
            similarityAlgo = "l2_norm"
        elif similarity is DistanceStrategy.DOT_PRODUCT:
            similarityAlgo = "dot_product"
        elif similarity is DistanceStrategy.MAX_INNER_PRODUCT:
            similarityAlgo = "max_inner_product"
        else:
            raise ValueError(f"Similarity {similarity} not supported.")

        return {
            "mappings": {
                "properties": {
                    vector_query_field: {
                        "type": "dense_vector",
                        "dims": dims_length,
                        "index": True,
                        "similarity": similarityAlgo,
                    },
                }
            }
        }


@deprecated("0.2.0", alternative="DenseVectorScriptScoreStrategy", pending=True)
class ExactRetrievalStrategy(BaseRetrievalStrategy):
    """Exact retrieval strategy using the `script_score` query."""

    def query(
        self,
        query_vector: Union[List[float], None],
        query: Union[str, None],
        k: int,
        fetch_k: int,
        vector_query_field: str,
        text_field: str,
        filter: Union[List[dict], None],
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        if similarity is DistanceStrategy.COSINE:
            similarityAlgo = (
                f"cosineSimilarity(params.query_vector, '{vector_query_field}') + 1.0"
            )
        elif similarity is DistanceStrategy.EUCLIDEAN_DISTANCE:
            similarityAlgo = (
                f"1 / (1 + l2norm(params.query_vector, '{vector_query_field}'))"
            )
        elif similarity is DistanceStrategy.DOT_PRODUCT:
            similarityAlgo = f"""
            double value = dotProduct(params.query_vector, '{vector_query_field}');
            return sigmoid(1, Math.E, -value);
            """
        else:
            raise ValueError(f"Similarity {similarity} not supported.")

        queryBool: Dict = {"match_all": {}}
        if filter:
            queryBool = {"bool": {"filter": filter}}

        return {
            "query": {
                "script_score": {
                    "query": queryBool,
                    "script": {
                        "source": similarityAlgo,
                        "params": {"query_vector": query_vector},
                    },
                },
            }
        }

    def index(
        self,
        dims_length: Union[int, None],
        vector_query_field: str,
        text_field: str,
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        """Create the mapping for the Elasticsearch index."""

        return {
            "mappings": {
                "properties": {
                    vector_query_field: {
                        "type": "dense_vector",
                        "dims": dims_length,
                        "index": False,
                    },
                }
            }
        }


@deprecated("0.2.0", alternative="SparseVectorStrategy", pending=True)
class SparseRetrievalStrategy(BaseRetrievalStrategy):
    """Sparse retrieval strategy using the `text_expansion` processor."""

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or ".elser_model_1"

    def query(
        self,
        query_vector: Union[List[float], None],
        query: Union[str, None],
        k: int,
        fetch_k: int,
        vector_query_field: str,
        text_field: str,
        filter: List[dict],
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        return {
            "query": {
                "bool": {
                    "must": [
                        {
                            "text_expansion": {
                                f"{vector_query_field}.tokens": {
                                    "model_id": self.model_id,
                                    "model_text": query,
                                }
                            }
                        }
                    ],
                    "filter": filter,
                }
            }
        }

    def _get_pipeline_name(self) -> str:
        return f"{self.model_id}_sparse_embedding"

    def before_index_setup(
        self, client: "Elasticsearch", text_field: str, vector_query_field: str
    ) -> None:
        if self.model_id:
            model_must_be_deployed(client, self.model_id)

            # Create a pipeline for the model
            client.ingest.put_pipeline(
                id=self._get_pipeline_name(),
                description="Embedding pipeline for langchain vectorstore",
                processors=[
                    {
                        "inference": {
                            "model_id": self.model_id,
                            "target_field": vector_query_field,
                            "field_map": {text_field: "text_field"},
                            "inference_config": {
                                "text_expansion": {"results_field": "tokens"}
                            },
                        }
                    }
                ],
            )

    def index(
        self,
        dims_length: Union[int, None],
        vector_query_field: str,
        text_field: str,
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        return {
            "mappings": {
                "properties": {
                    vector_query_field: {
                        "properties": {"tokens": {"type": "rank_features"}}
                    }
                }
            },
            "settings": {"default_pipeline": self._get_pipeline_name()},
        }

    def require_inference(self) -> bool:
        return False


@deprecated("0.2.0", alternative="BM25Strategy", pending=True)
class BM25RetrievalStrategy(BaseRetrievalStrategy):
    """Retrieval strategy using the native BM25 algorithm of Elasticsearch."""

    def __init__(self, k1: Union[float, None] = None, b: Union[float, None] = None):
        self.k1 = k1
        self.b = b

    def query(
        self,
        query_vector: Union[List[float], None],
        query: Union[str, None],
        k: int,
        fetch_k: int,
        vector_query_field: str,
        text_field: str,
        filter: List[dict],
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        return {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                text_field: {
                                    "query": query,
                                }
                            },
                        },
                    ],
                    "filter": filter,
                },
            },
        }

    def index(
        self,
        dims_length: Union[int, None],
        vector_query_field: str,
        text_field: str,
        similarity: Union[DistanceStrategy, None],
    ) -> Dict:
        mappings: Dict = {
            "properties": {
                text_field: {
                    "type": "text",
                    "similarity": "custom_bm25",
                },
            },
        }
        settings: Dict = {
            "similarity": {
                "custom_bm25": {
                    "type": "BM25",
                },
            },
        }

        if self.k1 is not None:
            settings["similarity"]["custom_bm25"]["k1"] = self.k1

        if self.b is not None:
            settings["similarity"]["custom_bm25"]["b"] = self.b

        return {"mappings": mappings, "settings": settings}

    def require_inference(self) -> bool:
        return False
