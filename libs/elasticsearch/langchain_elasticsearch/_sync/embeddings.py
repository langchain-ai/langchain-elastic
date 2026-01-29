from __future__ import annotations

from typing import List, Optional

from elasticsearch import Elasticsearch
from elasticsearch.helpers.vectorstore import EmbeddingService
from langchain_core.embeddings import Embeddings

from langchain_elasticsearch._utilities import (
    with_user_agent_header,
)
from langchain_elasticsearch.client import create_elasticsearch_client


class ElasticsearchEmbeddings(Embeddings):
    """`Elasticsearch` embedding models.

    This class provides an interface to generate embeddings using a model deployed
    in an Elasticsearch cluster. It requires an Elasticsearch connection and the
    model_id of the model deployed in the cluster.

    In Elasticsearch you need to have an embedding model loaded and deployed.
    - https://www.elastic.co/guide/en/elasticsearch/reference/current/infer-trained-model.html
    - https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-deploy-models.html

    Setup:
        Install `langchain_elasticsearch` and start Elasticsearch locally using
        the start-local script.

        ```bash
        pip install -qU langchain_elasticsearch
        curl -fsSL https://elastic.co/start-local | sh
        ```

        This will create an `elastic-start-local` folder. To start Elasticsearch
        and Kibana:
        ```bash
        cd elastic-start-local
        ./start.sh
        ```

        Elasticsearch will be available at `http://localhost:9200`. The password
        for the `elastic` user and API key are stored in the `.env` file in the
        `elastic-start-local` folder.

    Key init args:
        - `model_id` (str): The model_id of the model deployed in the Elasticsearch
          cluster.
        - `input_field` (str): The name of the key for the input text field in the
          document. Defaults to 'text_field'.

    Instantiate:
        ```python
        from langchain_elasticsearch import ElasticsearchEmbeddings

        embeddings = ElasticsearchEmbeddings(
            model_id="your_model_id",
            es_url="http://localhost:9200"
        )
        ```

        **Instantiate with API key (URL):**
        ```python
        from langchain_elasticsearch import ElasticsearchEmbeddings

        embeddings = ElasticsearchEmbeddings(
            model_id="your_model_id",
            es_url="http://localhost:9200",
            es_api_key="your-api-key"
        )
        ```

        **Instantiate with username/password (URL):**
        ```python
        from langchain_elasticsearch import ElasticsearchEmbeddings

        embeddings = ElasticsearchEmbeddings(
            model_id="your_model_id",
            es_url="http://localhost:9200",
            es_user="elastic",
            es_password="password"
        )
        ```

        If you want to use a cloud hosted Elasticsearch instance, you can pass in the
        es_cloud_id argument instead of the es_url argument.

        **Instantiate from cloud (with username/password):**
            ```python
            from langchain_elasticsearch import ElasticsearchEmbeddings

            embeddings = ElasticsearchEmbeddings(
                model_id="your_model_id",
                es_cloud_id="<cloud_id>",
                es_user="elastic",
                es_password="<password>"
            )
            ```

        **Instantiate from cloud (with API key):**
            ```python
            from langchain_elasticsearch import ElasticsearchEmbeddings

            embeddings = ElasticsearchEmbeddings(
                model_id="your_model_id",
                es_cloud_id="<cloud_id>",
                es_api_key="your-api-key"
            )
            ```

        You can also connect to an existing Elasticsearch instance by passing in a
        pre-existing Elasticsearch connection via the client argument.

        **Instantiate from existing connection:**
            ```python
            from langchain_elasticsearch import ElasticsearchEmbeddings
            from elasticsearch import Elasticsearch

            client = Elasticsearch("http://localhost:9200")
            embeddings = ElasticsearchEmbeddings(
                model_id="your_model_id",
                client=client
            )
            ```

    Generate embeddings:
        ```python
        documents = [
            "This is an example document.",
            "Another example document to generate embeddings for.",
        ]
        embeddings_list = embeddings.embed_documents(documents)
        ```

    Generate query embedding:
        ```python
        query_embedding = embeddings.embed_query("What is this about?")
        ```

    For synchronous applications, use the `ElasticsearchEmbeddings` class.
    For asynchronous applications, use the `AsyncElasticsearchEmbeddings` class.
    """  # noqa: E501

    def __init__(
        self,
        model_id: str,
        *,
        input_field: str = "text_field",
        client: Optional[Elasticsearch] = None,
        es_url: Optional[str] = None,
        es_cloud_id: Optional[str] = None,
        es_user: Optional[str] = None,
        es_api_key: Optional[str] = None,
        es_password: Optional[str] = None,
    ):
        """Initialize the ElasticsearchEmbeddings instance.

        Args:
            model_id (str): The model_id of the model deployed in the Elasticsearch
                cluster.
            input_field (str): The name of the key for the input text field in the
                document. Defaults to 'text_field'.
            client (AsyncElasticsearch or Elasticsearch, optional):
                Pre-existing Elasticsearch connection. Either provide this OR
                credentials.
            es_url (str, optional): URL of the Elasticsearch instance to connect to.
            es_cloud_id (str, optional): Cloud ID of the Elasticsearch instance.
            es_user (str, optional): Username to use when connecting to
                Elasticsearch.
            es_api_key (str, optional): API key to use when connecting to
                Elasticsearch.
            es_password (str, optional): Password to use when connecting to
                Elasticsearch.
        """
        # Accept either client OR credentials (one required)
        if client is not None:
            es_connection = client
        elif es_url is not None or es_cloud_id is not None:
            es_connection = create_elasticsearch_client(
                url=es_url,
                cloud_id=es_cloud_id,
                api_key=es_api_key,
                username=es_user,
                password=es_password,
            )
        else:
            raise ValueError(
                "Either 'client' or credentials (es_url, es_cloud_id, etc.) "
                "must be provided."
            )

        # Apply User-Agent for telemetry
        # (applies to both passed and internally created clients)
        self.client = with_user_agent_header(es_connection, "langchain-py-e")
        self.model_id = model_id
        self.input_field = input_field

    def _embedding_func(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the given texts using the Elasticsearch model.

        Args:
            texts (List[str]): A list of text strings to generate embeddings for.

        Returns:
            List[List[float]]: A list of embeddings, one for each text in the input
                list.
        """
        response = self.client.ml.infer_trained_model(
            model_id=self.model_id, docs=[{self.input_field: text} for text in texts]
        )

        embeddings = [doc["predicted_value"] for doc in response["inference_results"]]
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            texts (List[str]): A list of document text strings to generate embeddings
                for.

        Returns:
            List[List[float]]: A list of embeddings, one for each document in the input
                list.
        """
        return self._embedding_func(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query text.

        Args:
            text (str): The query text to generate an embedding for.

        Returns:
            List[float]: The embedding for the input query text.
        """
        return (self._embedding_func([text]))[0]


class EmbeddingServiceAdapter(EmbeddingService):
    """
    Adapter for LangChain Embeddings to support the EmbeddingService interface from
    elasticsearch.helpers.vectorstore.
    """

    def __init__(self, langchain_embeddings: Embeddings):
        self._langchain_embeddings = langchain_embeddings

    def __eq__(self, other):  # type: ignore[no-untyped-def]
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            texts (List[str]): A list of document text strings to generate embeddings
                for.

        Returns:
            List[List[float]]: A list of embeddings, one for each document in the input
                list.
        """
        return self._langchain_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query text.

        Args:
            text (str): The query text to generate an embedding for.

        Returns:
            List[float]: The embedding for the input query text.
        """
        return self._langchain_embeddings.embed_query(text)
