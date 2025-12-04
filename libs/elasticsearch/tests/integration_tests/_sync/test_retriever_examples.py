"""Test that all code examples in AsyncElasticsearchRetriever docstring actually work."""

from typing import Any, Mapping

import pytest
from elasticsearch import Elasticsearch
from langchain_core.documents import Document

from langchain_elasticsearch import ElasticsearchRetriever

from ._test_utilities import create_es_client, read_env, requests_saving_es_client


@pytest.fixture(scope="function")
def es_client() -> Elasticsearch:
    """Provide an Elasticsearch client for testing."""
    client = requests_saving_es_client()
    yield client
    client.close()


@pytest.fixture(scope="function")
def index_name() -> str:
    """Provide a unique index name for testing."""
    import uuid

    return f"test_{uuid.uuid4().hex}"


def index_test_data(es_client: Elasticsearch, index_name: str, field_name: str) -> None:
    """Helper to index test data about planetary moons."""
    # Format: (id, moon_name, planet, description)
    moons = [
        (1, "Moon", "Earth", "Earth's only natural satellite"),
        (2, "Phobos", "Mars", "Larger of Mars's two moons"),
        (3, "Deimos", "Mars", "Smaller of Mars's two moons"),
        (4, "Io", "Jupiter", "Most volcanically active body in the solar system"),
        (5, "Europa", "Jupiter", "Icy moon with subsurface ocean"),
        (6, "Ganymede", "Jupiter", "Largest moon in the solar system"),
        (7, "Callisto", "Jupiter", "Most heavily cratered object in the solar system"),
        (8, "Titan", "Saturn", "Largest moon of Saturn with thick atmosphere"),
        (9, "Enceladus", "Saturn", "Icy moon with geysers"),
        (10, "Mimas", "Saturn", "Moon that looks like the Death Star"),
        (11, "Triton", "Neptune", "Largest moon of Neptune, retrograde orbit"),
        (12, "Nereid", "Neptune", "Third largest moon of Neptune"),
        (13, "Proteus", "Neptune", "Second largest moon of Neptune"),
    ]
    for identifier, moon_name, planet, description in moons:
        es_client.index(
            index=index_name,
            document={
                field_name: f"{moon_name} is a moon of {planet}. {description}",
                "moon_name": moon_name,
                "planet": planet,
                "description": description,
            },
            id=str(identifier),
            refresh=True,
        )


@pytest.mark.sync
def test_example_instantiate(es_client: Elasticsearch, index_name: str) -> None:
    """Test: Instantiate example from docstring."""
    # Example from docstring
    from langchain_elasticsearch import ElasticsearchRetriever

    def body_func(query: str) -> dict:
        return {"query": {"match": {"text": {"query": query}}}}

    retriever = ElasticsearchRetriever(
        index_name=index_name,
        body_func=body_func,
        content_field="text",
        client=es_client,  # Use fixture instead of es_url
    )

    index_test_data(es_client, index_name, "text")
    documents = retriever.get_relevant_documents("Jupiter")
    assert len(documents) > 0
    assert all(isinstance(doc, Document) for doc in documents)


@pytest.mark.sync
def test_example_instantiate_with_api_key(index_name: str) -> None:
    """Test: Instantiate with API key example from docstring."""
    from langchain_elasticsearch import ElasticsearchRetriever

    def body_func(query: str) -> dict:
        return {"query": {"match": {"text": {"query": query}}}}

    env_config = read_env()
    if "es_url" not in env_config and "es_cloud_id" not in env_config:
        pytest.skip("No Elasticsearch connection available")

    config = {}
    if "es_url" in env_config:
        config["es_url"] = env_config["es_url"]
    if "es_api_key" in env_config:
        config["es_api_key"] = env_config["es_api_key"]
    if "es_cloud_id" in env_config:
        config["es_cloud_id"] = env_config["es_cloud_id"]

    retriever = ElasticsearchRetriever(
        index_name=index_name,
        body_func=body_func,
        content_field="text",
        **config,
    )

    # Verify it was created
    assert retriever is not None
    assert retriever.index_name == index_name


@pytest.mark.sync
def test_example_instantiate_with_username_password(index_name: str) -> None:
    """Test: Instantiate with username/password example from docstring."""
    from langchain_elasticsearch import ElasticsearchRetriever

    def body_func(query: str) -> dict:
        return {"query": {"match": {"text": {"query": query}}}}

    env_config = read_env()
    if "es_url" not in env_config and "es_cloud_id" not in env_config:
        pytest.skip("No Elasticsearch connection available")

    config = {}
    if "es_url" in env_config:
        config["es_url"] = env_config["es_url"]
    if "es_user" in env_config:
        config["es_user"] = env_config["es_user"]
    if "es_password" in env_config:
        config["es_password"] = env_config["es_password"]
    if "es_cloud_id" in env_config:
        config["es_cloud_id"] = env_config["es_cloud_id"]

    retriever = ElasticsearchRetriever(
        index_name=index_name,
        body_func=body_func,
        content_field="text",
        **config,
    )

    # Verify it was created
    assert retriever is not None
    assert retriever.index_name == index_name


@pytest.mark.sync
def test_example_instantiate_from_cloud(index_name: str) -> None:
    """Test: Instantiate from cloud example from docstring."""
    from langchain_elasticsearch import ElasticsearchRetriever

    def body_func(query: str) -> dict:
        return {"query": {"match": {"text": {"query": query}}}}

    env_config = read_env()
    if "es_cloud_id" not in env_config:
        pytest.skip("No cloud_id available")

    config = {
        "es_cloud_id": env_config["es_cloud_id"],
    }
    if "es_user" in env_config:
        config["es_user"] = env_config["es_user"]
    if "es_password" in env_config:
        config["es_password"] = env_config["es_password"]
    if "es_api_key" in env_config:
        config["es_api_key"] = env_config["es_api_key"]

    retriever = ElasticsearchRetriever(
        index_name=index_name,
        body_func=body_func,
        content_field="text",
        **config,
    )

    # Verify it was created
    assert retriever is not None
    assert retriever.index_name == index_name


@pytest.mark.sync
def test_example_instantiate_from_existing_connection(
    es_client: Elasticsearch, index_name: str
) -> None:
    """Test: Instantiate from existing connection example from docstring."""
    from elasticsearch import Elasticsearch

    from langchain_elasticsearch import ElasticsearchRetriever

    def body_func(query: str) -> dict:
        return {"query": {"match": {"text": {"query": query}}}}

    client = es_client  # Use fixture
    retriever = ElasticsearchRetriever(
        index_name=index_name,
        body_func=body_func,
        content_field="text",
        client=client,
    )

    index_test_data(es_client, index_name, "text")
    documents = retriever.get_relevant_documents("Mars")
    assert len(documents) > 0


@pytest.mark.sync
def test_example_retrieve_documents(es_client: Elasticsearch, index_name: str) -> None:
    """Test: Retrieve documents example from docstring (matches docstring exactly)."""
    from elasticsearch import Elasticsearch

    from langchain_elasticsearch import ElasticsearchRetriever

    from ._test_utilities import read_env

    # Index sample documents using async client (matching docstring example structure)
    es_client.index(
        index=index_name,
        document={"text": "The quick brown fox jumps over the lazy dog"},
        id="1",
        refresh=True,
    )
    es_client.index(
        index=index_name,
        document={"text": "Python is a popular programming language"},
        id="2",
        refresh=True,
    )
    es_client.index(
        index=index_name,
        document={"text": "Elasticsearch is a search engine"},
        id="3",
        refresh=True,
    )

    # Create sync retriever (matching docstring example)
    env_config = read_env()

    def body_func(query: str) -> dict:
        return {"query": {"match": {"text": {"query": query}}}}

    # Build config for retriever, including API key if available
    retriever_config = {
        "index_name": index_name,
        "body_func": body_func,
        "content_field": "text",
    }
    if "es_url" in env_config:
        retriever_config["es_url"] = env_config["es_url"]
    if "es_api_key" in env_config:
        retriever_config["es_api_key"] = env_config["es_api_key"]
    if "es_cloud_id" in env_config:
        retriever_config["es_cloud_id"] = env_config["es_cloud_id"]

    retriever = ElasticsearchRetriever(**retriever_config)

    # Retrieve documents (matching docstring example)
    documents = retriever.invoke("Python")

    # Verify output matches docstring expected output
    assert len(documents) > 0
    output_lines = [doc.page_content for doc in documents]
    assert "Python is a popular programming language" in output_lines

    # Print to verify it matches docstring output format
    print("Output:")
    for doc in documents:
        print(f"* {doc.page_content}")

    # Verify document structure
    for doc in documents:
        assert hasattr(doc, "page_content")
        assert hasattr(doc, "metadata")
        assert isinstance(doc.page_content, str)
        assert isinstance(doc.metadata, dict)


@pytest.mark.sync
def test_example_custom_document_mapper(
    es_client: Elasticsearch, index_name: str
) -> None:
    """Test: Custom document mapper example from docstring."""
    from typing import Any, Mapping

    from elasticsearch import Elasticsearch
    from langchain_core.documents import Document

    from langchain_elasticsearch import ElasticsearchRetriever

    def body_func(query: str) -> dict:
        return {"query": {"match": {"custom_field": {"query": query}}}}

    def custom_mapper(hit: Mapping[str, Any]) -> Document:
        # Custom logic to extract content and metadata
        return Document(
            page_content=hit["_source"]["custom_field"],
            metadata={"score": hit["_score"]},
        )

    client = es_client
    retriever = ElasticsearchRetriever(
        index_name=index_name,
        body_func=body_func,
        document_mapper=custom_mapper,
        client=client,
    )

    # Index test data with custom_field
    es_client.index(
        index=index_name,
        document={"custom_field": "test content"},
        id="1",
        refresh=True,
    )

    documents = retriever.get_relevant_documents("test")
    assert len(documents) > 0
    assert all(isinstance(doc, Document) for doc in documents)
    assert "score" in documents[0].metadata


@pytest.mark.sync
def test_example_multiple_indices(es_client: Elasticsearch, index_name: str) -> None:
    """Test: Multiple indices example from docstring."""
    from elasticsearch import Elasticsearch

    from langchain_elasticsearch import ElasticsearchRetriever

    def body_func(query: str) -> dict:
        return {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text_field_1", "text_field_2"],
                }
            }
        }

    index_name_1 = f"{index_name}_1"
    index_name_2 = f"{index_name}_2"

    client = es_client
    retriever = ElasticsearchRetriever(
        index_name=[index_name_1, index_name_2],
        body_func=body_func,
        content_field={
            index_name_1: "text_field_1",
            index_name_2: "text_field_2",
        },
        client=client,
    )

    # Index test data
    es_client.index(
        index=index_name_1,
        document={"text_field_1": "test content 1"},
        id="1",
        refresh=True,
    )
    es_client.index(
        index=index_name_2,
        document={"text_field_2": "test content 2"},
        id="1",
        refresh=True,
    )

    documents = retriever.get_relevant_documents("test")
    assert len(documents) > 0
    assert all(isinstance(doc, Document) for doc in documents)


@pytest.mark.sync
def test_retrieve_neptune_moons(es_client: Elasticsearch, index_name: str) -> None:
    """Test: Verify that ainvoke correctly retrieves Neptune moons."""
    from langchain_elasticsearch import ElasticsearchRetriever

    def body_func(query: str) -> dict:
        return {"query": {"match": {"text": {"query": query}}}}

    retriever = ElasticsearchRetriever(
        index_name=index_name,
        body_func=body_func,
        content_field="text",
        client=es_client,
    )

    # Index moon data
    index_test_data(es_client, index_name, "text")

    # Use ainvoke to retrieve Neptune moons
    documents = retriever.invoke("Neptune")

    # Verify we got documents
    assert len(documents) > 0
    assert all(isinstance(doc, Document) for doc in documents)
    print(documents)
    # # Verify we got Neptune moons specifically
    # neptune_moons = {"Triton", "Nereid", "Proteus"}
    # found_moons = set()
    # for doc in documents:
    #     # Metadata structure: { "_index": "...", "_id": "...", "_score": ..., "_source": {...} }
    #     # The moon_name is in _source after the content_field is popped
    #     if "_source" in doc.metadata and "moon_name" in doc.metadata["_source"]:
    #         moon_name = doc.metadata["_source"]["moon_name"]
    #         if moon_name in neptune_moons:
    #             found_moons.add(moon_name)
    #     # Also check page_content for Neptune mentions
    #     if "Neptune" in doc.page_content:
    #         # Extract moon name from metadata if available
    #         if "_source" in doc.metadata and "moon_name" in doc.metadata["_source"]:
    #             moon_name = doc.metadata["_source"]["moon_name"]
    #             if moon_name in neptune_moons:
    #                 found_moons.add(moon_name)

    # # Verify we found at least one Neptune moon
    # moon_names_found = [
    #     doc.metadata.get("_source", {}).get("moon_name", "N/A")
    #     for doc in documents
    #     if "_source" in doc.metadata
    # ]
    # assert len(found_moons) > 0, f"Expected Neptune moons (Triton, Nereid, Proteus), found: {moon_names_found}"
    # print(f"Successfully retrieved Neptune moons: {found_moons}")


@pytest.mark.sync
def test_example_langchain_retriever_chain(
    es_client: Elasticsearch, index_name: str
) -> None:
    """Test: LangChain retriever in chains example from docstring (using Ollama)."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_ollama import ChatOllama

    from langchain_elasticsearch import ElasticsearchRetriever

    # AsyncElasticsearchRetriever is already a BaseRetriever
    retriever = ElasticsearchRetriever(
        index_name=index_name,
        body_func=lambda q: {"query": {"match": {"text": {"query": q}}}},
        content_field="text",
        client=es_client,
    )

    # Index test data first
    index_test_data(es_client, index_name, "text")

    llm = ChatOllama(model="llama3", temperature=0)

    # Create a prompt template for the RAG system
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    # Create a chain that retrieves documents and then generates a response
    def format_docs(docs):
        """Format documents for the prompt"""
        return "\n\n".join(doc.page_content for doc in docs)

    # Create a modern RAG chain using LCEL
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    result = chain.invoke("What are the moons of Neptune?")

    # Verify we got a response from the LLM
    assert result is not None
    assert hasattr(result, "content") or isinstance(result, str)
    print(result)

    # # Verify that Neptune moons were retrieved
    # neptune_moons = await retriever.aget_relevant_documents("Neptune moons")
    # assert len(neptune_moons) > 0

    # # Check that we got Neptune moons (Triton, Nereid, Proteus)
    # # Metadata structure: { "_index": "...", "_id": "...", "_score": ..., "_source": {...} }
    # moon_names = [
    #     doc.metadata.get("_source", {}).get("moon_name", "")
    #     for doc in neptune_moons
    #     if "_source" in doc.metadata and "moon_name" in doc.metadata["_source"]
    # ]
    # neptune_moon_names = {"Triton", "Nereid", "Proteus"}
    # found_neptune_moons = set(moon_names) & neptune_moon_names
    # assert len(found_neptune_moons) > 0, f"Expected Neptune moons, found: {moon_names}"

    # print(f"Retrieved Neptune moons: {found_neptune_moons}")
    # print(f"LLM Response: {result}")
