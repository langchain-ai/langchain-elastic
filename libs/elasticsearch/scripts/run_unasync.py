import os
import subprocess
import sys
from glob import glob
from pathlib import Path

import unasync


def main(check=False):
    # the list of directories that need to be processed with unasync
    # each entry has two paths:
    #   - the source path with the async sources
    #   - the destination path where the sync sources should be written
    source_dirs = [
        (
            "langchain_elasticsearch/_async/",
            "langchain_elasticsearch/_sync/",
        ),
        ("tests/_async/", "tests/_sync/"),
        ("tests/integration_tests/_async/", "tests/integration_tests/_sync/"),
        ("tests/unit_tests/_async/", "tests/unit_tests/_sync/"),
    ]

    # Unasync all the generated async code
    additional_replacements = {
        "_async": "_sync",
        "AsyncElasticsearch": "Elasticsearch",
        "AsyncTransport": "Transport",
        "AsyncBM25Strategy": "BM25Strategy",
        "AsyncDenseVectorScriptScoreStrategy": "DenseVectorScriptScoreStrategy",
        "AsyncDenseVectorStrategy": "DenseVectorStrategy",
        "AsyncRetrievalStrategy": "RetrievalStrategy",
        "AsyncSparseVectorStrategy": "SparseVectorStrategy",
        "AsyncVectorStore": "VectorStore",
        "AsyncElasticsearchStore": "ElasticsearchStore",
        "AsyncElasticsearchEmbeddings": "ElasticsearchEmbeddings",
        "AsyncElasticsearchEmbeddingsCache": "ElasticsearchEmbeddingsCache",
        "AsyncEmbeddingServiceAdapter": "EmbeddingServiceAdapter",
        "AsyncEmbeddingService": "EmbeddingService",
        "AsyncElasticsearchRetriever": "ElasticsearchRetriever",
        "AsyncElasticsearchCache": "ElasticsearchCache",
        "AsyncElasticsearchChatMessageHistory": "ElasticsearchChatMessageHistory",
        "AsyncCallbackManagerForRetrieverRun": "CallbackManagerForRetrieverRun",
        "AsyncConsistentFakeEmbeddings": "ConsistentFakeEmbeddings",
        "AsyncRequestSavingTransport": "RequestSavingTransport",
        "AsyncMock": "Mock",
        "Embeddings": "Embeddings",
        "AsyncGenerator": "Generator",
        "AsyncIterator": "Iterator",
        "create_async_elasticsearch_client": "create_elasticsearch_client",
        "aadd_texts": "add_texts",
        "aadd_embeddings": "add_embeddings",
        "aadd_documents": "add_documents",
        "afrom_texts": "from_texts",
        "afrom_documents": "from_documents",
        "amax_marginal_relevance_search": "max_marginal_relevance_search",
        "asimilarity_search": "similarity_search",
        "asimilarity_search_by_vector_with_relevance_scores": "similarity_search_by_vector_with_relevance_scores",  # noqa: E501
        "asimilarity_search_with_score": "similarity_search_with_score",
        "asimilarity_search_with_relevance_scores": "similarity_search_with_relevance_scores",  # noqa: E501
        "adelete": "delete",
        "aclose": "close",
        "ainvoke": "invoke",
        "aembed_documents": "embed_documents",
        "aembed_query": "embed_query",
        "_aget_relevant_documents": "_get_relevant_documents",
        "aget_relevant_documents": "get_relevant_documents",
        "alookup": "lookup",
        "aupdate": "update",
        "aclear": "clear",
        "amget": "mget",
        "amset": "mset",
        "amdelete": "mdelete",
        "ayield_keys": "yield_keys",
        "asearch": "search",
        "aget_messages": "get_messages",
        "aadd_messages": "add_messages",
        "aadd_message": "add_message",
        "aencode_vector": "encode_vector",
        "assert_awaited_with": "assert_called_with",
        "async_es_client_fx": "es_client_fx",
        "async_es_embeddings_cache_fx": "es_embeddings_cache_fx",
        "async_es_cache_fx": "es_cache_fx",
        "async_bulk": "bulk",
        "async_with_user_agent_header": "with_user_agent_header",
        "asyncio": "sync",
    }
    rules = [
        unasync.Rule(
            fromdir=dir[0],
            todir=f"{dir[0]}_sync_check/" if check else dir[1],
            additional_replacements=additional_replacements,
        )
        for dir in source_dirs
    ]

    filepaths = []
    for root, _, filenames in os.walk(Path(__file__).absolute().parent.parent):
        if "/site-packages" in root or "/." in root or "__pycache__" in root:
            continue
        for filename in filenames:
            if filename.rpartition(".")[-1] in (
                "py",
                "pyi",
            ) and not filename.startswith("utils.py"):
                filepaths.append(os.path.join(root, filename))

    unasync.unasync_files(filepaths, rules)
    for dir in source_dirs:
        output_dir = f"{dir[0]}_sync_check/" if check else dir[1]
        subprocess.check_call(["ruff", "format", "--target-version=py38", output_dir])
        subprocess.check_call(["ruff", "check", "--fix", "--select", "I", output_dir])
        for file in glob("*.py", root_dir=dir[0]):
            subprocess.check_call(
                [
                    "sed",
                    "-i.bak",
                    "s/pytest.mark.asyncio/pytest.mark.sync/",
                    f"{output_dir}{file}",
                ]
            )
            subprocess.check_call(
                [
                    "sed",
                    "-i.bak",
                    "s/get_messages()/messages/",
                    f"{output_dir}{file}",
                ]
            )
            subprocess.check_call(["rm", f"{output_dir}{file}.bak"])

            if check:
                # make sure there are no differences between _sync and _sync_check
                subprocess.check_call(
                    [
                        "diff",
                        f"{dir[1]}{file}",
                        f"{output_dir}{file}",
                    ]
                )

        if check:
            subprocess.check_call(["rm", "-rf", output_dir])


if __name__ == "__main__":
    main(check="--check" in sys.argv)
