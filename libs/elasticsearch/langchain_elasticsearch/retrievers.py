from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from langchain_elasticsearch._async.retrievers import (
    AsyncElasticsearchRetriever as _AsyncElasticsearchRetriever,
)
from langchain_elasticsearch._sync.retrievers import (
    ElasticsearchRetriever,  # noqa: F401
)


# langchain defines some sync methods as abstract in its base class
# so we have to add dummy methods for them, even though we only use the async versions
class AsyncElasticsearchRetriever(_AsyncElasticsearchRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        raise NotImplementedError(
            "This class is asynchronous, use _aget_relevant_documents()"
        )
