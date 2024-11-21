from typing import List

from langchain_core.messages import BaseMessage

from langchain_elasticsearch._async.chat_history import (
    AsyncElasticsearchChatMessageHistory as _AsyncElasticsearchChatMessageHistory,
)
from langchain_elasticsearch._sync.chat_history import (
    ElasticsearchChatMessageHistory as _ElasticsearchChatMessageHistory,
)


# add the messages property which is only in the sync version
class ElasticsearchChatMessageHistory(_ElasticsearchChatMessageHistory):
    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore[override]
        return self.get_messages()


# langchain defines some sync methods as abstract in its base class
# so we have to add dummy methods for them, even though we only use the async versions
class AsyncElasticsearchChatMessageHistory(_AsyncElasticsearchChatMessageHistory):
    def clear(self) -> None:
        raise NotImplementedError("This class is asynchronous, use aclear()")
