from typing import Any, Iterator, List, Optional, Sequence, Tuple

from elasticsearch import Elasticsearch, exceptions, helpers
from elasticsearch.helpers import BulkIndexError
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.load import dumps, loads
from langchain_core.stores import ByteStore

from langchain_elasticsearch.client import create_elasticsearch_client, create_async_elasticsearch_client
from langchain_elasticsearch._async.cache import AsyncElasticsearchCache as _AsyncElasticsearchCache, AsyncElasticsearchEmbeddingsCache as _AsyncElasticsearchEmbeddingsCache
from langchain_elasticsearch._sync.cache import ElasticsearchCache, ElasticsearchEmbeddingsCache


# langchain defines some sync methods as abstract in its base class
# so we have to add dummy methods for them, even though we only use the async versions
class AsyncElasticsearchCache(_AsyncElasticsearchCache):
    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        raise NotImplemented("This class is asynchronous, use alookup()")

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        raise NotImplemented("This class is asynchronous, use aupdate()")

    def clear(self, **kwargs: Any) -> None:
        raise NotImplemented("This class is asynchronous, use aclear()")


# langchain defines some sync methods as abstract in its base class
# so we have to add dummy methods for them, even though we only use the async versions
class AsyncElasticsearchEmbeddingsCache(_AsyncElasticsearchEmbeddingsCache):   
    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        raise NotImplemented("This class is asynchronous, use amget()")

    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        raise NotImplemented("This class is asynchronous, use amset()")

    def mdelete(self, keys: Sequence[str]) -> None:
        raise NotImplemented("This class is asynchronous, use amdelete()")

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        raise NotImplemented("This class is asynchronous, use ayield_keys()")
