from typing import Any, Iterator, List, Optional, Sequence, Tuple

from elasticsearch import Elasticsearch, exceptions, helpers  # noqa: F401
from elasticsearch.helpers import BulkIndexError  # noqa: F401
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache  # noqa: F401
from langchain_core.load import dumps, loads  # noqa: F401
from langchain_core.stores import ByteStore  # noqa: F401

from langchain_elasticsearch._async.cache import (
    AsyncElasticsearchCache as _AsyncElasticsearchCache,
)
from langchain_elasticsearch._async.cache import (
    AsyncElasticsearchEmbeddingsCache as _AsyncElasticsearchEmbeddingsCache,
)
from langchain_elasticsearch._sync.cache import (
    ElasticsearchCache as _ElasticsearchCache,
)
from langchain_elasticsearch._sync.cache import (
    ElasticsearchEmbeddingsCache as _ElasticsearchEmbeddingsCache,
)
from langchain_elasticsearch.client import (  # noqa: F401
    create_async_elasticsearch_client,
    create_elasticsearch_client,
)


# langchain defines some sync methods as abstract in its base class
# so we have to add dummy methods for them, even though we only use the async versions
class AsyncElasticsearchCache(_AsyncElasticsearchCache):
    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        raise NotImplementedError("This class is asynchronous, use alookup()")

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        raise NotImplementedError("This class is asynchronous, use aupdate()")

    def clear(self, **kwargs: Any) -> None:
        raise NotImplementedError("This class is asynchronous, use aclear()")


# langchain defines some sync methods as abstract in its base class
# so we have to add dummy methods for them, even though we only use the async versions
class AsyncElasticsearchEmbeddingsCache(_AsyncElasticsearchEmbeddingsCache):
    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        raise NotImplementedError("This class is asynchronous, use amget()")

    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        raise NotImplementedError("This class is asynchronous, use amset()")

    def mdelete(self, keys: Sequence[str]) -> None:
        raise NotImplementedError("This class is asynchronous, use amdelete()")

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        raise NotImplementedError("This class is asynchronous, use ayield_keys()")


# these are only defined here so that they are picked up by Langchain's docs generator
class ElasticsearchCache(_ElasticsearchCache):
    pass


class ElasticsearchEmbeddingsCache(_ElasticsearchEmbeddingsCache):
    pass
