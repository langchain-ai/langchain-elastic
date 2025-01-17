from langgraph.store.base import PutOp
import pytest

from langgraph.store.elasticsearch.queries.base import namespace_to_text, text_to_namespace
from elastic_transport import ConnectionError
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
mocks = [
    {
        "_index": "langgraph-store",
        "_id": "test/foo/key1",
        "_source": {
            "namespace": "test/foo",
            "key": "key1",
            "value": {
                "data": "value1"
            },
            "created_at": "2021-10-01T00:00:00Z",
            "updated_at": "2021-10-01T00:00:00Z"
        }
    },
    {
        "_index": "langgraph-store",
        "_id": "test/foo/key2",
        "_source": {
            "namespace": "test/foo",
            "key": "key2",
            "value": {
                "data": "value2"
            },
            "created_at": "2021-10-01T00:00:00Z",
            "updated_at": "2021-10-01T00:00:00Z"
        }
    }
]

@pytest.mark.parametrize(
    "index_name",
    "langgraph-store",
)
@pytest.mark.mock
class TestPutOperation:
    @pytest.mark.parametrize(
        "op",
        PutOp(namespace=("test", "foo"), key="key1", value={"data": "value1"}),
    )    
    def test_one_put(self, factory_es_client, store, request, op, index_name):
        store.batch(op)
        _, _, kwargs = factory_es_client.update.mock_calls[0]

        assert kwargs["index"] == index_name
        assert kwargs["id"] == text_to_namespace(op.namespace + (op.key,))
        assert kwargs["body"]["upsert"]["namespace"] == namespace_to_text(op.namespace)
        assert kwargs["body"]["upsert"]["key"] == op.key
        assert kwargs["body"]["upsert"]["value"] == op.value
        assert kwargs["body"]["doc"]["value"] == op.value

    @pytest.mark.parametrize(
        "ops",
        [
            PutOp(namespace=("test", "foo"), key="key1", value={"data": "value1"}),
            PutOp(namespace=("test", "foo"), key="key2", value={"data": "value2"}),
        ],
    )
    def test_multiple_puts(self, factory_es_client, store, ops, index_name):
        ops = [
            PutOp(namespace=("test", "foo"), key="key1", value={"data": "value1"}),
            PutOp(namespace=("test", "foo"), key="key2", value={"data": "value2"}),
        ]

        store.batch(ops)
        for idx, calls in enumerate(factory_es_client.update.mock_calls):
            _, _, kwargs = calls
            assert kwargs["index"] == mocks[idx]["_index"]
            assert kwargs["id"] == mocks[idx]["_id"]
            assert kwargs["body"]["upsert"]["namespace"] == mocks[idx]["_source"]["namespace"]
            assert kwargs["body"]["upsert"]["key"] == mocks[idx]["_source"]["key"]
            assert kwargs["body"]["upsert"]["value"] == mocks[idx]["_source"]["value"]
            assert kwargs["body"]["doc"]["value"] == mocks[idx]["_source"]["value"]

    @pytest.mark.fail
    def test_fail_connection(self, store, ops, index_name):
        ops = [
            PutOp(namespace=("test", "foo"), key="key1", value={"data": "value1"}),
        ]
        with pytest.raises(ConnectionError):
            store.batch(ops)

@pytest.mark.integration
class TestIntegrationPutOperation(TestPutOperation):
    def test_one_put(self, factory_es_client, store, request):
        mock = mocks[0]
        op = [
            PutOp(
                namespace=text_to_namespace(mock["_source"]["namespace"]), 
                key=mock["_source"]["key"],
                value=mock["_source"]["value"],)
        ]

        store.batch(op)
        retrieved = factory_es_client.get(index=mock["_index"], id=mock["_id"])

        assert retrieved["_index"] == mock["_index"]
        assert retrieved["_id"] == mock["_id"]
        assert retrieved["_source"]["namespace"] == mock["_source"]["namespace"]
        assert retrieved["_source"]["key"] == mock["_source"]["key"]
        assert retrieved["_source"]["value"] == mock["_source"]["value"]

@pytest.mark.embed        
class TestPutOperationWithEmbeddings(TestPutOperation):
    @pytest.mark.fail
    def test_fail(self, store):
        ops = [
            PutOp(namespace=("test", "foo"), key="key1", value={"data": "value1"}),
        ]
        with pytest.raises(ConnectionError):
            store.batch(ops)



