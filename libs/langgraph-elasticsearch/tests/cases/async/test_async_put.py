import pytest
from tests.cases.test_put import TestPutOperation

@pytest.mark.asyncio
class TestSyncPutOperation(TestPutOperation):
    pass