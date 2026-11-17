import pytest
from langchain_core.structured_query import Comparator, Comparison, Operation, Operator, StructuredQuery
from langgraph.translator import ElasticsearchTranslator, ExtendedStructuredQuery

@pytest.fixture
def translator():
    return ElasticsearchTranslator()

def test_format_func_operator(translator):
    assert translator._format_func(Operator.AND) == "must"
    assert translator._format_func(Operator.OR) == "should"
    assert translator._format_func(Operator.NOT) == "must_not"

def test_format_func_comparator(translator):
    assert translator._format_func(Comparator.EQ) == "term"
    assert translator._format_func(Comparator.GT) == "gt"
    assert translator._format_func(Comparator.GTE) == "gte"
    assert translator._format_func(Comparator.LT) == "lt"
    assert translator._format_func(Comparator.LTE) == "lte"
    assert translator._format_func(Comparator.CONTAIN) == "wildcard"
    assert translator._format_func(Comparator.LIKE) == "match"
    assert translator._format_func(Comparator.IN) == "terms"
    assert translator._format_func(Comparator.NIN) == "must_not_terms"

def test_str_to_comparator(translator):
    assert translator.str_to_comparator("$eq") == Comparator.EQ
    assert translator.str_to_comparator("$ne") == Comparator.NE
    assert translator.str_to_comparator("$gt") == Comparator.GT
    assert translator.str_to_comparator("$gte") == Comparator.GTE
    assert translator.str_to_comparator("$lt") == Comparator.LT
    assert translator.str_to_comparator("$lte") == Comparator.LTE
    assert translator.str_to_comparator("$contain") == Comparator.CONTAIN
    assert translator.str_to_comparator("$like") == Comparator.LIKE
    assert translator.str_to_comparator("$in") == Comparator.IN
    assert translator.str_to_comparator("$nin") == Comparator.NIN

def test_visit_operation(translator):
    operation = Operation(operator=Operator.AND, arguments=[
        Comparison(attribute="age", comparator=Comparator.GT, value=30),
        Comparison(attribute="name", comparator=Comparator.LIKE, value="John")
    ])
    result = translator.visit_operation(operation)
    expected = {
        "bool": {
            "must": [
                {"range": {"age": {"gt": 30}}},
                {"match": {"name": {"query": "John", "fuzziness": "AUTO"}}}
            ]
        }
    }
    assert result == expected

def test_visit_comparison(translator):
    comparison = Comparison(attribute="age", comparator=Comparator.GT, value=30)
    result = translator.visit_comparison(comparison)
    expected = {"range": {"age": {"gt": 30}}}
    assert result == expected

    comparison = Comparison(attribute="name", comparator=Comparator.LIKE, value="John")
    result = translator.visit_comparison(comparison)
    expected = {"match": {"name": {"query": "John", "fuzziness": "AUTO"}}}
    assert result == expected

    comparison = Comparison(attribute="tags", comparator=Comparator.IN, value=["python", "elasticsearch"])
    result = translator.visit_comparison(comparison)
    expected = {"terms": {"tags": ["python", "elasticsearch"]}}
    assert result == expected

    comparison = Comparison(attribute="tags", comparator=Comparator.NIN, value=["python", "elasticsearch"])
    result = translator.visit_comparison(comparison)
    expected = {"bool": {"must_not_terms": [{"terms": {"tags": ["python", "elasticsearch"]}}]}}
    assert result == expected

def test_visit_structured_query(translator):
    structured_query = ExtendedStructuredQuery(
        query="test query",
        filter=Operation(operator=Operator.AND, arguments=[
            Comparison(attribute="age", comparator=Comparator.GT, value=30),
            Comparison(attribute="name", comparator=Comparator.LIKE, value="John")
        ]),
        limit=10
    )
    query, result = translator.visit_structured_query(structured_query)
    expected = {
        "query": {
            "bool": {
                "must": [
                    {"range": {"age": {"gt": 30}}},
                    {"match": {"name": {"query": "John", "fuzziness": "AUTO"}}}
                ]
            }
        },
        "size": 10
    }
    assert query == "test query"
    assert result == expected