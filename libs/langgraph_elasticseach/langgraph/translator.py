from typing import Any, Dict, Tuple, Union

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)
from pydantic import Extra


class ElasticsearchTranslator(Visitor):
    """
    Translates a structured query language into Elasticsearch's query DSL.
    """

    # Subset of allowed comparators and operators
    allowed_comparators = [
        Comparator.EQ,
        Comparator.GT,
        Comparator.GTE,
        Comparator.LT,
        Comparator.LTE,
        Comparator.CONTAIN,
        Comparator.LIKE,
        Comparator.IN,
        Comparator.NIN,
    ]
    allowed_operators = [Operator.AND, Operator.OR, Operator.NOT]

    # Mapping of comparators and operators to Elasticsearch syntax
    _operator_map = {
        Operator.OR: "should",
        Operator.NOT: "must_not",
        Operator.AND: "must",
    }
    _comparator_map = {
        Comparator.EQ: "term",
        Comparator.GT: "gt",
        Comparator.GTE: "gte",
        Comparator.LT: "lt",
        Comparator.LTE: "lte",
        Comparator.CONTAIN: "wildcard",
        Comparator.LIKE: "match",
        Comparator.IN: "terms",
        Comparator.NIN: "must_not_terms",  # Custom handling for NOT IN
    }

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        """
        Maps an operator or comparator to its Elasticsearch equivalent.

        Args:
            func (Union[Operator, Comparator]): The operator or comparator.

        Returns:
            str: The Elasticsearch equivalent.
        """
        self._validate_func(func)
        if isinstance(func, Operator):
            return self._operator_map[func]
        return self._comparator_map[func]

    def str_to_comparator(self, comparator: str) -> Comparator:
        """
        Converts a string representation of a comparator to its enum.

        Args:
            comparator (str): The string representation of the comparator.

        Returns:
            Comparator: The corresponding `Comparator` enum.
        """
        comparator_map = {
            "$eq": Comparator.EQ,
            "$ne": Comparator.NE,
            "$gt": Comparator.GT,
            "$gte": Comparator.GTE,
            "$lt": Comparator.LT,
            "$lte": Comparator.LTE,
            "$contain": Comparator.CONTAIN,
            "$like": Comparator.LIKE,
            "$in": Comparator.IN,
            "$nin": Comparator.NIN,
        }
        return comparator_map.get(comparator.lower(), Comparator.EQ)

    def visit_operation(self, operation: Operation) -> Dict:
        """
        Translates logical operations (AND, OR, NOT) into Elasticsearch bool queries.

        Args:
            operation (Operation): The logical operation.

        Returns:
            Dict: The translated Elasticsearch query.
        """
        args = [arg.accept(self) for arg in operation.arguments]
        
        return {"bool": {self._format_func(operation.operator): args}}

    def visit_comparison(self, comparison: Comparison) -> Dict:
        """
        Translates comparisons (EQ, GT, CONTAIN, etc.) into Elasticsearch filters.

        Args:
            comparison (Comparison): The comparison expression.

        Returns:
            Dict: The translated Elasticsearch query.
        """
        if comparison.comparator in [Comparator.GT, Comparator.GTE, Comparator.LT, Comparator.LTE]:
            value = comparison.value.get("date") if isinstance(comparison.value, dict) and "date" in comparison.value else comparison.value
            return {"range": {comparison.attribute: {self._format_func(comparison.comparator): value}}}

        if comparison.comparator == Comparator.CONTAIN:
            return {
                self._format_func(comparison.comparator): {
                    comparison.attribute: f"*{comparison.value}*" 
                        if not comparison.value.startswith("*") and not comparison.value.endswith("*")
                        else comparison.value
                }
            }

        if comparison.comparator == Comparator.LIKE:
            return {
                self._format_func(comparison.comparator): {
                    comparison.attribute: {"query": comparison.value, "fuzziness": "AUTO"}
                }
            }

        if comparison.comparator == Comparator.IN:
            return {
                self._format_func(comparison.comparator): {
                    comparison.attribute: comparison.value
                }
            }

        if comparison.comparator == Comparator.NIN:
            return {
                "bool": {
                    self._format_func(Comparator.NIN): [
                        {"terms": {comparison.attribute: comparison.value}}
                    ]
                }
            }

        value = comparison.value.get("date") if isinstance(comparison.value, dict) and "date" in comparison.value else comparison.value
        return {self._format_func(comparison.comparator): {comparison.attribute: value}}

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        """
        Translates a structured query into Elasticsearch's query format.

        Args:
            structured_query (StructuredQuery): The structured query to translate.

        Returns:
            Tuple[str, dict]: A tuple containing the query string and the translated filters.
        """
        
        elastic_query = {"query": structured_query.filter.accept(self)} if structured_query.filter else {}
        
        if structured_query.limit is not None:
            elastic_query["size"] = structured_query.limit

        if hasattr(structured_query, "args"):
            elastic_query.update(structured_query.model_extra)

        return structured_query.query, elastic_query


class ExtendedStructuredQuery(StructuredQuery):
    def __init__(self, **kwargs):
        self.extra_attributes = {k: v for k, v in kwargs.items() if k not in self.model_fields}
        super().__init__(**{k: v for k, v in kwargs.items() if k in self.model_fields})

    def __getattr__(self, name: str):
        if name in self.extra_attributes:
            return self.extra_attributes[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def dict(self, *args, **kwargs):
        base_dict = super().model_dump(*args, **kwargs)
        return {**base_dict, **self.extra_attributes}


# class ExtendedStructuredQuery(StructuredQuery):
#     class Config:
#         extra = "allow"
