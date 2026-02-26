"""Safe arithmetic calculator using Python's ast module (never eval)."""

from __future__ import annotations

import ast
import operator
from typing import Union

# Supported operators
_OPERATORS: dict = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> Union[int, float]:
    """Recursively evaluate a single AST node.

    Only numeric literals and the whitelisted operators are permitted.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported literal type: {type(node.value)}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_node(node.operand)
        return _OPERATORS[op_type](operand)

    raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression.

    Args:
        expression: A string like ``"2 + 2"`` or ``"(10 * 3) / 4"``.

    Returns:
        The result as a string, or an error message if the expression is
        invalid or uses unsupported operations.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        return f"Error: invalid expression syntax — {exc}"

    try:
        result = _eval_node(tree.body)
    except (ValueError, ZeroDivisionError) as exc:
        return f"Error: {exc}"

    # Return integer representation when the result is a whole number
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)
