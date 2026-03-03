"""Unit tests for the AST-based scientific calculator."""

import pytest

from src.tools.calculator import calculate


# ── Arithmetic ────────────────────────────────────────────────────────────────

def test_addition():
    assert calculate("2 + 3") == "5"


def test_subtraction():
    assert calculate("10 - 4") == "6"


def test_multiplication():
    assert calculate("3 * 4") == "12"


def test_true_division():
    assert calculate("10 / 4") == "2.5"


def test_power():
    assert calculate("2 ** 10") == "1024"


def test_modulo():
    assert calculate("10 % 3") == "1"


def test_floor_division():
    assert calculate("10 // 3") == "3"


def test_unary_minus():
    assert calculate("-5") == "-5"


def test_unary_minus_expression():
    assert calculate("-(3 + 2)") == "-5"


# ── Math functions ────────────────────────────────────────────────────────────

def test_sqrt():
    assert calculate("sqrt(4)") == "2"


def test_abs():
    assert calculate("abs(-7)") == "7"


def test_floor():
    assert calculate("floor(3.9)") == "3"


def test_ceil():
    assert calculate("ceil(3.1)") == "4"


def test_log_natural():
    # log(1) = 0
    assert calculate("log(1)") == "0"


def test_exp():
    # exp(0) = 1
    assert calculate("exp(0)") == "1"


def test_sin_zero():
    assert calculate("sin(0)") == "0"


def test_cos_zero():
    assert calculate("cos(0)") == "1"


# ── Physics constants ─────────────────────────────────────────────────────────

def test_proton_mass():
    # m_p = 938.27208816 MeV/c²  →  f"{938.27208816:.6g}" = "938.272"
    assert calculate("m_p") == "938.272"


def test_pi():
    result = calculate("pi")
    assert result.startswith("3.14159")


def test_hbar_scientific_notation():
    # hbar is ~1e-34, so .6g produces scientific notation
    result = calculate("hbar")
    assert "e" in result.lower()


def test_speed_of_light():
    result = calculate("c")
    assert result == "299792458"


def test_hbarc():
    # hbarc = 197.3269804 MeV·fm  →  "197.327"
    assert calculate("hbarc") == "197.327"


def test_physics_expression():
    # sqrt(m_p**2 + 500**2) should produce a plausible number, not an error
    result = calculate("sqrt(m_p**2 + 500**2)")
    assert not result.startswith("Error")
    assert float(result) > 900  # momentum adds to m_p ≈ 938, result > 938


# ── Error handling ────────────────────────────────────────────────────────────

def test_division_by_zero():
    result = calculate("1 / 0")
    assert result.startswith("Error")


def test_unknown_variable():
    result = calculate("undefined_var")
    assert result.startswith("Error")
    assert "undefined_var" in result


def test_unknown_function():
    result = calculate("exec(1)")
    assert result.startswith("Error")
    assert "exec" in result


def test_invalid_syntax():
    result = calculate("2 +* 3")
    assert result.startswith("Error")


def test_string_literal_rejected():
    result = calculate('"hello"')
    assert result.startswith("Error")


# ── Output formatting ─────────────────────────────────────────────────────────

def test_integer_output_no_decimal():
    # Integers < 1e15 must be returned without a decimal point
    assert calculate("2 ** 10") == "1024"
    assert calculate("3 * 4") == "12"


def test_float_output_six_significant_figures():
    # m_p has more than 6 sig figs; result must be trimmed
    result = calculate("m_p")
    # .6g gives at most 6 significant figures
    assert len(result.replace(".", "").replace("-", "").rstrip("0")) <= 6
