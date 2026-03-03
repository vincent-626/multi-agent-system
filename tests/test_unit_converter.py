"""Unit tests for the particle-physics unit converter."""

import pytest

from src.tools.unit_converter import convert


# ── Same-unit round-trip ──────────────────────────────────────────────────────

def test_same_unit_returns_value_unchanged():
    result = convert(5, "MeV", "MeV")
    assert "5" in result


# ── Energy ────────────────────────────────────────────────────────────────────

def test_gev_to_mev():
    result = convert(1, "GeV", "MeV")
    assert "1000" in result


def test_mev_to_gev():
    result = convert(500, "MeV", "GeV")
    assert "0.5" in result


def test_tev_to_gev():
    result = convert(7, "TeV", "GeV")
    assert "7000" in result


def test_ev_to_j():
    # 1 eV = 1.60218e-19 J
    result = convert(1, "eV", "J")
    assert "1.60218e-19" in result


# ── Cross-section ─────────────────────────────────────────────────────────────

def test_pb_to_fb():
    result = convert(1, "pb", "fb")
    assert "1000" in result


def test_nb_to_pb():
    result = convert(1, "nb", "pb")
    assert "1000" in result


def test_barn_to_mb():
    result = convert(1, "b", "mb")
    assert "1000" in result


# ── Length ────────────────────────────────────────────────────────────────────

def test_m_to_cm():
    result = convert(1, "m", "cm")
    assert "100" in result


def test_nm_to_m():
    result = convert(1, "nm", "m")
    assert "1e-09" in result


def test_fm_to_m():
    result = convert(1, "fm", "m")
    assert "1e-15" in result


# ── Mass ──────────────────────────────────────────────────────────────────────

def test_gev_c2_to_mev_c2():
    result = convert(1, "GeV/c2", "MeV/c2")
    assert "1000" in result


def test_u_to_mev_c2():
    # 1 u = 931.494 MeV/c²
    result = convert(1, "u", "MeV/c2")
    assert "931.494" in result


# ── Momentum ──────────────────────────────────────────────────────────────────

def test_gev_c_to_mev_c():
    result = convert(1, "GeV/c", "MeV/c")
    assert "1000" in result


# ── Time ──────────────────────────────────────────────────────────────────────

def test_s_to_ms():
    result = convert(1, "s", "ms")
    assert "1000" in result


def test_ns_to_ps():
    result = convert(1, "ns", "ps")
    assert "1000" in result


# ── Temperature ↔ Energy ─────────────────────────────────────────────────────

def test_kelvin_to_ev():
    # 1 K → kB * 1 = 8.617e-5 eV
    result = convert(1, "K", "eV")
    assert "8.617" in result


def test_ev_to_kelvin():
    # 1 eV / kB ≈ 11604 K
    result = convert(1, "eV", "K")
    assert "11604" in result


# ── Error cases ───────────────────────────────────────────────────────────────

def test_unknown_from_unit():
    result = convert(1, "banana", "MeV")
    assert result.startswith("Error")
    assert "banana" in result


def test_unknown_to_unit():
    result = convert(1, "MeV", "banana")
    assert result.startswith("Error")
    assert "banana" in result


def test_family_mismatch_energy_vs_length():
    result = convert(1, "MeV", "fm")
    assert result.startswith("Error")
    assert "mismatch" in result.lower()


def test_temperature_to_non_energy_unit():
    result = convert(300, "K", "m")
    assert result.startswith("Error")


def test_non_energy_to_temperature():
    result = convert(1, "fm", "K")
    assert result.startswith("Error")


# ── Case insensitivity ────────────────────────────────────────────────────────

def test_case_insensitive_from_unit():
    lower = convert(1, "gev", "MeV")
    upper = convert(1, "GeV", "MeV")
    assert "1000" in lower
    assert "1000" in upper
