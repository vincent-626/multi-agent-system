"""Unit converter for particle physics quantities.

Supported quantity families and their units
-------------------------------------------
Energy        : eV, keV, MeV, GeV, TeV, PeV, J, erg
Cross-section : b/barn, mb, μb/ub, nb, pb, fb, ab  (and cm², m²)
Length        : m, cm, mm, μm/um, nm, pm, fm/fermi, Å/angstrom, ly, pc, kpc, Mpc
Mass          : eV/c², keV/c², MeV/c², GeV/c², TeV/c², u/amu/Da, kg, g
Momentum      : eV/c, keV/c, MeV/c, GeV/c, TeV/c
Time          : s, ms, μs/us, ns, ps, fs
Temperature ↔ Energy: K ↔ any energy unit (via Boltzmann constant)
"""

# ── Physical constants ────────────────────────────────────────────────────────

_KB_EV_PER_K = 8.617333262145e-5    # Boltzmann constant (eV / K)
_MEV_PER_U   = 931.49410242          # 1 atomic mass unit in MeV/c²
_MEV_PER_KG  = 5.609588604e29        # 1 kg·c² in MeV
_EV_PER_J    = 6.241509074460763e18  # 1 J in eV

# ── Unit tables ───────────────────────────────────────────────────────────────
# factor[unit] = how many *base units* equal one of this unit.
# Base units: energy→eV | cross_section→barn | length→m |
#             mass→MeV/c² | momentum→MeV/c | time→s

_FAMILIES: dict[str, dict[str, float]] = {
    "energy": {
        "ev":  1.0,
        "kev": 1e3,
        "mev": 1e6,
        "gev": 1e9,
        "tev": 1e12,
        "pev": 1e15,
        "j":   _EV_PER_J,
        "erg": _EV_PER_J * 1e-7,
    },
    "cross_section": {
        "b":         1.0,
        "barn":      1.0,
        "mb":        1e-3,
        "millibarn": 1e-3,
        "ub":        1e-6,
        "μb":        1e-6,
        "microbarn": 1e-6,
        "nb":        1e-9,
        "nanobarn":  1e-9,
        "pb":        1e-12,
        "picobarn":  1e-12,
        "fb":        1e-15,
        "femtobarn": 1e-15,
        "ab":        1e-18,
        "attobarn":  1e-18,
        "cm2":       1e24,   # 1 barn = 1e-24 cm²
        "cm²":       1e24,
        "m2":        1e28,   # 1 barn = 1e-28 m²
        "m²":        1e28,
    },
    "length": {
        "m":        1.0,
        "cm":       1e-2,
        "mm":       1e-3,
        "um":       1e-6,
        "μm":       1e-6,
        "nm":       1e-9,
        "pm":       1e-12,
        "fm":       1e-15,
        "fermi":    1e-15,
        "a":        1e-10,
        "å":        1e-10,
        "angstrom": 1e-10,
        "ly":       9.4607304725808e15,
        "pc":       3.085677581491367e16,
        "kpc":      3.085677581491367e19,
        "mpc":      3.085677581491367e22,
    },
    "mass": {
        "ev/c2":   1e-6,
        "ev/c²":   1e-6,
        "kev/c2":  1e-3,
        "kev/c²":  1e-3,
        "mev/c2":  1.0,
        "mev/c²":  1.0,
        "gev/c2":  1e3,
        "gev/c²":  1e3,
        "tev/c2":  1e6,
        "tev/c²":  1e6,
        "u":       _MEV_PER_U,
        "amu":     _MEV_PER_U,
        "da":      _MEV_PER_U,
        "kg":      _MEV_PER_KG,
        "g":       _MEV_PER_KG * 1e-3,
    },
    "momentum": {
        "ev/c":   1e-6,
        "kev/c":  1e-3,
        "mev/c":  1.0,
        "gev/c":  1e3,
        "tev/c":  1e6,
    },
    "time": {
        "s":   1.0,
        "ms":  1e-3,
        "us":  1e-6,
        "μs":  1e-6,
        "ns":  1e-9,
        "ps":  1e-12,
        "fs":  1e-15,
    },
}

_TEMPERATURE_UNITS = {"k", "kelvin"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise(unit: str) -> str:
    """Lowercase and canonicalise a unit string for lookup."""
    return (
        unit.strip()
        .lower()
        .replace("^2", "2")   # MeV/c^2 → mev/c2
        .replace("²", "2")    # MeV/c² → mev/c2 (after lower())
        .replace(" ", "")
    )


def _find(unit: str) -> tuple[str, float] | None:
    """Return (family, factor) for a normalised unit, or None if unknown."""
    for family, table in _FAMILIES.items():
        if unit in table:
            return family, table[unit]
    return None


def _fmt(value: float) -> str:
    """Format a float with up to 6 significant figures."""
    return f"{value:.6g}"


# ── Public API ────────────────────────────────────────────────────────────────

def convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert *value* from *from_unit* to *to_unit*.

    Handles all unit families listed in the module docstring, plus
    temperature ↔ energy conversions via the Boltzmann constant.

    Args:
        value:     Numeric value to convert.
        from_unit: Source unit (case-insensitive, e.g. ``"MeV"``, ``"fb"``).
        to_unit:   Target unit (case-insensitive, e.g. ``"GeV"``, ``"pb"``).

    Returns:
        Human-readable result string, or an error message on failure.
    """
    fu = _normalise(from_unit)
    tu = _normalise(to_unit)

    if fu == tu:
        return f"{_fmt(value)} {from_unit} = {_fmt(value)} {to_unit}"

    # ── Temperature ↔ Energy ──────────────────────────────────────────────────
    if fu in _TEMPERATURE_UNITS:
        ev_value = value * _KB_EV_PER_K
        if tu in _TEMPERATURE_UNITS:
            return f"{_fmt(value)} {from_unit} = {_fmt(value)} {to_unit}"
        target = _find(tu)
        if target and target[0] == "energy":
            result = ev_value / target[1]
            return f"{_fmt(value)} {from_unit} = {_fmt(result)} {to_unit}  [kB = {_KB_EV_PER_K:.4e} eV/K]"
        return f"Error: cannot convert temperature to '{to_unit}' (only energy units supported)"

    from_info = _find(fu)
    if from_info is None:
        return f"Error: unknown unit '{from_unit}'"
    from_family, from_factor = from_info

    if tu in _TEMPERATURE_UNITS:
        if from_family != "energy":
            return f"Error: cannot convert {from_family} unit '{from_unit}' to temperature"
        ev_value = value * from_factor
        result = ev_value / _KB_EV_PER_K
        return f"{_fmt(value)} {from_unit} = {_fmt(result)} K  [kB = {_KB_EV_PER_K:.4e} eV/K]"

    to_info = _find(tu)
    if to_info is None:
        return f"Error: unknown unit '{to_unit}'"
    to_family, to_factor = to_info

    if from_family != to_family:
        return (
            f"Error: unit mismatch — '{from_unit}' is a {from_family} unit "
            f"but '{to_unit}' is a {to_family} unit"
        )

    result = (value * from_factor) / to_factor
    return f"{_fmt(value)} {from_unit} = {_fmt(result)} {to_unit}"
