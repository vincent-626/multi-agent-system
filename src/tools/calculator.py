"""Scientific calculator for particle physics expressions.

Extends basic arithmetic with math functions and HEP/SM physics constants.
Uses AST-based evaluation — never calls eval() on arbitrary input.

Available functions
-------------------
sqrt, cbrt, exp, log, log10, log2,
sin, cos, tan, asin, acos, atan, atan2,
sinh, cosh, tanh, abs, floor, ceil, round, factorial

Available constants
-------------------
Mathematical : pi, euler (e), inf
Kinematic    : c          — speed of light, m/s
               hbar       — reduced Planck constant, J·s
               hbar_eV    — reduced Planck constant, eV·s
               hbarc      — ℏc, MeV·fm
Particle masses (MeV/c²):
               m_e        — electron
               m_p        — proton
               m_n        — neutron
               m_mu       — muon
               m_W        — W boson
               m_Z        — Z boson
               m_H        — Higgs boson
Couplings/misc:
               e_charge   — elementary charge, C
               k_B        — Boltzmann constant, eV/K
               N_A        — Avogadro's number
               alpha      — fine-structure constant
               G          — gravitational constant, m³/(kg·s²)
"""

import ast
import math
import operator

# ── Allowed functions ─────────────────────────────────────────────────────────

_FUNCTIONS: dict[str, object] = {
    "sqrt":      math.sqrt,
    "cbrt":      lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "exp":       math.exp,
    "log":       math.log,        # log(x) → ln(x); log(x, base) → log_base(x)
    "log10":     math.log10,
    "log2":      math.log2,
    "sin":       math.sin,
    "cos":       math.cos,
    "tan":       math.tan,
    "asin":      math.asin,
    "acos":      math.acos,
    "atan":      math.atan,
    "atan2":     math.atan2,
    "sinh":      math.sinh,
    "cosh":      math.cosh,
    "tanh":      math.tanh,
    "abs":       abs,
    "floor":     math.floor,
    "ceil":      math.ceil,
    "round":     round,
    "factorial": math.factorial,
}

# ── Physics and math constants ────────────────────────────────────────────────

_CONSTANTS: dict[str, float] = {
    # Mathematical
    "pi":       math.pi,
    "euler":    math.e,
    "inf":      math.inf,
    # Kinematic
    "c":        299_792_458.0,        # speed of light, m/s
    "hbar":     1.054571817e-34,      # reduced Planck constant, J·s
    "hbar_eV":  6.582119569e-16,      # reduced Planck constant, eV·s
    "hbarc":    197.3269804,          # ℏc, MeV·fm
    # Particle masses, MeV/c²
    "m_e":      0.51099895,           # electron
    "m_p":      938.27208816,         # proton
    "m_n":      939.56542052,         # neutron
    "m_mu":     105.6583755,          # muon
    "m_W":      80_377.0,             # W boson
    "m_Z":      91_187.6,             # Z boson
    "m_H":      125_250.0,            # Higgs boson
    # Couplings and misc
    "e_charge": 1.602176634e-19,      # elementary charge, C
    "k_B":      8.617333262e-5,       # Boltzmann constant, eV/K
    "N_A":      6.02214076e23,        # Avogadro's number
    "alpha":    7.2973525693e-3,      # fine-structure constant (~1/137)
    "G":        6.67430e-11,          # gravitational constant, m³/(kg·s²)
}

# ── AST evaluator ─────────────────────────────────────────────────────────────

_BINOPS: dict = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNOPS: dict = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported literal: {node.value!r}")

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(
            f"Unknown name '{node.id}'. "
            f"Available constants: {', '.join(sorted(_CONSTANTS))}"
        )

    if isinstance(node, ast.BinOp):
        op = type(node.op)
        if op not in _BINOPS:
            raise ValueError(f"Unsupported operator: {op.__name__}")
        return _BINOPS[op](_eval(node.left), _eval(node.right))

    if isinstance(node, ast.UnaryOp):
        op = type(node.op)
        if op not in _UNOPS:
            raise ValueError(f"Unsupported unary operator: {op.__name__}")
        return _UNOPS[op](_eval(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported (e.g. sqrt(2))")
        fname = node.func.id
        if fname not in _FUNCTIONS:
            raise ValueError(
                f"Unknown function '{fname}'. "
                f"Available: {', '.join(sorted(_FUNCTIONS))}"
            )
        args = [_eval(arg) for arg in node.args]
        return _FUNCTIONS[fname](*args)  # type: ignore[operator]

    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


# ── Public API ────────────────────────────────────────────────────────────────

def calculate(expression: str) -> str:
    """Evaluate a scientific expression and return a human-readable result.

    Supports math functions and particle physics constants (see module docstring).

    Args:
        expression: Expression string, e.g. ``"hbarc / (m_e * 2)"`` or
                    ``"sqrt(m_p**2 + (500)**2)"`` (momentum in MeV/c).

    Returns:
        Result string, or an error message if the expression is invalid.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        return f"Error: invalid expression syntax — {exc}"

    try:
        result = _eval(tree.body)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        return f"Error: {exc}"

    # Format: use scientific notation for very large/small values
    if result == int(result) and abs(result) < 1e15:
        return str(int(result))
    return f"{result:.6g}"
