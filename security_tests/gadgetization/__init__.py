"""Reference nonlinear gadgetization implementations."""

from . import nonlinear193, nonlinear291
from .nonlinear193 import Circuit, E, maj
from .nonlinear291 import Weight2Circuit

run_nonlinear193 = nonlinear193.run_gate
run_nonlinear291 = nonlinear291.run_gate
build_nonlinear193_chain = nonlinear193.build_chain
build_nonlinear291_chain = nonlinear291.build_chain
NONLINEAR193_R57_GATE_COUNT = nonlinear193.CANONICAL_R57_GATE_COUNT
NONLINEAR291_R57_GATE_COUNT = nonlinear291.CANONICAL_R57_GATE_COUNT

__all__ = [
    "Circuit",
    "E",
    "NONLINEAR193_R57_GATE_COUNT",
    "NONLINEAR291_R57_GATE_COUNT",
    "Weight2Circuit",
    "build_nonlinear193_chain",
    "build_nonlinear291_chain",
    "maj",
    "nonlinear193",
    "nonlinear291",
    "run_nonlinear193",
    "run_nonlinear291",
]
