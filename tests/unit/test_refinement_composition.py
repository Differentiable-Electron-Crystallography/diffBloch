"""Python/API composition of a refinement problem: ``build_refinement_problem`` + riding.

Hydrogen handling is scientific composition expressed as typed Python values, not config: riding
freezes the hydrogens out of the trainable groups (the constraint derives them each step) and adds a
perceived :class:`HydrogenRiding`. Pinned against the abiraterone fixture (a molecular crystal with
hydrogens) and a hydrogen-free control (quartz).
"""

from __future__ import annotations

from pathlib import Path

from diffBloch.engine import (
    AtomSelection,
    HydrogenRiding,
    TrainableSpec,
    build_refinement_problem,
    with_hydrogen_riding,
)
from diffBloch.io import read_structure
from diffBloch.io.record import StructureRecord
from diffBloch.params import RefinableParams
from diffBloch.preprocess.experiment import RefinementSetup

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load(
    material: str, cif: str, *, load_hydrogens: bool
) -> tuple[StructureRecord, RefinableParams]:
    structure = read_structure(FIXTURES / material / cif, load_hydrogens=load_hydrogens)
    return structure, RefinementSetup.from_structure(structure).params


# --- build_refinement_problem: the pure factory passes its fields through unchanged ---


def test_build_refinement_problem_passes_fields_through() -> None:
    structure, initial = _load("abiraterone_anchor", "abiraterone.cif", load_hydrogens=True)
    trainable = TrainableSpec(positions=AtomSelection.all(), adp=AtomSelection.none())
    problem = build_refinement_problem(initial=initial, trainable=trainable)
    assert problem.initial is initial
    assert problem.trainable == trainable
    assert problem.constraints == () and problem.penalties == ()


# --- with_hydrogen_riding: freeze H + perceive the riding constraint ---


def test_riding_freezes_hydrogens_and_adds_the_constraint() -> None:
    structure, _ = _load("abiraterone_anchor", "abiraterone.cif", load_hydrogens=True)
    base = TrainableSpec(positions=AtomSelection.all(), adp=AtomSelection.all())
    trainable, constraints = with_hydrogen_riding(structure, base)
    assert trainable.positions.element_exclude == ("H",)  # H frozen out of positions...
    assert trainable.adp.element_exclude == ("H",)  # ...and ADPs (the constraint owns both)
    assert len(constraints) == 1 and isinstance(constraints[0], HydrogenRiding)


def test_riding_on_a_hydrogen_free_structure_adds_no_constraint() -> None:
    structure, _ = _load("quartz_anchor", "enantiomer_1.cif", load_hydrogens=False)
    base = TrainableSpec(positions=AtomSelection.all(), adp=AtomSelection.all())
    trainable, constraints = with_hydrogen_riding(structure, base)
    assert constraints == ()  # no hydrogens -> nothing to ride
    assert trainable.positions.selects_any and trainable.adp.selects_any  # rest still trains


def test_riding_leaves_an_untrained_group_untouched() -> None:
    structure, _ = _load("abiraterone_anchor", "abiraterone.cif", load_hydrogens=True)
    base = TrainableSpec(positions=AtomSelection.none(), adp=AtomSelection.all())
    trainable, _ = with_hydrogen_riding(structure, base)
    assert trainable.positions.mode == "none" and not trainable.positions.has_element_filter
    assert trainable.adp.element_exclude == ("H",)


def test_riding_preserves_an_api_callers_element_filter() -> None:
    # An API caller can pass a richer selection than config's all/none; freezing H must ADD H to the
    # exclusion set, not replace the filter (which would broaden "C only" to "all non-H atoms").
    structure, _ = _load("abiraterone_anchor", "abiraterone.cif", load_hydrogens=True)
    base = TrainableSpec(positions=AtomSelection.include_elements("C"), adp=AtomSelection.all())
    trainable, _ = with_hydrogen_riding(structure, base)
    assert trainable.positions.element_include == ("C",)  # C-only intent preserved
    assert trainable.positions.element_exclude == ("H",)  # H additionally excluded
