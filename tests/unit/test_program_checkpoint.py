"""The checkpoint/resume driver in ``run_experiment`` (``_prepare``): reuse / resume / regen.

Exercises the decision logic with cheap spy steps (no physics) on a copied experiment dir, asserting
*which* steps actually run and that ``plan.npz`` + ``plan.lock`` are written/regenerated. The
end-to-end physics path is covered by the anchor e2e; here we pin the driver.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from tests.unit.synthetic import built_seed_system

from diffBloch.app.program import _prepare
from diffBloch.config import load_experiment
from diffBloch.preprocess import as_step, fork
from diffBloch.preprocess.plan import Plan

LOCKED = Path(__file__).parent.parent / "fixtures" / "locked_min"


def _experiment(tmp_path: Path) -> Path:
    exp = tmp_path / "experiment"
    exp.mkdir()
    for name in ("experiment.yaml", "experiment.lock", "enantiomer_1.cif", "exp_data.cif_pets"):
        shutil.copy(LOCKED / name, exp / name)
    return exp


def _spy(calls: dict[str, int], name: str, params: dict | None = None):
    def run(plan: Plan) -> Plan:
        calls[name] = calls.get(name, 0) + 1
        return plan

    return as_step(name, params, run)


def _run(exp, base, calls, names_params, *, checkpoint=True, refresh=False) -> Plan:
    steps = [_spy(calls, n, p) for n, p in names_params]
    cfg, _lock = load_experiment(exp)
    return _prepare(base, steps, root=exp, cfg=cfg, checkpoint=checkpoint, refresh=refresh)


def test_first_run_computes_and_writes_the_checkpoint(tmp_path: Path) -> None:
    exp = _experiment(tmp_path)
    _, base = built_seed_system()
    calls: dict[str, int] = {}
    _run(exp, base, calls, [("a", None), ("b", None)])
    assert calls == {"a": 1, "b": 1}  # both steps ran
    assert (exp / "plan.npz").exists() and (exp / "plan.lock").exists()


def test_identical_recipe_reuses_without_running_steps(tmp_path: Path) -> None:
    exp = _experiment(tmp_path)
    _, base = built_seed_system()
    _run(exp, base, {}, [("a", None), ("b", None)])  # seed the checkpoint
    calls: dict[str, int] = {}
    out = _run(exp, base, calls, [("a", None), ("b", None)])
    assert calls == {}  # nothing re-ran -- full reuse
    assert [r.name for r in out.provenance] == ["a", "b"]


def test_appended_step_resumes_running_only_the_suffix(tmp_path: Path) -> None:
    exp = _experiment(tmp_path)
    _, base = built_seed_system()
    _run(exp, base, {}, [("a", None), ("b", None)])
    calls: dict[str, int] = {}
    out = _run(exp, base, calls, [("a", None), ("b", None), ("c", None)])
    assert calls == {"c": 1}  # only the appended tail ran
    assert [r.name for r in out.provenance] == ["a", "b", "c"]  # snapshot prefix + suffix
    # the checkpoint ratcheted forward: an identical re-run is now a full reuse
    again: dict[str, int] = {}
    _run(exp, base, again, [("a", None), ("b", None), ("c", None)])
    assert again == {}


def test_changed_middle_step_recomputes(tmp_path: Path) -> None:
    exp = _experiment(tmp_path)
    _, base = built_seed_system()
    _run(exp, base, {}, [("a", {"x": 1}), ("b", None)])
    calls: dict[str, int] = {}
    _run(exp, base, calls, [("a", {"x": 2}), ("b", None)])  # a's params changed
    assert calls == {"a": 1, "b": 1}  # not a prefix -> full recompute


def test_refresh_recomputes_even_when_fresh(tmp_path: Path) -> None:
    exp = _experiment(tmp_path)
    _, base = built_seed_system()
    _run(exp, base, {}, [("a", None), ("b", None)])
    calls: dict[str, int] = {}
    _run(exp, base, calls, [("a", None), ("b", None)], refresh=True)
    assert calls == {"a": 1, "b": 1}  # forced recompute despite a valid checkpoint
    assert (exp / "plan.lock").exists()  # ...still regenerated


def test_no_checkpoint_neither_reads_nor_writes(tmp_path: Path) -> None:
    exp = _experiment(tmp_path)
    _, base = built_seed_system()
    calls: dict[str, int] = {}
    _run(exp, base, calls, [("a", None), ("b", None)], checkpoint=False)
    assert calls == {"a": 1, "b": 1}
    assert not (exp / "plan.npz").exists() and not (exp / "plan.lock").exists()


def test_fork_resolves_against_the_grid_and_stays_checkpointable(tmp_path: Path) -> None:
    # _prepare compiles a fork away against base.grid before locking, so a forked recipe checkpoints
    # through the ordinary flat-recipe lock: same branch -> reuse; a flipped branch -> stale.
    exp = _experiment(tmp_path)
    _, base = built_seed_system()
    calls: dict[str, int] = {}
    cfg, _lock = load_experiment(exp)

    def recipe(pred):  # a -> fork(coarse | exact); the fork's chosen branch is what gets recorded
        return [
            _spy(calls, "a"),
            fork(pred, when_true=[_spy(calls, "coarse")], when_false=[_spy(calls, "exact")]),
        ]

    # Seed with the when_true branch; the recorded recipe is the resolved [a, coarse].
    out = _prepare(base, recipe(lambda g: True), root=exp, cfg=cfg, checkpoint=True, refresh=False)
    assert calls == {"a": 1, "coarse": 1}
    assert [r.name for r in out.provenance] == ["a", "coarse"]

    # Same fork resolves to the same branch -> full reuse, nothing re-runs.
    calls.clear()
    reused = _prepare(
        base, recipe(lambda g: True), root=exp, cfg=cfg, checkpoint=True, refresh=False
    )
    assert calls == {}
    assert [r.name for r in reused.provenance] == ["a", "coarse"]

    # A flipped predicate resolves to the other branch -> different recipe -> stale -> recompute.
    calls.clear()
    reforked = _prepare(
        base, recipe(lambda g: False), root=exp, cfg=cfg, checkpoint=True, refresh=False
    )
    assert calls == {"a": 1, "exact": 1}
    assert [r.name for r in reforked.provenance] == ["a", "exact"]
