"""The checkpoint/resume driver in ``run_experiment`` (``_prepare``): reuse / resume / regen.

Exercises the decision logic with cheap spy steps (no physics) on a copied experiment dir, asserting
*which* steps actually run and that each dataset's ``plan.<stem>.npz`` + ``plan.<stem>.lock`` are
written/regenerated -- and that one dataset's checkpoint never restales another's. The end-to-end
physics path is covered by the anchor e2e; here we pin the driver.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from tests.unit.synthetic import built_seed_system

from diffBloch.app.program import _prepare
from diffBloch.config import input_lock_for, load_experiment
from diffBloch.preprocess import as_step, fork
from diffBloch.preprocess.plan import Plan

LOCKED = Path(__file__).parent.parent / "fixtures" / "locked_min"
EXP_REF = "exp_data.cif_pets"  # locked_min's single inputs.exp_data ref -> stem "exp_data"


def _experiment(tmp_path: Path) -> Path:
    exp = tmp_path / "experiment"
    exp.mkdir()
    (exp / "reproducibility").mkdir()
    for name in ("experiment.yaml", "enantiomer_1.cif", "exp_data.cif_pets"):
        shutil.copy(LOCKED / name, exp / name)
    shutil.copy(
        LOCKED / "reproducibility" / "experiment.lock", exp / "reproducibility" / "experiment.lock"
    )
    return exp


def _spy(calls: dict[str, int], name: str, params: dict | None = None):
    def run(plan: Plan) -> Plan:
        calls[name] = calls.get(name, 0) + 1
        return plan

    return as_step(name, params, run)


def _run(
    exp,
    base,
    calls,
    names_params,
    *,
    checkpoint=True,
    refresh=False,
    dataset_ref=EXP_REF,
    ignored_rotations=(),
) -> Plan:
    """Drive ``_prepare`` for one dataset the way ``_preprocess``'s per-dataset loop does."""
    steps = [_spy(calls, n, p) for n, p in names_params]
    cfg, _lock = load_experiment(exp)
    plan, _lock_sha = _prepare(
        base,
        steps,
        root=exp,
        cfg=cfg,
        dataset_ref=dataset_ref,
        ignored_rotations=ignored_rotations,
        structure_lock=input_lock_for(exp / cfg.inputs.structure, ref=cfg.inputs.structure),
        dataset_lock=input_lock_for(exp / dataset_ref, ref=dataset_ref),
        checkpoint=checkpoint,
        refresh=refresh,
    )
    return plan


def test_first_run_computes_and_writes_the_checkpoint(tmp_path: Path) -> None:
    exp = _experiment(tmp_path)
    _, base = built_seed_system()
    calls: dict[str, int] = {}
    _run(exp, base, calls, [("a", None), ("b", None)])
    assert calls == {"a": 1, "b": 1}  # both steps ran
    assert (exp / "reproducibility" / "plan.exp_data.npz").exists()
    assert (exp / "reproducibility" / "plan.exp_data.lock").exists()


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
    assert (exp / "reproducibility" / "plan.exp_data.lock").exists()  # ...still regenerated


def test_no_checkpoint_neither_reads_nor_writes(tmp_path: Path) -> None:
    exp = _experiment(tmp_path)
    _, base = built_seed_system()
    calls: dict[str, int] = {}
    _run(exp, base, calls, [("a", None), ("b", None)], checkpoint=False)
    assert calls == {"a": 1, "b": 1}
    assert not (exp / "reproducibility" / "plan.exp_data.npz").exists()
    assert not (exp / "reproducibility" / "plan.exp_data.lock").exists()


def test_fork_resolves_against_the_grid_and_stays_checkpointable(tmp_path: Path) -> None:
    # _prepare compiles a fork away against base.structure_factor_grid before locking, so a forked recipe checkpoints
    # through the ordinary flat-recipe lock: same branch -> reuse; a flipped branch -> stale.
    exp = _experiment(tmp_path)
    _, base = built_seed_system()
    calls: dict[str, int] = {}
    cfg, _lock = load_experiment(exp)

    def prepare(pred) -> Plan:
        # a -> fork(coarse | exact); the fork's chosen branch is what gets recorded
        steps = [
            _spy(calls, "a"),
            fork(pred, when_true=[_spy(calls, "coarse")], when_false=[_spy(calls, "exact")]),
        ]
        plan, _lock_sha = _prepare(
            base,
            steps,
            root=exp,
            cfg=cfg,
            dataset_ref=EXP_REF,
            ignored_rotations=(),
            structure_lock=input_lock_for(exp / cfg.inputs.structure, ref=cfg.inputs.structure),
            dataset_lock=input_lock_for(exp / EXP_REF, ref=EXP_REF),
            checkpoint=True,
            refresh=False,
        )
        return plan

    # Seed with the when_true branch; the recorded recipe is the resolved [a, coarse].
    out = prepare(lambda g: True)
    assert calls == {"a": 1, "coarse": 1}
    assert [r.name for r in out.provenance] == ["a", "coarse"]

    # Same fork resolves to the same branch -> full reuse, nothing re-runs.
    calls.clear()
    reused = prepare(lambda g: True)
    assert calls == {}
    assert [r.name for r in reused.provenance] == ["a", "coarse"]

    # A flipped predicate resolves to the other branch -> different recipe -> stale -> recompute.
    calls.clear()
    reforked = prepare(lambda g: False)
    assert calls == {"a": 1, "exact": 1}
    assert [r.name for r in reforked.provenance] == ["a", "exact"]


def test_per_dataset_checkpoints_restale_independently(tmp_path: Path) -> None:
    """Tampering one pooled dataset's bytes recomputes only that dataset's checkpoint."""
    exp = _experiment(tmp_path)
    shutil.copy(exp / EXP_REF, exp / "b.cif_pets")
    _, base = built_seed_system()
    recipe = [("a", None), ("b", None)]
    _run(exp, base, {}, recipe, dataset_ref=EXP_REF)
    _run(exp, base, {}, recipe, dataset_ref="b.cif_pets")
    assert (exp / "reproducibility" / "plan.exp_data.npz").exists()
    assert (exp / "reproducibility" / "plan.b.npz").exists()

    (exp / "b.cif_pets").write_bytes(b"tampered dataset bytes")
    first: dict[str, int] = {}
    _run(exp, base, first, recipe, dataset_ref=EXP_REF)
    assert first == {}  # untouched dataset: full reuse
    second: dict[str, int] = {}
    _run(exp, base, second, recipe, dataset_ref="b.cif_pets")
    assert second == {"a": 1, "b": 1}  # tampered dataset: recompute


def test_changed_file_local_ignore_restales_only_that_dataset(tmp_path: Path) -> None:
    exp = _experiment(tmp_path)
    shutil.copy(exp / EXP_REF, exp / "b.cif_pets")
    _, base = built_seed_system()
    recipe = [("a", None), ("b", None)]
    _run(exp, base, {}, recipe, dataset_ref=EXP_REF)
    _run(exp, base, {}, recipe, dataset_ref="b.cif_pets")

    # A pooled ignore edit that lands only on dataset b translates to a changed file-local set
    # for b and an unchanged (empty) one for exp_data.
    first: dict[str, int] = {}
    _run(exp, base, first, recipe, dataset_ref=EXP_REF, ignored_rotations=())
    assert first == {}
    second: dict[str, int] = {}
    _run(exp, base, second, recipe, dataset_ref="b.cif_pets", ignored_rotations=(1,))
    assert second == {"a": 1, "b": 1}


def test_unparsable_lock_recomputes_instead_of_crashing(tmp_path: Path) -> None:
    exp = _experiment(tmp_path)
    _, base = built_seed_system()
    _run(exp, base, {}, [("a", None)])
    (exp / "reproducibility" / "plan.exp_data.lock").write_text('{"not": "a preprocess lock"}\n')
    calls: dict[str, int] = {}
    _run(exp, base, calls, [("a", None)])
    assert calls == {"a": 1}  # treated as stale -> recomputed
    # ...and the regenerated lock is healthy again: an identical re-run fully reuses
    again: dict[str, int] = {}
    _run(exp, base, again, [("a", None)])
    assert again == {}
