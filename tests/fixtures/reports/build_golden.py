"""Build the golden report: one committed JSONL file exercising every event the library defines.

The report format is a contract between the run that writes it and tools that read it later --
possibly much later, from an older checkout's output. ``golden-v1.jsonl`` is that contract pinned
to disk: a small, plausible ``converge -> preprocess -> infer -> refine`` run over two pooled
datasets with a validation split, carrying at least one instance of *every* class in
:data:`diffBloch.observability.EVENT_TYPES`. ``tests/unit/test_golden_report.py`` reads it back
with the current code and renders every table and figure from it.

Regenerate after a deliberate schema change::

    uv run python tests/fixtures/reports/build_golden.py

and commit the diff -- that diff *is* the review of the schema change. A test also checks the
committed file matches what this script produces, so the fixture cannot drift from the code
unnoticed, and a new event class fails the coverage test until it is added to :func:`build_events`.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

from diffBloch.observability import (
    EVENT_TYPES,
    ConvergencePassStarted,
    ConvergenceSweepStarted,
    ConvergenceTrial,
    CouplingSummary,
    DeviceSelected,
    Event,
    EventRecord,
    ExperimentDeclared,
    InferenceCompleted,
    ObjectiveManifest,
    ObjectiveTerm,
    OrientationOptimizationStarted,
    OrientationOptimizationSummary,
    OrientationOptimized,
    OrientationSearchTrace,
    PlanSeeded,
    PlanStepCompleted,
    PreprocessCompleted,
    RefinedRotationMetrics,
    RefinementCompleted,
    RefinementOrientationStep,
    RefinementOutputsWritten,
    RefinementStarted,
    RefinementStep,
    RotationCoupling,
    RotationCouplingSegments,
    RotationReflections,
    RotationScored,
    RunStageStarted,
    RunStageStopped,
    ThicknessOptimizationStarted,
    ThicknessOptimized,
    ThicknessProfile,
    event_record_from_event,
)

GOLDEN = Path(__file__).with_name("golden-v1.jsonl")
RUN_ID = "golden-v1"
EXPERIMENT = "/data/quartz"
DATASETS = ("a.cif_pets", "b.cif_pets")
_T0 = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)


def _stage(name: str, seconds: float, *events: Event) -> Iterator[Event]:
    yield RunStageStarted(stage=name, experiment_directory=EXPERIMENT)  # type: ignore[arg-type]
    yield from events
    yield RunStageStopped(
        stage=name,  # type: ignore[arg-type]
        status="completed",
        elapsed_seconds=seconds,
        experiment_directory=EXPERIMENT,
    )


def _preprocess_dataset(dataset: str) -> Iterator[Event]:
    """One dataset's recipe: two rotations (indices within the dataset), orientation then thickness."""
    rotations = (0, 1)
    yield PlanSeeded(measurements={"n_orientations": 2.0, "n_observed_hkl": 60.0})
    yield PlanStepCompleted(
        channel="build_orientation_plans",
        index=0,
        measurements={"n_orientations": 2.0, "n_union_beams": 48.0},
    )
    yield OrientationOptimizationStarted(total_rotations=2, dataset=dataset)
    for k, rotation in enumerate(rotations):
        yield OrientationOptimized(
            rotation_index=rotation,
            score=0.040 + 0.005 * k,
            seed_score=0.061 + 0.004 * k,
            alpha=0.010 * (k + 1),
            beta=-0.020,
            omega=0.005,
            residual="wr2",
            n_matched_hkl=40 - k,
            n_trials=14,
            n_passes=5,
            pass_cap=60,
            dataset=dataset,
        )
        yield OrientationSearchTrace(
            rotation_index=rotation,
            residual="wr2",
            alpha=(0.0, 0.05, 0.010 * (k + 1)),
            beta=(0.0, -0.05, -0.020),
            omega=(0.0, 0.05, 0.005),
            score=(0.061 + 0.004 * k, 0.09, 0.040 + 0.005 * k),
            comparable_score=(0.061 + 0.004 * k, 0.095, 0.040 + 0.005 * k),
            n_matched_hkl=(42, 38, 40 - k),
            is_seed=(1, 0, 0),
            is_final=(0, 0, 1),
            dataset=dataset,
        )
    yield OrientationOptimizationSummary(
        n_orientations=2,
        mean_score=0.0425,
        residual="wr2",
        unique_matched_hkl=70,
        unique_strong_hkl=52,
        unique_observed_hkl=110,
        total_trials=28,
        max_passes=5,
    )
    yield PlanStepCompleted(
        channel="optimize_orientation", index=1, measurements={"mean_wr2": 0.0425}
    )
    yield ThicknessOptimizationStarted(total_rotations=2, dataset=dataset)
    grid = (900.0, 950.0, 1000.0, 1050.0, 1100.0)
    for k, rotation in enumerate(rotations):
        scores = (0.12, 0.07, 0.04 + 0.01 * k, 0.06, 0.11)
        yield ThicknessOptimized(
            rotation_index=rotation,
            score=min(scores),
            residual="wr2",
            thickness=1000.0,
            candidate_thicknesses=grid,
            candidate_score=scores,
            dataset=dataset,
        )
    yield PlanStepCompleted(channel="optimize_thickness", index=2, measurements={"mean_wr2": 0.045})


def _coupling(dataset: str, offset: int) -> Iterator[Event]:
    """``index`` is the position in the pooled plan; ``rotation_index`` the index within the dataset."""
    for k, rotation in enumerate((0, 1)):
        yield RotationCoupling(
            index=offset + k,
            n_coupling_segments=2,
            n_tilts=8,
            max_tilts_per_segment=4,
            n_union_beams=48 + k,
            max_beams_per_segment=30,
            dataset=dataset,
            rotation_index=rotation,
        )
        yield RotationCouplingSegments(
            rotation_index=rotation,
            first_tilt_index=(0, 4),
            last_tilt_index=(3, 7),
            n_tilts=(4, 4),
            n_segment_beams=(30, 28 + k),
            n_union_beams=48 + k,
            n_total_tilts=8,
            dataset=dataset,
        )


def build_events() -> list[Event]:
    """The whole golden run, in emission order."""
    declared = ExperimentDeclared(
        name="golden-quartz",
        structure="quartz.cif",
        experimental_data="a.cif_pets, b.cif_pets",
        optimizer="adam",
        seed_thicknesses_by_dataset=(("a.cif_pets", (1000.0,)), ("b.cif_pets", (1000.0,))),
        integration_semiangles=(0.5, 0.5),
        rocking_curve_sampling=21,
        dsg=0.0015,
        rsg=0.9,
        solve_g_max=2.0,
        sg_max=0.01,
        absorption=False,
        steps=3,
        learning_rate=0.001,
        experiment_directory=EXPERIMENT,
    )
    converge = _stage(
        "converge",
        4.0,
        DeviceSelected(requested="cuda", selected="cpu", cuda_available=False),
        ConvergencePassStarted(
            pass_index=1,
            g_max=1.5,
            sg_max=0.02,
            tilt_steps=11,
            r_factor_threshold=0.01,
            n_orientations=1,
        ),
        ConvergenceSweepStarted(control="g_max", pass_index=1),
        ConvergenceTrial(
            control="g_max",
            trial_index=0,
            pass_index=1,
            previous=1.5,
            candidate=1.75,
            r_factor=0.030,
            n_compared_hkl=80,
        ),
        ConvergenceTrial(
            control="g_max",
            trial_index=1,
            pass_index=1,
            previous=1.75,
            candidate=2.0,
            r_factor=0.008,
            n_compared_hkl=96,
        ),
    )
    preprocess = _stage(
        "preprocess",
        30.0,
        DeviceSelected(requested="cuda", selected="cpu", cuda_available=False),
        declared,
        *_preprocess_dataset("a.cif_pets"),
        *_preprocess_dataset("b.cif_pets"),
        *_coupling("a.cif_pets", 0),
        *_coupling("b.cif_pets", 2),
        CouplingSummary(
            measurements={
                "n_orientations": 4.0,
                "n_observed_hkl": 110.0,
                "n_matched_hkl": 70.0,
                "max_union_beams": 49.0,
            }
        ),
        PreprocessCompleted(
            n_rotations=4,
            n_stages=3,
            total_hkl=110,
            matched_hkl=70,
            steps=(
                (
                    "build_orientation_plans",
                    {"rocking": {"__type__": "RockingCurve", "sampling": 21}},
                ),
                (
                    "optimize_orientation",
                    {
                        "search": {
                            "__type__": "NelderMeadSearch",
                            "step_size": 0.05,
                            "max_iterations": 60,
                        },
                        "absorption": {"__type__": "Absorption", "enabled": False},
                    },
                ),
                (
                    "optimize_thickness",
                    {
                        "grid": {
                            "__type__": "ThicknessGrid",
                            "min_thickness": 900.0,
                            "max_thickness": 1100.0,
                            "n_steps": 5,
                        }
                    },
                ),
            ),
        ),
    )
    infer = _stage(
        "infer",
        2.0,
        *(
            RotationScored(
                index=k,
                r_obs=0.050 + 0.002 * k,
                wr2=0.040 + 0.002 * k,
                n_matched=35,
                dataset=DATASETS[k // 2],
                rotation_index=k % 2,
            )
            for k in range(4)
        ),
        InferenceCompleted(n_rotations=4, n_evaluated=4, mean_r_obs=0.053, mean_wr2=0.043),
    )
    steps = [
        RefinementStep(
            iteration=i,
            loss=0.30 - 0.05 * i,
            wr2=0.045 - 0.003 * i,
            r_obs=0.055 - 0.003 * i,
            diff_loss=0.28 - 0.05 * i,
            objective_total=0.30 - 0.05 * i,
            components={
                "diffraction": {
                    "raw": 0.28 - 0.05 * i,
                    "weight": 1.0,
                    "contribution": 0.28 - 0.05 * i,
                },
                "bond_length": {"raw": 0.01, "weight": 2.0, "contribution": 0.02},
            },
            n_rotations=3,
            n_wr2_evaluated=3,
            n_r_obs_evaluated=3,
            val_wr2=0.050 - 0.002 * i,
            val_r_obs=0.060 - 0.002 * i,
            val_n_rotations=1,
            val_n_wr2_evaluated=1,
            val_n_r_obs_evaluated=1,
        )
        for i in range(3)
    ]
    refine = _stage(
        "refine",
        60.0,
        ObjectiveManifest(
            penalties=(ObjectiveTerm(name="bond_length", weight=2.0),),
            constraints=("hydrogen_riding",),
            components=("diffraction", "bond_length"),
        ),
        RefinementStarted(total_steps=3),
        steps[0],
        *(
            RefinementOrientationStep(
                iteration=0,
                rotation_index=k,
                wr2=0.046 + 0.002 * k,
                r_obs=0.056 + 0.002 * k,
                diff_loss=0.28,
                dataset="a.cif_pets",
            )
            for k in (0, 1)
        ),
        steps[1],
        *(
            RefinementOrientationStep(
                iteration=1,
                rotation_index=k,
                wr2=0.043 + 0.002 * k,
                r_obs=0.053 + 0.002 * k,
                diff_loss=0.23,
                dataset="a.cif_pets",
            )
            for k in (0, 1)
        ),
        steps[2],
        RefinementCompleted(
            n_steps=3,
            best_step=2,
            best_loss=0.046,
            selection="validation",
            reflection_counts={"matched": 70, "matched_i_gt_3sigma": 52},
        ),
        *(
            RefinedRotationMetrics(
                rotation_index=k % 2,
                wr2=0.039 + 0.002 * k,
                r_obs=0.049 + 0.002 * k,
                n_matched=35,
                is_validation=(k == 3),
                dataset=DATASETS[k // 2],
            )
            for k in range(4)
        ),
        *(
            RotationReflections(
                rotation_index=k % 2,
                dataset=DATASETS[k // 2],
                h=(1, 1, 2, 0, 3),
                k=(0, 1, 0, 2, 1),
                l=(0, 0, 1, 1, 1),
                i_obs=(120.0, 40.0, 9.0, 2.0 + k, -1.0),
                sigma=(4.0, 3.0, 2.0, 1.5, 1.0),
                i_calc=(118.0 + k, 43.0, 8.0, 3.0, 0.5),
                d_spacing=(4.91, 3.47, 2.46, 2.13, 1.54),
                scale=0.8 + 0.05 * k,
                thickness=1000.0,
            )
            for k in range(4)
        ),
        *(
            ThicknessProfile(
                form="linear",
                min_thickness=900.0,
                max_thickness=1100.0,
                rotation_indices=(0, 1),
                alphas=(-10.0, 10.0),
                thicknesses=(980.0, 1020.0 + 20.0 * slot),
                label=dataset,
            )
            for slot, dataset in enumerate(DATASETS)
        ),
        RefinementOutputsWritten(
            structure="refined_structure.cif",
            artifacts={
                "refined_structure": "refined_structure.cif",
                "refined_parameters": "reproducibility/refined_parameters.npz",
                "refinement_lock": "reproducibility/refinement.lock",
            },
            experiment_directory=EXPERIMENT,
        ),
    )
    return [*converge, *preprocess, *infer, *refine]


def build_records() -> list[EventRecord]:
    """The golden run as records, with fixed identity and timestamps so the file is reproducible."""
    return [
        event_record_from_event(
            event, run_id=RUN_ID, sequence=index, timestamp=_T0 + timedelta(seconds=index)
        )
        for index, event in enumerate(build_events())
    ]


def render(records: list[EventRecord]) -> str:
    return "".join(record.model_dump_json() + "\n" for record in records)


def main() -> int:
    records = build_records()
    missing = set(EVENT_TYPES) - {record.event_type for record in records}
    if missing:
        print(f"build_events() emits no instance of: {sorted(missing)}", file=sys.stderr)
        return 1
    GOLDEN.write_text(render(records))
    print(f"wrote {GOLDEN} ({len(records)} records, {len(EVENT_TYPES)} event types)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
