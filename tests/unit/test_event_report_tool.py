from __future__ import annotations

from datetime import UTC, datetime

from tools.event_report.event_report import render_html

from diffBloch.observability import (
    ExperimentDeclared,
    OrientationOptimized,
    PreprocessCompleted,
    RefinedRotationMetrics,
    RefinementCompleted,
    RefinementOutputsWritten,
    RefinementStep,
    RotationScored,
    RunStageStarted,
    RunStageStopped,
    ThicknessOptimized,
    event_record_from_event,
)


def _record(event: object, sequence: int):
    return event_record_from_event(
        event, run_id="run", sequence=sequence, timestamp=datetime(2026, 1, 1, tzinfo=UTC)
    )


def test_event_report_renders_named_sections_from_jsonl_events() -> None:
    records = [
        _record(
            ExperimentDeclared(
                name="quartz",
                structure="structure.cif",
                experimental_data="exp_data.cif_pets",
                optimizer="adam",
                seed_thicknesses_by_dataset=(("exp_data.cif_pets", (1000.0,)),),
                integration_semiangles=(0.01,),
                rocking_curve_sampling=21,
                dsg=0.1,
                rsg=0.1,
                solve_g_max=0.7,
                sg_max=1.4,
                absorption=False,
                steps=2,
                learning_rate=0.001,
                experiment_directory="/tmp/quartz-no-abs",
            ),
            0,
        ),
        _record(RunStageStarted(stage="preprocess", experiment_directory="/tmp/quartz-no-abs"), 1),
        _record(PreprocessCompleted(n_rotations=4, total_hkl=100, matched_hkl=80), 2),
        _record(
            OrientationOptimized(
                rotation_index=0,
                score=0.04,
                residual="wr2",
                n_matched_hkl=40,
                n_trials=12,
                n_passes=4,
                pass_cap=2000,
                dataset="a.cif_pets",
                seed_score=0.06,
                alpha=0.01,
                beta=0.02,
                omega=0.03,
            ),
            3,
        ),
        _record(
            ThicknessOptimized(
                rotation_index=0,
                score=0.04,
                residual="wr2",
                thickness=1200.0,
                candidate_thicknesses=(1000.0, 1200.0),
                candidate_score=(0.05, 0.04),
                dataset="a.cif_pets",
            ),
            4,
        ),
        _record(
            RunStageStopped(
                stage="preprocess",
                status="completed",
                elapsed_seconds=1.0,
                experiment_directory="/tmp/quartz-no-abs",
            ),
            5,
        ),
        _record(RunStageStarted(stage="infer", experiment_directory="/tmp/quartz-no-abs"), 6),
        _record(RotationScored(index=0, r_obs=0.05, n_observed=20, n_beams=64), 7),
        _record(
            RunStageStopped(
                stage="infer",
                status="completed",
                elapsed_seconds=2.0,
                experiment_directory="/tmp/quartz-no-abs",
            ),
            8,
        ),
        _record(RunStageStarted(stage="refine", experiment_directory="/tmp/quartz-no-abs"), 9),
        _record(
            RefinementStep(
                iteration=0,
                loss=1.0,
                wr2=0.05,
                r_obs=0.06,
                n_rotations=3,
                n_wr2_evaluated=3,
                n_r_obs_evaluated=2,
                val_wr2=0.07,
                val_r_obs=0.08,
                val_n_rotations=1,
                val_n_wr2_evaluated=1,
                val_n_r_obs_evaluated=1,
            ),
            10,
        ),
        _record(
            RefinedRotationMetrics(
                rotation_index=0,
                wr2=0.05,
                r_obs=0.06,
                n_matched=40,
                is_validation=False,
                dataset="a.cif_pets",
            ),
            11,
        ),
        _record(
            RefinedRotationMetrics(
                rotation_index=3,
                wr2=0.07,
                r_obs=0.08,
                n_matched=40,
                is_validation=True,
                dataset="b.cif_pets",
            ),
            12,
        ),
        _record(RefinementCompleted(n_steps=2, best_step=0, best_loss=1.0), 13),
        _record(
            RefinementOutputsWritten(
                structure="/tmp/quartz-no-abs/refined_structure.cif",
                artifacts={"refined_structure": "/tmp/quartz-no-abs/refined_structure.cif"},
            ),
            14,
        ),
        _record(
            RunStageStopped(
                stage="refine",
                status="completed",
                elapsed_seconds=3.0,
                experiment_directory="/tmp/quartz-no-abs",
            ),
            15,
        ),
    ]

    html = render_html(records)

    assert "Preprocess Summary" in html
    assert "Run Stages" in html
    assert "Preprocess" in html
    assert "Inference" in html
    assert "Refinement" in html
    assert "Experiment" in html
    assert "/tmp/quartz-no-abs" in html
    assert "structure.cif" in html
    assert "Epoch Curve" in html
    assert "Orientation Optimization" in html
    assert "Delta alpha deg" in html
    assert "Validation wR2" in html
    assert "Per-Dataset Summary" in html
    assert "a.cif_pets" in html
    assert "b.cif_pets" in html
    assert "Thickness Grids" in html
    assert "refined_structure" in html
