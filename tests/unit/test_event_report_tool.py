"""The ``tools/event_report`` consumer: the report reader and the figures over it.

The figures live in an importable module rather than in notebook cells precisely so they can be
asserted on here -- a plot function inside an ``.ipynb`` is code nothing in CI ever runs.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # headless: no display, and no figure windows left open by the suite

from tools.event_report import reader  # noqa: E402
from tools.event_report.figures import (  # noqa: E402
    build_figures,
    build_sections,
    export_figures,
    plot_convergence_sweeps,
    plot_coupling_segment_heatmap,
    plot_dataset_summary,
    plot_epoch_curve,
    plot_orientation_optimization,
    plot_orientation_search_trace,
    plot_refined_rotation_scores,
    plot_thickness_grids,
    plot_thickness_heatmap,
    plot_thickness_model,
)

from diffBloch.observability import (  # noqa: E402
    ConvergencePassStarted,
    ConvergenceTrial,
    EventRecord,
    ExperimentDeclared,
    OrientationOptimized,
    OrientationSearchTrace,
    PreprocessCompleted,
    RefinedRotationMetrics,
    RefinementCompleted,
    RefinementOutputsWritten,
    RefinementStep,
    RotationCoupling,
    RotationCouplingSegments,
    RotationScored,
    RunStageStarted,
    RunStageStopped,
    ThicknessOptimized,
    ThicknessProfile,
    event_record_from_event,
)


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    """pyplot retains every figure until closed; a module of figure tests otherwise leaks them."""
    yield
    matplotlib.pyplot.close("all")


def _records(*events: object) -> list[EventRecord]:
    return [
        event_record_from_event(
            event, run_id="run", sequence=index, timestamp=datetime(2026, 1, 1, tzinfo=UTC)
        )
        for index, event in enumerate(events)
    ]


def _refinement_step(iteration: int, *, validation: bool = False) -> RefinementStep:
    return RefinementStep(
        iteration=iteration,
        loss=1.0 - 0.1 * iteration,
        wr2=0.05 - 0.001 * iteration,
        r_obs=0.06,
        n_rotations=3,
        n_wr2_evaluated=3,
        n_r_obs_evaluated=2,
        val_wr2=0.07 if validation else None,
        val_r_obs=0.08 if validation else None,
        val_n_rotations=1 if validation else None,
        val_n_wr2_evaluated=1 if validation else None,
        val_n_r_obs_evaluated=1 if validation else None,
    )


def _orientation(rotation_index: int, dataset: str = "a.cif_pets") -> OrientationOptimized:
    return OrientationOptimized(
        rotation_index=rotation_index,
        score=0.04,
        residual="wr2",
        n_matched_hkl=40,
        n_trials=12,
        n_passes=4,
        pass_cap=2000,
        dataset=dataset,
        seed_score=0.06,
        alpha=0.01,
        beta=0.02,
        omega=0.03,
    )


def _refined(rotation_index: int, dataset: str, *, validation: bool) -> RefinedRotationMetrics:
    return RefinedRotationMetrics(
        rotation_index=rotation_index,
        wr2=0.05,
        r_obs=0.06,
        n_matched=40,
        is_validation=validation,
        dataset=dataset,
    )


def _thickness(rotation_index: int) -> ThicknessOptimized:
    return ThicknessOptimized(
        rotation_index=rotation_index,
        score=0.04,
        residual="wr2",
        thickness=1200.0,
        candidate_thicknesses=(1000.0, 1200.0),
        candidate_score=(0.05, 0.04),
        dataset="a.cif_pets",
    )


def _trace(rotation_index: int) -> OrientationSearchTrace:
    return OrientationSearchTrace(
        rotation_index=rotation_index,
        residual="wr2",
        alpha=(0.0, 0.1, 0.08),
        beta=(0.0, 0.0, 0.02),
        omega=(0.0, 0.0, 0.0),
        score=(0.06, 0.05, 0.04),
        comparable_score=(0.06, 0.05, 0.04),
        n_matched_hkl=(40, 41, 41),
        is_seed=(1, 0, 0),
        is_final=(0, 0, 1),
        dataset="a.cif_pets",
    )


def _segments(rotation_index: int) -> RotationCouplingSegments:
    return RotationCouplingSegments(
        rotation_index=rotation_index,
        first_tilt_index=(0, 3),
        last_tilt_index=(2, 5),
        n_tilts=(3, 3),
        n_segment_beams=(120, 90),
        n_union_beams=180,
        n_total_tilts=6,
        dataset="a.cif_pets",
    )


def _coupling(rotation_index: int) -> RotationCoupling:
    return RotationCoupling(
        index=rotation_index,
        n_coupling_segments=2,
        n_tilts=6,
        max_tilts_per_segment=3,
        n_union_beams=180,
        max_beams_per_segment=120,
        dataset="a.cif_pets",
        rotation_index=rotation_index,
    )


def _full_report() -> list[EventRecord]:
    return _records(
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
        RunStageStarted(stage="preprocess", experiment_directory="/tmp/quartz-no-abs"),
        PreprocessCompleted(n_rotations=4, total_hkl=100, matched_hkl=80),
        _orientation(0),
        _orientation(1),
        _trace(0),
        _thickness(0),
        _coupling(0),
        _segments(0),
        RunStageStopped(
            stage="preprocess",
            status="completed",
            elapsed_seconds=1.0,
            experiment_directory="/tmp/quartz-no-abs",
        ),
        RunStageStarted(stage="infer", experiment_directory="/tmp/quartz-no-abs"),
        RotationScored(index=0, r_obs=0.05, n_observed=20, n_beams=64),
        RunStageStopped(
            stage="infer",
            status="completed",
            elapsed_seconds=2.0,
            experiment_directory="/tmp/quartz-no-abs",
        ),
        RunStageStarted(stage="refine", experiment_directory="/tmp/quartz-no-abs"),
        _refinement_step(0, validation=True),
        _refinement_step(1, validation=True),
        _refined(0, "a.cif_pets", validation=False),
        _refined(3, "b.cif_pets", validation=True),
        ThicknessProfile(
            form="linear",
            min_thickness=1000.0,
            max_thickness=1300.0,
            rotation_indices=(0, 1),
            alphas=(-10.0, 10.0),
            thicknesses=(1000.0, 1300.0),
            label="a.cif_pets",
        ),
        RefinementCompleted(n_steps=2, best_step=0, best_loss=1.0),
        RefinementOutputsWritten(
            structure="/tmp/quartz-no-abs/refined_structure.cif",
            artifacts={"refined_structure": "/tmp/quartz-no-abs/refined_structure.cif"},
        ),
        RunStageStopped(
            stage="refine",
            status="completed",
            elapsed_seconds=3.0,
            experiment_directory="/tmp/quartz-no-abs",
        ),
    )


# --- reader ---------------------------------------------------------------------------------


def test_reader_round_trips_a_written_report(tmp_path: Path) -> None:
    path = tmp_path / "report.jsonl"
    written = _full_report()
    path.write_text("".join(record.model_dump_json() + "\n" for record in written))

    loaded = reader.read_records(path)

    assert [record.sequence for record in loaded] == [record.sequence for record in written]
    assert [record.event_type for record in loaded] == [r.event_type for r in written]


def test_reader_resolves_a_repository_relative_path(tmp_path: Path) -> None:
    """A notebook launched from its own directory must still resolve a repo-relative path."""
    root = reader.repository_root()
    relative = Path("pyproject.toml")

    assert reader.resolve_event_log_path(root / relative) == root / relative
    with pytest.raises(FileNotFoundError, match="Tried:"):
        reader.resolve_event_log_path("no/such/report.jsonl")


def test_reader_slices_by_type_and_dataset() -> None:
    records = _full_report()

    assert len(reader.records_of(records, "OrientationOptimized")) == 2
    assert reader.records_of(records, "NoSuchEvent") == []
    assert sorted(reader.by_dataset(reader.records_of(records, "RefinedRotationMetrics"))) == [
        "a.cif_pets",
        "b.cif_pets",
    ]
    ordered = reader.sorted_by_rotation(reader.records_of(records, "OrientationOptimized"))
    assert [record.rotation_index for record in ordered] == [0, 1]


def test_reader_finite_mean_ignores_none_and_non_finite() -> None:
    assert reader.finite_mean([1.0, 3.0, None, float("nan"), float("inf")]) == 2.0
    assert reader.finite_mean([]) is None
    assert reader.finite_mean([None, float("nan")]) is None


# --- figures --------------------------------------------------------------------------------


def test_build_figures_renders_every_figure_the_report_has_events_for() -> None:
    built = build_figures(_full_report())

    assert set(built) == {
        "epoch_curve",
        "orientation_optimization",
        "orientation_search_trace",
        "refined_rotation_scores",
        "per_dataset_summary",
        "thickness_grids",
        "thickness_heatmap",
        "thickness_model",
        "coupling_geometry",
        "coupling_segment_heatmap",
    }
    # A refine report declares no convergence sweep, so that figure is absent rather than empty.
    assert "convergence_sweeps" not in built
    assert built["epoch_curve"].axes[0].get_xlabel() == "epoch"
    # Two epochs, and validation reported -> four series, not two.
    assert len(built["epoch_curve"].axes[0].lines) == 4


def test_build_sections_heads_each_stage_separately() -> None:
    sections = build_sections(_full_report())

    assert [title for title, _ in sections] == [
        "Preprocess — orientation optimization",
        "Preprocess — per-rotation thickness fit",
        "Preprocess — coupled solve geometry",
        "Refinement — epoch history",
        "Refinement — per-rotation scores",
        "Refinement — datasets",
        "Refinement — learned thickness model",
    ]
    assert dict(sections)["Preprocess — per-rotation thickness fit"].keys() == {
        "thickness_grids",
        "thickness_heatmap",
    }


def test_build_sections_follows_the_order_the_run_actually_ran() -> None:
    """``preprocess.stage_order: thickness_first`` swaps the two fits, so the order is derived."""
    thickness_first = _records(_thickness(0), _orientation(0))
    orientation_first = _records(_orientation(0), _thickness(0))

    assert [title for title, _ in build_sections(thickness_first)] == [
        "Preprocess — per-rotation thickness fit",
        "Preprocess — orientation optimization",
    ]
    assert [title for title, _ in build_sections(orientation_first)] == [
        "Preprocess — orientation optimization",
        "Preprocess — per-rotation thickness fit",
    ]


def test_build_sections_drops_a_section_whose_figures_all_declined() -> None:
    """A single-dataset run has metrics but no dataset comparison, so that heading is absent."""
    sections = dict(build_sections(_records(_refined(0, "a.cif_pets", validation=False))))

    assert "Refinement — per-rotation scores" in sections
    assert "Refinement — datasets" not in sections


def test_every_figure_is_none_when_its_events_are_absent() -> None:
    """A preprocess-only report renders fewer figures rather than raising."""
    empty: list[EventRecord] = []
    for plot in (
        plot_convergence_sweeps,
        plot_epoch_curve,
        plot_orientation_optimization,
        plot_orientation_search_trace,
        plot_refined_rotation_scores,
        plot_dataset_summary,
        plot_thickness_grids,
        plot_thickness_heatmap,
        plot_thickness_model,
        plot_coupling_segment_heatmap,
    ):
        assert plot(empty) is None, plot.__name__
    assert build_figures(empty) == {}


def test_per_dataset_summary_needs_more_than_one_dataset() -> None:
    """One dataset has nothing to compare against, so the comparison figure is omitted."""
    single = _records(_refined(0, "a.cif_pets", validation=False))

    assert plot_dataset_summary(single) is None


def test_orientation_search_trace_plots_the_longest_search() -> None:
    short = OrientationSearchTrace(
        rotation_index=1,
        residual="wr2",
        alpha=(0.0,),
        beta=(0.0,),
        omega=(0.0,),
        score=(0.09,),
        comparable_score=(0.09,),
        n_matched_hkl=(40,),
        is_seed=(1,),
        is_final=(1,),
        dataset="a.cif_pets",
    )

    figure = plot_orientation_search_trace(_records(short, _trace(0)))

    assert figure is not None
    assert "a.cif_pets:0" in figure.axes[0].get_title()  # the 3-trial trace, not the 1-trial one
    assert figure.axes[1].get_xlabel() == "trial"


def test_refined_rotation_scores_marks_held_out_rotations() -> None:
    records = _records(
        _refined(0, "a.cif_pets", validation=False),
        _refined(1, "a.cif_pets", validation=True),
    )

    figure = plot_refined_rotation_scores(records)

    assert figure is not None
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert "a.cif_pets validation" in labels


def _converge_report() -> list[EventRecord]:
    """One pass sweeping two controls, each settling on its third candidate."""
    events: list[object] = [
        ConvergencePassStarted(
            pass_index=1,
            g_max=2.25,
            sg_max=0.01,
            tilt_steps=42,
            r_factor_threshold=0.01,
            n_orientations=1,
        )
    ]
    for control, values in (
        ("g_max", ((2.25, 2.45, 0.0312), (2.45, 2.65, 0.0180), (2.65, 2.85, 0.0071))),
        ("sg_max", ((0.01, 0.02, 0.0500), (0.02, 0.03, 0.0210), (0.03, 0.04, 0.0042))),
    ):
        for trial_index, (previous, candidate, r_factor) in enumerate(values):
            events.append(
                ConvergenceTrial(
                    control=control,
                    trial_index=trial_index,
                    pass_index=1,
                    previous=previous,
                    candidate=candidate,
                    r_factor=r_factor,
                    n_compared_hkl=612,
                )
            )
    return _records(*events)


def test_convergence_sweeps_panels_each_control_against_the_threshold() -> None:
    figure = plot_convergence_sweeps(_converge_report())

    assert figure is not None
    assert [ax.get_title() for ax in figure.axes] == ["g_max", "sg_max"]
    assert figure.axes[-1].get_xlabel() == "candidate value"
    # Log scale: the R-factors span orders of magnitude as a control converges.
    assert figure.axes[0].get_yscale() == "log"
    # One ladder line per pass, and the threshold drawn as a rule beneath it.
    assert len(figure.axes[0].lines) == 2
    assert figure.axes[0].lines[1].get_ydata()[0] == pytest.approx(0.01)


def test_convergence_sweeps_marks_the_first_candidate_under_the_threshold() -> None:
    """The crossing is the settled value -- the reason the ladder is plotted at all."""
    figure = plot_convergence_sweeps(_converge_report())

    assert figure is not None
    settled = [
        collection.get_offsets().tolist()
        for collection in figure.axes[0].collections
        if len(collection.get_offsets())
    ]
    assert settled == [[[2.85, 0.0071]]]  # the third g_max trial, the first under 0.01


def test_convergence_sweeps_stays_linear_when_an_r_factor_is_not_positive() -> None:
    """A log axis silently drops non-positive points; fall back rather than hide a trial."""
    records = _records(
        ConvergenceTrial(
            control="g_max",
            trial_index=0,
            pass_index=1,
            previous=2.25,
            candidate=2.45,
            r_factor=0.0,
            n_compared_hkl=612,
        )
    )

    figure = plot_convergence_sweeps(records)

    assert figure is not None
    assert figure.axes[0].get_yscale() == "linear"


def test_thickness_heatmap_traces_the_fitted_thickness_per_rotation() -> None:
    records = _records(_thickness(0), _thickness(1), _thickness(2))

    figure = plot_thickness_heatmap(records)

    assert figure is not None
    assert figure.axes[0].get_ylabel() == "rotation"
    # axes[0] is the single data panel; the colorbar is appended after it.
    assert figure.axes[0].get_xlabel() == "thickness"
    trace = figure.axes[0].lines[0]
    assert list(trace.get_xdata()) == [1200.0, 1200.0, 1200.0]
    assert list(trace.get_ydata()) == [0, 1, 2]
    assert figure.axes[0].images[0].get_array().shape == (3, 2)


def test_thickness_heatmap_clips_the_colour_scale_to_keep_the_basin_readable() -> None:
    """The grid's far ends score arbitrarily badly and would flatten the minimum into one tone."""
    wide = ThicknessOptimized(
        rotation_index=0,
        score=0.02,
        residual="wr2",
        thickness=1000.0,
        candidate_thicknesses=(900.0, 1000.0, 1100.0, 1200.0),
        candidate_score=(0.05, 0.02, 0.06, 9.0),
        dataset="a.cif_pets",
    )

    figure = plot_thickness_heatmap(_records(wide))

    assert figure is not None
    assert figure.axes[0].images[0].get_clim()[1] < 9.0


def test_thickness_heatmap_gives_each_dataset_its_own_panel() -> None:
    """Pooled datasets may be gridded over different ranges, so they cannot share an axis."""
    other = ThicknessOptimized(
        rotation_index=0,
        score=0.03,
        residual="wr2",
        thickness=800.0,
        candidate_thicknesses=(700.0, 800.0),
        candidate_score=(0.06, 0.03),
        dataset="b.cif_pets",
    )

    figure = plot_thickness_heatmap(_records(_thickness(0), other))

    assert figure is not None
    titles = [ax.get_title() for ax in figure.axes if ax.get_title()]
    assert titles == ["Thickness score grid: a.cif_pets", "Thickness score grid: b.cif_pets"]


def test_thickness_heatmap_skips_rotations_gridded_differently_from_their_dataset() -> None:
    """A mismatched grid is dropped rather than stretched onto the wrong thickness axis."""
    mismatched = ThicknessOptimized(
        rotation_index=1,
        score=0.03,
        residual="wr2",
        thickness=1100.0,
        candidate_thicknesses=(1000.0, 1100.0, 1200.0),
        candidate_score=(0.06, 0.03, 0.05),
        dataset="a.cif_pets",
    )

    figure = plot_thickness_heatmap(_records(_thickness(0), mismatched))

    assert figure is not None
    assert figure.axes[0].images[0].get_array().shape == (1, 2)  # only the matching rotation


def test_export_figures_writes_one_file_per_figure_and_format(tmp_path: Path) -> None:
    built = build_figures(_full_report())

    written = export_figures(built, tmp_path / "figures", formats=("svg", "png"))

    assert len(written) == 2 * len(built)
    assert all(path.exists() and path.stat().st_size > 0 for path in written)
