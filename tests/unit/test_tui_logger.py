"""``TuiLogger``: the live terminal dashboard backend (the ``diffBloch[tui]`` extra).

Rendering is asserted by printing the composed renderable to an off-screen ``Console`` and reading
the text back, so these cover layout content without needing a real terminal.
"""

from __future__ import annotations

import pytest

from diffBloch.observability import (
    ConvergenceTrial,
    ExperimentDeclared,
    ObjectiveManifest,
    ObjectiveTerm,
    PlanSeeded,
    PlanStepCompleted,
    RefinedRotationMetrics,
    RefinementCompleted,
    RefinementOutputsWritten,
    RefinementStarted,
    RefinementStep,
    RotationScored,
)

pytest.importorskip("rich", reason="optional diffBloch[tui] extra")

from diffBloch.app.loggers.tui import TuiLogger  # noqa: E402


def _rendered(logger: TuiLogger, width: int = 110) -> str:
    """The composed dashboard as plain text (no tty required)."""
    from rich.console import Console

    console = Console(file=None, width=width, record=True, force_terminal=False)
    console.begin_capture()
    console.print(logger._render())
    return console.end_capture()


def _rendered_window(logger: TuiLogger, window: int | None, width: int = 110) -> str:
    """The dashboard at an explicit window size, bypassing terminal-height detection."""
    from rich.console import Console

    console = Console(width=width, force_terminal=False)
    console.begin_capture()
    console.print(logger._render(window=window))
    return console.end_capture()


def _experiment() -> ExperimentDeclared:
    return ExperimentDeclared(
        name="quartz",
        structure="quartz.cif",
        experimental_data="quartz.cif_pets",
        optimizer="lbfgs",
        seed_thicknesses=(820.0,),
        integration_semiangle=1.0,
        rocking_curve_sampling=42,
        dsg=0.0015,
        rsg=0.9,
        solve_g_max=2.25,
        sg_max=0.01,
        absorption=False,
        steps=2,
        learning_rate=0.001,
    )


def _epoch(iteration: int, wr2: float) -> RefinementStep:
    return RefinementStep(
        iteration=iteration,
        loss=1.0,
        wr2=wr2,
        r_obs=0.1,
        diff_loss=1.0,
        n_rotations=90,
        n_wr2_evaluated=90,
        n_r_obs_evaluated=90,
        components={
            "diffraction": {"raw": 1.0, "weight": 1.0, "contribution": 1.0},
            "bond_length": {"raw": 0.05, "weight": 3.0, "contribution": 0.15},
        },
    )


def test_off_a_terminal_the_display_never_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    """An in-place dashboard has no meaning in a log file, so it degrades to nothing."""
    logger = TuiLogger()
    logger.report(_experiment())  # pytest captures stdout -> not a terminal
    assert logger._live is None
    logger.close()  # idempotent, safe with nothing started
    assert logger._live is None


def test_unmodelled_events_surface_generically_instead_of_vanishing() -> None:
    """No event is dropped by omission -- an unstyled row beats an unseen observation.

    The convergence sweep is the case that matters: for a `run converge` the terminal is often the
    only sink attached, so an event the dashboard has no dedicated view for still has to appear.
    """
    logger = TuiLogger()
    assert logger._absorb(RotationScored(index=0, r_obs=0.1, n_observed=5, n_beams=9)) is True
    assert (
        logger._absorb(
            ConvergenceTrial(
                control="g_max",
                trial_index=0,
                pass_index=0,
                previous=2.25,
                candidate=2.45,
                r_factor=0.031,
                n_compared_hkl=612,
            )
        )
        is True
    )

    text = _rendered(logger)
    assert "other events" in text
    assert "convergence g_max" in text
    assert "r_factor=0.031" in text and "n_compared_hkl=612" in text


def test_tables_say_when_they_are_showing_a_window() -> None:
    """A silently truncated table misreads as a complete one."""
    logger = TuiLogger()
    logger._absorb(RefinementStarted(total_steps=12))
    for iteration in range(12):
        logger._absorb(_epoch(iteration, 0.2))

    assert "refinement  (last 4 of 12)" in _rendered_window(logger, 4)
    # On close the display releases the terminal, so the settled tables are rendered in full.
    assert "refinement  (12)" in _rendered_window(logger, None)


def test_dashboard_renders_the_declaration_and_objective() -> None:
    logger = TuiLogger()
    logger._absorb(_experiment())
    logger._absorb(
        ObjectiveManifest(
            penalties=(ObjectiveTerm(name="bond_length", weight=3.0),),
            components=("apparent_thickness",),
        )
    )

    text = _rendered(logger)
    assert "quartz.cif + quartz.cif_pets" in text
    assert "lbfgs lr=0.001" in text
    assert "bond_length (w=3)" in text
    # "none" is rendered, never omitted: an empty category is a fact worth stating.
    assert "constraints none" in text


def test_stage_table_shows_a_missing_count_as_absent_not_zero() -> None:
    """The seed has no alignment, so its matched count is "-"; a 0 would mean "matched nothing"."""
    logger = TuiLogger()
    logger._absorb(
        PlanSeeded(
            measurements={
                "n_orientations": 99.0,
                "n_solve_beams_total": 538263.0,
                "n_solve_beams_max": 5437.0,
                "n_observed_hkl": 6666.0,
            }
        )
    )
    logger._absorb(
        PlanStepCompleted(
            channel="build_orientation_plans",
            index=0,
            measurements={
                "n_orientations": 99.0,
                "n_solve_beams_total": 9988.0,
                "n_solve_beams_max": 111.0,
                "n_observed_hkl": 6666.0,
                "n_matched_hkl": 2567.0,
            },
        )
    )

    lines = _rendered(logger).splitlines()
    (seed,) = [line for line in lines if "seed" in line]
    (built,) = [line for line in lines if "build orientation" in line]
    assert seed.rstrip().rstrip("│").rstrip().endswith("-")  # absent, not zero
    assert "2567" in built
    assert "538263" in seed and "9988" in built  # the survival counts read as a prune


def test_epoch_table_marks_the_best_epoch_and_carries_denominators() -> None:
    logger = TuiLogger()
    logger._absorb(RefinementStarted(total_steps=3))
    for iteration, wr2 in enumerate((0.9, 0.1, 0.5)):
        logger._absorb(_epoch(iteration, wr2))
    logger._absorb(RefinementCompleted(n_steps=3, best_step=1, best_loss=1.0))

    lines = _rendered(logger).splitlines()
    best = next(line for line in lines if "2 *" in line)
    assert "0.100000" in best  # epoch 2 (1-based) is the starred, selected one
    assert "[90/90]" in best  # every mean states its denominator
    assert "bond_length" in "\n".join(lines)  # composed penalties get their own column


def test_rotation_table_labels_held_out_rotations() -> None:
    logger = TuiLogger()
    for index, is_validation in ((0, False), (1, True)):
        logger._absorb(
            RefinedRotationMetrics(
                rotation_index=index,
                wr2=0.2,
                r_obs=0.15,
                n_matched=5,
                is_validation=is_validation,
            )
        )

    text = _rendered(logger)
    assert "validation" in text
    assert "mean wR2 0.200000" in text and "[2/2]" in text


def test_terminal_event_closes_the_display() -> None:
    logger = TuiLogger()
    logger.report(_experiment())
    logger.report(RefinementOutputsWritten(structure="/tmp/refined_structure.cif"))
    assert logger._live is None
