"""The human-readable run summary (``refinement_report.txt``) as an ordinary logger backend.

Named for what it produces, not for how it is fed: ``report`` is already the :class:`Logger`
protocol's own method, so a ``ReportLogger.report(...)`` would read as a tautology.

The summary is built from the event stream and nothing else: the run's
knobs arrive as :class:`~diffBloch.observability.ExperimentDeclared`, its objective as
:class:`~diffBloch.observability.ObjectiveManifest`, the epoch curve as
:class:`~diffBloch.observability.RefinementStep`, the settled per-rotation scores as
:class:`~diffBloch.observability.RefinedRotationMetrics`, and the trained thickness curve as
:class:`~diffBloch.observability.ThicknessProfile`. So it composes through
:class:`~diffBloch.observability.MultiLogger` beside the console/CSV/W&B sinks, and a caller who
does not want the file simply does not attach it.

Writing happens once, on :class:`~diffBloch.observability.RefinementOutputsWritten` -- the run's
terminal event. That is what lets a write-at-the-end artifact be a plain one-method ``Logger``
rather than forcing a ``close``/``finalize`` hook into the protocol. That event also carries the
path of the committed ``refined_structure.cif``, which this reads back rather than being handed
parsed values, so the reported cell / space group / atom-site tables are byte-consistent with the
file on disk by construction.

Unlike the vendor backends this needs no optional dependency: gemmi is already a core dependency
and the tables are rendered as plain text.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import gemmi

from diffBloch.core.crystal import cell_volume
from diffBloch.io import read_structure
from diffBloch.observability import (
    Event,
    ExperimentDeclared,
    ObjectiveManifest,
    RefinedRotationMetrics,
    RefinementCompleted,
    RefinementOutputsWritten,
    RefinementStep,
    ThicknessProfile,
)

__all__ = ["SummaryLogger", "ascii_table"]


def ascii_table(headers: list[str], rows: list[list[str]]) -> str:
    """Column-aligned plain-text table: a header row, a rule, then the data rows.

    Widths are measured from the content, so a long cell widens its column rather than overflowing
    it.
    """
    widths = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) for i in range(len(headers))
    ]

    def fmt(cells: list[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))

    lines = [fmt(headers), "  ".join("-" * width for width in widths)]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def _cif_loop_as_table(block: gemmi.cif.Block, first_tag: str) -> str | None:
    """Render a CIF loop (identified by its first column tag) as a plain aligned ASCII table.

    ``None`` if the loop is absent (e.g. no ``_atom_site_aniso_*`` loop when every ADP is Uiso).
    """
    column = block.find_loop(first_tag)
    if not column:
        return None
    loop = column.get_loop()
    tags = list(loop.tags)
    width = loop.width()
    rows = [
        [loop.values[row * width + col] for col in range(width)] for row in range(loop.length())
    ]
    return ascii_table(tags, rows)


@dataclass
class SummaryLogger:
    """Accumulate a refinement run's events and write ``refinement_report.txt`` when it ends.

    ``path`` is the report file. Every event is buffered until
    :class:`~diffBloch.observability.RefinementOutputsWritten` arrives, at which point the report is
    rendered and written; a run that never emits that event writes nothing, which is the correct
    behaviour for an aborted run (there is no committed structure to describe).

    ``plot`` (default ``True``) additionally writes the thickness-NN shape PNG beside the report when
    a :class:`~diffBloch.observability.ThicknessProfile` was seen and matplotlib is installed; the
    report notes the omission otherwise, exactly as before.
    """

    path: Path
    plot: bool = True
    _experiment: ExperimentDeclared | None = field(default=None, init=False, repr=False)
    _manifest: ObjectiveManifest | None = field(default=None, init=False, repr=False)
    _steps: list[RefinementStep] = field(default_factory=list, init=False, repr=False)
    _completed: RefinementCompleted | None = field(default=None, init=False, repr=False)
    _rotations: list[RefinedRotationMetrics] = field(default_factory=list, init=False, repr=False)
    _profiles: list[ThicknessProfile] = field(default_factory=list, init=False, repr=False)
    _started_at: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        # The sink times the run itself rather than being told how long it took: it sees the first
        # event and the last, which is the same span the caller would have measured.
        self._started_at = time.perf_counter()

    def report(self, event: Event) -> None:
        if isinstance(event, ExperimentDeclared):
            self._experiment = event
        elif isinstance(event, ObjectiveManifest):
            self._manifest = event
        elif isinstance(event, RefinementStep):
            self._steps.append(event)
        elif isinstance(event, RefinementCompleted):
            self._completed = event
        elif isinstance(event, RefinedRotationMetrics):
            self._rotations.append(event)
        elif isinstance(event, ThicknessProfile):
            self._profiles.append(event)
        elif isinstance(event, RefinementOutputsWritten):
            self._write(event)

    def _write(self, outputs: RefinementOutputsWritten) -> None:
        elapsed = time.perf_counter() - self._started_at
        lines: list[str] = []

        def banner(title: str) -> None:
            lines.append("=" * 78)
            lines.append(f" {title}")
            lines.append("=" * 78)

        def rule(title: str) -> None:
            lines.append("")
            lines.append(f"--- {title} " + "-" * max(0, 74 - len(title)))

        experiment = self._experiment
        banner(f"diffBloch refinement report -- {experiment.name if experiment else 'run'}")
        lines.append(f" structure   : {experiment.structure if experiment else 'n/a'}")
        lines.append(f" exp_data    : {experiment.experimental_data if experiment else 'n/a'}")
        lines.append(f" generated   : {datetime.now(UTC).isoformat(timespec='seconds')}")
        lines.append(f" elapsed     : {elapsed:.1f} s ({elapsed / 60.0:.2f} min)")

        self._simulation_parameters(lines, rule)
        self._crystallographic_parameters(lines, rule, Path(outputs.structure))
        self._objective_terms(lines, rule)
        self._objective_components(lines, rule)
        self._rotation_metrics(lines, rule)
        self._thickness_profile(lines, rule)
        self._refined_structure(lines, rule, Path(outputs.structure))

        self.path.write_text("\n".join(lines) + "\n")

    @property
    def _best(self) -> RefinementStep | None:
        """The epoch the run selected, or ``None`` if no step stream was seen."""
        if not self._steps:
            return None
        if self._completed is not None:
            for step in self._steps:
                if step.iteration == self._completed.best_step:
                    return step
        return self._steps[-1]

    def _simulation_parameters(self, lines: list[str], rule: Callable[[str], None]) -> None:
        rule("Simulation / refinement parameters")
        experiment, best, completed = self._experiment, self._best, self._completed
        if experiment is None:
            lines.append(" n/a (no recorded experiment declaration)")
            return

        def epoch_mean_pct(value: float | None, evaluated: int | None) -> str:
            """A best-epoch mean as a percentage, with the rotation count it was averaged over.

            A non-finite mean renders ``n/a``, the same spelling the per-rotation means below this
            table use: the objective averages only finite scores, so non-finite means nothing was
            evaluated -- already stated by the count beside it.
            """
            if best is None or value is None:
                return "n/a"
            rendered = f"{100.0 * value:.2f}" if math.isfinite(value) else "n/a"
            if evaluated is None or best.n_rotations is None:
                return rendered
            return f"{rendered} [{evaluated}/{best.n_rotations}]"

        counts = completed.reflection_counts if completed else {}
        hkls_matched = (
            f"{counts['matched_strong']} / {counts['matched']}"
            if "matched" in counts and "matched_strong" in counts
            else "n/a"
        )
        best_epoch = f"{completed.best_step + 1:g}" if completed else "n/a"
        rows = [
            ["Integration semiangle (deg)", f"{experiment.integration_semiangle:g}"],
            ["Rocking-curve sampling", f"{experiment.rocking_curve_sampling:g}"],
            ["D_sg", f"{experiment.dsg:g}"],
            ["R_sg", f"{experiment.rsg:g}"],
            ["g_max (solve cutoff, A^-1)", f"{experiment.solve_g_max:g}"],
            [
                "structure-factor support (2 x g_max, A^-1)",
                f"{2.0 * experiment.solve_g_max:g}",
            ],
            ["sg_max (A^-1)", f"{experiment.sg_max:g}"],
            ["Absorption (T/F)", "T" if experiment.absorption else "F"],
            ["Seed thickness (A)", ", ".join(f"{t:g}" for t in experiment.seed_thicknesses)],
            ["Epochs (configured)", f"{experiment.steps:g}"],
            ["Best epoch", best_epoch],
            ["Optimizer", experiment.optimizer],
            ["Learning rate", f"{experiment.learning_rate:g}"],
            [
                "R_obs (%)",
                epoch_mean_pct(
                    best.r_obs if best else None, best.n_r_obs_evaluated if best else None
                ),
            ],
            [
                "wR2 (%)",
                epoch_mean_pct(best.wr2 if best else None, best.n_wr2_evaluated if best else None),
            ],
            ["Matched HKLs (strong I>3sigma / total)", hkls_matched],
        ]
        lines.append(ascii_table(["Parameter", "Value"], rows))

    def _crystallographic_parameters(
        self, lines: list[str], rule: Callable[[str], None], structure_path: Path
    ) -> None:
        rule("Crystallographic parameters")
        structure = read_structure(structure_path)
        a, b, c, alpha, beta, gamma = (float(v) for v in structure.cell_parameters)
        lines.append(
            ascii_table(
                ["Parameter", "Value"],
                [
                    ["Space group", f"{structure.spacegroup_hm} (#{structure.spacegroup_number})"],
                    ["a, b, c (A)", f"{a:.4f}, {b:.4f}, {c:.4f}"],
                    ["alpha, beta, gamma (deg)", f"{alpha:.2f}, {beta:.2f}, {gamma:.2f}"],
                    ["Volume (A^3)", f"{cell_volume(structure.unit_cell):.1f}"],
                    ["N atoms (ASU)", f"{len(structure.labels)}"],
                ],
            )
        )

    def _objective_terms(self, lines: list[str], rule: Callable[[str], None]) -> None:
        rule("Objective terms (declared)")
        manifest = self._manifest
        if manifest is None:
            lines.append(" n/a (no recorded objective manifest)")
            return
        # "none" rather than an omitted line: the default CLI path composes no penalties, and that
        # is a scientific fact a reader should not have to infer from a missing section.
        penalties = ", ".join(f"{t.name} (weight {t.weight:g})" for t in manifest.penalties)
        lines.append(f" penalties  : {penalties or 'none'}")
        lines.append(f" constraints: {', '.join(manifest.constraints) or 'none'}")
        lines.append(f" components : {', '.join(manifest.components) or 'none'}")

    def _objective_components(self, lines: list[str], rule: Callable[[str], None]) -> None:
        rule("Objective components (best epoch)")
        best = self._best
        if best is None or not best.components:
            lines.append(" n/a (no recorded objective components)")
            return
        lines.append(
            ascii_table(
                ["Component", "Raw", "Weight", "Contribution"],
                [
                    [
                        term,
                        f"{values['raw']:.6g}",
                        f"{values['weight']:g}",
                        f"{values['contribution']:.6g}",
                    ]
                    for term, values in best.components.items()
                ],
            )
        )
        lines.append("")
        total = "n/a" if best.objective_total is None else f"{best.objective_total:.6g}"
        lines.append(f" objective total = {total}")
        # Every row is a term the objective actually composed. A restraint that was not composed
        # has no row, so this table never reports an inactive term as a satisfied zero.
        lines.append(" (terms not composed into the objective have no row)")

    def _rotation_metrics(self, lines: list[str], rule: Callable[[str], None]) -> None:
        def block(rows: Sequence[RefinedRotationMetrics]) -> None:
            lines.append(
                ascii_table(
                    ["Rotation", "wR2", "R_obs", "N matched"],
                    [
                        [
                            str(row.rotation_index),
                            f"{row.wr2:.6f}",
                            f"{row.r_obs:.6f}",
                            str(row.n_matched),
                        ]
                        for row in rows
                    ],
                )
            )
            finite_wr2 = [row.wr2 for row in rows if math.isfinite(row.wr2)]
            finite_r_obs = [row.r_obs for row in rows if math.isfinite(row.r_obs)]
            lines.append("")

            def mean_line(label: str, finite: list[float]) -> str:
                # Each mean states the denominator it was taken over: a mean that covers fewer
                # rotations is a different quantity, not a better one.
                if not finite:
                    return f" mean {label} = n/a [0/{len(rows)}]"
                return (
                    f" mean {label} = {sum(finite) / len(finite):.6f} [{len(finite)}/{len(rows)}]"
                )

            lines.append(mean_line("wR2  ", finite_wr2))
            lines.append(mean_line("R_obs", finite_r_obs))

        rule("Per-rotation wR2 / R_obs (final refined model)")
        block(self._rotations)
        validation = [row for row in self._rotations if row.is_validation]
        if validation:
            rule("Validation set (held out from the refinement objective)")
            lines.append(f" n_rotations = {len(validation)}")
            lines.append("")
            block(validation)

    def _thickness_profile(self, lines: list[str], rule: Callable[[str], None]) -> None:
        """Render one "Thickness NN" section per trained component.

        Ordinarily exactly one, unlabeled (``profile.label is None``) -- rendered exactly as before,
        same section title, same fixed ``thickness_nn_shape.png`` filename. A combined
        (``inputs.multi_dataset``) run reports one *labeled* profile per dataset (each an
        independently-trained NN, see :class:`~diffBloch.engine.components.ApparentThicknessNN`'s
        ``rotation_range``) -- each gets its own titled sub-section and its own PNG, named from the
        label so they never collide.
        """
        if not self._profiles:
            rule("Thickness NN")
            lines.append(" enabled: no (preprocess-baked per-rotation thickness only)")
            return
        for profile in self._profiles:
            title = "Thickness NN" if profile.label is None else f"Thickness NN -- {profile.label}"
            rule(title)
            lines.append(
                f" enabled: yes ({profile.form}, bounds "
                f"[{profile.min_thickness:g}, {profile.max_thickness:g}] A)"
            )
            lines.append("")
            lines.append(
                ascii_table(
                    ["Rotation", "alpha (degrees)", "Thickness (A)"],
                    [
                        [str(index), f"{alpha:.4f}", f"{thickness:.2f}"]
                        for index, alpha, thickness in zip(
                            profile.rotation_indices,
                            profile.alphas,
                            profile.thicknesses,
                            strict=True,
                        )
                    ],
                )
            )
            if not self.plot:
                continue
            suffix = "" if profile.label is None else f"_{profile.label.replace('/', '_')}"
            plot_path = self.path.parent / f"thickness_nn_shape{suffix}.png"
            try:
                from diffBloch.app.loggers.plotting import plot_thickness_nn_shape

                plot_thickness_nn_shape(
                    list(zip(profile.alphas, profile.thicknesses, strict=True)), plot_path
                )
                lines.append("")
                lines.append(f" plot: {plot_path.name}")
            except ModuleNotFoundError:
                lines.append("")
                lines.append(
                    " plot: skipped (matplotlib not installed -- see the diffBloch[plot] extra)"
                )

    def _refined_structure(
        self, lines: list[str], rule: Callable[[str], None], structure_path: Path
    ) -> None:
        rule("Refined structure -- atom_site")
        block = gemmi.cif.read_file(str(structure_path)).sole_block()
        atom_table = _cif_loop_as_table(block, "_atom_site_label")
        if atom_table is not None:
            lines.append(atom_table)
        aniso_table = _cif_loop_as_table(block, "_atom_site_aniso_label")
        if aniso_table is not None:
            lines.append("")
            lines.append(aniso_table)
