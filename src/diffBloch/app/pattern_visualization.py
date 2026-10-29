"""Render calculated diffraction patterns at the application boundary.

This module is deliberately part of :mod:`diffBloch.app`: it consumes settled
``Plan`` and ``BlochSolution`` values but is not imported by the numerical core.
Matplotlib is an optional output dependency and is imported only when rendering
is requested.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from diffBloch.core.crystal import orientation_basis
from diffBloch.core.losses import optimal_scale, rbragg
from diffBloch.core.products import BlochSolution, align
from diffBloch.core.reciprocal import g_vectors
from diffBloch.preprocess.plan import Plan, require_built_plans
from diffBloch.preprocess.steps.beams import klar_beam_mask
from diffBloch.specs import BeamSelection

__all__ = ["PatternVisualization", "PatternVisualizationOutput", "render_patterns"]


@dataclass(frozen=True)
class PatternVisualization:
    """Execution-only styling for calculated-pattern PNG output."""

    source: Literal["simulated", "experimental"] = "simulated"
    side_by_side: bool = False
    normalization: Literal["global", "self"] = "global"
    image_size: int = 1024
    sigma_pixels: float = 2.0
    gamma: float = 0.5
    intensity_normalization: float = 10.0
    intensity_exponent: float = 3.0
    n_label_hkls: int = 2
    hkl_label_fontsize: int = 24
    color_norm_percentile: float = 99.5
    header: str | None = None
    theme: Literal["dark", "light"] = "light"
    scale_bar: float = 0.5
    cmap: str = "gray"
    rotate_pattern_degrees: float = 0.0

    def __post_init__(self) -> None:
        if self.image_size < 64:
            raise ValueError("image_size must be >= 64")
        if self.sigma_pixels < 0:
            raise ValueError("sigma_pixels must be non-negative")
        if self.gamma <= 0 or self.intensity_exponent <= 0:
            raise ValueError("gamma and intensity_exponent must be positive")
        if self.intensity_normalization <= 0:
            raise ValueError("intensity_normalization must be positive")
        if self.n_label_hkls < 0 or self.hkl_label_fontsize < 1:
            raise ValueError("label count must be non-negative and font size must be positive")
        if not 0 < self.color_norm_percentile <= 100:
            raise ValueError("color_norm_percentile must be in (0, 100]")
        if self.scale_bar <= 0:
            raise ValueError("scale_bar must be positive")


@dataclass(frozen=True)
class PatternVisualizationOutput:
    """Paths written by :func:`render_patterns`."""

    frames: tuple[Path, ...]


@dataclass(frozen=True)
class _Pattern:
    rotation_index: int
    dataset: str | None
    hkl: NDArray[np.int64]
    intensities: NDArray[np.float64]
    detector_xy: NDArray[np.float64]


def _closest_rotation(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Remove scale/shear from a PETS orientation by polar decomposition."""
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def _detector_xy(
    cell: NDArray[np.float64], orientation: NDArray[np.float64], hkl: NDArray[np.int64]
) -> NDArray[np.float64]:
    basis = orientation_basis(cell, _closest_rotation(orientation))
    return np.asarray(hkl @ basis, dtype=np.float64)[:, :2]


def _rotate_xy(xy: NDArray[np.float64], degrees: float) -> NDArray[np.float64]:
    angle = math.radians(degrees)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    return xy @ rotation.T


def _patterns(
    plan: Plan,
    solutions: tuple[BlochSolution, ...] | None,
    *,
    source: Literal["simulated", "experimental"],
    experimental_selections: Mapping[str, BeamSelection] | None = None,
) -> tuple[_Pattern, ...]:
    orientations = require_built_plans(plan)
    if source == "simulated" and solutions is None:
        raise ValueError("simulated pattern visualization requires BlochSolution values")
    if solutions is not None and len(orientations) != len(solutions):
        raise ValueError("one BlochSolution is required for every Plan orientation")
    cell = np.asarray(plan.structure_factor_grid.cell, dtype=np.float64)
    rows: list[_Pattern] = []
    for index, orientation in enumerate(orientations):
        if source == "experimental":
            if experimental_selections is None:
                raise ValueError("experimental visualization requires rsg/dsg selections")
            selection = experimental_selections[orientation.pattern.dataset]
            all_hkl = np.asarray(orientation.pattern.hkl.detach().cpu(), dtype=np.int64)
            basis = orientation_basis(cell, np.asarray(orientation.orientation, dtype=np.float64))
            frame_selected = klar_beam_mask(
                g_vectors(all_hkl, basis),
                energy=orientation.energy,
                u0=orientation.u0,
                rsg=selection.rsg,
                dsg=selection.dsg,
                semiangle=selection.integration.semiangle,
                geometry=selection.integration.geometry,
            )
            observed = np.asarray(orientation.pattern.intensities.detach().cpu(), dtype=np.float64)
            sigmas = np.asarray(orientation.pattern.sigmas.detach().cpu(), dtype=np.float64)
            keep = frame_selected & (observed > 3 * sigmas)
            hkl = all_hkl[keep]
            intensity = observed[keep]
        else:
            assert solutions is not None  # narrowed by the construction guard above
            solution = solutions[index]
            # Settled plans carry one fitted thickness. If a custom plan has a grid,
            # show its first thickness explicitly rather than silently mixing them.
            intensity = np.asarray(solution.intensities[0].detach().cpu(), dtype=np.float64)
            hkl = np.asarray(solution.beam_hkl.detach().cpu(), dtype=np.int64)
            if experimental_selections is not None:
                selection = experimental_selections[orientation.pattern.dataset]
                basis = orientation_basis(
                    cell, np.asarray(orientation.orientation, dtype=np.float64)
                )
                frame_selected = klar_beam_mask(
                    g_vectors(hkl, basis),
                    energy=orientation.energy,
                    u0=orientation.u0,
                    rsg=selection.rsg,
                    dsg=selection.dsg,
                    semiangle=selection.integration.semiangle,
                    geometry=selection.integration.geometry,
                )
                frame_selected |= np.all(hkl == 0, axis=1)
                hkl = hkl[frame_selected]
                intensity = intensity[frame_selected]
        rows.append(
            _Pattern(
                rotation_index=int(orientation.pattern.rotation_index),
                dataset=orientation.pattern.dataset,
                hkl=hkl,
                intensities=intensity,
                detector_xy=_detector_xy(
                    cell, np.asarray(orientation.orientation, dtype=np.float64), hkl
                ),
            )
        )
    return tuple(rows)


def _hkl_label(hkl: NDArray[np.int64]) -> str:
    def part(value: int) -> str:
        return rf"\overline{{{abs(value)}}}" if value < 0 else str(value)

    return r"$({}\,{}\,{})$".format(*(part(int(value)) for value in hkl))


def _render_progress(current: int, total: int, started_at: float, rotation_index: int) -> None:
    """Render one in-place app-shell progress line while PNG frames are produced."""
    if not sys.stdout.isatty():
        return
    fraction = current / total
    width = 30
    filled = int(width * fraction)
    elapsed = time.perf_counter() - started_at
    eta = ""
    if current < total:
        remaining = elapsed / current * (total - current)
        minutes, seconds = divmod(max(0, int(remaining)), 60)
        eta = f" eta {minutes}:{seconds:02d}"
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(
        f"\r[{bar}] {current}/{total} ({100 * fraction:.0f}%){eta} rotation {rotation_index}\033[K"
    )
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def render_patterns(
    plan: Plan,
    solutions: tuple[BlochSolution, ...] | None,
    output_dir: str | Path,
    *,
    name: str = "pattern-vis",
    style: PatternVisualization | None = None,
    experimental_selections: Mapping[str, BeamSelection] | None = None,
) -> PatternVisualizationOutput:
    """Render calculated patterns without changing or feeding back into the simulation.

    The display scale is shared across rotations, derived from the largest detector-plane
    reciprocal-space radius. PNG frames are written under ``output_dir/frames``. Plotting
    dependencies are lazy so ordinary diffBloch use does not import a graphics stack.
    """
    style = PatternVisualization() if style is None else style
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colors, patheffects
        from matplotlib.patches import Circle
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:  # pragma: no cover - depends on the installed optional extra
        raise RuntimeError(
            "pattern visualization requires the 'visualization' extra: "
            "uv sync --extra visualization"
        ) from exc

    sources: tuple[Literal["simulated", "experimental"], ...] = (
        ("experimental", "simulated") if style.side_by_side else (style.source,)
    )
    pattern_groups = tuple(
        _patterns(
            plan,
            solutions,
            source=source,
            experimental_selections=experimental_selections,
        )
        for source in sources
    )
    if not pattern_groups[0]:
        raise ValueError("cannot visualize a Plan with no orientations")
    rotated_groups = tuple(
        tuple(_rotate_xy(pattern.detector_xy, style.rotate_pattern_degrees) for pattern in patterns)
        for patterns in pattern_groups
    )
    max_radius = max(
        float(np.linalg.norm(xy, axis=1).max(initial=0.0))
        for rotated in rotated_groups
        for xy in rotated
    )
    if not math.isfinite(max_radius) or max_radius <= 0:
        raise ValueError("calculated patterns have no finite non-zero reciprocal-space radius")

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(name).stem

    size = style.image_size
    center = size / 2
    # A real inset in detector-image coordinates, not merely figure padding: the circle stroke was
    # previously one pixel from the axes boundary, so Matplotlib clipped it like a screenshot crop.
    detector_margin = max(16.0, 0.04 * size)
    radius = center - detector_margin
    pixels_per_ainv = radius / max_radius
    yy, xx = np.mgrid[:size, :size]
    circle = (xx - center) ** 2 + (yy - center) ** 2 <= radius**2
    # Keep the transmitted beam's blurred footprint out of the robust colour-limit estimate so
    # weaker diffracted spots retain contrast. This affects only display normalization: the
    # simulated (0, 0, 0) intensity is still rendered unchanged below.
    central_exclusion_radius = max(3 * style.sigma_pixels, 1.0)
    central = (xx - center) ** 2 + (yy - center) ** 2 <= central_exclusion_radius**2

    def raster(pattern: _Pattern, xy: NDArray[np.float64]) -> NDArray[np.float64]:
        # float32 is ample for display output and halves the working set. Side-by-side 1024px
        # output otherwise retains well over a gigabyte across 99 rotations before frame export.
        image = np.zeros((size, size), dtype=np.float32)
        px = center + xy[:, 0] * pixels_per_ainv
        py = center + xy[:, 1] * pixels_per_ainv
        valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        np.add.at(
            image,
            (np.rint(py[valid]).astype(int), np.rint(px[valid]).astype(int)),
            np.maximum(pattern.intensities[valid], 0) ** style.intensity_exponent,
        )
        return np.asarray(
            gaussian_filter(image, sigma=style.sigma_pixels, output=np.float32),
            dtype=np.float32,
        )

    raster_groups = tuple(
        tuple(raster(pattern, xy) for pattern, xy in zip(patterns, rotated, strict=True))
        for patterns, rotated in zip(pattern_groups, rotated_groups, strict=True)
    )

    def normalize_group(
        rasters: tuple[NDArray[np.float64], ...],
    ) -> tuple[tuple[NDArray[np.float64], ...], colors.Normalize]:
        # Experimental and simulated intensities have independent arbitrary scales. Normalize each
        # source over all rotations, while keeping every frame within that source comparable.
        global_max = max(float(image.max(initial=0.0)) for image in rasters) or 1.0
        # Normalize in place: these rasters have no second consumer, so allocating an equally large
        # normalized copy only doubles peak memory and slows export through memory pressure.
        for image in rasters:
            np.sqrt(image, out=image)
            image /= math.sqrt(global_max)
            np.power(image, style.gamma, out=image)
            image *= style.intensity_normalization
        positive_chunks = tuple(
            values[values > 0] for image in rasters if (values := image[circle & ~central]).size
        )
        positive = (
            np.concatenate(positive_chunks) if positive_chunks else np.empty(0, dtype=np.float32)
        )
        vmax = float(np.percentile(positive, style.color_norm_percentile)) if positive.size else 1.0
        # A hard ``Normalize(vmax=percentile)`` clips every stronger pixel to the same solid colour,
        # while x/(x+knee) still asymptotically flattens beams many orders above the knee. ``asinh``
        # stays approximately linear for weak spots and logarithmic for strong ones, preserving the
        # Gaussian radial gradient instead of turning its bright core into a black square.
        knee = max(float(vmax), float(np.finfo(np.float32).eps))
        tone_max = max(float(image.max(initial=0.0)) for image in rasters)
        tone_denominator = math.asinh(tone_max / knee) or 1.0
        for image in rasters:
            image /= knee
            np.arcsinh(image, out=image)
            image /= tone_denominator
        return rasters, colors.Normalize(vmin=0.0, vmax=1.0)

    if style.normalization == "global":
        normalized_groups, group_norms = zip(
            *(normalize_group(rasters) for rasters in raster_groups), strict=True
        )
        norm_groups = tuple(
            tuple(norm for _ in rasters)
            for norm, rasters in zip(group_norms, normalized_groups, strict=True)
        )
    else:
        normalized_group_rows: list[tuple[NDArray[np.float64], ...]] = []
        norm_group_rows: list[tuple[colors.Normalize, ...]] = []
        for rasters in raster_groups:
            images: list[NDArray[np.float64]] = []
            image_norms: list[colors.Normalize] = []
            for raster_image in rasters:
                (normalized_image,), image_norm = normalize_group((raster_image,))
                images.append(normalized_image)
                image_norms.append(image_norm)
            normalized_group_rows.append(tuple(images))
            norm_group_rows.append(tuple(image_norms))
        normalized_groups = tuple(normalized_group_rows)
        norm_groups = tuple(norm_group_rows)
    comparison_metrics: tuple[tuple[int, int, float], ...] | None = None
    if style.side_by_side:
        if solutions is None:
            raise ValueError("side-by-side visualization requires simulated solutions")
        orientations = require_built_plans(plan)
        metric_rows: list[tuple[int, int, float]] = []
        for index, (orientation, solution) in enumerate(zip(orientations, solutions, strict=True)):
            aligned = align(solution, orientation.pattern, orientation.alignment)
            _, r_obs_tensor = optimal_scale(
                aligned.calculated[0], aligned.observed[0], aligned.sigmas[0], metric=rbragg
            )
            n_observed_matches = int((aligned.observed[0] > 3 * aligned.sigmas[0]).sum())
            n_unmatched = max(0, len(pattern_groups[0][index].hkl) - n_observed_matches)
            metric_rows.append((n_observed_matches, n_unmatched, float(r_obs_tensor)))
        comparison_metrics = tuple(metric_rows)
    if style.side_by_side:
        # The two sources need independent normalization because their raw intensity scales are
        # unrelated. Copy the *display-normalized* simulated 000 footprint so it has exactly the
        # same rendered shape/contrast in the experimental panel instead of disappearing after the
        # experimental normalization.
        central_copy_radius = max(12 * style.sigma_pixels, 12.0)
        central_copy = (xx - center) ** 2 + (yy - center) ** 2 <= central_copy_radius**2
        for experimental_image, simulated_image in zip(
            normalized_groups[0], normalized_groups[1], strict=True
        ):
            # Both images are already in the same final display range [0, 1]. Copying these pixels
            # exactly gives the two panels the identical 000 size, shape, and contrast. At twelve
            # Gaussian widths the copied beam is effectively zero, so the boundary is invisible.
            experimental_image[central_copy] = simulated_image[central_copy]
    dark = style.theme == "dark"
    figure_color, annotation_color, outline_color = (
        ("black", "white", "black") if dark else ("white", "black", "white")
    )
    cmap = ("gray" if dark else "gray_r") if style.cmap.lower() in {"gray", "grey"} else style.cmap
    frame_paths: list[Path] = []
    render_started_at = time.perf_counter()
    n_frames = len(pattern_groups[0])
    for frame_index in range(n_frames):
        fig, axes = plt.subplots(
            1,
            len(sources),
            # One data pixel per output pixel. The previous 12x6in @ 180dpi side-by-side canvas
            # rendered 2160x1080 and then ran an expensive tight-bbox layout on every frame.
            figsize=(size * len(sources) / 100, size / 100),
            dpi=100,
            facecolor=figure_color,
            squeeze=False,
        )
        # Reserve a clean title band and a small lower margin. Filling the entire figure caused the
        # detector-circle stroke to clip at the bottom and the title to crowd its upper edge.
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.07, top=0.84, wspace=0.04)
        if comparison_metrics is not None:
            n_matched, n_unmatched, r_obs_value = comparison_metrics[frame_index]
            rendered_r = f"{100.0 * r_obs_value:.2f}%" if math.isfinite(r_obs_value) else "n/a"
            fig.suptitle(
                f"Matched: {n_matched}    Unmatched: {n_unmatched}    R_obs: {rendered_r}",
                fontsize=24,
                color=annotation_color,
                y=0.98,
            )
        for panel, source in enumerate(sources):
            pattern = pattern_groups[panel][frame_index]
            xy = rotated_groups[panel][frame_index]
            image = normalized_groups[panel][frame_index]
            display = np.zeros_like(image)
            display[circle] = image[circle]
            ax = axes[0, panel]
            ax.set_facecolor(figure_color)
            ax.imshow(
                display,
                cmap=cmap,
                origin="lower",
                norm=norm_groups[panel][frame_index],
                interpolation="bilinear",
                resample=True,
            )
            ax.add_patch(
                Circle((center, center), radius, color=annotation_color, fill=False, linewidth=2)
            )
            # Side-by-side panels share this exact pixels-per-A^-1 mapping and extent, so one bar
            # on the right-hand panel applies to both. A single-source frame keeps its own bar.
            if not style.side_by_side or panel == len(sources) - 1:
                bar = style.scale_bar * pixels_per_ainv
                bar_x, bar_y = size - 15 - bar, 10
                ax.plot(
                    [bar_x, bar_x + bar],
                    [bar_y, bar_y],
                    color=annotation_color,
                    linewidth=2,
                )
                ax.text(
                    bar_x + bar / 2,
                    bar_y + 12,
                    f"{style.scale_bar:g} Å⁻¹",
                    color=annotation_color,
                    fontsize=18,
                    ha="center",
                )
            title = (
                f"{style.header} — {source.title()}"
                if style.header and style.side_by_side
                else style.header or f"{source.title()} rotation {pattern.rotation_index}"
            )
            ax.set_title(
                title,
                color=annotation_color,
                fontsize=28,
                pad=20,
                path_effects=[patheffects.withStroke(linewidth=3, foreground=outline_color)],
            )
            label_candidates = np.flatnonzero(np.any(pattern.hkl != 0, axis=1))
            n_labels = min(style.n_label_hkls, label_candidates.size)
            top = (
                label_candidates[np.argsort(pattern.intensities[label_candidates])[-n_labels:]]
                if n_labels
                else np.empty(0, dtype=np.int64)
            )
            for index in top:
                x = center + xy[index, 0] * pixels_per_ainv
                y = center + xy[index, 1] * pixels_per_ainv
                ax.text(
                    x,
                    y + 24,
                    _hkl_label(pattern.hkl[index]),
                    color=annotation_color,
                    fontsize=style.hkl_label_fontsize,
                    ha="center",
                    path_effects=[patheffects.withStroke(linewidth=2, foreground=outline_color)],
                )
            ax.axis("off")
        pattern = pattern_groups[0][frame_index]
        dataset = f"{Path(pattern.dataset).stem}_" if pattern.dataset else ""
        frame_path = frames_dir / f"{safe_name}_{dataset}{pattern.rotation_index:04d}.png"
        fig.savefig(frame_path, dpi=100, facecolor=figure_color)
        frame_paths.append(frame_path)
        plt.close(fig)
        _render_progress(frame_index + 1, n_frames, render_started_at, pattern.rotation_index)

    return PatternVisualizationOutput(frames=tuple(frame_paths))
