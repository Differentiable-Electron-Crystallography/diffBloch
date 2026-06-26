# Stage 8 decision — Bloch-wave propagators: swappable, default `matrix_exp`

**Status:** accepted · **Scope:** `core/solver.py`, `core/dynamical/assembly.py` (`BlochSystem`)

## Context

`BlochSystem` (`A`, `Mii`, `psi0`, `k_n`, `mask`) is propagator-agnostic; `propagate(system, T,
method=...)` integrates it to the exit wavefunction. Two methods exist (ported from
`diffBloch_private::calculate_dynamical_scattering_batched`, no-absorption path):

- **`matrix_exp`** — `psi(t) = matrix_exp(A · iπt/k_n) @ psi0`. One dense matrix exponential.
- **`bloch_eigen`** — `eigh(A)` once, then each thickness is a cheap phase multiply; the diagonal is
  un-symmetrised by `Mii`.

They are **not two solvers for one answer** — they differ on *what* they return: `matrix_exp`
propagates the **symmetrised** (Hermitian-basis) wavefunction; `bloch_eigen` un-symmetrises to
**physical** amplitudes. This is a model/convention difference, not a numerical-method difference.

## Decision

1. **Both methods are first-class and swappable** off the same `BlochSystem` value (strategy-as-value,
   `method=` literal — no registry/plugin framework until a third method earns it). The
   symmetrised-vs-physical difference is a feature a scientist can experiment with, not a defect to
   hide.
2. **Default = `matrix_exp`** for differentiable refinement: a single autograd-stable primitive with
   no `eigh`-backward conditioning risk near the (near-)degenerate eigenvalues that symmetric crystals
   routinely produce.
3. **`bloch_eigen` is first-class for evaluation**: it amortises one eigendecomposition over many
   thicknesses, and returns physical amplitudes — the right tool for fast multi-thickness eval and for
   studying the physical/symmetrised distinction.

## Evidence

`scripts/stage8_propagator_experiment.py` (zone-axis `Mii==1` and oblique `Mii!=1` α-quartz fixtures,
`T ∈ {1,8,42,500} Å`, 200 keV, CPU):

| regime | forward `max|Δψ|` | flux `matrix_exp` | flux `bloch_eigen` | timing |
| --- | --- | --- | --- | --- |
| zone (`Mii==1`) | `~1e-15` (machine) | `1.000000` | `1.000000` | `be ≈ 0.4–0.5 × me` |
| oblique (`Mii!=1`) | `4.9e-5` (`O(g_z/k_n)`) | `1.000000` (unitary) | `≠ 1` (physical) | `be ≈ 0.5–0.7 × me` |

So the methods coincide to machine precision **only at `Mii==1`**; off zone axis they differ by the
obliquity, exactly as the symmetrised-vs-physical framing predicts. `bloch_eigen` is the faster
forward.

## Explicitly not claimed

A numerically airtight **gradient-stability** comparison is **deferred**. An early attempt used a
flux loss (`Σ|ψ|²`), which is identically `1` for unitary `matrix_exp` and so has a structurally zero
gradient — a degenerate probe; a follow-up finite-difference cross-check hit a complex-Wirtinger
convention error. The default rests on the *qualitative* argument (single autograd-stable primitive
vs `eigh`-backward conditioning), which is sufficient and textbook; the numerical proof is a separate,
careful task if/when it's worth it.

## Deferred (oracle methodology — recorded for later)

- **Real-CIF oracle.** Replace synthetic Friedel-symmetric `Fgb` with `Fgb` derived from a real CIF
  through the new `io` parser + `core.scattering` (Friedel symmetry then comes for free; exercises the
  full `CIF → parser → expand_asu → scattering → A → propagate` chain). Golden `A`/`psi` stay
  private-generated (independence in the *golden*, not the *input*; do **not** reintroduce `abtem`).
- **R-factor / intensity oracle.** Compare an R-loss (calculated vs observed intensities) against the
  private path, not just raw `ψ`.
- **Composite-step breakdown.** Per-step oracle parity (gather → A → propagate → intensity → loss),
  not only end-to-end.
