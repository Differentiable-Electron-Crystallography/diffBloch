# Lessons carried over from `diffBloch_private`

The diffBloch 2.0 rewrite is a clean-room port of `diffBloch_private`. Beyond reproducing the
physics, the rewrite deliberately **fixes structural/design problems** the original ran into. This
file records those lessons -- what the old codebase did, why it hurt, and how 2.0 is shaped to avoid
it -- so the *reasoning* behind the new structure stays discoverable.

It is distinct from its sibling ledgers: [`DIVERGENCE.md`](DIVERGENCE.md) records intentional
*behaviour* differences (corrected bugs, generalizations); [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md)
records open latent bugs / deferred fixes; [`REFERENCES.md`](REFERENCES.md) records literature; and
`design/decisions/` holds the forward-looking per-stage decisions. This file is the index of
*architectural* lessons. It is living -- add an entry whenever a 2.0 structure exists specifically
to avoid a private-codebase trap.

## Config: a validated schema with defaults-as-code, not composed Hydra trees

**The trap.** The private used Hydra, whose power -- multi-file composition, defaults lists,
`${...}` interpolation, command-line overrides -- let config sprawl. A `DictConfig` (dynamic,
untyped, mutable) reached deep into the code, so the *structure* of config was emergent from many
files rather than declared in one place. Worse, unrelated concerns piled into the same blocks. The
real private `preprocess.orientation` block:

```yaml
orientation:
  optim: true
  use_optim_thicknesses: false   # cross-step flag => path-dependence between steps
  min_search_angle: 0.001        # Palatinus-only ┐ mutually-exclusive options
  max_search_angle: 0.4          # Palatinus-only │  coexisting in one block
  step_size: 0.05                # Nelder-Mead-only ┘
  n_steps: 6
  save_name: "..._inverted.csv"  # persistence leaking into algorithm config
  save_path: ???                 # Hydra "mandatory missing" placeholder magic
  method: Nelder-Mead            # stringly-typed dispatch
  num_workers: 8                 # execution concern in algorithm config
```

The damage was never the *nesting* -- it was the dynamism (`???`, composition, interpolation),
mutually-exclusive fields coexisting, persistence/execution bleeding into algorithm config, and
cross-step flags (`use_optim_*`) encoding history-dependence.

**What 2.0 does.** One pydantic `ExperimentConfig` (`src/diffBloch/config/schema.py`), validated at
the boundary by `load_config` -- *no Hydra, and no `DictConfig` reaches the core*. Every field has a
sensible default ("defaults-as-code"), so an `experiment.yaml` carries only input references and
overrides. Nesting is used for **grouping** (`refinement.optimizer`, `refinement.split`), never for
**composition**. The distilled equivalent of the block above is three consumed fields:

```python
class OrientationFitConfig(BaseModel):
    """Palatinus hexagonal-tilt search bounds for fit_orientation."""
    max_search_angle: float = 0.4    # deg, initial hex radius (private 0.038*8)
    min_search_angle: float = 0.001  # deg, stop below this radius
    n_steps: int = 6                 # hex azimuths 0,60,...,300 (Palatinus)
```

Each dropped field maps to a 2.0 rule, not an omission:

- `save_name` / `save_path` -> **gone**: persistence is not algorithm config (see the persistence
  lesson below); preprocess steps return `Plan`s, you checkpoint the whole `Plan`.
- `num_workers` -> execution concern, not config.
- `method` / `step_size` -> we chose one method (Palatinus); a second method would be a
  **discriminated union** (`Literal` + a per-method block), never coexisting optional fields.
  Optional / shed fields are an anti-pattern.
- `use_optim_*` / `optim` -> the `Plan -> Plan` pipeline composition *is* the dependency and the
  enablement; steps never coordinate through config flags (see the path-independence lesson).

**The split that keeps the core clean.** The config block is the declarative, sweepable *home* at
the boundary; the pure functions take **plain values**. `fit_orientation(refinement, *,
max_search_angle=0.4, min_search_angle=0.001, n_steps=6)` never sees a pydantic model -- the
preprocess boundary unpacks `cfg.preprocess.orientation -> fit_orientation(...)`, exactly as the
refine boundary unpacks `cfg.refinement.optimizer ->` plain optimizer args.

*Enforced in:* `src/diffBloch/config/schema.py`, `tests/fixtures/quartz_anchor/experiment.yaml`.

## Persistence and effects live at the boundary, not inside algorithm code

**The trap.** Private preprocess steps wrote per-facet CSV side-cars (`save_name` / `save_path`) and
later steps read them back (`use_optim_orientations` / `use_optim_thicknesses`), so the algorithm
was entangled with the filesystem and with a specific on-disk handoff format.

**What 2.0 does.** `fit_orientation` / `fit_thickness` return sharpened `Plan`s and never write
files; persistence means checkpointing the whole `Plan`, and CSV/visualization are boundary
reporters behind a pluggable logger. A deterministic simulation inside a step is *not* a side
effect; only observation of it (logging, checkpoints, events) is.

*Decided in:*
[`design/decisions/effects-and-observability.md`](design/decisions/effects-and-observability.md).

## State is a function of values, not history

**The trap.** Because steps handed off through files and toggled each other with flags
(`use_optim_thicknesses`, ...), the post-preprocess state depended on the *route* taken, not just
the values -- two paths to the same physics could leave the system in different shapes.

**What 2.0 does.** `Plan`s are fully-populated, self-describing values; equal values are
indistinguishable regardless of how they were produced. A quantity gets **one home decided by its
role** (e.g. per-rotation thickness is frozen conditioning in `OrientationPlan`, read through a
`None`-default provider seam; the learned-thickness path is an explicit opt-in *mode*, not a second
home). "Was this fitted in preprocess?" is a question of value, never of location.

*Decided in:*
[`design/decisions/plan-shape-and-step-ordering.md`](design/decisions/plan-shape-and-step-ordering.md);
see also the thickness two-mode note in [`ROADMAP.md`](ROADMAP.md) and
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).
