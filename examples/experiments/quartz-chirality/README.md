# Example: quartz chirality check

This proof-of-concept compares two quartz starting structures against the same observed PETS2
`.cif_pets` data:

- `matching-hand/` uses the P3_2 21 quartz hand from the canonical quartz example.
- `opposite-hand/` uses the P3_1 21 enantiomorphic setting with the same cell and asymmetric-unit
  coordinates.

The intended test is simple: score both committed settled `Plan` checkpoints with `infer` and
compare the terminal R-loss. `infer` runs the Bloch-wave simulation and scoring path without
optimizer updates. If the dynamical multiple-scattering model is discriminating handedness for this
dataset, the opposite-hand candidate should score at a larger loss than the matching-hand candidate.

This is a handedness-discrimination example, not a claim that optimizer traces are
hardware-independent. The comparison is the score from the committed checkpoints.

## Run

From the repository root:

```bash
# Optional sanity checks: verify config and input locks.
uv run diffbloch validate examples/experiments/quartz-chirality/matching-hand/experiment.yaml
uv run diffbloch validate examples/experiments/quartz-chirality/opposite-hand/experiment.yaml

# Score the committed settled plans without optimizer updates. These runs reuse the bundled
# plan.npz/plan.lock checkpoints unless --refresh is passed.
uv run diffbloch run infer examples/experiments/quartz-chirality/matching-hand
uv run diffbloch run infer examples/experiments/quartz-chirality/opposite-hand

# Expected committed-checkpoint result:
# matching-hand: evaluated 99 rotations; mean R_obs = 0.0507
# opposite-hand: evaluated 99 rotations; mean R_obs = 0.2192
```

To print a compact comparison summary:

```bash
uv run python -c 'from pathlib import Path; from diffBloch.app import run_experiment; print("running quartz chirality comparison...", flush=True); root = Path("examples/experiments/quartz-chirality"); matching = run_experiment(root / "matching-hand", checkpoint=True); opposite = run_experiment(root / "opposite-hand", checkpoint=True); print(f"matching-hand mean R_obs: {matching.mean_r_obs:.4f}"); print(f"opposite-hand mean R_obs: {opposite.mean_r_obs:.4f}"); print(f"opposite/matching ratio: {opposite.mean_r_obs / matching.mean_r_obs:.2f}x")'
```

Each experiment has its own `experiment.lock` so the CIF and `.cif_pets` input bytes are verified
before a run starts. Each experiment also includes a committed preprocess checkpoint:

- `plan.npz` stores the settled `Plan` arrays for that handedness candidate.
- `plan.lock` binds the checkpoint to the input bytes, resolved preprocess config, software version,
  and recipe identity.

Pass `--refresh` to recompute a plan from scratch and overwrite the local checkpoint.

## Interpretation

One diffBloch simulation of the two preprocessed `Plan`s gives these R-losses:

```text
matching-hand: mean R_obs = 0.0507
opposite-hand: mean R_obs = 0.2192
```

The opposite-hand structure scores about 4.3x worse against the same observed `.cif_pets` data,
indicating that the observed diffraction is much more consistent with the P3_2 21 handedness than
with the enantiomorphic P3_1 21 candidate.

If a future dataset gives losses that are too close, treat that as a scientific result to
investigate rather than a test failure. Possible follow-ups include checking the enantiomorph
construction, increasing solve precision, changing trainable groups, or using a dataset with stronger
handedness signal.
