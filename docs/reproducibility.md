# Reproducibility and checkpoints

Reproducing a specific result means starting from the exact same fitted `Plan`. diffBloch checkpoints
that fitted geometry and locks it to everything that determined it.

**`experiment.lock`** pins the raw input files -- the structure CIF and the experimental data -- by
content hash (SHA256).

**`plan.lock`** pins the checkpointed `plan.npz` itself -- the optimized orientations and thicknesses
that came out of preprocessing. It's checked against four separate things, and all four have to
match for the checkpoint to be reused as-is:

1. **The input lock** -- `experiment.lock` must match (see above).
2. **The resolved config** -- a hash (`config_digest`) of exactly the settings that could have
   changed the fitted `Plan`. See below for exactly what's in and out of this hash.
3. **The code version** -- the release version of diffBloch that produced the checkpoint. Commits
   within the same release don't invalidate it (see "code version" below), but a version bump does.
4. **The recipe** -- the exact ordered list of preprocessing steps that ran, each with its own
   parameters (`select_beams`, `optimize_orientation`, `optimize_thickness`, etc., in whatever
   order they ran).

Additionally, the `plan.npz` file itself is hashed and checked against the hash recorded in
the lock, so even direct tampering or corruption of the checkpoint file is caught.

The config fields that can change the fitted `Plan` include:

- `inputs`, `sample`, `blochwave` -- these shape the diffraction geometry and the beam grid, so any
  change here can change the fit.
- `preprocess` settings.
- `refinement.split` -- this decides how the fitted rotations get grouped into train/test, which is
  baked into the checkpointed `Plan`.

Everything else in `refinement` -- `objective`, `optimizer`, `steps`, `trainable`, and so on -- is
**exempt**. Those only control what happens *after* preprocessing, during refinement, and refinement
never touches the checkpoint. You can double the number of refinement epochs, swap optimizers, or
change loss weights, and your preprocess checkpoint stays valid -- because none of those settings
were ever read while building it.

## Code version

The full build stamp (`version+g<git-sha>[.dirty]`) is recorded in the lock for provenance, but the
reuse check only compares the release version. Committing changes within the same release (no
version bump) doesn't invalidate existing checkpoints -- deliberately, since not every commit
changes the physics. The trade-off: an un-released physics change technically reuses a checkpoint
it maybe shouldn't. `--refresh` is the manual escape hatch for that case.

## Can it be turned off?

`experiment.lock` verification cannot be disabled -- it runs on every `load_experiment` call, no
flag skips it. If the input files don't match, the run stops.

`plan.lock` has two explicit, deliberate bypasses, both opt-in per run:

- `--no-checkpoint` -- don't read or write `plan.npz`/`plan.lock` at all. Preprocessing always runs
  from scratch and nothing is saved. Use this when you don't want checkpointing in the loop at all.
- `--refresh` -- force preprocessing to run from scratch and overwrite the existing checkpoint,
  regardless of whether the current lock would have matched. Use this when you know you want a new
  checkpoint (e.g. after a genuine, unreleased physics change that a version bump hasn't caught up
  with yet).

Short of those two flags, there's no way to make a stale or mismatched checkpoint get reused
quietly -- a mismatch on any of the four axes, or a corrupted `.npz`, makes the run refuse to reuse
the checkpoint and fall back to running preprocessing fresh.

In committed examples, `plan.npz` and `.cif_pets` files are Git LFS artifacts; `plan.lock` verifies
the realized file bytes after LFS checkout, not the small pointer file.
