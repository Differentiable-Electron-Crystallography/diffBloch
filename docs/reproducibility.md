# Reproducibility and checkpoints

Reproducing a specific result means starting from the exact same fitted `Plan`. diffBloch
checkpoints that fitted geometry **per dataset** and locks each checkpoint to everything that
determined it.

**`experiment.lock`** pins the raw input files -- the structure CIF and the experimental data (one
entry per `inputs.exp_data` file for a combined experiment) -- by content hash (SHA256).

**`plan.<stem>.lock`** pins that one dataset's checkpointed `plan.<stem>.npz` -- the optimized
orientations and thicknesses that came out of preprocessing that file. The lock's identity is
deliberately *per dataset*: nothing in it hashes the whole `experiment.lock` or the full `exp_data`
list, so adding, removing, reordering, or editing *other* datasets in a combined experiment never
invalidates this one's checkpoint. The one deliberate shared axis is the authoritative PETS cell:
in a combined experiment, the first file's cell shapes every checkpointed grid. Six things are
checked, and all six have to match for the checkpoint to be reused as-is:

1. **The input bytes** -- the structure CIF and *this dataset's* `.cif_pets`, by content hash and
   size (not filename: a byte-identical rename moves the checkpoint to a new stem but is still the
   same measurement).
2. **The dataset-scoped config** -- a hash (`dataset_config_digest`) of exactly the settings that
   could have changed this dataset's fitted `Plan`. See below for what's in and out.
3. **The file-local ignored rotations** -- the slice of `blochwave.ignore_orientations` that lands
   on this file, translated to its own rotation indices. An ignore edit restales exactly the
   datasets it touches.
4. **The authoritative PETS cell** -- the first `.cif_pets` file's cell for a combined experiment,
   or the only `.cif_pets` file's cell for a single-dataset experiment. This keeps every pooled
   checkpoint tied to the shared grid and ADP metric, even for non-anchor datasets.
5. **The code version** -- the release version of diffBloch that produced the checkpoint. Commits
   within the same release don't invalidate it (see "code version" below), but a version bump does.
6. **The recipe** -- the exact ordered list of preprocessing steps that ran, each with its own
   parameters (`select_beams`, `optimize_orientation`, `optimize_thickness`, etc., in whatever
   order they ran).

Additionally, the `.npz` file itself is hashed and checked against the hash recorded in the lock,
so even direct tampering or corruption of the checkpoint file is caught. A lock file that no longer
parses is treated as stale (recompute), never as an error.

The config fields that can change a dataset's fitted `Plan` include:

- `inputs` -- reduced to *this dataset's* view: the structure ref, this one `exp_data` entry, and
  `load_hydrogens`. The `multi_dataset` flag and the rest of the list are excluded -- a dataset's
  settled plan doesn't depend on what it's pooled with.
- `sample`, `blochwave` -- these shape the diffraction geometry and the beam grid (minus
  `ignore_orientations`, which is covered by axis 3 above).
- `preprocess` settings.
- `loss_metrics` -- the residual the orientation/thickness searches minimise.

Everything in `refinement` -- including `split` -- is **exempt**. The train/validation split
partitions rotations when the pooled plan is handed to refinement; it never shapes a checkpointed
per-dataset plan. You can double the number of refinement epochs, swap optimizers, change loss
weights, or re-cut the split, and your preprocess checkpoints stay valid -- because none of those
settings were ever read while building them.

## Code version

The full build stamp (`version+g<git-sha>[.dirty]`) is recorded in the lock for provenance, but the
reuse check only compares the release version. Committing changes within the same release (no
version bump) doesn't invalidate existing checkpoints -- deliberately, since not every commit
changes the physics. The trade-off: an un-released physics change technically reuses a checkpoint
it maybe shouldn't. `--refresh` is the manual escape hatch for that case.

## Can it be turned off?

`experiment.lock` verification cannot be disabled -- it runs on every `load_experiment` call, no
flag skips it. If the input files don't match, the run stops.

The plan locks have two explicit, deliberate bypasses, both opt-in per run:

- `--no-checkpoint` -- don't read or write any `plan.<stem>.npz`/`.lock` at all. Preprocessing
  always runs from scratch and nothing is saved. Use this when you don't want checkpointing in the
  loop at all.
- `--refresh` -- force preprocessing to run from scratch and overwrite the existing checkpoints,
  regardless of whether the current locks would have matched. Use this when you know you want new
  checkpoints (e.g. after a genuine, unreleased physics change that a version bump hasn't caught up
  with yet).

Short of those two flags, there's no way to make a stale or mismatched checkpoint get reused
quietly -- a mismatch on any axis, or a corrupted `.npz`, makes the run refuse to reuse that
dataset's checkpoint and fall back to preprocessing it fresh (other datasets' valid checkpoints
still reuse).

Checkpoint files whose dataset left `inputs.exp_data` are pruned on the next checkpointing run. A
bare `plan.npz`/`plan.lock` pair from an older diffBloch has no stem segment, is never read or
pruned, and can be deleted by hand.

In committed examples, plan checkpoints and `.cif_pets` files are Git LFS artifacts; each lock
verifies the realized file bytes after LFS checkout, not the small pointer file.

## `refinement.lock`

`refinement.lock` is the refinement-stage counterpart to the plan locks, written alongside
`refined_structure.cif` whenever a checkpointing `run refine` completes. It chains only to the
plan locks **this run verified or wrote** -- a `--no-checkpoint` run writes no `refinement.lock`
at all, even if lock files from an earlier run are still on disk, because those are not the
provenance of the plans this run actually refined.
Refinement runs on top of already-settled per-dataset `Plan`s, so everything that determines
*those* -- inputs, sample, blochwave, preprocess config, recipe -- is already pinned by the plan
locks this run refined from. `refinement.lock` adds exactly what refinement itself contributes on
top of that:

- **`plan_lock_sha256s`** -- the hashes of the exact `plan.<stem>.lock` files this run refined
  from, in `inputs.exp_data` order. A recorded fact about that run, not a live re-check: it doesn't
  require the plan locks to still be present or still match to be verified.
- **`refinement_config_digest`** -- the hash of the refinement-determining config (`steps`,
  `optimizer`, `trainable`, `thickness_nn`, `split`; everything `dataset_config_digest` excludes
  under `refinement` because it can't change a settled per-dataset `Plan` -- see the config-scope
  split above).
- **`code_version`** -- the build that produced this refinement.
- **`refined_structure`/`refined_parameters`** -- content hashes of `refined_structure.cif` and
  `refined_parameters.npz`, so the outputs themselves are pinned, not just the config that produced
  them.

Unlike the plan locks, `refinement.lock` is not a reuse gate -- there's no "reuse this refined
structure if the config matches" path the way a preprocess checkpoint is reused. It exists purely
as a verifiable provenance record: given a `refined_structure.cif`, `refinement.lock` states
exactly which `Plan`s, which refinement config, and which code version produced it, and lets you
confirm the file bytes haven't changed since.
