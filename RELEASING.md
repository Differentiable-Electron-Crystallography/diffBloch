# Releasing diffBloch

Releases are cut by **pushing a version tag**. Merging to `main` publishes nothing — the release
workflow (`.github/workflows/pypi.yml`) only triggers on `v*` tags, so `main` can move freely
between releases.

Pre-release tags (`v0.3.0rc1`) go to **TestPyPI**; final tags (`v0.3.0`) go to **PyPI**. The
routing is automatic: the workflow classifies the built version with PEP 440 and picks the index.

`main` is protected by the `protected-main` ruleset — pull request required, one approving
review, all review threads resolved, and no bypass actors. The version bump therefore goes
through review like any other change. Tags are not covered by a ruleset, so the tag push itself
is direct.

## Cutting a release

1. **Branch.**

   ```bash
   git switch -c release/0.3.0
   ```

2. **Bump the version.** `__version__` in `src/diffBloch/__init__.py` is the single source of
   truth — hatchling reads it via `[tool.hatch.version]`.

3. **Mirror it in the two hand-maintained citation strings.** Neither is derived, and nothing
   fails if you forget:
   - `README.md` — the Citation section (`diffBloch, version 0.2.0`)
   - `docs/index.md` — the matching Citation section

   (`docs/conf.py` reads the installed version through `importlib.metadata`, so it needs no edit.)

4. **Open the PR and get it approved.**

   ```bash
   git commit -am "diffBloch 0.3.0"
   git push -u origin release/0.3.0
   gh pr create --title "diffBloch 0.3.0" --fill
   ```

5. **Merge it**, then tag the *merged* commit:

   ```bash
   git switch main
   git pull                       # REQUIRED -- see below
   git tag v0.3.0
   git push origin v0.3.0
   ```

   > The ruleset allows only **squash** and **rebase** merges, so the commit that lands on `main`
   > is a new object with a different SHA from your branch tip. Tagging before pulling would
   > point the release at a commit that is not on `main`. The workflow cannot detect this — the
   > version would match and it would publish happily — so the `git pull` is load-bearing.

6. **Watch the `Release` run.** It builds, re-runs the unit tests, verifies the sdist scope,
   checks the tag against the built metadata, and publishes.

Rehearsing first is cheap and strongly recommended for anything non-routine: tag `v0.3.0rc1` off
the same merged commit, let it land on TestPyPI, install it in a clean venv, then cut the real
tag. **A PyPI filename can never be reused**, even after deleting the release, so a botched
upload costs you that version number permanently.

## What the workflow guards

- **Tag vs. metadata.** Because `__version__` is a static string, a tag can disagree with what
  actually ships. The build compares the tag against the *built wheel's* version (normalised
  through PEP 440, so `v0.3.0-rc1` and `v0.3.0rc1` are equal) and fails before publishing.
- **sdist scope.** `[tool.hatch.build.targets.sdist]` limits the sdist to `src/diffBloch` plus
  `README.md`, `LICENSE`, and `REFERENCES.md`. The workflow asserts the resulting file list.
  This matters because `tests/` and `examples/` hold Git LFS files (`*.npz`, `*.cif_pets`); if
  they ever reached the sdist from a pointer-only checkout they would ship as ~130-byte stubs —
  an archive that installs cleanly and then fails at read time.
- **Tests.** Tags can be pushed to any commit without review, so the fast suite runs again before
  anything is uploaded.
- **Renderable metadata.** `twine check --strict` catches a README that would fail to render on
  the project page — unfixable without burning another version number.

It does **not** check that the tagged commit is on `main`; see the warning in step 5.

## Side effect worth knowing

`config/manifest.py` derives `code_version()` from `__version__`, and `_release()` keys
**checkpoint reuse** on it. A version bump therefore invalidates existing checkpoints: it is a
physics/caching event, not only a packaging one. This is also why the project deliberately does
*not* use `hatch-vcs` to derive the version from git tags.

## One-time index setup

Both indexes use **Trusted Publishing** (OIDC). There are no PyPI API tokens in this repo's
secrets, and nothing to rotate.

On <https://pypi.org/manage/account/publishing/>, add a pending publisher:

| Field | Value |
| --- | --- |
| PyPI project name | `diffBloch` |
| Owner | `Differentiable-Electron-Crystallography` |
| Repository name | `diffBloch` |
| Workflow name | `pypi.yml` |
| Environment name | `pypi` |

Repeat on <https://test.pypi.org/manage/account/publishing/> with environment name `testpypi`.

"Pending publisher" is the right form until the first upload — it is what lets the workflow
create a project that does not exist yet. After the first successful release it becomes a normal
publisher on the project.

The `pypi` and `testpypi` GitHub environments are what the OIDC claim is scoped to. Adding
required reviewers to the `pypi` environment turns every production release into a second
approval gate; leave it unprotected for hands-off releases.

## Verifying a build by hand

```bash
uv build
tar -tzf dist/*.tar.gz | grep -Ev '^[^/]+/(src/diffBloch/|README|LICENSE|REFERENCES|PKG-INFO|pyproject|\.gitignore)'
uv run --with twine twine check --strict dist/*

uv venv /tmp/vv && VIRTUAL_ENV=/tmp/vv uv pip install dist/*.whl
cd /tmp && /tmp/vv/bin/diffbloch --help          # console script resolves
/tmp/vv/bin/python -c "import diffBloch; print(diffBloch.__version__)"
```

Running from `/tmp` matters: it is the only way to exercise the no-git fallback in
`code_version()` that every installed user hits.

## Optional backends

`wandb` and `comet_ml` are extras, imported lazily so that neither is needed to
`import diffBloch.app`:

```bash
pip install "diffBloch[wandb]"
pip install "diffBloch[comet]"
```
