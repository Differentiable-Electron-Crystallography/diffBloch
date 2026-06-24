"""Experiment locks and run manifests keep input identity separate from generated artifacts."""

import tarfile
from pathlib import Path
from zipfile import ZipFile

import pytest

from diffBloch.config import (
    RunManifest,
    artifact_hash_for,
    load_experiment,
    pack_run,
    sha256_file,
    write_run_manifest,
)

LOCKED = Path(__file__).parent.parent / "fixtures" / "locked_min"


def test_load_experiment_verifies_locked_input_bytes() -> None:
    cfg, lock = load_experiment(LOCKED)
    assert cfg.name == "locked-min"
    assert lock.structure.ref == "enantiomer_1.cif"
    assert lock.structure.sha256 == sha256_file(LOCKED / "enantiomer_1.cif")


def test_load_experiment_detects_input_drift(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    for name in ["experiment.yaml", "experiment.lock", "enantiomer_1.cif", "exp_data.cif_pets"]:
        (experiment / name).write_bytes((LOCKED / name).read_bytes())
    (experiment / "enantiomer_1.cif").write_text("changed\n")

    with pytest.raises(ValueError, match="input drift"):
        load_experiment(experiment)


def test_run_manifest_hashes_generated_artifacts(tmp_path: Path) -> None:
    run = tmp_path / "run_001"
    run.mkdir()
    for name in [
        "resolved_config.yaml",
        "config.diff.yaml",
        "data_used.parquet",
        "history.jsonl",
        "parameter_table.parquet",
    ]:
        (run / name).write_text(f"{name}\n")

    manifest = RunManifest(
        experiment_lock_sha256="input-lock-hash",
        resolved_config=artifact_hash_for(run / "resolved_config.yaml", root=run),
        config_diff=artifact_hash_for(run / "config.diff.yaml", root=run),
        data_used=artifact_hash_for(run / "data_used.parquet", root=run),
        history=artifact_hash_for(run / "history.jsonl", root=run),
        parameter_table=artifact_hash_for(run / "parameter_table.parquet", root=run),
        code_version="test",
        environment={"python": "test"},
    )
    write_run_manifest(run / "run_manifest.json", manifest)

    loaded = RunManifest.model_validate_json((run / "run_manifest.json").read_text())
    assert loaded.resolved_config.path == "resolved_config.yaml"
    assert loaded.experiment_lock_sha256 == "input-lock-hash"


def test_pack_run_exports_zip_preserving_full_run_name(tmp_path: Path) -> None:
    run = tmp_path / "run.v2"
    run.mkdir()
    (run / "run_manifest.json").write_text("{}\n")
    (run / "history.jsonl").write_text("{}\n")

    package = pack_run(run, format="zip")
    assert package.is_file()
    assert package.name == "run.v2.zip"
    with ZipFile(package) as archive:
        assert "run.v2/run_manifest.json" in archive.namelist()


@pytest.mark.parametrize(
    ("format", "expected_name"),
    [
        ("tar", "run_001.tar"),
        ("bagit", "run_001.bagit.zip"),
        ("ro-crate", "run_001.ro-crate.zip"),
    ],
)
def test_pack_run_export_formats(tmp_path: Path, format: str, expected_name: str) -> None:
    run = tmp_path / "run_001"
    run.mkdir()
    (run / "run_manifest.json").write_text("{}\n")
    (run / "history.jsonl").write_text("{}\n")

    package = pack_run(run, format=format)
    assert package.name == expected_name
    if format == "tar":
        with tarfile.open(package) as archive:
            assert "run_001/run_manifest.json" in archive.getnames()
    else:
        with ZipFile(package) as archive:
            names = archive.namelist()
        assert any(name.endswith("run_manifest.json") for name in names)


def test_pack_run_requires_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pack_run(tmp_path / "missing_manifest")
