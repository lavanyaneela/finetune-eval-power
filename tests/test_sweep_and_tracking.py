"""The sweep driver and the tracking wrapper, exercised on a toy configuration.

The sweep is the only module that trains, so it is tested at a size that finishes in
seconds rather than re-measuring the committed fixture. What matters here is the
shape of what it produces and the ordering guarantee that cost a whole capture run
when it was absent.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from finetune_eval_power.config import SWEEP_SEEDS, ArmConfig
from finetune_eval_power.runs import ARRAY_NAMES, load_runs
from finetune_eval_power.sweep import _build_manifest, run_sweep

# Collection of this module is skipped entirely when the training extra is
# absent; see pytest_ignore_collect in conftest.py.
pytestmark = pytest.mark.slow


@pytest.fixture
def toy_arm(small_arm: ArmConfig) -> ArmConfig:
    return dataclasses.replace(small_arm, freeze_depths=(0, 4))


def test_a_sweep_produces_one_row_per_seed_and_depth(tmp_path, toy_arm):
    runs = run_sweep(toy_arm, (3, 5), tracking_directory=tmp_path / "mlruns")
    assert runs.suite_before.shape[:2] == (2, len(toy_arm.freeze_depths))
    assert runs.probe_after.shape[:2] == (2, len(toy_arm.freeze_depths))


def test_a_swept_measurement_round_trips_through_disk(tmp_path, toy_arm):
    runs = run_sweep(toy_arm, (3, 5), tracking_directory=tmp_path / "mlruns")
    runs.write(tmp_path / "out")
    loaded = load_runs(tmp_path / "out")
    assert loaded.seeds == (3, 5)
    assert set(loaded.manifest["array_sha256"]) == {f"{name}.npy" for name in ARRAY_NAMES}


def test_the_manifest_records_which_slices_the_suite_never_samples(toy_arm):
    from finetune_eval_power.models.training import build_fixed_cases

    manifest = _build_manifest(toy_arm, build_fixed_cases(toy_arm))
    assert manifest["held_out_slices"]
    assert set(manifest["held_out_slices"]) <= set(manifest["slice_names"])


def test_the_manifest_counts_suite_cases_for_every_named_slice(toy_arm):
    from finetune_eval_power.models.training import build_fixed_cases

    manifest = _build_manifest(toy_arm, build_fixed_cases(toy_arm))
    assert set(manifest["suite_slice_case_counts"]) == set(manifest["slice_names"])
    assert all(
        count == 0
        for name, count in manifest["suite_slice_case_counts"].items()
        if name in manifest["held_out_slices"]
    )


def test_the_manifest_is_json_serialisable_before_any_training_happens(toy_arm):
    """Pins a real incident. The manifest used to be built after the sweep, a
    refactor renamed an attribute it reads, and the failure surfaced twelve minutes
    of training later with nothing written. Anything that can fail cheaply has to
    fail before anything expensive starts."""
    from finetune_eval_power.models.training import build_fixed_cases

    manifest = _build_manifest(toy_arm, build_fixed_cases(toy_arm))
    assert json.loads(json.dumps(manifest))["config"]["name"] == toy_arm.name


def test_a_manifest_that_cannot_serialise_fails_before_the_first_epoch(
    monkeypatch, tmp_path, toy_arm
):
    """The second half of the same lesson: if the cheap step is broken, no epoch
    should run at all. A sentinel in the training function proves none did."""
    import finetune_eval_power.sweep as sweep_module

    trained: list[int] = []

    def record(arm, seed, cases):  # pragma: no cover - must never be reached
        trained.append(seed)
        raise AssertionError("training ran despite a broken manifest")

    def broken(arm, cases):
        return {"unserialisable": object()}

    monkeypatch.setattr(sweep_module, "_build_manifest", broken)
    monkeypatch.setattr("finetune_eval_power.models.training.train_seed", record, raising=True)
    with pytest.raises(TypeError):
        run_sweep(toy_arm, (3,), tracking_directory=tmp_path / "mlruns")
    assert trained == []


def test_the_declared_sweep_seeds_are_distinct():
    assert len(set(SWEEP_SEEDS)) == len(SWEEP_SEEDS)


def test_tracking_writes_a_local_store_when_mlflow_is_installed(tmp_path):
    pytest.importorskip("mlflow", reason="needs the training extra")
    from finetune_eval_power.tracking import open_tracker

    tracker = open_tracker(tmp_path / "mlruns")
    assert tracker.enabled
    with tracker.run("a-run", {"arm": "toy", "seed": 1}) as handle:
        handle.log_metrics({"suite_accuracy_before": 0.5})
    assert (tmp_path / "mlruns" / "tracking.db").exists()


def test_a_tracking_failure_never_breaks_the_run(tmp_path):
    """Observability that can take down the thing it observes is a liability."""
    from finetune_eval_power.tracking import SweepTracker

    class Hostile:
        def start_run(self, **_kwargs):
            raise RuntimeError("backend down")

        def log_params(self, _params):
            raise RuntimeError("backend down")

        def log_metrics(self, _metrics):
            raise RuntimeError("backend down")

    tracker = SweepTracker(Hostile())
    with tracker.run("a-run", {"x": 1}) as handle:
        handle.log_metrics({"y": 2.0})


def test_a_broken_tracking_backend_disables_itself_rather_than_raising(monkeypatch, tmp_path):
    import finetune_eval_power.tracking.mlflow_local as module

    class Broken:
        @staticmethod
        def set_tracking_uri(_uri):
            raise RuntimeError("no backend")

    monkeypatch.setitem(__import__("sys").modules, "mlflow", Broken)
    assert not module.open_tracker(tmp_path / "mlruns").enabled
