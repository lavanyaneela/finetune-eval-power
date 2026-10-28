"""Properties of the committed measurement: its shape invariants and its hashes."""

from __future__ import annotations

import json

import numpy as np
import pytest
from tests.conftest import build_synthetic_runs

from finetune_eval_power.errors import ToolError
from finetune_eval_power.runs import ARRAY_NAMES, MANIFEST_NAME, SweepRuns, load_runs


def test_a_measurement_reports_the_case_counts_of_its_arrays(synthetic_runs):
    assert synthetic_runs.suite_size == synthetic_runs.suite_before.shape[2]
    assert synthetic_runs.probe_size == synthetic_runs.probe_before.shape[2]


def test_depth_and_seed_lookups_return_positions_in_the_stored_order(synthetic_runs):
    assert synthetic_runs.depth_index(synthetic_runs.freeze_depths[2]) == 2
    assert synthetic_runs.seed_index(synthetic_runs.seeds[1]) == 1


def test_an_unknown_freeze_depth_is_refused_by_name(synthetic_runs):
    with pytest.raises(ToolError, match="freeze depth 99"):
        synthetic_runs.depth_index(99)


def test_an_unknown_seed_is_refused_by_name(synthetic_runs):
    with pytest.raises(ToolError, match="seed 12345"):
        synthetic_runs.seed_index(12345)


def _rebuild(runs: SweepRuns, **overrides) -> SweepRuns:
    payload = {name: getattr(runs, name) for name in ARRAY_NAMES}
    payload.update(overrides)
    return SweepRuns(
        arm=runs.arm,
        seeds=runs.seeds,
        freeze_depths=runs.freeze_depths,
        manifest=runs.manifest,
        **payload,
    )


def test_a_non_boolean_correctness_array_is_refused(synthetic_runs):
    floats = synthetic_runs.suite_before.astype(np.float32)
    with pytest.raises(ToolError, match="boolean correctness"):
        _rebuild(synthetic_runs, suite_before=floats)


def test_an_array_whose_leading_shape_disagrees_with_the_seeds_is_refused(synthetic_runs):
    trimmed = synthetic_runs.suite_before[:-1]
    with pytest.raises(ToolError, match="expected"):
        _rebuild(synthetic_runs, suite_before=trimmed)


def test_before_and_after_must_align_case_for_case(synthetic_runs):
    shorter = synthetic_runs.suite_after[:, :, :-1]
    with pytest.raises(ToolError, match="align case for case"):
        _rebuild(synthetic_runs, suite_after=shorter)


def test_probe_before_and_after_must_align_case_for_case(synthetic_runs):
    shorter = synthetic_runs.probe_after[:, :, :-1]
    with pytest.raises(ToolError, match="align case for case"):
        _rebuild(synthetic_runs, probe_after=shorter)


def test_a_curve_with_the_wrong_leading_shape_is_refused(synthetic_runs):
    with pytest.raises(ToolError, match="suite_curve has shape"):
        _rebuild(synthetic_runs, suite_curve=np.ones((1, 1, 3), dtype=np.float32))


def test_writing_then_loading_returns_the_same_arrays(tmp_path, synthetic_runs):
    synthetic_runs.write(tmp_path)
    loaded = load_runs(tmp_path)
    for name in ARRAY_NAMES:
        assert np.array_equal(getattr(loaded, name), getattr(synthetic_runs, name)), name


def test_writing_records_a_hash_for_every_array(tmp_path, synthetic_runs):
    hashes = synthetic_runs.write(tmp_path)
    assert set(hashes) == {f"{name}.npy" for name in ARRAY_NAMES}
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert manifest["array_sha256"] == hashes


def test_the_manifest_records_the_shapes_it_wrote(tmp_path, synthetic_runs):
    synthetic_runs.write(tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert manifest["shapes"]["suite_before"] == list(synthetic_runs.suite_before.shape)


def test_writing_the_same_measurement_twice_produces_identical_bytes(tmp_path, vision_arm):
    """Byte reproducibility is why the arrays are separate .npy files.

    A single .npz would embed a modification time in its zip container and the
    fixture would differ from itself on the next write, which makes a fixture diff
    in CI useless.
    """
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_synthetic_runs(vision_arm).write(first)
    build_synthetic_runs(vision_arm).write(second)
    for name in ARRAY_NAMES:
        assert (first / f"{name}.npy").read_bytes() == (second / f"{name}.npy").read_bytes()
    assert (first / MANIFEST_NAME).read_bytes() == (second / MANIFEST_NAME).read_bytes()


def test_a_tampered_array_fails_its_hash_check(tmp_path, synthetic_runs):
    synthetic_runs.write(tmp_path)
    tampered = np.load(tmp_path / "suite_after.npy")
    tampered[0, 0, 0] = ~tampered[0, 0, 0]
    np.save(tmp_path / "suite_after.npy", tampered, allow_pickle=False)
    with pytest.raises(ToolError, match="does not match its manifest hash"):
        load_runs(tmp_path)


def test_hash_checking_can_be_turned_off_for_a_deliberately_edited_fixture(
    tmp_path, synthetic_runs
):
    synthetic_runs.write(tmp_path)
    tampered = np.load(tmp_path / "suite_after.npy")
    tampered[0, 0, 0] = ~tampered[0, 0, 0]
    np.save(tmp_path / "suite_after.npy", tampered, allow_pickle=False)
    assert load_runs(tmp_path, verify_hashes=False).arm == synthetic_runs.arm


def test_a_missing_manifest_is_refused_with_the_path(tmp_path):
    with pytest.raises(ToolError, match="no measurement manifest"):
        load_runs(tmp_path)


def test_a_missing_array_is_refused_with_the_path(tmp_path, synthetic_runs):
    synthetic_runs.write(tmp_path)
    (tmp_path / "probe_after.npy").unlink()
    with pytest.raises(ToolError, match="probe_after.npy is missing"):
        load_runs(tmp_path)


def test_a_manifest_that_hashes_nothing_is_refused(tmp_path, synthetic_runs):
    synthetic_runs.write(tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    del manifest["array_sha256"]
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(ToolError, match="does not hash"):
        load_runs(tmp_path)
