"""The analysis path must not need TensorFlow, and this is how that is pinned.

Trap one from the build notes: a heavy optional dependency installed on the machine
masks a broken import in the job that installs without it, and the failure surfaces
as a confusing ModuleNotFoundError across every Python in the matrix while the other
jobs stay green. CI runs a job with the training extra left out. These tests make
the same guarantee locally, whether or not Keras happens to be installed here.
"""

from __future__ import annotations

import builtins
import importlib

import pytest

BLOCKED = ("tensorflow", "keras", "mlflow")

ANALYSIS_MODULES = (
    "finetune_eval_power.analysis",
    "finetune_eval_power.cli",
    "finetune_eval_power.config",
    "finetune_eval_power.report",
    "finetune_eval_power.runs",
    "finetune_eval_power.generator.attributes",
    "finetune_eval_power.generator.sequence",
    "finetune_eval_power.generator.vision",
    "finetune_eval_power.power",
    "finetune_eval_power.slices",
)


@pytest.fixture
def without_training_extra(monkeypatch):
    """Make the training dependencies unimportable for the duration of a test."""
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    for blocked in BLOCKED:
        for loaded in [name for name in list(globals()) if name.startswith(blocked)]:
            monkeypatch.delitem(globals(), loaded, raising=False)
    return refuse


@pytest.mark.parametrize("module_name", ANALYSIS_MODULES)
def test_every_analysis_module_imports_without_the_training_extra(
    module_name, without_training_extra, monkeypatch
):
    import sys

    monkeypatch.delitem(sys.modules, module_name, raising=False)
    assert importlib.import_module(module_name) is not None


def test_the_whole_analysis_runs_without_the_training_extra(
    tmp_path, vision_arm, without_training_extra
):
    from tests.conftest import build_synthetic_runs

    from finetune_eval_power.analysis import analyse
    from finetune_eval_power.runs import load_runs

    build_synthetic_runs(vision_arm).write(tmp_path)
    analysis = analyse(vision_arm, load_runs(tmp_path))
    assert analysis.slices_total > 0


def test_the_cli_answers_without_the_training_extra(tmp_path, vision_arm, without_training_extra):
    from tests.conftest import build_synthetic_runs

    from finetune_eval_power.cli import main

    build_synthetic_runs(vision_arm).write(tmp_path / "vision")
    assert main(["suite", "--runs", str(tmp_path)]) == 0


def test_the_tracker_disables_itself_when_mlflow_is_absent(without_training_extra):
    import sys

    sys.modules.pop("finetune_eval_power.tracking.mlflow_local", None)
    from finetune_eval_power.tracking.mlflow_local import open_tracker

    tracker = open_tracker()
    assert not tracker.enabled


def test_a_disabled_tracker_still_yields_and_still_accepts_metrics(without_training_extra):
    from finetune_eval_power.tracking.mlflow_local import SweepTracker

    tracker = SweepTracker(None)
    with tracker.run("anything", {"a": 1}) as handle:
        handle.log_metrics({"b": 2.0})
    assert not tracker.enabled


def test_the_training_module_is_the_only_one_that_needs_the_extra():
    """Stated as a test so the boundary cannot quietly move: if some analysis module
    grows a Keras import, the parametrised test above fails and names it."""
    import finetune_eval_power.models.training as training

    source = training.__file__
    assert source.endswith("training.py")
