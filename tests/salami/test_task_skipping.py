"""Skipping tasks mid-run, without disturbing the task list.

Two controls:
  - `enabled` (protocol)  -- off from now on, survives a resume
  - `skip_once` (status)  -- skip the next occurrence only

Both deliberately leave `config.tasks` untouched, so `status.next_task_index`
stays valid and resume is unaffected.
"""

import pytest

from fibsem.applications.salami import experiment as experiment_module
from fibsem.applications.salami.experiment import (
    CONFIG_FILENAME,
    STATUS_FILENAME,
    SalamiExperiment,
    SalamiExperimentConfig,
    SalamiExperimentStatus,
)
from fibsem.applications.salami.structures import RunSchedule, VolumeAdvanceConfig
from fibsem.applications.salami.tasks import AcquisitionTaskConfig, MillingTaskConfig


@pytest.fixture
def protocol() -> SalamiExperimentConfig:
    return SalamiExperimentConfig(
        name="Test Protocol",
        n_steps=10,
        advance=VolumeAdvanceConfig(step_size=10e-9),
        tasks=[
            MillingTaskConfig(name="Slice_Task"),
            AcquisitionTaskConfig(name="View_Task"),
        ],
    )


@pytest.fixture
def experiment(protocol, tmp_path) -> SalamiExperiment:
    exp = SalamiExperiment.create(
        parent_dir=str(tmp_path), name="exp", template=protocol
    )
    exp.microscope = object()  # never dereferenced; run_task is stubbed
    return exp


@pytest.fixture
def ran(monkeypatch):
    """Record which task configs the run loop actually executed."""
    calls = []
    monkeypatch.setattr(
        experiment_module,
        "run_task",
        lambda microscope, cfg, ctx: calls.append(cfg.name),
    )
    return calls


# ---------------------------------------------------------------------------
# enabled
# ---------------------------------------------------------------------------


def test_disabled_task_is_not_run(experiment, ran):
    experiment.set_task_enabled("Slice_Task", False)

    experiment._run_step(0)

    assert ran == ["View_Task"]


def test_disabled_task_still_advances_the_resume_index(experiment, ran):
    """Otherwise resume would re-enter the step at the disabled task forever."""
    experiment.set_task_enabled("Slice_Task", False)

    experiment._run_step(0)

    assert experiment.status.next_task_index == 2


def test_enabled_survives_a_reload(experiment):
    experiment.set_task_enabled("View_Task", False)

    reopened = SalamiExperiment.open(experiment.path)

    assert reopened.find_task("View_Task").enabled is False
    assert reopened.find_task("Slice_Task").enabled is True


def test_enabled_is_reset_when_reused_on_a_new_sample(experiment, tmp_path):
    """A detector fault on this sample shouldn't silently disable tomorrow's run."""
    experiment.set_task_enabled("View_Task", False)

    template = SalamiExperimentConfig.load(experiment.path + "/" + CONFIG_FILENAME)
    reused = SalamiExperiment.create(
        parent_dir=str(tmp_path), name="next-sample", template=template
    )

    assert all(t.enabled for t in reused.config.tasks)


def test_old_configs_without_the_flag_default_to_enabled():
    data = MillingTaskConfig(name="T").to_dict()
    del data["enabled"]
    assert MillingTaskConfig.from_dict(data).enabled is True


def test_disabling_does_not_disturb_the_schedule(experiment):
    task = experiment.find_task("Slice_Task")
    task.schedule = RunSchedule(every_n_steps=5)

    experiment.set_task_enabled("Slice_Task", False)
    experiment.set_task_enabled("Slice_Task", True)

    # the cadence is intact; using schedule.end_step as an off-switch would
    # have destroyed it
    assert task.schedule.every_n_steps == 5
    assert task.should_run(0) and not task.should_run(1)


# ---------------------------------------------------------------------------
# skip once
# ---------------------------------------------------------------------------


def test_skip_once_skips_exactly_one_occurrence(experiment, ran):
    experiment.skip_task_once("Slice_Task")

    experiment._run_step(0)
    assert ran == ["View_Task"]

    experiment.status.next_task_index = 0
    experiment._run_step(1)
    assert ran == ["View_Task", "Slice_Task", "View_Task"]


def test_skip_once_is_consumed_and_persisted(experiment, ran):
    experiment.skip_task_once("Slice_Task")
    experiment._run_step(0)

    assert experiment.status.skip_once == []
    # committed to disk, so a crash here doesn't skip a second time
    assert (
        SalamiExperimentStatus.load(experiment.path + "/" + STATUS_FILENAME).skip_once
        == []
    )


def test_pending_skip_survives_a_crash_before_the_task_ran(experiment):
    experiment.skip_task_once("Slice_Task")

    reopened = SalamiExperiment.open(experiment.path)

    assert reopened.status.skip_once == ["Slice_Task"]


def test_skip_once_is_not_consumed_when_the_task_was_not_scheduled(experiment, ran):
    """ "Skip the next one" means the next actual run, not the next step."""
    experiment.find_task("Slice_Task").schedule = RunSchedule(every_n_steps=2)
    experiment.skip_task_once("Slice_Task")

    experiment._run_step(1)  # Slice_Task not scheduled on odd steps
    assert ran == ["View_Task"]
    assert experiment.status.skip_once == ["Slice_Task"]  # still pending

    experiment.status.next_task_index = 0
    experiment._run_step(2)  # now it would have run, so the skip is spent
    assert experiment.status.skip_once == []


def test_skip_once_advances_the_resume_index(experiment, ran):
    experiment.skip_task_once("Slice_Task")
    experiment._run_step(0)
    assert experiment.status.next_task_index == 2


def test_cancel_skip_once(experiment, ran):
    experiment.skip_task_once("Slice_Task")
    experiment.cancel_skip_once("Slice_Task")

    experiment._run_step(0)

    assert ran == ["Slice_Task", "View_Task"]


def test_skip_once_is_idempotent(experiment):
    experiment.skip_task_once("Slice_Task")
    experiment.skip_task_once("Slice_Task")
    assert experiment.status.skip_once == ["Slice_Task"]


def test_unknown_task_names_are_rejected(experiment):
    with pytest.raises(KeyError, match="No task named"):
        experiment.skip_task_once("Nope")
    with pytest.raises(KeyError, match="No task named"):
        experiment.set_task_enabled("Nope", False)


def test_skip_once_does_not_leak_onto_a_new_sample(experiment, tmp_path):
    """It lives in status, and a new experiment gets a fresh status."""
    experiment.skip_task_once("Slice_Task")

    template = SalamiExperimentConfig.load(experiment.path + "/" + CONFIG_FILENAME)
    reused = SalamiExperiment.create(
        parent_dir=str(tmp_path), name="next-sample", template=template
    )

    assert reused.status.skip_once == []


# ---------------------------------------------------------------------------
# UI summary
# ---------------------------------------------------------------------------


def test_task_states_reports_both_controls(experiment):
    experiment.set_task_enabled("Slice_Task", False)
    experiment.skip_task_once("View_Task")

    states = {s["name"]: s for s in experiment.task_states()}

    assert states["Slice_Task"]["enabled"] is False
    assert states["Slice_Task"]["runs_this_step"] is False
    assert states["View_Task"]["skip_next"] is True
    assert states["View_Task"]["enabled"] is True
