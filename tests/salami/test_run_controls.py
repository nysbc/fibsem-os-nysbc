"""Running the experiment: the continuous loop, single stepping, and stopping.

Most of these stub `run_task`, because what is under test is the step state
machine — which step ran, what `run_state` ended up as, whether resume is left
pointing at the right place — not the acquisition itself. The last test does run
a real (simulated) acquisition, to check images actually land on disk.
"""

import os

import pytest

from fibsem.applications.salami import experiment as experiment_module
from fibsem.applications.salami.experiment import (
    SalamiExperiment,
    SalamiExperimentConfig,
)
from fibsem.applications.salami.structures import VolumeAdvanceConfig
from fibsem.applications.salami.tasks import AcquisitionTaskConfig


@pytest.fixture
def protocol() -> SalamiExperimentConfig:
    """Acquisition only — no milling, so nothing here is destructive."""
    return SalamiExperimentConfig(
        name="Acquire Only",
        n_steps=10,
        advance=VolumeAdvanceConfig(step_size=10e-9),
        tasks=[AcquisitionTaskConfig(name="View_Task")],
    )


@pytest.fixture
def experiment(protocol, tmp_path) -> SalamiExperiment:
    exp = SalamiExperiment.create(
        parent_dir=str(tmp_path), name="run", template=protocol
    )
    exp.microscope = object()  # never dereferenced; run_task is stubbed
    exp.status.setup_complete = True
    return exp


@pytest.fixture
def ran(monkeypatch):
    calls = []
    monkeypatch.setattr(
        experiment_module,
        "run_task",
        lambda microscope, cfg, ctx: calls.append(cfg.name),
    )
    return calls


# ---------------------------------------------------------------------------
# Single stepping
# ---------------------------------------------------------------------------


def test_run_next_step_advances_exactly_one_step(experiment, ran):
    assert experiment.run_next_step() is True

    assert experiment.status.current_step == 1
    assert len(ran) == 1
    assert experiment.status.run_state == "paused"


def test_run_next_step_can_be_repeated(experiment, ran):
    for _ in range(3):
        experiment.run_next_step()

    assert experiment.status.current_step == 3
    assert len(ran) == 3


def test_stepping_by_hand_then_handing_over_to_the_loop(experiment, ran):
    """The unit of work is the same either way, so the two must agree."""
    experiment.run_next_step()
    experiment.run_next_step()
    experiment.run()

    assert experiment.status.current_step == 10
    assert len(ran) == 10  # no step run twice, none skipped
    assert experiment.status.run_state == "completed"


def test_last_step_completes_the_experiment(experiment, ran):
    experiment.status.current_step = 9
    experiment.run_next_step()

    assert experiment.status.current_step == 10
    assert experiment.status.run_state == "completed"


def test_run_next_step_is_a_noop_once_complete(experiment, ran):
    experiment.status.current_step = 10

    assert experiment.run_next_step() is False
    assert ran == []


# ---------------------------------------------------------------------------
# The continuous loop
# ---------------------------------------------------------------------------


def test_run_executes_every_remaining_step(experiment, ran):
    experiment.run()

    assert len(ran) == 10
    assert experiment.status.current_step == 10
    assert experiment.status.run_state == "completed"


def test_run_resumes_from_where_it_stopped(experiment, ran):
    experiment.status.current_step = 7
    experiment.run()

    assert len(ran) == 3
    assert experiment.status.current_step == 10


def test_stopping_pauses_without_losing_the_position(experiment, ran, monkeypatch):
    """Stop is checked between tasks, so the run halts at a step boundary."""

    class StopAfter:
        """Stands in for the UI's stop event, tripping after 4 steps."""

        def is_set(self):
            return experiment.status.current_step >= 4

    class FakeUI:
        _workflow_stop_event = StopAfter()
        workflow_update_signal = type(
            "S", (), {"emit": staticmethod(lambda _p: None)}
        )()

    experiment.parent_ui = FakeUI()
    experiment.run()

    assert experiment.status.current_step == 4
    assert experiment.status.run_state == "paused"
    assert len(ran) == 4
    # Resume picks up cleanly rather than repeating step 4.
    assert experiment.status.next_task_index == 0


def test_a_failing_task_marks_the_run_failed(experiment, monkeypatch):
    def boom(microscope, cfg, ctx):
        raise RuntimeError("detector fell over")

    monkeypatch.setattr(experiment_module, "run_task", boom)

    with pytest.raises(RuntimeError, match="detector fell over"):
        experiment.run()

    assert experiment.status.run_state == "failed"
    # in_flight_task is left set: the operator has to say what happened to it.
    assert experiment.has_interrupted_task


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_run_refuses_without_a_microscope(experiment, ran):
    experiment.microscope = None

    with pytest.raises(RuntimeError, match="no microscope"):
        experiment.run()
    with pytest.raises(RuntimeError, match="no microscope"):
        experiment.run_next_step()


def test_run_next_step_refuses_while_a_task_is_unresolved(experiment, ran):
    experiment.status.in_flight_task = "View_Task"

    with pytest.raises(RuntimeError, match="interrupted"):
        experiment.run_next_step()
    assert ran == []


def test_is_complete_tracks_the_step_count(experiment):
    assert not experiment.is_complete
    experiment.status.current_step = 10
    assert experiment.is_complete


# ---------------------------------------------------------------------------
# End to end, with a real acquisition
# ---------------------------------------------------------------------------


def test_running_saves_one_image_per_step(tmp_path):
    """The milestone: press run, get one image per slice on disk."""
    from fibsem import utils

    scope, _settings = utils.setup_session(manufacturer="Demo")

    task = AcquisitionTaskConfig(name="View_Task")
    # Auto-functions add seconds per step and are not what this is testing.
    task.autofocus.enabled = False
    task.autocontrast.enabled = False

    protocol = SalamiExperimentConfig(
        name="Acquire Only",
        n_steps=3,
        advance=VolumeAdvanceConfig(step_size=10e-9),
        tasks=[task],
    )
    exp = SalamiExperiment.create(
        parent_dir=str(tmp_path), name="e2e", template=protocol, microscope=scope
    )
    exp.status.setup_complete = True

    exp.run()

    images = sorted(os.listdir(os.path.join(exp.path, "View_Task")))
    assert images == [
        "View_Task-Image-000.tif",
        "View_Task-Image-001.tif",
        "View_Task-Image-002.tif",
    ]
    assert exp.status.run_state == "completed"
