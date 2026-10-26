"""Lifecycle tests for the Salami experiment: create, open, reuse, resume.

These cover the two scenarios the app has to support:
  1. run an experiment, crash or stop, resume it correctly
  2. reuse a protocol on a different sample without touching the original

No microscope is involved — this is all config/status handling.
"""

import os

import pytest
import yaml

from fibsem.applications.salami.experiment import (
    CONFIG_FILENAME,
    STATUS_FILENAME,
    SalamiExperiment,
    SalamiExperimentConfig,
    SalamiExperimentStatus,
)
from fibsem.applications.salami.structures import VolumeAdvanceConfig
from fibsem.applications.salami.tasks import (
    AcquisitionTaskConfig,
    MillingTaskConfig,
    runtime_task_config,
)

STEP_SIZE = 10e-9  # 10 nm slices


@pytest.fixture
def protocol() -> SalamiExperimentConfig:
    """A two-task protocol: mill a slice, then image the face."""
    return SalamiExperimentConfig(
        name="Test Protocol",
        description="two-task slice and view",
        n_steps=100,
        advance=VolumeAdvanceConfig(step_size=STEP_SIZE),
        tasks=[
            MillingTaskConfig(name="Slice_Task"),
            AcquisitionTaskConfig(name="View_Task"),
        ],
    )


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------


def test_config_round_trips_through_yaml(protocol, tmp_path):
    protocol.path = str(tmp_path)
    protocol.save()

    loaded = SalamiExperimentConfig.load(str(tmp_path / CONFIG_FILENAME))

    assert loaded.name == protocol.name
    assert loaded.n_steps == protocol.n_steps
    assert loaded.advance.step_size == pytest.approx(STEP_SIZE)
    assert [t.name for t in loaded.tasks] == ["Slice_Task", "View_Task"]


def test_load_derives_path_from_file_location(protocol, tmp_path):
    """A config carrying someone else's absolute path must not send us there."""
    protocol.path = r"C:\Users\someone-else\Documents\SALAMI"
    target = tmp_path / CONFIG_FILENAME
    with open(target, "w") as f:
        yaml.dump(protocol.to_dict(), f)

    loaded = SalamiExperimentConfig.load(str(target))

    assert loaded.path == str(tmp_path)


def test_volume_depth_matches_slices_times_thickness(protocol):
    protocol.n_steps = 5000
    protocol.advance.step_size = 10e-9
    assert protocol.volume_depth == pytest.approx(50e-6)


# ---------------------------------------------------------------------------
# Scenario 2: reuse a protocol on a new sample
# ---------------------------------------------------------------------------


def test_create_from_template_does_not_touch_the_source(protocol, tmp_path):
    """Reusing a protocol must leave the original experiment's files alone."""
    source = tmp_path / "original-experiment"
    source.mkdir()
    protocol.path = str(source)
    protocol.save()
    # the original is part-way through a run
    SalamiExperimentStatus(current_step=317, setup_complete=True).save(str(source))

    before = (source / STATUS_FILENAME).read_text()

    template = SalamiExperimentConfig.load(str(source / CONFIG_FILENAME))
    SalamiExperiment.create(
        parent_dir=str(tmp_path), name="new-sample", template=template
    )

    # the original status is untouched: still resumable at step 317
    assert (source / STATUS_FILENAME).read_text() == before
    assert (
        SalamiExperimentStatus.load(str(source / STATUS_FILENAME)).current_step == 317
    )


def test_create_from_template_starts_a_fresh_run(protocol, tmp_path):
    protocol.path = str(tmp_path / "src")
    os.makedirs(protocol.path)
    protocol.save()
    SalamiExperimentStatus(current_step=317, setup_complete=True).save(protocol.path)

    template = SalamiExperimentConfig.load(os.path.join(protocol.path, CONFIG_FILENAME))
    experiment = SalamiExperiment.create(
        parent_dir=str(tmp_path), name="new-sample", template=template
    )

    assert experiment.status.current_step == 0
    assert experiment.status.setup_complete is False
    assert experiment.status.run_state == "created"
    assert experiment.config.name == "new-sample"
    assert experiment.path == str(tmp_path / "new-sample")


def test_as_template_strips_sample_and_machine_state(protocol, tmp_path):
    """A reused protocol must not carry the old sample's position or paths."""
    from fibsem.structures import FibsemStagePosition

    acq = protocol.tasks[1]
    acq.state.stage_position = FibsemStagePosition(x=1e-3, y=2e-3, z=3e-3)
    acq.image.path = r"C:\old-sample\View_Task"

    template = protocol.as_template(name="reused", path=str(tmp_path))

    assert template.tasks[1].state.stage_position is None
    assert template.tasks[1].image.path is None
    # the source is untouched
    assert protocol.tasks[1].state.stage_position is not None


def test_create_refuses_to_clobber_an_existing_experiment(protocol, tmp_path):
    SalamiExperiment.create(parent_dir=str(tmp_path), name="exp", template=protocol)

    with pytest.raises(FileExistsError):
        SalamiExperiment.create(parent_dir=str(tmp_path), name="exp", template=protocol)


# ---------------------------------------------------------------------------
# Scenario 1: resume
# ---------------------------------------------------------------------------


def _milled_y(pattern) -> float:
    """Where the pattern's shape actually sits, as the milling engine sees it.

    Asserting on the config field instead would miss the failure this guards
    against: `LinePattern.define()` ignores `point`, so a pattern can report a
    moved position while still cutting the same plane.
    """
    shape = pattern.define()[0]
    # Explicit None check: centre_y is legitimately 0.0 at step 0, and `or`
    # would fall through to start_y, which a rectangle does not have.
    centre_y = getattr(shape, "centre_y", None)
    return centre_y if centre_y is not None else shape.start_y


def test_advance_is_derived_from_step_not_accumulated(protocol, tmp_path):
    """Geometry at step N depends only on N, so resuming lands at the right depth."""
    milling = protocol.tasks[0]
    baseline_y = _milled_y(milling.milling.stages[0].pattern)

    at_300 = runtime_task_config(milling, str(tmp_path), 300, protocol.advance)
    assert _milled_y(at_300.milling.stages[0].pattern) == pytest.approx(
        baseline_y + 300 * STEP_SIZE
    )

    # and the stored protocol is still the step-0 baseline
    assert _milled_y(milling.milling.stages[0].pattern) == pytest.approx(baseline_y)

    # deriving twice gives the same answer (the old cumulative version did not)
    again = runtime_task_config(milling, str(tmp_path), 300, protocol.advance)
    assert _milled_y(again.milling.stages[0].pattern) == pytest.approx(
        _milled_y(at_300.milling.stages[0].pattern)
    )


def test_the_milled_shape_moves_not_just_the_stored_position(protocol, tmp_path):
    """Regression: the advance used to write only to `pattern.point`.

    The default slice is a LinePattern, whose define() ignores `point` — so the
    position advanced correctly while every slice was milled in one plane and no
    volume was collected.
    """
    milling = protocol.tasks[0]
    at_0 = runtime_task_config(milling, str(tmp_path), 0, protocol.advance)
    at_100 = runtime_task_config(milling, str(tmp_path), 100, protocol.advance)

    moved = _milled_y(at_100.milling.stages[0].pattern) - _milled_y(
        at_0.milling.stages[0].pattern
    )
    assert moved == pytest.approx(100 * STEP_SIZE)


def test_advance_applies_to_every_milling_stage(protocol, tmp_path):
    """A coarse+fine slice pair has to stay on the same plane."""
    from copy import deepcopy

    milling = protocol.tasks[0]
    milling.milling.stages.append(deepcopy(milling.milling.stages[0]))
    milling.milling.stages[1].name = "Fine Slice"

    baseline = _milled_y(milling.milling.stages[0].pattern)
    advanced = runtime_task_config(milling, str(tmp_path), 50, protocol.advance)
    ys = [_milled_y(s.pattern) for s in advanced.milling.stages]

    assert ys[0] == pytest.approx(ys[1])
    assert ys[0] == pytest.approx(baseline + 50 * STEP_SIZE)


def test_open_restores_step_and_leaves_protocol_untouched(protocol, tmp_path):
    experiment = SalamiExperiment.create(
        parent_dir=str(tmp_path), name="exp", template=protocol
    )
    experiment.status.current_step = 42
    experiment.status.setup_complete = True
    experiment.status.save(experiment.path)

    config_before = (tmp_path / "exp" / CONFIG_FILENAME).read_text()
    reopened = SalamiExperiment.open(experiment.path)

    assert reopened.status.current_step == 42
    assert reopened.status.setup_complete is True
    assert (tmp_path / "exp" / CONFIG_FILENAME).read_text() == config_before


def test_open_accepts_a_file_inside_the_experiment(protocol, tmp_path):
    experiment = SalamiExperiment.create(
        parent_dir=str(tmp_path), name="exp", template=protocol
    )
    reopened = SalamiExperiment.open(os.path.join(experiment.path, STATUS_FILENAME))
    assert reopened.config.name == "exp"


def test_open_rejects_a_directory_with_no_experiment(tmp_path):
    with pytest.raises(FileNotFoundError):
        SalamiExperiment.open(str(tmp_path))


# ---------------------------------------------------------------------------
# Interrupted destructive tasks
# ---------------------------------------------------------------------------


def test_interrupted_task_is_detected_on_open(protocol, tmp_path):
    experiment = SalamiExperiment.create(
        parent_dir=str(tmp_path), name="exp", template=protocol
    )
    experiment.status.current_step = 10
    experiment.status.next_task_index = 0
    experiment.status.in_flight_task = "Slice_Task"
    experiment.status.save(experiment.path)

    reopened = SalamiExperiment.open(experiment.path)
    assert reopened.has_interrupted_task
    assert reopened.status.in_flight_task == "Slice_Task"


def test_run_refuses_while_a_task_is_unresolved(protocol, tmp_path):
    experiment = SalamiExperiment.create(
        parent_dir=str(tmp_path), name="exp", template=protocol
    )
    experiment.microscope = object()  # never reached; the guard fires first
    experiment.status.in_flight_task = "Slice_Task"

    with pytest.raises(RuntimeError, match="interrupted"):
        experiment.run()


def test_resolve_retry_reruns_the_same_task(protocol, tmp_path):
    experiment = SalamiExperiment.create(
        parent_dir=str(tmp_path), name="exp", template=protocol
    )
    experiment.status.next_task_index = 0
    experiment.status.in_flight_task = "Slice_Task"

    experiment.resolve_interrupted_task("retry")

    assert experiment.status.in_flight_task is None
    assert experiment.status.next_task_index == 0  # task 0 runs again


def test_resolve_skip_moves_past_the_task(protocol, tmp_path):
    experiment = SalamiExperiment.create(
        parent_dir=str(tmp_path), name="exp", template=protocol
    )
    experiment.status.next_task_index = 0
    experiment.status.in_flight_task = "Slice_Task"

    experiment.resolve_interrupted_task("skip")

    assert experiment.status.in_flight_task is None
    assert experiment.status.next_task_index == 1  # straight to the acquisition


# ---------------------------------------------------------------------------
# Status schema
# ---------------------------------------------------------------------------


def test_status_round_trips(tmp_path):
    status = SalamiExperimentStatus(
        current_step=7,
        next_task_index=1,
        setup_complete=True,
        run_state="paused",
        in_flight_task="Slice_Task",
    )
    status.save(str(tmp_path))

    loaded = SalamiExperimentStatus.load(str(tmp_path / STATUS_FILENAME))

    assert loaded.current_step == 7
    assert loaded.next_task_index == 1
    assert loaded.setup_complete is True
    assert loaded.run_state == "paused"
    assert loaded.in_flight_task == "Slice_Task"
    assert loaded.last_updated is not None


def test_status_loads_pre_rename_files(tmp_path):
    """Status files written before `state` was renamed to `sample_state`."""
    legacy = {
        "current_step": 12,
        "setup_complete": True,
        "name": "Experiment_Status",
        "state": None,
    }
    target = tmp_path / STATUS_FILENAME
    with open(target, "w") as f:
        yaml.dump(legacy, f)

    loaded = SalamiExperimentStatus.load(str(target))

    assert loaded.current_step == 12
    assert loaded.setup_complete is True
    # new fields fall back to sensible defaults
    assert loaded.next_task_index == 0
    assert loaded.run_state == "created"
    assert loaded.in_flight_task is None


def test_default_task_configs_do_not_share_milling_stages():
    """DEFAULT_SLICE is a module singleton; each task needs its own copy."""
    a = MillingTaskConfig(name="A")
    b = MillingTaskConfig(name="B")

    a.milling.stages[0].pattern.point.y = 1.23e-6

    assert b.milling.stages[0].pattern.point.y != pytest.approx(1.23e-6)


def test_advance_moves_a_rectangle_pattern_too(protocol, tmp_path):
    """Rectangles are positioned by `point`, lines by their endpoints."""
    from fibsem.milling.patterning.patterns2 import RectanglePattern

    milling = protocol.tasks[0]
    milling.milling.stages[0].pattern = RectanglePattern()

    advanced = runtime_task_config(milling, str(tmp_path), 200, protocol.advance)

    assert _milled_y(advanced.milling.stages[0].pattern) == pytest.approx(
        200 * STEP_SIZE
    )


def test_a_pattern_that_cannot_be_moved_is_reported(protocol, tmp_path, caplog):
    """Silently not advancing is the failure mode worth shouting about."""
    from fibsem.applications.salami.tasks import advance_pattern_y

    class Unmovable:
        point = None

    advance_pattern_y(Unmovable(), 1e-6)

    assert "will not advance" in caplog.text


# ---------------------------------------------------------------------------
# The slice pattern: a wide, one-slice-thick slab
# ---------------------------------------------------------------------------


def test_the_slice_pattern_can_be_positioned_and_sized(protocol):
    """It has to carry a width, a height and a point for any of this to work.

    A line pattern satisfies none of those: no height to be the slice
    thickness, and `define()` ignores `point` so it cannot be placed.
    """
    pattern = protocol.tasks[0].milling.stages[0].pattern

    assert hasattr(pattern, "width")
    assert hasattr(pattern, "height")
    assert pattern.define()[0].centre_y == pytest.approx(pattern.point.y)


def test_a_slice_is_wide_and_thin(protocol):
    """The slab spans the face laterally and is a slice thick, not the reverse."""
    pattern = protocol.tasks[0].milling.stages[0].pattern

    assert pattern.width > pattern.height * 10, "slice is not much wider than thick"
    assert pattern.height > 0.0


@pytest.mark.parametrize("step_size", [5e-9, 50e-9, 500e-9])
def test_pattern_height_follows_the_slice_thickness(protocol, tmp_path, step_size):
    """One quantity, so it tracks whatever the thickness is set to."""
    protocol.advance.step_size = step_size

    cfg = runtime_task_config(protocol.tasks[0], str(tmp_path), 7, protocol.advance)

    assert cfg.milling.stages[0].pattern.define()[0].height == pytest.approx(step_size)


def test_consecutive_slices_are_contiguous(protocol, tmp_path):
    """The gap between one slab and the next must be exactly zero.

    Advancing further than the pattern is tall leaves unmilled material between
    slices; advancing less re-cuts what the last step removed. Either shows up
    only once the stack is reconstructed.
    """
    stages = [
        runtime_task_config(protocol.tasks[0], str(tmp_path), step, protocol.advance)
        .milling.stages[0]
        .pattern.define()[0]
        for step in (10, 11)
    ]

    advanced = stages[1].centre_y - stages[0].centre_y
    assert advanced == pytest.approx(stages[0].height)


def test_a_pattern_without_a_height_is_left_alone(protocol, tmp_path):
    """A trench states upper/lower heights; guessing which to resize would be wrong."""
    from fibsem.milling.patterning.patterns2 import TrenchPattern

    milling = protocol.tasks[0]
    milling.milling.stages[0].pattern = TrenchPattern()
    before = milling.milling.stages[0].pattern.to_dict()

    cfg = runtime_task_config(milling, str(tmp_path), 0, protocol.advance)

    assert cfg.milling.stages[0].pattern.to_dict() == before


def test_the_shipped_default_protocol_can_actually_slice(tmp_path):
    """The template on disk is what new experiments get, not DEFAULT_SLICE.

    `MillingTaskConfig.__post_init__` only falls back to DEFAULT_SLICE when a
    task has no stages, and the shipped config has one serialised. So changing
    the code default without regenerating the YAML would leave every new
    experiment on the old pattern, silently.
    """
    experiment = SalamiExperiment.create(parent_dir=str(tmp_path), name="from-template")
    pattern = experiment.find_task("Slice_Task").milling.stages[0].pattern

    # positionable: define() has to read `point`, or the slice cannot be placed
    assert pattern.define()[0].centre_y == pytest.approx(pattern.point.y)
    # and sized, so the slice thickness has somewhere to live
    assert hasattr(pattern, "height") and hasattr(pattern, "width")

    # and it advances: the milled shape moves, not just the stored position
    advance = experiment.config.advance
    task = experiment.find_task("Slice_Task")
    moved = _milled_y(
        runtime_task_config(task, str(tmp_path), 20, advance).milling.stages[0].pattern
    ) - _milled_y(
        runtime_task_config(task, str(tmp_path), 0, advance).milling.stages[0].pattern
    )
    assert moved == pytest.approx(20 * advance.step_size)
