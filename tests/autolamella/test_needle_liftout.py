"""The general needle liftout task's configuration.

The movement itself is covered in test_needle.py. What is here is the part that
decides what the task will do on the instrument: which steps carry milling, how a
protocol round-trips through its file, what an old protocol becomes, and what is
refused before anything moves.
"""

from pathlib import Path

import numpy as np
import pytest

import fibsem.utils as utils
from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.workflows.tasks import load_config
from fibsem.applications.autolamella.workflows.tasks.needle_liftout import (
    NeedleLiftoutTask,
    NeedleLiftoutTaskConfig,
    NeedleLoopStep,
    resolve_feature_name,
)
from fibsem.detection import detection
from fibsem.microscopes.simulator import DemoMicroscope
from fibsem.structures import BeamType


def _config(**kwargs) -> NeedleLiftoutTaskConfig:
    return NeedleLiftoutTaskConfig(task_name="Needle Liftout", **kwargs)


def _task(config: NeedleLiftoutTaskConfig, microscope=None) -> NeedleLiftoutTask:
    """Enough of a task to validate a config: validation reads only the config."""
    lamella = Lamella(path=Path("/tmp/needle-liftout-test"), number=1, petname="01")
    return NeedleLiftoutTask(microscope=microscope, config=config, lamella=lamella)


@pytest.fixture(scope="module")
def microscope() -> DemoMicroscope:
    return DemoMicroscope(utils.load_microscope_configuration().system)


def _aimed(**kwargs) -> NeedleLoopStep:
    """A step with something to steer by in the FIB view."""
    return NeedleLoopStep(to_feature_ib="LandingPost", **kwargs)


# ---------------------------------------------------------------------------
# Defaults: a starting point, not a recipe
# ---------------------------------------------------------------------------


def test_the_default_is_one_step_and_no_milling():
    config = _config()

    assert len(config.steps) == 1
    assert config.steps[0].milling_enabled is False
    assert config.milling == {}


# ---------------------------------------------------------------------------
# Per-step milling
# ---------------------------------------------------------------------------


def test_ticking_milling_on_a_step_creates_its_entry():
    config = _config()
    config.steps.append(NeedleLoopStep(name="Weld", field_of_view=80e-6))
    config.steps[1].milling_enabled = True

    added = config.sync_step_milling()

    assert added == ["Weld"]
    assert config.milling["Weld"].field_of_view == pytest.approx(80e-6)


def test_unticking_or_renaming_never_deletes_an_entry():
    """A slip in the step editor must not destroy patterns that took time to place."""
    config = _config(
        steps=[NeedleLoopStep(name="Weld", milling_enabled=True)],
    )
    config.steps[0].milling_enabled = False
    config.steps[0].name = "Weld (renamed)"

    assert config.sync_step_milling() == []
    assert "Weld" in config.milling


def test_a_milling_key_keeps_a_step_on_its_entry_across_a_rename():
    config = _config(
        steps=[NeedleLoopStep(name="Weld", milling_key="weld", milling_enabled=True)]
    )
    config.steps[0].name = "Weld (renamed)"

    assert config.sync_step_milling() == []
    assert list(config.milling) == ["weld"]


# ---------------------------------------------------------------------------
# The protocol file
# ---------------------------------------------------------------------------


def test_a_protocol_round_trips():
    config = _config(
        steps=[
            _aimed(name="Approach", field_of_view=300e-6),
            NeedleLoopStep(name="Weld", field_of_view=80e-6, milling_enabled=True),
        ]
    )

    again = load_config(config.task_type, config.to_dict())

    assert [s.name for s in again.steps] == ["Approach", "Weld"]
    assert again.steps[1].milling_enabled is True
    assert again.steps[1].field_of_view == pytest.approx(80e-6)
    assert list(again.milling) == ["Weld"]


def test_an_old_protocol_keeps_its_align_feature():
    """The previous codebase kept it on every step, and read only the first."""
    ddict = _config().to_dict()
    del ddict["parameters"]["align_feature"]
    ddict["parameters"]["steps"][0]["align_feature"] = "VolumeBlockCentre"

    assert load_config(ddict["task_type"], ddict).align_feature == "VolumeBlockCentre"


def test_an_old_protocol_drops_relative_move_dy():
    """It was never applied: relative moves run in the FIB frame, where the second
    axis is z. Loading it is not an error; carrying it on would imply it works."""
    ddict = _config().to_dict()
    ddict["parameters"]["steps"][0]["relative_move_dy"] = 5e-6

    step = load_config(ddict["task_type"], ddict).steps[0]

    assert not hasattr(step, "relative_move_dy")


# ---------------------------------------------------------------------------
# Refused before anything moves
# ---------------------------------------------------------------------------


def test_a_valid_config_passes():
    config = _config(steps=[_aimed()])
    _task(config)._validate_config(config.enabled_steps)


def test_no_enabled_steps_is_refused():
    config = _config(steps=[_aimed(enabled=False)])
    with pytest.raises(ValueError, match="no enabled HFW steps"):
        _task(config)._validate_config(config.enabled_steps)


def test_a_step_with_nothing_to_aim_at_is_refused():
    """The default step has no To Feature, so the first one must be chosen."""
    config = _config()
    with pytest.raises(ValueError, match="nothing to aim at"):
        _task(config)._validate_config(config.enabled_steps)


def test_a_step_that_skips_the_needle_needs_no_target():
    config = _config(steps=[NeedleLoopStep(skip_needle_move=True)])
    _task(config)._validate_config(config.enabled_steps)


def test_milling_without_its_entry_is_refused():
    """Never filled in at run time: a blank entry would be milled unseen."""
    config = _config(steps=[_aimed(name="Weld")])
    config.steps[0].milling_enabled = True  # ticked, but never synced
    with pytest.raises(ValueError, match="no milling entry"):
        _task(config)._validate_config(config.enabled_steps)


def test_an_unknown_feature_is_refused():
    config = _config(steps=[_aimed(from_feature_eb="NotAFeature")])
    with pytest.raises(ValueError, match="invalid From Feature"):
        _task(config)._validate_config(config.enabled_steps)


# ---------------------------------------------------------------------------
# Scan rotation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scan_rotation, edge", [(0.0, "right"), (np.pi, "left")])
def test_the_needle_tip_follows_scan_rotation(microscope, scan_rotation, edge):
    """At 180 degrees the image is rotated, so the needle comes in from the other
    side and its tip is its leftmost point, not its rightmost."""
    microscope.set_scan_rotation(scan_rotation, beam_type=BeamType.ION)
    task = _task(_config(), microscope=microscope)

    feature = task._feature("NeedleTip")

    assert feature.name == "NeedleTip"
    assert feature.edge == edge


def test_the_needle_tip_flip_can_be_turned_off(microscope):
    microscope.set_scan_rotation(np.pi, beam_type=BeamType.ION)
    task = _task(_config(flip_features_with_scan_rotation=False), microscope)

    assert task._feature("NeedleTip").edge == "right"


@pytest.mark.parametrize(
    "name, flipped",
    [
        ("NeedleTipBottom", "NeedleTipBottom"),  # there is no NeedleTipTop
        ("LamellaLeftEdge", "LamellaLeftEdge"),  # only the needle swaps sides
        ("VolumeBlockBottomEdge", "VolumeBlockTopEdge"),  # Top/Bottom as before
    ],
)
def test_what_flips_at_180(name, flipped):
    assert resolve_feature_name(name, flip=True).name == flipped


def test_a_left_needle_tip_finds_the_end_away_from_the_border():
    """A needle entering from the right: the default right edge finds where it
    enters the image instead of its tip."""
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[40:51, 60:] = 3  # manipulator from x=60 out to the right border
    img = np.zeros_like(mask)

    left = detection.NeedleTip(edge="left").detect(img=img, mask=mask)
    right = detection.NeedleTip().detect(img=img, mask=mask)

    assert (left.x, left.y) == (60, 45)
    assert (right.x, right.y) == (199, 45)
