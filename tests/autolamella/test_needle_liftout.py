"""The general needle liftout task's configuration.

The movement itself is covered in test_needle.py. What is here is the part that
decides what the task will do on the instrument: which steps carry milling, how a
protocol round-trips through its file, what an old protocol becomes, and what is
refused before anything moves.
"""

from pathlib import Path

import pytest

from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.workflows.tasks import load_config
from fibsem.applications.autolamella.workflows.tasks.needle_liftout import (
    NeedleLiftoutTask,
    NeedleLiftoutTaskConfig,
    NeedleLoopStep,
)


def _config(**kwargs) -> NeedleLiftoutTaskConfig:
    return NeedleLiftoutTaskConfig(task_name="Needle Liftout", **kwargs)


def _task(config: NeedleLiftoutTaskConfig) -> NeedleLiftoutTask:
    """Enough of a task to validate a config: validation reads only the config."""
    lamella = Lamella(path=Path("/tmp/needle-liftout-test"), number=1, petname="01")
    return NeedleLiftoutTask(microscope=None, config=config, lamella=lamella)


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
