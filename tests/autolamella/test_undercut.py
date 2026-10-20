"""The undercut task's configurable features.

What is here is the part decided before anything moves: which feature the task
aligns to and at what field of view, which edge each pattern is placed from at each
scan rotation, which settings are refused, and which alignment features Auto Abort
and the trench-size gate apply to. The milling loop itself needs a microscope and
is not covered.
"""

from pathlib import Path

import numpy as np
import pytest

from fibsem import config as fcfg
from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.workflows.tasks import load_config
from fibsem.applications.autolamella.workflows.tasks.needle_liftout import (
    NeedleLiftoutTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.trench import (
    MillTrenchTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.undercut import (
    MillUndercutTask,
    MillUndercutTaskConfig,
    get_placement_edge,
)
from fibsem.detection.detection import (
    LamellaBottomEdge,
    LamellaTopEdge,
    VolumeBlockBottomEdge,
    VolumeBlockTopEdge,
)


def _config(**kwargs) -> MillUndercutTaskConfig:
    return MillUndercutTaskConfig(task_name="Undercut Milling", **kwargs)


def _task(config: MillUndercutTaskConfig) -> MillUndercutTask:
    """Enough of a task to validate a config: validation reads only the config."""
    lamella = Lamella(path=Path("/tmp/undercut-test"), number=1, petname="01")
    return MillUndercutTask(microscope=None, config=config, lamella=lamella)


# ---------------------------------------------------------------------------
# Defaults: what undercut did before these were settings
# ---------------------------------------------------------------------------


def test_the_defaults_are_what_undercut_did_before_they_were_settings():
    config = _config()

    assert config.align_feature == "LamellaCentre"
    assert config.align_feature_hfw == pytest.approx(fcfg.REFERENCE_HFW_HIGH)
    assert config.pattern_placement_feature == "LamellaEdge"


@pytest.mark.parametrize(
    "config_cls",
    [MillUndercutTaskConfig, MillTrenchTaskConfig, NeedleLiftoutTaskConfig],
)
def test_for_liftout_is_no_longer_a_setting(config_cls):
    """Which lamellae a step runs on is the workflow's Runs On, not a task flag."""
    assert "for_liftout" not in config_cls().parameters


# ---------------------------------------------------------------------------
# The protocol file
# ---------------------------------------------------------------------------


def test_the_new_settings_round_trip_through_the_protocol_file():
    config = _config(
        align_feature="VolumeBlockCentre",
        align_feature_hfw=600e-6,
        pattern_placement_feature="VolumeBlockEdge",
    )

    loaded = load_config(MillUndercutTaskConfig.task_type, config.to_dict())

    assert loaded.align_feature == "VolumeBlockCentre"
    assert loaded.align_feature_hfw == pytest.approx(600e-6)
    assert loaded.pattern_placement_feature == "VolumeBlockEdge"


def test_a_protocol_saved_with_for_liftout_loads_with_the_default_hfw():
    """for_liftout is dropped on load, not translated: a protocol saved with it
    set gets the default alignment field of view until it is set again."""
    ddict = _config().to_dict()
    ddict["parameters"]["for_liftout"] = True

    loaded = load_config(MillUndercutTaskConfig.task_type, ddict)

    assert loaded.align_feature_hfw == pytest.approx(fcfg.REFERENCE_HFW_HIGH)
    assert "for_liftout" not in loaded.to_dict()["parameters"]


# ---------------------------------------------------------------------------
# Pattern placement: the edge follows the scan rotation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, scan_rotation, expected",
    [
        ("LamellaEdge", 0.0, LamellaTopEdge),
        ("LamellaEdge", np.pi, LamellaBottomEdge),
        ("VolumeBlockEdge", 0.0, VolumeBlockTopEdge),
        ("VolumeBlockEdge", np.pi, VolumeBlockBottomEdge),
    ],
)
def test_the_placement_edge_follows_the_scan_rotation(name, scan_rotation, expected):
    assert type(get_placement_edge(name, scan_rotation)) is expected


def test_each_placement_edge_is_a_fresh_instance():
    """Detection writes its results onto the feature, so one must not be reused."""
    first = get_placement_edge("LamellaEdge", 0.0)
    second = get_placement_edge("LamellaEdge", 0.0)

    assert first is not second


# ---------------------------------------------------------------------------
# Auto Abort and the trench-size gate: lamella features only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "align_feature, expected",
    [("LamellaCentre", True), ("VolumeBlockCentre", False)],
)
def test_the_detection_checks_apply_to_lamella_features_only(align_feature, expected):
    task = _task(_config(align_feature=align_feature))

    assert task._lamella_feature is expected


# ---------------------------------------------------------------------------
# Refused before anything moves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"align_feature": "NeedleTip"}, "unknown Alignment Feature"),
        # the per-edge names are not choices: the edge follows the scan rotation
        ({"pattern_placement_feature": "LamellaTopEdge"}, "unknown Pattern Placement"),
        ({"align_feature_hfw": 0.0}, "must be positive"),
    ],
)
def test_a_bad_setting_is_refused_before_anything_moves(kwargs, message):
    """_run with no microscope: reaching the stage move would be an AttributeError,
    so a ValueError proves the check came first."""
    task = _task(_config(**kwargs))

    with pytest.raises(ValueError, match=message):
        task._run()
