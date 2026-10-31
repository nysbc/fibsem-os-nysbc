"""The undercut task's configurable features.

What is here is the part decided before anything moves: which feature the task
aligns to and at what field of view, which edge each pattern is placed from and
which way it is offset, which settings are refused, and which alignment features
Auto Abort and the trench-size gate apply to. The milling loop itself needs a
microscope and is not covered.
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
    placement_offset_sign,
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
    assert config.pattern_placement_feature == "LamellaTopEdge"
    assert config.flip_placement_with_scan_rotation is True


@pytest.mark.parametrize(
    "scan_rotation, expected_edge, expected_sign",
    [(0.0, LamellaTopEdge, 1), (np.pi, LamellaBottomEdge, -1)],
)
def test_the_default_placement_matches_the_old_hard_coded_rule(
    scan_rotation, expected_edge, expected_sign
):
    """Before it was a setting: the top edge and +height/2 at 0 degrees, the
    bottom edge and -height/2 otherwise."""
    config = _config()

    edge = get_placement_edge(
        config.pattern_placement_feature,
        scan_rotation,
        config.flip_placement_with_scan_rotation,
    )

    assert type(edge) is expected_edge
    assert placement_offset_sign(edge) == expected_sign


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
        align_feature="VolumeBlockTopEdge",
        align_feature_hfw=600e-6,
        pattern_placement_feature="VolumeBlockBottomEdge",
        flip_placement_with_scan_rotation=False,
    )

    loaded = load_config(MillUndercutTaskConfig.task_type, config.to_dict())

    assert loaded.align_feature == "VolumeBlockTopEdge"
    assert loaded.align_feature_hfw == pytest.approx(600e-6)
    assert loaded.pattern_placement_feature == "VolumeBlockBottomEdge"
    assert loaded.flip_placement_with_scan_rotation is False


@pytest.mark.parametrize(
    "saved, expected",
    [("LamellaEdge", "LamellaTopEdge"), ("VolumeBlockEdge", "VolumeBlockTopEdge")],
)
def test_a_protocol_saved_with_a_paired_edge_name_loads_as_the_same_rule(
    saved, expected
):
    """The paired names meant the top edge at 0 degrees and the bottom otherwise:
    the top edge, flipped with scan rotation."""
    ddict = _config().to_dict()
    ddict["parameters"]["pattern_placement_feature"] = saved
    ddict["parameters"]["flip_placement_with_scan_rotation"] = False

    loaded = load_config(MillUndercutTaskConfig.task_type, ddict)

    assert loaded.pattern_placement_feature == expected
    assert loaded.flip_placement_with_scan_rotation is True


def test_a_protocol_saved_with_for_liftout_loads_with_the_default_hfw():
    """for_liftout is dropped on load, not translated: a protocol saved with it
    set gets the default alignment field of view until it is set again."""
    ddict = _config().to_dict()
    ddict["parameters"]["for_liftout"] = True

    loaded = load_config(MillUndercutTaskConfig.task_type, ddict)

    assert loaded.align_feature_hfw == pytest.approx(fcfg.REFERENCE_HFW_HIGH)
    assert "for_liftout" not in loaded.to_dict()["parameters"]


# ---------------------------------------------------------------------------
# Pattern placement: the chosen edge, optionally flipped at 180 degrees
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, flip, scan_rotation, expected",
    [
        # flip on: Top and Bottom swap when scan rotated
        ("LamellaTopEdge", True, 0.0, LamellaTopEdge),
        ("LamellaTopEdge", True, np.pi, LamellaBottomEdge),
        ("LamellaBottomEdge", True, np.pi, LamellaTopEdge),
        ("VolumeBlockTopEdge", True, np.pi, VolumeBlockBottomEdge),
        ("VolumeBlockBottomEdge", True, np.pi, VolumeBlockTopEdge),
        # flip off: the chosen edge at any scan rotation
        ("LamellaTopEdge", False, np.pi, LamellaTopEdge),
        ("LamellaBottomEdge", False, 0.0, LamellaBottomEdge),
        ("VolumeBlockBottomEdge", False, np.pi, VolumeBlockBottomEdge),
    ],
)
def test_the_placement_edge_is_the_chosen_one_flipped_if_asked(
    name, flip, scan_rotation, expected
):
    assert type(get_placement_edge(name, scan_rotation, flip)) is expected


def test_each_placement_edge_is_a_fresh_instance():
    """Detection writes its results onto the feature, so one must not be reused."""
    first = get_placement_edge("LamellaTopEdge", 0.0, True)
    second = get_placement_edge("LamellaTopEdge", 0.0, True)

    assert first is not second


@pytest.mark.parametrize(
    "edge, expected",
    [
        (LamellaTopEdge(), 1),
        (VolumeBlockTopEdge(), 1),
        (LamellaBottomEdge(), -1),
        (VolumeBlockBottomEdge(), -1),
    ],
)
def test_the_offset_moves_the_pattern_outside_its_edge(edge, expected):
    assert placement_offset_sign(edge) == expected


def test_a_bottom_edge_at_0_degrees_is_offset_down_not_up():
    """The case that keying the offset on the scan rotation got wrong: a bottom
    edge chosen at 0 degrees would have been offset up, over the lamella."""
    edge = get_placement_edge("LamellaBottomEdge", 0.0, False)

    assert placement_offset_sign(edge) == -1


# ---------------------------------------------------------------------------
# Auto Abort and the trench-size gate: lamella features only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "align_feature, expected",
    [
        ("LamellaCentre", True),
        ("LamellaTopEdge", True),
        ("LamellaBottomEdge", True),
        ("VolumeBlockCentre", False),
        ("VolumeBlockTopEdge", False),
        ("VolumeBlockBottomEdge", False),
    ],
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
        # a centre is not an edge to place from
        ({"pattern_placement_feature": "LamellaCentre"}, "unknown Pattern Placement"),
        ({"align_feature_hfw": 0.0}, "must be positive"),
    ],
)
def test_a_bad_setting_is_refused_before_anything_moves(kwargs, message):
    """_run with no microscope: reaching the stage move would be an AttributeError,
    so a ValueError proves the check came first."""
    task = _task(_config(**kwargs))

    with pytest.raises(ValueError, match=message):
        task._run()
