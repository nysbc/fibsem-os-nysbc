"""Tests for the headless microscope <-> config sync in salami/tasks.py.

No Qt here: the sync logic deliberately lives outside the UI so it can be driven
from a script or a test, and this is that test. Widget behaviour is covered in
test_task_config_editor.py.
"""

import pytest

from fibsem import utils
from fibsem.applications.salami.experiment import SalamiExperiment
from fibsem.applications.salami.tasks import (
    SYNCED_BEAM_FIELDS,
    SYNCED_IMAGE_FIELDS,
    AcquisitionTaskConfig,
    acquire_preview_image,
    apply_beam_settings,
    beam_block_for,
    resolve_beam_defaults,
    sync_config_from_microscope,
    sync_config_to_microscope,
)
from fibsem.structures import BeamSettings, BeamType


@pytest.fixture(scope="module")
def microscope():
    scope, _settings = utils.setup_session(manufacturer="Demo")
    return scope


@pytest.fixture
def config(tmp_path):
    """A task config from a real experiment, so it has been through YAML."""
    experiment = SalamiExperiment.create(parent_dir=str(tmp_path), name="SyncTest")
    return experiment, experiment.find_task("Acquisition_Task")


# ---------------------------------------------------------------------------
# apply_beam_settings — the None-safe, stage-free apply
# ---------------------------------------------------------------------------


def test_all_unset_beam_block_applies_nothing(microscope):
    """A fresh protocol leaves beam fields unset, meaning "leave the column alone".

    microscope.set_beam_settings would forward every None into a setter and raise
    once it reached set_scan_rotation(None).
    """
    beam = BeamSettings(beam_type=BeamType.ELECTRON)
    assert apply_beam_settings(microscope, beam) == []


def test_apply_beam_settings_handles_none_block(microscope):
    assert apply_beam_settings(microscope, None) == []


def test_only_set_fields_are_applied(microscope):
    before_wd = microscope.get_working_distance(BeamType.ELECTRON)
    beam = BeamSettings(beam_type=BeamType.ELECTRON, voltage=5000)

    applied = apply_beam_settings(microscope, beam)

    assert any("voltage" in a for a in applied)
    assert not any("working distance" in a for a in applied)
    assert microscope.get_beam_voltage(BeamType.ELECTRON) == 5000
    # An unset field must not be touched.
    assert microscope.get_working_distance(BeamType.ELECTRON) == before_wd


def test_apply_never_moves_the_stage(microscope, config):
    """Slice-and-view must not move the stage mid-volume.

    set_microscope_state would: MicroscopeState.stage_position defaults to a
    non-None all-None FibsemStagePosition, so its `is not None` guard always
    passes and the position reaches movement.angle_difference.
    """
    _experiment, cfg = config
    before = microscope.get_stage_position()

    cfg.state.electron_beam.voltage = 5000
    sync_config_to_microscope(microscope, cfg, acquire=False)

    assert microscope.get_stage_position() == before


def test_stigmation_is_never_applied(microscope, config):
    """A null stigmation in YAML loads as Point(0, 0), not None.

    The protocol neither edits nor reads stigmation back, so applying it could
    only ever push that artifact and zero the stigmator on a tuned column.
    """
    _experiment, cfg = config
    beam = beam_block_for(cfg)
    # Confirm the artifact this guards against actually exists after a load.
    assert beam.stigmation is not None

    applied = apply_beam_settings(microscope, beam)
    assert not any("stigmation" in a for a in applied)
    assert not any("scan rotation" in a for a in applied)


# ---------------------------------------------------------------------------
# Sync from the microscope
# ---------------------------------------------------------------------------


def test_sync_from_microscope_captures_live_settings(microscope, config):
    _experiment, cfg = config
    microscope.set_field_of_view(220e-6, BeamType.ELECTRON)
    microscope.set_dwell_time(3e-6, BeamType.ELECTRON)
    microscope.set_beam_voltage(10000, BeamType.ELECTRON)

    result = sync_config_from_microscope(microscope, cfg, acquire=False)

    assert cfg.image.hfw == pytest.approx(220e-6)
    assert cfg.image.dwell_time == pytest.approx(3e-6)
    assert cfg.state.electron_beam.voltage == 10000
    assert result.direction == "from_microscope"
    assert result.changes


def test_sync_from_microscope_is_idempotent(microscope, config):
    """A second sync with nothing changed should report no changes."""
    _experiment, cfg = config
    sync_config_from_microscope(microscope, cfg, acquire=False)
    again = sync_config_from_microscope(microscope, cfg, acquire=False)
    assert again.changes == []
    assert "already in sync" in again.summary


def test_sync_from_microscope_leaves_derived_fields_alone(microscope, config):
    """path and filename are set per step at run time, not by a sync.

    get_imaging_settings returns them from the microscope's last-used settings,
    so copying the whole object would contaminate the protocol.
    """
    _experiment, cfg = config
    cfg.image.path = None
    cfg.image.filename = "should-not-change"

    sync_config_from_microscope(microscope, cfg, acquire=False)

    assert cfg.image.path is None
    assert cfg.image.filename == "should-not-change"


def test_hfw_is_not_duplicated_into_the_beam_block(microscope, config):
    """ImageSettings owns hfw/resolution/dwell_time; the beam block must not.

    Two homes for one parameter would leave it ambiguous which wins at run time.
    """
    _experiment, cfg = config
    sync_config_from_microscope(microscope, cfg, acquire=False)

    for name in ("hfw", "resolution", "dwell_time"):
        assert name not in SYNCED_BEAM_FIELDS
        assert getattr(cfg.state.electron_beam, name) is None
    for name in SYNCED_IMAGE_FIELDS:
        assert getattr(cfg.image, name) is not None


# ---------------------------------------------------------------------------
# Sync to the microscope
# ---------------------------------------------------------------------------


def test_sync_to_microscope_applies_config(microscope, config):
    _experiment, cfg = config
    cfg.image.hfw = 60e-6
    cfg.image.dwell_time = 4e-6
    cfg.state.electron_beam.voltage = 5000

    result = sync_config_to_microscope(microscope, cfg, acquire=False)

    assert microscope.get_field_of_view(BeamType.ELECTRON) == pytest.approx(60e-6)
    assert microscope.get_dwell_time(BeamType.ELECTRON) == pytest.approx(4e-6)
    assert microscope.get_beam_voltage(BeamType.ELECTRON) == 5000
    assert result.direction == "to_microscope"


def test_round_trip_through_the_microscope_is_stable(microscope, config):
    _experiment, cfg = config
    cfg.image.hfw = 120e-6
    cfg.state.electron_beam.voltage = 5000

    sync_config_to_microscope(microscope, cfg, acquire=False)
    sync_config_from_microscope(microscope, cfg, acquire=False)

    assert cfg.image.hfw == pytest.approx(120e-6)
    assert cfg.state.electron_beam.voltage == 5000


# ---------------------------------------------------------------------------
# The acquired image
# ---------------------------------------------------------------------------


def test_sync_acquires_an_image_by_default(microscope, config):
    _experiment, cfg = config
    result = sync_config_from_microscope(microscope, cfg)
    assert result.image is not None
    assert result.image.data is not None


def test_sync_can_skip_the_acquisition(microscope, config):
    _experiment, cfg = config
    assert sync_config_from_microscope(microscope, cfg, acquire=False).image is None
    assert sync_config_to_microscope(microscope, cfg, acquire=False).image is None


def test_preview_image_is_not_written_to_disk(microscope, config, tmp_path):
    """A sync is a preview, not a slice.

    Saving it would drop an image into the task's output directory where it would
    be indistinguishable from run data.
    """
    import os

    experiment, cfg = config
    task_dir = os.path.join(experiment.path, cfg.name)
    cfg.image.save = True  # even when the config asks to save
    cfg.image.path = task_dir

    acquire_preview_image(microscope, cfg.image)

    assert os.listdir(task_dir) == []


def test_preview_does_not_mutate_the_config(microscope, config):
    """acquire_preview_image works on a copy; ImageSettings is shared state."""
    _experiment, cfg = config
    cfg.image.save = True
    cfg.image.path = "/somewhere"

    acquire_preview_image(microscope, cfg.image)

    assert cfg.image.save is True
    assert cfg.image.path == "/somewhere"


# ---------------------------------------------------------------------------
# Defaults for the dropdowns
# ---------------------------------------------------------------------------


def test_resolve_beam_defaults_fills_unset_fields(microscope):
    """The dropdowns have no "unset" entry, so the config must hold a real value."""
    cfg = AcquisitionTaskConfig(name="fresh")
    beam = beam_block_for(cfg)
    assert beam.voltage is None and beam.beam_current is None

    filled = resolve_beam_defaults(microscope, cfg)

    assert "voltage" in filled and "beam_current" in filled
    assert (
        beam.voltage == microscope.get_available_values("voltage", BeamType.ELECTRON)[0]
    )
    assert (
        beam.beam_current
        == microscope.get_available_values("current", BeamType.ELECTRON)[0]
    )
    assert beam.working_distance is not None


def test_resolve_beam_defaults_preserves_existing_values(microscope):
    cfg = AcquisitionTaskConfig(name="fresh")
    beam = beam_block_for(cfg)
    beam.voltage = 30000

    resolve_beam_defaults(microscope, cfg)

    assert beam.voltage == 30000


def test_resolve_beam_defaults_is_idempotent(microscope):
    cfg = AcquisitionTaskConfig(name="fresh")
    resolve_beam_defaults(microscope, cfg)
    assert resolve_beam_defaults(microscope, cfg) == []


# ---------------------------------------------------------------------------
# Beam block resolution
# ---------------------------------------------------------------------------


def test_beam_block_follows_the_imaging_beam():
    cfg = AcquisitionTaskConfig(name="ion task")
    cfg.image.beam_type = BeamType.ION
    assert beam_block_for(cfg) is cfg.state.ion_beam

    cfg.image.beam_type = BeamType.ELECTRON
    assert beam_block_for(cfg) is cfg.state.electron_beam


def test_beam_block_created_when_null():
    """A YAML file with a null beam block would leave nowhere to store values."""
    cfg = AcquisitionTaskConfig(name="task")
    cfg.state.electron_beam = None

    beam = beam_block_for(cfg)

    assert beam is not None
    assert beam.beam_type is BeamType.ELECTRON
    assert cfg.state.electron_beam is beam


def test_log_summary_is_ascii_safe(microscope, config):
    """format_value emits U+03BC, which a cp1252 log handler cannot encode.

    These runs last days; the log is the record of them.
    """
    _experiment, cfg = config
    microscope.set_dwell_time(5e-6, BeamType.ELECTRON)
    result = sync_config_from_microscope(microscope, cfg, acquire=False)

    result.log_summary.encode("cp1252")  # must not raise
    assert "μ" not in result.log_summary
