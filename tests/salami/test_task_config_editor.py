"""Headless tests for the generated Salami task config editors.

Uses PyQt5 directly with the offscreen platform (no pytest-qt dependency),
mirroring tests/fm/test_autofocus_widget.py.

The editors are generated from the config dataclasses, so what is worth testing
is the generation rules rather than any individual row: that the right control
type is chosen, that display units convert back to SI, that `None` survives as
"leave the microscope alone", and that fields which must not be edited from a
protocol are absent.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox

from fibsem.applications.salami.experiment import (
    CONFIG_FILENAME,
    SalamiExperiment,
    SalamiExperimentConfig,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def experiment(tmp_path):
    return SalamiExperiment.create(parent_dir=str(tmp_path), name="EditorTest")


@pytest.fixture
def editor(qapp, experiment):
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget

    widget = TaskConfigEditorWidget()
    widget.set_task_config(experiment.find_task("Acquisition_Task"))
    return widget


@pytest.fixture(scope="module")
def microscope():
    """The simulator implements the same get_available_values contract."""
    from fibsem import utils

    scope, _settings = utils.setup_session(manufacturer="Demo")
    return scope


def _control(editor, path):
    _owner, _name, param_widget = editor.editor.field_widgets[path]
    return param_widget


def _set(editor, path, value):
    """Drive the control the way a user would, so write-back is exercised."""
    param_widget = _control(editor, path)
    inner = getattr(param_widget, "inner", None)
    if inner is not None:  # optional_control: a checkbox around the control
        param_widget.check.setChecked(True)
        inner.widget.setValue(value)
        return
    widget = param_widget.widget
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        widget.setValue(value)
    elif isinstance(widget, QCheckBox):
        widget.setChecked(value)
    elif isinstance(widget, QComboBox):
        widget.setCurrentIndex(value)


# ---------------------------------------------------------------------------
# Field generation
# ---------------------------------------------------------------------------


def test_generates_editors_for_nested_dataclasses(editor):
    """Nested configs are walked recursively, not skipped."""
    paths = editor.editor.field_widgets
    assert "schedule.every_n_steps" in paths
    assert "autofocus.schedule.every_n_steps" in paths
    assert "image.dwell_time" in paths
    assert "state.electron_beam.voltage" in paths
    # Three levels deep: task -> auto-function -> region.
    assert "autofocus.area_of_interest.left" in paths


def test_control_types_follow_field_types(editor):
    assert isinstance(_control(editor, "enabled").widget, QCheckBox)
    assert isinstance(_control(editor, "schedule.every_n_steps").widget, QSpinBox)
    assert isinstance(_control(editor, "image.hfw").widget, QDoubleSpinBox)
    # Tuple[int, int] with a choices hint becomes a combo, not a text box.
    assert isinstance(_control(editor, "image.resolution").widget, QComboBox)


def test_string_annotations_are_resolved(editor):
    """fibsem/structures.py uses `from __future__ import annotations`.

    Its fields therefore arrive as strings like 'bool'. If they were not run
    through get_type_hints, every one of them would fall through to a text box.
    """
    assert isinstance(_control(editor, "image.drift_correction").widget, QCheckBox)
    assert isinstance(_control(editor, "autofocus.enabled").widget, QCheckBox)


def test_excludes_fields_that_are_not_protocol(editor):
    """Sample state, run-time paths and duplicated parameters stay out.

    stage_position matters most: it is sample-specific, and it is what makes
    set_microscope_state drive the stage.
    """
    paths = editor.editor.field_widgets
    for excluded in (
        "state.stage_position",
        "state.stage_position.x",
        "state.timestamp",
        "state.ion_beam.voltage",
        "image.path",
        "image.filename",
        "image.save",
        # BeamSettings carries these too, but ImageSettings owns them.
        "state.electron_beam.hfw",
        "state.electron_beam.dwell_time",
        "state.electron_beam.resolution",
        # Rewritten every step by apply_advance.
        "state.electron_beam.shift",
        # Forced False in AcquisitionTask.run, so offering them would be a lie.
        # Per-slice contrast is the `autocontrast` AutoFunctionConfig instead.
        "image.autocontrast",
        "image.autogamma",
    ):
        assert excluded not in paths, f"{excluded} should not be editable"


def test_per_slice_autocontrast_is_still_editable(editor):
    """Excluding image.autocontrast must not hide the scheduled auto-function."""
    assert "autocontrast.enabled" in editor.editor.field_widgets
    assert "autocontrast.schedule.every_n_steps" in editor.editor.field_widgets


def test_task_name_is_read_only(editor):
    """The name is the output directory and the skip_once key."""
    widget = _control(editor, "name")
    assert widget.kind == "read_only"
    assert not widget.editable
    assert not widget.widget.isEnabled()


# ---------------------------------------------------------------------------
# Write-back
# ---------------------------------------------------------------------------


def test_display_units_convert_back_to_si(editor, experiment):
    """Spinboxes show μs/μm/kV/nA/mm; the config must keep SI."""
    config = experiment.find_task("Acquisition_Task")

    _set(editor, "image.dwell_time", 5.0)  # μs
    _set(editor, "image.hfw", 80.0)  # μm
    _set(editor, "state.electron_beam.voltage", 2.0)  # kV
    _set(editor, "state.electron_beam.beam_current", 0.4)  # nA
    _set(editor, "state.electron_beam.working_distance", 4.0)  # mm

    assert config.image.dwell_time == pytest.approx(5e-6)
    assert config.image.hfw == pytest.approx(80e-6)
    assert config.state.electron_beam.voltage == pytest.approx(2000.0)
    assert config.state.electron_beam.beam_current == pytest.approx(0.4e-9)
    assert config.state.electron_beam.working_distance == pytest.approx(4e-3)


def test_unset_optional_stays_none(editor, experiment):
    """None means "leave the microscope as it is", so it must be reachable.

    Overloading zero would make "don't touch the voltage" indistinguishable
    from "set 0 V".
    """
    config = experiment.find_task("Acquisition_Task")
    assert config.state.electron_beam.voltage is None

    _set(editor, "state.electron_beam.voltage", 2.0)
    assert config.state.electron_beam.voltage == pytest.approx(2000.0)

    _control(editor, "state.electron_beam.voltage").check.setChecked(False)
    assert config.state.electron_beam.voltage is None


def test_resolution_round_trips_as_tuple(editor, experiment):
    """Declared Tuple[int, int] but arrives from YAML as a list."""
    config = experiment.find_task("Acquisition_Task")
    combo = _control(editor, "image.resolution").widget
    # Scan rather than findData: Qt does not match a Python tuple through QVariant.
    index = next(
        i for i in range(combo.count()) if tuple(combo.itemData(i)) == (3072, 2048)
    )
    combo.setCurrentIndex(index)
    assert tuple(config.image.resolution) == (3072, 2048)


def test_edits_emit_a_save_nudge(editor):
    received = []
    editor.config_changed.connect(lambda: received.append(1))
    _set(editor, "schedule.every_n_steps", 10)
    assert received


def test_edits_survive_save_and_reload(editor, experiment):
    config = experiment.find_task("Acquisition_Task")

    _set(editor, "image.dwell_time", 3.0)
    _set(editor, "schedule.every_n_steps", 25)
    _set(editor, "autofocus.enabled", False)
    _set(editor, "enabled", False)
    experiment.config.save()

    reloaded = SalamiExperimentConfig.load(
        os.path.join(experiment.path, CONFIG_FILENAME)
    )
    task = next(t for t in reloaded.tasks if t.name == config.name)
    assert task.image.dwell_time == pytest.approx(3e-6)
    assert task.schedule.every_n_steps == 25
    assert task.autofocus.enabled is False
    assert task.enabled is False


# ---------------------------------------------------------------------------
# Task switching
# ---------------------------------------------------------------------------


def test_switching_task_rebuilds_the_form(editor, experiment):
    editor.set_task_config(experiment.find_task("Slice_Task"))
    assert "state.ion_beam.hfw" in editor.editor.field_widgets
    # The acquisition-only rows are gone rather than left behind.
    assert "image.dwell_time" not in editor.editor.field_widgets
    assert "state.electron_beam.voltage" not in editor.editor.field_widgets

    editor.set_task_config(experiment.find_task("Acquisition_Task"))
    assert "image.dwell_time" in editor.editor.field_widgets
    assert "state.ion_beam.hfw" not in editor.editor.field_widgets


def test_clearing_shows_a_prompt_and_no_fields(editor):
    editor.set_task_config(None)
    assert not editor.editor.field_widgets
    # isVisibleTo, not isVisible: the widget was never shown, so isVisible() is
    # False regardless of what we set.
    assert editor.placeholder.isVisibleTo(editor)
    assert editor.placeholder.text()


def test_readout_reports_pixel_size_and_acquisition_time(editor):
    """Voxel isotropy is the thing to watch while setting imaging parameters."""
    _set(editor, "image.hfw", 150.0)
    text = editor.info_label.text()
    assert "Pixel size" in text
    assert "Acquisition" in text


# ---------------------------------------------------------------------------
# Microscope-backed choices
# ---------------------------------------------------------------------------


def test_voltage_and_current_offer_only_supported_values(qapp, experiment, microscope):
    """Both must come from the microscope, not be free numbers.

    A protocol that stores an unsupported current would fail at step 0.
    """
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget
    from fibsem.structures import BeamType

    widget = TaskConfigEditorWidget(microscope=microscope)
    widget.set_task_config(experiment.find_task("Acquisition_Task"))

    for path, key in (
        ("state.electron_beam.voltage", "voltage"),
        ("state.electron_beam.beam_current", "current"),
    ):
        param_widget = _control(widget, path)
        assert param_widget.kind == "choice", f"{path} should be a dropdown"
        combo = param_widget.widget
        offered = [combo.itemData(i) for i in range(combo.count())]
        # Exactly what the microscope reports — no placeholder entry.
        assert offered == list(microscope.get_available_values(key, BeamType.ELECTRON))


def test_selecting_a_supported_value_stores_si(qapp, experiment, microscope):
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget
    from fibsem.structures import BeamType

    config = experiment.find_task("Acquisition_Task")
    widget = TaskConfigEditorWidget(microscope=microscope)
    widget.set_task_config(config)

    expected = microscope.get_available_values("voltage", BeamType.ELECTRON)[2]
    combo = _control(widget, "state.electron_beam.voltage").widget
    combo.setCurrentIndex(2)
    assert config.state.electron_beam.voltage == expected


def test_unset_value_shows_and_stores_the_first_choice(qapp, experiment, microscope):
    """No "unset" entry, so the config is seeded to match what is displayed.

    Otherwise the form would show the first supported value while the config still
    held None, and the run would apply neither.
    """
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget
    from fibsem.structures import BeamType

    config = experiment.find_task("Acquisition_Task")
    config.state.electron_beam.voltage = None
    assert config.state.electron_beam.voltage is None

    widget = TaskConfigEditorWidget(microscope=microscope)
    widget.set_task_config(config)

    first = microscope.get_available_values("voltage", BeamType.ELECTRON)[0]
    param_widget = _control(widget, "state.electron_beam.voltage")
    assert param_widget.read() == first
    # Displayed == stored.
    assert config.state.electron_beam.voltage == first


def test_unsupported_stored_value_stays_visible(qapp, experiment, microscope):
    """A value set offline, or on another column, must read back as itself.

    Snapping to the nearest supported value would silently rewrite the protocol.
    """
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget

    config = experiment.find_task("Acquisition_Task")
    config.state.electron_beam.voltage = 2500.0

    widget = TaskConfigEditorWidget(microscope=microscope)
    widget.set_task_config(config)

    param_widget = _control(widget, "state.electron_beam.voltage")
    assert param_widget.read() == 2500.0
    assert "2.50 kV" in param_widget.widget.currentText()


def test_falls_back_to_numeric_entry_with_no_microscope(editor):
    """Offline protocol editing must still work.

    Also covers a backend that reports nothing for a key — Tescan has no
    `voltage` branch, so it returns an empty list.
    """
    assert _control(editor, "state.electron_beam.voltage").kind == "optional"
    assert _control(editor, "state.electron_beam.beam_current").kind == "optional"


def test_connecting_swaps_numeric_entry_for_dropdowns(qapp, experiment, microscope):
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget

    widget = TaskConfigEditorWidget()
    widget.set_task_config(experiment.find_task("Acquisition_Task"))
    assert _control(widget, "state.electron_beam.voltage").kind == "optional"

    widget.set_microscope(microscope)
    assert _control(widget, "state.electron_beam.voltage").kind == "choice"

    widget.set_microscope(None)
    assert _control(widget, "state.electron_beam.voltage").kind == "optional"


def test_available_values_are_queried_per_beam(qapp, microscope):
    """The key is looked up for the beam that owns the field.

    The simulator reports different voltages for the two columns, so an ion
    BeamSettings must not be offered the electron list.
    """
    from fibsem.applications.salami.ui.config_editor import fetch_available_values
    from fibsem.structures import BeamType

    electron = fetch_available_values(microscope, "voltage", BeamType.ELECTRON)
    ion = fetch_available_values(microscope, "voltage", BeamType.ION)
    assert electron and ion and electron != ion


# ---------------------------------------------------------------------------
# Sync row
# ---------------------------------------------------------------------------


def _find(widget, target) -> bool:
    layout = widget.layout()
    if layout is None:
        return False
    for i in range(layout.count()):
        item = layout.itemAt(i)
        child = item.widget() if item else None
        if child is None:
            continue
        if child is target or _find(child, target):
            return True
    return False


@pytest.mark.parametrize(
    "task,spec_name",
    [("Acquisition_Task", "ACQUISITION_SPEC"), ("Slice_Task", "MILLING_SPEC")],
)
def test_sync_row_sits_below_the_divider(qapp, experiment, microscope, task, spec_name):
    """Directly above the settings the microscope and the protocol share.

    Both task types sync — the acquisition task against the SEM, the milling
    task against the ion column.
    """
    from fibsem.applications.salami.ui import task_editor as te

    widget = te.TaskConfigEditorWidget(microscope=microscope)
    widget.set_task_config(experiment.find_task(task))

    spec = getattr(te, spec_name)
    # The buttons anchor to the same section the divider precedes.
    assert spec["dividers"] & set(te.SYNC_ROW_ANCHORS)
    assert _find(widget.editor, widget.sync_row)


def test_sync_row_survives_a_rebuild(qapp, experiment, microscope):
    """It is owned by the editor, not the generated form.

    The form is torn down and rebuilt on every task switch; deleting a
    caller-owned widget with it would drop its signal connections.
    """
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget

    widget = TaskConfigEditorWidget(microscope=microscope)
    widget.set_task_config(experiment.find_task("Acquisition_Task"))
    row = widget.sync_row

    widget.set_task_config(experiment.find_task("Slice_Task"))
    widget.set_task_config(experiment.find_task("Acquisition_Task"))

    assert widget.sync_row is row
    assert _find(widget.editor, row)
    assert row.button_from.receivers(row.button_from.clicked) > 0


def test_sync_buttons_need_a_microscope(qapp, experiment):
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget

    widget = TaskConfigEditorWidget()
    widget.set_task_config(experiment.find_task("Acquisition_Task"))
    assert not widget.sync_row.button_from.isEnabled()
    assert not widget.sync_row.button_to.isEnabled()


def test_busy_locks_the_buttons_and_shows_the_spinner(qapp, experiment, microscope):
    """The spinner is the only feedback during a multi-second acquisition."""
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget

    widget = TaskConfigEditorWidget(microscope=microscope)
    widget.set_task_config(experiment.find_task("Acquisition_Task"))
    row = widget.sync_row

    assert not row.spinner.isVisibleTo(row)
    row.set_busy(True, "Reading settings...")
    assert row.spinner.isVisibleTo(row)
    assert not row.button_from.isEnabled()
    assert not row.button_to.isEnabled()
    assert row.status_label.text() == "Reading settings..."

    row.set_busy(False)
    assert not row.spinner.isVisibleTo(row)
    assert row.button_from.isEnabled()
    assert row.button_to.isEnabled()


def test_buttons_follow_the_image_settings_widget_styling(qapp, experiment, microscope):
    """Same convention as FibsemImageSettingsWidget's acquire buttons.

    Primary while idle; secondary and renamed "Acquiring..." while the column is
    scanning. The style is never cleared when disabled, because both stylesheets
    carry their own `:disabled` rule.
    """
    from fibsem.applications.salami.ui.task_editor import (
        SYNC_BUSY_LABEL,
        SYNC_FROM_LABEL,
        SYNC_TO_LABEL,
        TaskConfigEditorWidget,
    )
    from fibsem.ui import stylesheets

    widget = TaskConfigEditorWidget(microscope=microscope)
    widget.set_task_config(experiment.find_task("Acquisition_Task"))
    row = widget.sync_row
    buttons = (row.button_from, row.button_to)

    for button in buttons:
        assert button.styleSheet() == stylesheets.PRIMARY_BUTTON_STYLESHEET
    assert row.button_from.text() == SYNC_FROM_LABEL
    assert row.button_to.text() == SYNC_TO_LABEL

    row.set_busy(True)
    for button in buttons:
        assert button.styleSheet() == stylesheets.SECONDARY_BUTTON_STYLESHEET
        assert button.text() == SYNC_BUSY_LABEL

    row.set_busy(False)
    for button in buttons:
        assert button.styleSheet() == stylesheets.PRIMARY_BUTTON_STYLESHEET

    # Disabled keeps the primary sheet; its :disabled rule handles the look.
    widget.set_microscope(None)
    for button in buttons:
        assert not button.isEnabled()
        assert button.styleSheet() == stylesheets.PRIMARY_BUTTON_STYLESHEET


def test_status_and_info_labels_use_the_shared_style(qapp, experiment, microscope):
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget
    from fibsem.ui import stylesheets

    widget = TaskConfigEditorWidget(microscope=microscope)
    widget.set_task_config(experiment.find_task("Acquisition_Task"))

    assert (
        widget.sync_row.status_label.styleSheet()
        == stylesheets.LABEL_INSTRUCTIONS_STYLE
    )
    assert widget.info_label.styleSheet() == stylesheets.LABEL_INSTRUCTIONS_STYLE


def test_sync_requests_are_re_emitted(qapp, experiment, microscope):
    """The row asks; the main window does the work on a thread."""
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget

    widget = TaskConfigEditorWidget(microscope=microscope)
    widget.set_task_config(experiment.find_task("Acquisition_Task"))

    received = []
    widget.sync_from_microscope_requested.connect(lambda: received.append("from"))
    widget.sync_to_microscope_requested.connect(lambda: received.append("to"))

    widget.sync_row.button_from.click()
    widget.sync_row.button_to.click()
    assert received == ["from", "to"]


def test_refresh_picks_up_values_changed_behind_the_form(qapp, experiment, microscope):
    """A sync from the microscope rewrites the config; the form must re-read it."""
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget

    config = experiment.find_task("Acquisition_Task")
    widget = TaskConfigEditorWidget(microscope=microscope)
    widget.set_task_config(config)

    config.image.hfw = 275e-6
    widget.refresh()

    assert _control(widget, "image.hfw").read() == pytest.approx(275e-6)


def test_missing_electron_beam_block_is_created(qapp, experiment):
    """A YAML file with a null beam block would otherwise have nothing to bind."""
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget

    config = experiment.find_task("Acquisition_Task")
    config.state.electron_beam = None

    widget = TaskConfigEditorWidget()
    widget.set_task_config(config)

    assert config.state.electron_beam is not None
    assert "state.electron_beam.voltage" in widget.editor.field_widgets


# ---------------------------------------------------------------------------
# Locking during a microscope sync
# ---------------------------------------------------------------------------


def test_fields_are_disabled_while_syncing(editor):
    """A sync rewrites the config then refreshes, so an edit made mid-sync
    would be silently discarded. Grey the form out instead, as the image
    widget does while acquiring."""
    editor.set_editing_enabled(False)

    assert not _control(editor, "image.hfw").widget.isEnabled()
    assert not _control(editor, "schedule.every_n_steps").widget.isEnabled()
    assert not _control(editor, "state.electron_beam.voltage").widget.isEnabled()


def test_fields_are_re_enabled_afterwards(editor):
    editor.set_editing_enabled(False)
    editor.set_editing_enabled(True)

    assert _control(editor, "image.hfw").widget.isEnabled()
    assert _control(editor, "schedule.every_n_steps").widget.isEnabled()


def test_read_only_fields_stay_disabled_after_unlocking(editor):
    """Re-enabling everything indiscriminately would undo the view spec."""
    editor.set_editing_enabled(False)
    editor.set_editing_enabled(True)

    assert not _control(editor, "name").widget.isEnabled()


def test_sync_row_keeps_its_own_state_while_fields_are_locked(editor):
    """The row owns its buttons and has to keep its spinner animating, so
    locking the generated fields must not reach into it."""
    editor.sync_row.set_microscope_available(True)
    editor.set_editing_enabled(False)

    assert editor.sync_row.isEnabled()
    assert editor.sync_row.button_from.isEnabled()


# ---------------------------------------------------------------------------
# Optional integration features
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,limits",
    [
        ("line_integration", (2, 255)),
        ("scan_interlacing", (2, 8)),
        ("frame_integration", (2, 512)),
    ],
)
def test_integration_fields_read_as_a_feature_toggle(editor, field, limits):
    """Checkbox says Enable, and the spinbox offers only the hardware range."""
    widget = _control(editor, f"image.{field}")
    assert widget.check.text() == "Enable"
    assert (widget.inner.widget.minimum(), widget.inner.widget.maximum()) == limits


def test_integration_spinbox_is_inert_until_enabled(editor):
    widget = _control(editor, "image.line_integration")
    assert not widget.check.isChecked()
    assert not widget.inner.widget.isEnabled()

    widget.check.setChecked(True)
    assert widget.inner.widget.isEnabled()


def test_enabling_integration_stores_a_valid_count(editor, experiment):
    """Never 0 or 1: integrating one line is the same as not integrating.

    An unset field's inner control is built from a 0 placeholder, so this
    depends on the range clamping it up to the minimum.
    """
    config = experiment.find_task("Acquisition_Task")
    assert config.image.line_integration is None

    _control(editor, "image.line_integration").check.setChecked(True)
    assert config.image.line_integration == 2


def test_disabling_integration_restores_none(editor, experiment):
    config = experiment.find_task("Acquisition_Task")
    widget = _control(editor, "image.line_integration")

    widget.check.setChecked(True)
    widget.inner.widget.setValue(16)
    assert config.image.line_integration == 16

    widget.check.setChecked(False)
    assert config.image.line_integration is None


def test_beam_fields_keep_the_set_wording(editor):
    """`None` there means "leave the column alone", which is not a feature toggle."""
    assert _control(editor, "state.electron_beam.voltage").check.text() == "Set"


def test_integration_changes_the_acquisition_time_readout(editor, experiment):
    """ImageSettings.estimated_time multiplies by the integration counts."""
    config = experiment.find_task("Acquisition_Task")
    before = config.image.estimated_time

    widget = _control(editor, "image.line_integration")
    widget.check.setChecked(True)
    widget.inner.widget.setValue(8)

    assert config.image.estimated_time == pytest.approx(before * 8)


def test_working_distance_has_no_unset_checkbox(editor):
    """The spinbox is the value; an unset state there is a click in the way."""
    widget = _control(editor, "state.electron_beam.working_distance")
    assert widget.kind == "plain"
    assert isinstance(widget.widget, QDoubleSpinBox)


def test_working_distance_renders_even_when_unset(editor, experiment):
    """Regression: the float control calls float(value) and raises on None.

    _build_for catches that and drops the row, so opting out of the unset
    checkbox silently removed the field offline until a concrete value was
    substituted for building.
    """
    assert (
        experiment.find_task("Acquisition_Task").state.electron_beam.working_distance
        is None
    )
    assert "state.electron_beam.working_distance" in editor.editor.field_widgets


def test_unset_working_distance_is_not_written_back(editor, experiment):
    """Displaying 0 must not store 0 — apply_beam_settings would then drive the
    column to zero working distance instead of leaving it alone."""
    config = experiment.find_task("Acquisition_Task")
    assert config.state.electron_beam.working_distance is None


def test_working_distance_still_writes_si(editor, experiment):
    config = experiment.find_task("Acquisition_Task")
    _control(editor, "state.electron_beam.working_distance").widget.setValue(3.5)
    assert config.state.electron_beam.working_distance == pytest.approx(3.5e-3)


def test_other_optional_beam_fields_keep_their_checkbox(editor):
    """Only working_distance opted out; voltage offline still means "leave alone"."""
    assert _control(editor, "state.electron_beam.voltage").check.text() == "Set"


# ---------------------------------------------------------------------------
# The milling task: ion-column viewing options only
# ---------------------------------------------------------------------------


@pytest.fixture
def milling_editor(qapp, experiment, microscope):
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget

    widget = TaskConfigEditorWidget(microscope=microscope)
    widget.set_task_config(experiment.find_task("Slice_Task"))
    return widget


def test_milling_form_shows_only_ion_viewing_options(milling_editor):
    """Pattern geometry and milling current live on the Milling tab."""
    paths = milling_editor.editor.field_widgets
    for shown in (
        "state.ion_beam.voltage",
        "state.ion_beam.beam_current",
        "state.ion_beam.working_distance",
        "state.ion_beam.hfw",
        "state.ion_beam.resolution",
        "state.ion_beam.dwell_time",
    ):
        assert shown in paths, f"{shown} should be editable"

    for hidden in (
        "milling",
        "milling.field_of_view",
        "milling.channel",
        "milling.alignment.enabled",
        "milling.acquisition.imaging.dwell_time",
        "state.electron_beam",
        "state.stage_position",
    ):
        assert hidden not in paths, f"{hidden} should not be on this form"


def test_milling_form_has_no_beam_type_choice(milling_editor):
    """A milling task always images on the ion column; there is nothing to pick."""
    assert not [
        p for p in milling_editor.editor.field_widgets if p.endswith("beam_type")
    ]


def test_milling_viewing_options_have_no_unset_checkbox(milling_editor):
    """Same as the acquisition task: the control is the value."""
    for path in (
        "state.ion_beam.voltage",
        "state.ion_beam.beam_current",
        "state.ion_beam.hfw",
        "state.ion_beam.resolution",
        "state.ion_beam.dwell_time",
    ):
        widget = _control(milling_editor, path)
        assert widget.kind != "optional", f"{path} still has a checkbox"


def test_milling_working_distance_is_read_only(milling_editor):
    """Focus is set by focusing the column, then captured by the sync."""
    widget = _control(milling_editor, "state.ion_beam.working_distance")
    assert widget.kind == "read_only"
    assert not widget.widget.isEnabled()


def test_read_only_values_are_shown_in_display_units(milling_editor, experiment):
    """A stored 0.004 must read as 4.00 mm, or showing it is no use."""
    config = experiment.find_task("Slice_Task")
    config.state.ion_beam.working_distance = 0.004
    milling_editor.refresh()

    assert (
        "mm"
        in _control(milling_editor, "state.ion_beam.working_distance").widget.text()
    )


def test_unset_read_only_value_says_so(qapp, experiment):
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget

    config = experiment.find_task("Slice_Task")
    config.state.ion_beam.working_distance = None
    widget = TaskConfigEditorWidget()  # offline, so nothing seeds it
    widget.set_task_config(config)

    assert (
        _control(widget, "state.ion_beam.working_distance").widget.text() == "not set"
    )


# ---------------------------------------------------------------------------
# The SenseAI acquisition task
# ---------------------------------------------------------------------------
#
# Worth its own set, because the way this form fails is silent: a field the
# editor cannot build a control for is logged and dropped, so a broken hint
# looks like a setting that was never added rather than an error.


@pytest.fixture
def senseai_editor(qapp):
    from fibsem.applications.salami.tasks import SenseAIAcquisitionTaskConfig
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget

    widget = TaskConfigEditorWidget()
    widget.config = SenseAIAcquisitionTaskConfig(name="SenseAI_Task")
    widget.set_task_config(widget.config)
    return widget


def test_every_scan_generator_setting_is_editable(senseai_editor):
    """A dropped field is only a debug log, so assert each one is present."""
    paths = senseai_editor.editor.field_widgets
    for field in ("scan_size", "pattern", "sampling", "dwell_time", "reset_buffer"):
        assert f"scan.{field}" in paths, f"scan.{field} has no control"


def test_scan_pattern_offers_what_the_generator_supports(senseai_editor):
    from fibsem.applications.salami.tasks import SENSEAI_SCAN_PATTERNS

    widget = _control(senseai_editor, "scan.pattern").widget
    offered = [widget.itemText(i) for i in range(widget.count())]

    assert offered == list(SENSEAI_SCAN_PATTERNS)


def test_choosing_a_scan_pattern_reaches_the_config(senseai_editor):
    from fibsem.applications.salami.tasks import SENSEAI_SCAN_PATTERNS

    widget = _control(senseai_editor, "scan.pattern").widget
    widget.setCurrentIndex(list(SENSEAI_SCAN_PATTERNS).index("Raster-Random"))

    assert senseai_editor.config.scan.pattern == "Raster-Random"


def test_scan_size_stays_a_tuple_through_the_form(senseai_editor):
    """It is handed to the generator as ScanSize, so the pair has to survive."""
    from fibsem.applications.salami.tasks import SenseAIAcquisitionTaskConfig

    widget = _control(senseai_editor, "scan.scan_size").widget
    widget.setCurrentIndex(widget.findText("768x512"))

    assert tuple(senseai_editor.config.scan.scan_size) == (768, 512)
    restored = SenseAIAcquisitionTaskConfig.from_dict(senseai_editor.config.to_dict())
    assert tuple(restored.scan.scan_size) == (768, 512)


# ---------------------------------------------------------------------------
# Milling current
# ---------------------------------------------------------------------------
#
# The one milling parameter on this form. Pattern geometry stays on the Milling
# tab so two widgets never draw on one FIB canvas; the current has no geometry.


def test_milling_stage_panel_only_shows_for_milling_tasks(editor, experiment):
    assert not editor.milling_stage_panel.isVisibleTo(editor)

    editor.set_task_config(experiment.find_task("Slice_Task"))
    assert editor.milling_stage_panel.isVisibleTo(editor)


def test_a_row_per_stage(qapp, experiment):
    """A slice task can carry a coarse and a fine stage; they do not share a current."""
    from copy import deepcopy

    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget

    config = experiment.find_task("Slice_Task")
    config.milling.stages.append(deepcopy(config.milling.stages[0]))
    config.milling.stages[1].name = "Fine Slice"

    widget = TaskConfigEditorWidget()
    widget.set_task_config(config)

    assert len(widget.milling_stage_panel._rows) == 2


def test_offline_entry_writes_si_to_the_config(qapp, experiment):
    """Displayed in nA, stored in A — a protocol is prepared away from the column."""
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget

    config = experiment.find_task("Slice_Task")
    widget = TaskConfigEditorWidget()  # no microscope
    widget.set_task_config(config)

    _stage, control = widget.milling_stage_panel._rows[0]
    control.setValue(3.5)

    assert config.milling.stages[0].milling.milling_current == pytest.approx(3.5e-9)


def test_connected_offers_only_currents_the_column_supports(
    qapp, experiment, microscope
):
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget
    from fibsem.structures import BeamType

    config = experiment.find_task("Slice_Task")
    widget = TaskConfigEditorWidget(microscope=microscope)
    widget.set_task_config(config)

    _stage, combo = widget.milling_stage_panel._rows[0]
    offered = [combo.itemData(i) for i in range(combo.count())]

    assert offered
    assert offered == list(
        microscope.get_available_values_cached("current", BeamType.ION)
    )

    combo.setCurrentIndex(1)
    assert config.milling.stages[0].milling.milling_current == offered[1]


def test_editing_the_current_asks_for_a_save(qapp, experiment):
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget

    config = experiment.find_task("Slice_Task")
    widget = TaskConfigEditorWidget()
    widget.set_task_config(config)

    nudges = []
    widget.config_changed.connect(lambda: nudges.append(1))
    _stage, control = widget.milling_stage_panel._rows[0]
    control.setValue(control.value() + 1.0)

    assert nudges, "an edit must ask the owner to persist the protocol"


def test_the_current_survives_save_and_reload(qapp, experiment):
    from fibsem.applications.salami.experiment import SalamiExperiment
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget

    config = experiment.find_task("Slice_Task")
    widget = TaskConfigEditorWidget()
    widget.set_task_config(config)
    _stage, control = widget.milling_stage_panel._rows[0]
    control.setValue(2.25)
    experiment.config.save()

    reopened = SalamiExperiment.open(experiment.config.path)
    stage = reopened.find_task("Slice_Task").milling.stages[0]
    assert stage.milling.milling_current == pytest.approx(2.25e-9)


def test_milling_depth_is_editable_in_microns(qapp, experiment):
    """Depth is how far each cut reaches into the face, in um on the form."""
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget

    config = experiment.find_task("Slice_Task")
    widget = TaskConfigEditorWidget()
    widget.set_task_config(config)

    _stage, control = widget.milling_stage_panel._depth_rows[0]
    assert control.value() == pytest.approx(
        config.milling.stages[0].pattern.depth * 1e6
    )

    control.setValue(3.0)
    assert config.milling.stages[0].pattern.depth == pytest.approx(3e-6)


def test_milling_depth_is_independent_of_the_region(qapp, experiment):
    """The confusion this guards: resizing the volume box must not touch depth.

    Box height is the distance sliced through and sets the slice count; a
    stage's depth is how far into the face each cut goes and is fixed.
    """
    from fibsem.applications.salami.tasks import ImageGeometry, apply_milling_region
    from fibsem.structures import FibsemRectangle

    config = experiment.find_task("Slice_Task")
    config.milling.stages[0].pattern.depth = 5e-6
    geometry = ImageGeometry(pixel_size=10e-9, width_px=1536, height_px=1024)

    small = apply_milling_region(
        config,
        FibsemRectangle(left=0.4, top=0.4, width=0.2, height=0.1),
        experiment.config.advance,
        geometry,
    )
    depth_after_small = config.milling.stages[0].pattern.depth

    large = apply_milling_region(
        config,
        FibsemRectangle(left=0.2, top=0.1, width=0.6, height=0.8),
        experiment.config.advance,
        geometry,
    )

    # A taller box means more slices ...
    assert large > small
    # ... and the same milling depth.
    assert depth_after_small == pytest.approx(5e-6)
    assert config.milling.stages[0].pattern.depth == pytest.approx(5e-6)


def test_editing_the_depth_asks_for_a_save(qapp, experiment):
    from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget

    config = experiment.find_task("Slice_Task")
    widget = TaskConfigEditorWidget()
    widget.set_task_config(config)

    nudges = []
    widget.config_changed.connect(lambda: nudges.append(1))
    _stage, control = widget.milling_stage_panel._depth_rows[0]
    control.setValue(control.value() + 1.0)

    assert nudges
