"""Per-auto-function regions, and editing them by dragging on the preview.

Each auto-function has its own `area_of_interest`: focus wants a feature with
detail in it, contrast wants a region representative of the whole face, and
those are rarely the same box.

The same region is reachable two ways — four spinboxes in the form, and a
draggable rectangle over the preview image — so the sync between them, in both
directions and without a feedback loop, is the thing worth pinning.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import yaml

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.applications.salami.experiment import CONFIG_FILENAME, SalamiExperiment
from fibsem.applications.salami.structures import AutoFunctionConfig
from fibsem.structures import FibsemRectangle

# ---------------------------------------------------------------------------
# The config field
# ---------------------------------------------------------------------------


def test_bare_config_defaults_to_the_whole_frame():
    """The class default. Which fraction suits which function is decided by
    AcquisitionTaskConfig, since one class serves both."""
    area = AutoFunctionConfig().area_of_interest
    assert (area.left, area.top, area.width, area.height) == (0.0, 0.0, 1.0, 1.0)


def test_round_trips_through_yaml():
    config = AutoFunctionConfig()
    config.area_of_interest = FibsemRectangle(left=0.2, top=0.3, width=0.4, height=0.1)

    loaded = AutoFunctionConfig.from_dict(
        yaml.safe_load(yaml.safe_dump(config.to_dict()))
    )

    assert loaded.area_of_interest.left == pytest.approx(0.2)
    assert loaded.area_of_interest.height == pytest.approx(0.1)


def test_protocols_written_before_the_field_still_load():
    """A KeyError here would break every existing experiment on disk."""
    legacy = {"enabled": True, "schedule": {"every_n_steps": 2}}

    loaded = AutoFunctionConfig.from_dict(legacy)

    assert loaded.schedule.every_n_steps == 2
    assert loaded.area_of_interest == FibsemRectangle()


def test_each_auto_function_has_its_own_region(tmp_path):
    experiment = SalamiExperiment.create(parent_dir=str(tmp_path), name="e")
    task = experiment.find_task("Acquisition_Task")
    before = task.autocontrast.area_of_interest

    task.autofocus.area_of_interest = FibsemRectangle(0.1, 0.1, 0.2, 0.2)

    assert task.autocontrast.area_of_interest == before


def test_regions_survive_a_reload(tmp_path):
    experiment = SalamiExperiment.create(parent_dir=str(tmp_path), name="e")
    task = experiment.find_task("Acquisition_Task")
    task.autofocus.area_of_interest = FibsemRectangle(0.25, 0.25, 0.5, 0.5)
    experiment.config.save()

    reopened = SalamiExperiment.open(experiment.path)
    area = reopened.find_task("Acquisition_Task").autofocus.area_of_interest

    assert (area.left, area.width) == pytest.approx((0.25, 0.5))


def test_the_run_uses_each_function_own_region(tmp_path, monkeypatch):
    """Not the shared alignment_area, which is reserved for drift correction."""
    from fibsem.applications.salami import tasks as tasks_module

    calls = {}
    monkeypatch.setattr(
        tasks_module,
        "auto_focus_basic",
        lambda scope, beam, area: calls.__setitem__("focus", area),
    )
    monkeypatch.setattr(
        tasks_module,
        "auto_contrast_basic",
        lambda scope, beam, area: calls.__setitem__("contrast", area),
    )

    from fibsem import utils

    scope, _ = utils.setup_session(manufacturer="Demo")
    experiment = SalamiExperiment.create(
        parent_dir=str(tmp_path), name="e", microscope=scope
    )
    task = experiment.find_task("Acquisition_Task")
    task.autofocus.area_of_interest = FibsemRectangle(0.1, 0.1, 0.2, 0.2)
    task.autocontrast.area_of_interest = FibsemRectangle(0.5, 0.5, 0.4, 0.4)
    task.alignment_area = FibsemRectangle(0.9, 0.9, 0.05, 0.05)  # must not be used

    experiment.status.setup_complete = True
    experiment.run_next_step()

    assert calls["focus"].left == pytest.approx(0.1)
    assert calls["contrast"].left == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Drawing and editing on the preview
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def microscope():
    from fibsem import utils

    scope, _settings = utils.setup_session(manufacturer="Demo")
    return scope


@pytest.fixture
def window(qapp, microscope, tmp_path):
    """A shell with an experiment loaded and a preview image on the canvas."""
    from fibsem import utils
    from fibsem.applications.salami.tasks import sync_config_from_microscope
    from fibsem.applications.salami.ui.main_window import SalamiMainWindow

    w = SalamiMainWindow()
    ui = w.salami_ui
    _scope, settings = utils.setup_session(manufacturer="Demo")
    ui.microscope, ui.settings = microscope, settings
    ui.update_microscope_ui()
    ui.task_config_editor.set_microscope(microscope)

    experiment = SalamiExperiment.create(
        parent_dir=str(tmp_path), name="e", microscope=microscope, parent_ui=ui
    )
    ui._adopt_experiment(experiment)
    ui.comboBoxTasks.setCurrentIndex(ui.comboBoxTasks.findData("Acquisition_Task"))

    # The overlays need an image to position against.
    ui._on_sync_finished(
        sync_config_from_microscope(
            microscope, experiment.find_task("Acquisition_Task")
        )
    )
    return w


def _drag(overlay, x0, y0, width, height):
    """Move a rectangle as a user would: reposition, then release."""
    overlay.set_rect(x0, y0, width, height)
    overlay.rect_changed.emit(overlay.get_rect())


def test_an_acquisition_task_draws_both_regions(window):
    assert window.protocol_page.preview.rectangle_keys() == [
        "autofocus",
        "autocontrast",
    ]


def test_a_milling_task_draws_its_volume_and_no_auto_function_areas(window):
    """A milling task has no auto-functions; what it has is a volume to cut."""
    ui = window.salami_ui
    ui.comboBoxTasks.setCurrentIndex(ui.comboBoxTasks.findData("Slice_Task"))

    keys = window.protocol_page.preview.rectangle_keys()
    assert keys == ["milling_region"]


def test_each_region_has_a_legend_entry(window):
    legend = window.protocol_page.preview._legend_labels
    assert "Autofocus" in legend["autofocus"].text()
    assert "Autocontrast" in legend["autocontrast"].text()


def test_dragging_writes_the_region_back_to_the_config(window):
    ui = window.salami_ui
    task = ui.task_config_editor.task_config
    overlay = window.protocol_page.preview._rectangles["autofocus"]

    _drag(overlay, 200, 150, 400, 300)

    area = task.autofocus.area_of_interest
    assert area.left == pytest.approx(200 / 1536)
    assert area.width == pytest.approx(400 / 1536)


def test_dragging_one_region_leaves_the_other_alone(window):
    ui = window.salami_ui
    task = ui.task_config_editor.task_config
    before = task.autocontrast.area_of_interest

    _drag(window.protocol_page.preview._rectangles["autofocus"], 100, 100, 200, 200)

    assert task.autocontrast.area_of_interest == before


def test_dragging_updates_the_numbers_in_the_form(window):
    """The region is four spinboxes as well as a box; they must agree."""
    ui = window.salami_ui
    overlay = window.protocol_page.preview._rectangles["autofocus"]

    _drag(overlay, 0, 0, 768, 512)

    _owner, _name, widget = ui.task_config_editor.editor.field_widgets[
        "autofocus.area_of_interest.width"
    ]
    assert widget.read() == pytest.approx(0.5)


def test_editing_the_numbers_moves_the_rectangle(window):
    ui = window.salami_ui
    task = ui.task_config_editor.task_config

    task.autofocus.area_of_interest = FibsemRectangle(0.1, 0.1, 0.2, 0.2)
    ui._sync_rectangles_from_config()

    rect = window.protocol_page.preview._rectangles["autofocus"].get_rect()
    assert rect["x0"] == pytest.approx(round(0.1 * 1536))
    assert rect["width"] == pytest.approx(round(0.2 * 1536))


def test_dragging_persists_the_protocol(window, tmp_path):
    ui = window.salami_ui
    _drag(window.protocol_page.preview._rectangles["autofocus"], 300, 200, 300, 200)

    with open(os.path.join(ui.salamiExperiment.path, CONFIG_FILENAME)) as f:
        on_disk = yaml.safe_load(f)
    acquisition = next(t for t in on_disk["tasks"] if t["type"] == "acquisition")

    assert acquisition["autofocus"]["area_of_interest"]["left"] == pytest.approx(
        300 / 1536
    )


def test_switching_task_rebinds_the_rectangles(window, tmp_path, microscope):
    """A stale callback would write a drag into the task you left."""
    ui = window.salami_ui
    first = ui.task_config_editor.task_config

    ui.comboBoxTasks.setCurrentIndex(ui.comboBoxTasks.findData("Slice_Task"))
    ui.comboBoxTasks.setCurrentIndex(ui.comboBoxTasks.findData("Acquisition_Task"))

    _drag(window.protocol_page.preview._rectangles["autofocus"], 50, 50, 100, 100)

    # Same task object, and it did receive the edit.
    assert ui.task_config_editor.task_config is first
    assert first.autofocus.area_of_interest.left == pytest.approx(50 / 1536)


def test_alignment_area_is_hidden_from_the_form(window):
    """Reserved for drift correction; nothing reads it yet."""
    paths = window.salami_ui.task_config_editor.editor.field_widgets
    assert not [p for p in paths if p.startswith("alignment_area")]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


# The exact fractions are tuning and expected to change. What must hold is that
# a default region is physical and usable: inside the frame, not degenerate, and
# not silently the whole frame.
MIN_USABLE_FRACTION = 0.05


@pytest.mark.parametrize("function", ["autofocus", "autocontrast"])
def test_default_regions_are_physical(tmp_path, function):
    experiment = SalamiExperiment.create(parent_dir=str(tmp_path), name="e")
    area = getattr(experiment.find_task("Acquisition_Task"), function).area_of_interest

    assert area.left >= 0.0 and area.top >= 0.0, "region starts outside the frame"
    assert area.left + area.width <= 1.0, "region runs off the right edge"
    assert area.top + area.height <= 1.0, "region runs off the bottom edge"
    for extent in (area.width, area.height):
        assert MIN_USABLE_FRACTION <= extent < 1.0, "region is degenerate or full-frame"


@pytest.mark.parametrize("function", ["autofocus", "autocontrast"])
def test_default_regions_are_centred(tmp_path, function):
    """Centred, whatever size they are tuned to."""
    experiment = SalamiExperiment.create(parent_dir=str(tmp_path), name="e")
    area = getattr(experiment.find_task("Acquisition_Task"), function).area_of_interest

    assert area.left + area.width / 2 == pytest.approx(0.5)
    assert area.top + area.height / 2 == pytest.approx(0.5)


def test_focus_gets_a_larger_region_than_contrast(tmp_path):
    """The relationship is the design intent; the fractions themselves are not.

    Focus needs enough structure in view to score sharpness against; contrast
    only needs a patch representative of the face.
    """
    task = SalamiExperiment.create(parent_dir=str(tmp_path), name="e").find_task(
        "Acquisition_Task"
    )

    assert (
        task.autofocus.area_of_interest.width > task.autocontrast.area_of_interest.width
    )


def test_legacy_protocols_get_the_same_defaults_as_a_new_one():
    """A protocol predating the field should behave like one written today,
    whatever today's defaults happen to be."""
    from fibsem.applications.salami.tasks import AcquisitionTaskConfig

    data = AcquisitionTaskConfig(name="T").to_dict()
    del data["autofocus"]["area_of_interest"]
    del data["autocontrast"]["area_of_interest"]

    loaded = AcquisitionTaskConfig.from_dict(data)
    fresh = AcquisitionTaskConfig(name="T")

    assert loaded.autofocus.area_of_interest == fresh.autofocus.area_of_interest
    assert loaded.autocontrast.area_of_interest == fresh.autocontrast.area_of_interest


def test_centred_area_helper():
    from fibsem.applications.salami.structures import centred_area

    area = centred_area(0.5)
    assert (area.left, area.top, area.width, area.height) == (0.25, 0.25, 0.5, 0.5)


# ---------------------------------------------------------------------------
# Visibility follows the function's enabled flag
# ---------------------------------------------------------------------------


def _visible(window, key):
    return window.protocol_page.preview._rectangles[key]._area_visible


def _interactive(window, key):
    return window.protocol_page.preview._rectangles[key]._interactive


def test_both_regions_show_while_enabled(window):
    assert _visible(window, "autofocus")
    assert _visible(window, "autocontrast")


def test_disabling_a_function_hides_its_region(window):
    ui = window.salami_ui
    ui.task_config_editor.task_config.autofocus.enabled = False
    ui._on_task_config_changed()

    assert not _visible(window, "autofocus")
    assert _visible(window, "autocontrast")  # the other is unaffected


def test_a_hidden_region_cannot_be_dragged(window):
    """RectOverlay decides from _interactive alone, so a hidden-but-live box
    would swallow clicks meant for the one underneath it."""
    ui = window.salami_ui
    ui.task_config_editor.task_config.autofocus.enabled = False
    ui._on_task_config_changed()

    assert not _interactive(window, "autofocus")


def test_hiding_a_region_keeps_its_value(window):
    ui = window.salami_ui
    task = ui.task_config_editor.task_config
    _drag(window.protocol_page.preview._rectangles["autofocus"], 100, 100, 500, 400)
    dragged = task.autofocus.area_of_interest.width

    task.autofocus.enabled = False
    ui._on_task_config_changed()
    task.autofocus.enabled = True
    ui._on_task_config_changed()

    assert task.autofocus.area_of_interest.width == pytest.approx(dragged)
    assert _visible(window, "autofocus")


def test_a_task_selected_with_a_function_disabled_starts_hidden(window):
    """Not only on toggle — the initial draw has to respect it too."""
    ui = window.salami_ui
    ui.task_config_editor.task_config.autocontrast.enabled = False

    ui._on_selected_task_changed()  # redraw as if reselected

    assert not _visible(window, "autocontrast")
    assert _visible(window, "autofocus")


# ---------------------------------------------------------------------------
# The milling volume: a region that means metres
# ---------------------------------------------------------------------------
#
# The box is never stored. It is reconstructed from the pattern and the step
# count, so the properties worth pinning are the conversions and the round trip
# — not any particular size.

from fibsem.applications.salami.structures import VolumeAdvanceConfig
from fibsem.applications.salami.tasks import (
    ImageGeometry,
    MillingTaskConfig,
    apply_milling_region,
    milling_region_for,
)

GEOMETRY = ImageGeometry(pixel_size=97.65625e-9, width_px=1536, height_px=1024)


@pytest.fixture
def slice_task():
    return MillingTaskConfig(name="Slice_Task")


@pytest.fixture
def advance():
    return VolumeAdvanceConfig(step_size=10e-9)


def _box(left=0.35, top=0.30, width=0.30, height=0.40):
    return FibsemRectangle(left=left, top=top, width=width, height=height)


def test_a_drawn_box_round_trips(slice_task, advance):
    """Draw it, read it back, and it is the same box."""
    box = _box()
    n_steps = apply_milling_region(slice_task, box, advance, GEOMETRY)

    back = milling_region_for(slice_task, n_steps, advance, GEOMETRY)

    for attr in ("left", "top", "width", "height"):
        assert getattr(back, attr) == pytest.approx(getattr(box, attr), abs=1e-6)


def test_slice_count_is_the_box_depth_over_the_thickness(slice_task, advance):
    box = _box(height=0.40)
    expected = box.height * GEOMETRY.height_m / advance.step_size

    n_steps = apply_milling_region(slice_task, box, advance, GEOMETRY)

    assert n_steps == pytest.approx(expected, rel=1e-3)


def test_thinner_slices_mean_more_of_them_over_the_same_volume(slice_task, advance):
    box = _box()
    coarse = apply_milling_region(slice_task, box, advance, GEOMETRY)

    advance.step_size /= 2
    fine = apply_milling_region(slice_task, box, advance, GEOMETRY)

    assert fine == pytest.approx(coarse * 2, rel=1e-3)


def test_the_volume_swept_matches_the_box(slice_task, advance):
    """Across the whole run, the slabs cover the drawn box exactly.

    Start at the bottom edge, advance once per step, and the top of the last
    slab should land on the top edge — no unmilled margin, no overrun.
    """
    box = _box()
    n_steps = apply_milling_region(slice_task, box, advance, GEOMETRY)
    pattern = slice_task.milling.stages[0].pattern
    thickness = advance.step_size

    bottom = pattern.point.y - thickness / 2
    top_of_last = pattern.point.y + (n_steps - 1) * thickness + thickness / 2

    assert top_of_last - bottom == pytest.approx(
        box.height * GEOMETRY.height_m, rel=1e-3
    )


def test_box_width_becomes_the_slice_width(slice_task, advance):
    box = _box(width=0.30)

    apply_milling_region(slice_task, box, advance, GEOMETRY)

    assert slice_task.milling.stages[0].pattern.width == pytest.approx(
        box.width * GEOMETRY.width_m
    )


def test_a_centred_box_leaves_the_pattern_centred(slice_task, advance):
    apply_milling_region(slice_task, _box(left=0.25, width=0.5), advance, GEOMETRY)

    assert slice_task.milling.stages[0].pattern.point.x == pytest.approx(0.0)


def test_moving_the_box_right_moves_the_pattern_right(slice_task, advance):
    apply_milling_region(slice_task, _box(left=0.1), advance, GEOMETRY)
    left_x = slice_task.milling.stages[0].pattern.point.x

    apply_milling_region(slice_task, _box(left=0.6), advance, GEOMETRY)

    assert slice_task.milling.stages[0].pattern.point.x > left_x


def test_every_stage_shares_the_box(slice_task, advance):
    """A coarse and a fine stage cut the same plane, so one box governs both."""
    from copy import deepcopy

    slice_task.milling.stages.append(deepcopy(slice_task.milling.stages[0]))

    apply_milling_region(slice_task, _box(), advance, GEOMETRY)

    widths = {s.pattern.width for s in slice_task.milling.stages}
    ys = {s.pattern.point.y for s in slice_task.milling.stages}
    assert len(widths) == 1 and len(ys) == 1


def test_a_volume_always_has_at_least_one_slice(slice_task):
    """A degenerate thickness must not produce zero or negative slices."""
    assert (
        apply_milling_region(
            slice_task, _box(), VolumeAdvanceConfig(step_size=0.0), GEOMETRY
        )
        == 1
    )


def test_geometry_agrees_with_the_image_it_came_from():
    from fibsem import utils
    from fibsem.applications.salami.tasks import (
        AcquisitionTaskConfig,
        acquire_preview_image,
    )

    scope, _ = utils.setup_session(manufacturer="Demo")
    image = acquire_preview_image(scope, AcquisitionTaskConfig(name="T").image)

    geometry = ImageGeometry.from_image(image)

    assert geometry.width_m == pytest.approx(image.metadata.image_settings.hfw)
    # centre of the frame is the pattern origin
    assert geometry.to_pattern_x(0.5) == pytest.approx(0.0)
    assert geometry.to_pattern_y(0.5) == pytest.approx(0.0)
    # y is inverted: the top of the frame is positive
    assert geometry.to_pattern_y(0.0) > 0.0


# ---------------------------------------------------------------------------
# The volume box, driven through the UI
# ---------------------------------------------------------------------------


@pytest.fixture
def milling_window(window):
    """The shell with the slice task selected and a preview on the canvas."""
    ui = window.salami_ui
    ui.comboBoxTasks.setCurrentIndex(ui.comboBoxTasks.findData("Slice_Task"))
    return window


def _volume_overlay(window):
    return window.protocol_page.preview._rectangles["milling_region"]


def test_the_slice_panel_shows_only_for_a_milling_task(window):
    ui = window.salami_ui
    panel = ui.task_config_editor.slice_panel
    assert not panel.isVisibleTo(ui.task_config_editor)  # acquisition selected

    ui.comboBoxTasks.setCurrentIndex(ui.comboBoxTasks.findData("Slice_Task"))
    assert panel.isVisibleTo(ui.task_config_editor)


def test_the_panel_shows_the_current_thickness(milling_window):
    ui = milling_window.salami_ui
    panel = ui.task_config_editor.slice_panel

    assert panel.thickness_spinbox.value() == pytest.approx(
        ui.salamiExperimentConfig.advance.step_size * 1e9
    )


def test_dragging_the_volume_updates_the_slice_count(milling_window):
    ui = milling_window.salami_ui
    config = ui.salamiExperimentConfig
    before = config.n_steps

    _drag(_volume_overlay(milling_window), 200, 100, 800, 600)

    assert config.n_steps != before
    # and it is the drawn depth over the thickness
    geometry = milling_window.protocol_page.image_geometry
    expected = (600 / geometry.height_px) * geometry.height_m / config.advance.step_size
    assert config.n_steps == pytest.approx(expected, rel=1e-3)


def test_dragging_the_volume_moves_the_pattern(milling_window):
    ui = milling_window.salami_ui
    pattern = ui.task_config_editor.task_config.milling.stages[0].pattern
    before = (pattern.width, pattern.point.y)

    _drag(_volume_overlay(milling_window), 200, 100, 800, 600)

    assert (pattern.width, pattern.point.y) != before


def test_dragging_the_volume_persists_the_protocol(milling_window):
    ui = milling_window.salami_ui
    _drag(_volume_overlay(milling_window), 250, 150, 700, 500)

    with open(os.path.join(ui.salamiExperiment.path, CONFIG_FILENAME)) as f:
        on_disk = yaml.safe_load(f)
    assert on_disk["n_steps"] == ui.salamiExperimentConfig.n_steps


def test_changing_thickness_re_derives_the_count_for_the_same_volume(milling_window):
    """The box is the volume asked for; halve the slices and you need twice as many."""
    ui = milling_window.salami_ui
    config = ui.salamiExperimentConfig
    panel = ui.task_config_editor.slice_panel

    _drag(_volume_overlay(milling_window), 200, 100, 800, 600)
    coarse_steps = config.n_steps
    coarse_depth = coarse_steps * config.advance.step_size

    panel.thickness_spinbox.setValue(panel.thickness_spinbox.value() / 2)

    assert config.n_steps == pytest.approx(coarse_steps * 2, rel=1e-2)
    # the volume itself is unchanged
    assert config.n_steps * config.advance.step_size == pytest.approx(
        coarse_depth, rel=1e-2
    )


def test_the_panel_reports_the_volume(milling_window):
    ui = milling_window.salami_ui
    _drag(_volume_overlay(milling_window), 200, 100, 800, 600)

    text = ui.task_config_editor.slice_panel.summary_label.text()
    config = ui.salamiExperimentConfig
    assert str(config.n_steps) in text
    # The distance sliced through, which is what the box height means. Not
    # called "deep": a stage's milling depth is a different axis entirely — how
    # far each cut reaches into the face — and conflating the two is what sent
    # an operator looking for the depth control on the volume box.
    assert f"{config.volume_depth * 1e6:.2f}" in text


def test_the_volume_survives_a_reload(milling_window, tmp_path, microscope):
    """The box is not stored, so it has to be reconstructible from the protocol."""
    ui = milling_window.salami_ui
    _drag(_volume_overlay(milling_window), 200, 100, 800, 600)
    drawn = _volume_overlay(milling_window).get_area()

    reopened = SalamiExperiment.open(ui.salamiExperiment.path, microscope=microscope)
    from fibsem.applications.salami.tasks import milling_region_for

    rebuilt = milling_region_for(
        reopened.find_task("Slice_Task"),
        reopened.config.n_steps,
        reopened.config.advance,
        milling_window.protocol_page.image_geometry,
    )

    for attr in ("left", "top", "width", "height"):
        assert getattr(rebuilt, attr) == pytest.approx(getattr(drawn, attr), abs=1e-3)


def test_without_a_preview_the_panel_says_so_instead_of_failing(
    qapp, tmp_path, microscope
):
    """Fractions of a frame that is not there cannot become metres."""
    from fibsem import utils
    from fibsem.applications.salami.ui.main_window import SalamiMainWindow

    w = SalamiMainWindow()
    ui = w.salami_ui
    _scope, settings = utils.setup_session(manufacturer="Demo")
    ui.microscope, ui.settings = microscope, settings
    ui.update_microscope_ui()
    ui._adopt_experiment(
        SalamiExperiment.create(
            parent_dir=str(tmp_path),
            name="no-preview",
            microscope=microscope,
            parent_ui=ui,
        )
    )
    ui.comboBoxTasks.setCurrentIndex(ui.comboBoxTasks.findData("Slice_Task"))

    assert w.protocol_page.preview.rectangle_keys() == []
    assert "Sync" in ui.task_config_editor.slice_panel.summary_label.text()


def test_switching_from_a_milling_task_does_not_crash_the_form(milling_window):
    """Regression: selecting a task seeds its beam defaults, which emits
    config_changed before the rectangles have been redrawn — so the sync ran
    with the new task and the previous task's rectangles, and asked an
    acquisition config for `.milling`."""
    ui = milling_window.salami_ui
    assert "milling_region" in milling_window.protocol_page.preview.rectangle_keys()

    ui.comboBoxTasks.setCurrentIndex(ui.comboBoxTasks.findData("Acquisition_Task"))
    ui._on_task_config_changed()  # what the seeding emits

    assert ui.task_config_editor.task_config.name == "Acquisition_Task"
