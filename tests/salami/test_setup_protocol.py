"""The protocol's setup section: stored, edited on the Protocol page, skippable."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.applications.salami.experiment import (
    SalamiExperiment,
    SalamiExperimentConfig,
)
from fibsem.applications.salami.setup_config import (
    FIDUCIAL_KEY,
    TRENCH_KEY,
    SalamiSetupConfig,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    from fibsem.applications.salami.ui.main_window import SalamiMainWindow

    return SalamiMainWindow()


# ---------------------------------------------------------------------------
# Stored in the protocol
# ---------------------------------------------------------------------------


def test_setup_section_round_trips_through_the_protocol_file(tmp_path):
    experiment = SalamiExperiment.create(parent_dir=str(tmp_path), name="e")
    config = experiment.config
    trench = config.setup.trench
    trench.skip = True
    trench.milling.stages.append(
        trench.milling.stages[0].__class__.from_dict(trench.milling.stages[0].to_dict())
    )
    config.setup.fiducial.milling.stages[0].milling.milling_current = 3e-9
    config.save()

    loaded = SalamiExperimentConfig.load(
        os.path.join(config.path, "experiment_config.yaml")
    )
    assert loaded.setup.trench.skip
    assert len(loaded.setup.trench.milling.stages) == 2
    assert loaded.setup.fiducial.milling.stages[0].milling.milling_current == 3e-9


def test_protocols_without_a_setup_section_get_the_defaults():
    config = SalamiExperimentConfig.from_dict({"name": "old"})
    assert config.setup.to_dict() == SalamiSetupConfig().to_dict()


def test_saved_protocols_keep_the_setup_section(tmp_path):
    experiment = SalamiExperiment.create(parent_dir=str(tmp_path), name="e")
    experiment.config.setup.fiducial.skip = True
    path = experiment.config.export(str(tmp_path / "protocol.yaml"))
    assert SalamiExperimentConfig.load(path).setup.fiducial.skip


def test_skipped_steps_are_not_milled(tmp_path, monkeypatch):
    experiment = SalamiExperiment.create(parent_dir=str(tmp_path), name="e")
    experiment.config.setup.trench.skip = True
    milled = []
    monkeypatch.setattr(
        experiment,
        "_mill_interactively",
        lambda config, workflow_msg, msg: milled.append(workflow_msg) or config,
    )
    experiment._prep_area()
    experiment._create_fiducial()
    assert milled == ["Mill Fiducial"]


def test_what_was_milled_is_written_back_to_the_protocol(tmp_path, monkeypatch):
    experiment = SalamiExperiment.create(parent_dir=str(tmp_path), name="e")

    def mill(config, workflow_msg, msg):
        config.stages[0].milling.milling_current = 5e-9  # as if edited in the widget
        return config

    monkeypatch.setattr(experiment, "_mill_interactively", mill)
    experiment._prep_area()
    loaded = SalamiExperimentConfig.load(
        os.path.join(experiment.path, "experiment_config.yaml")
    )
    assert loaded.setup.trench.milling.stages[0].milling.milling_current == 5e-9


# ---------------------------------------------------------------------------
# Edited on the Protocol page
# ---------------------------------------------------------------------------


def test_setup_is_in_the_task_dropdown_and_swaps_the_editor(window, tmp_path):
    ui = window.salami_ui
    ui._adopt_experiment(SalamiExperiment.create(parent_dir=str(tmp_path), name="e"))
    assert ui.comboBoxTasks.itemText(0) == "Setup"
    assert ui.comboBoxTasks.currentIndex() == 1  # opens on the first task

    ui.comboBoxTasks.setCurrentIndex(0)
    assert not ui.setup_editor.isHidden()
    assert ui._task_editor_scroll.isHidden()
    page = window.protocol_page
    assert page.preview_stack.currentWidget() is ui.setup_editor.preview_widget
    assert not ui.pushButtonRemoveTask.isEnabled()  # Setup is not a task

    ui.comboBoxTasks.setCurrentIndex(1)
    assert ui.setup_editor.isHidden()
    assert page.preview_stack.currentWidget() is page.preview


def test_step_dropdown_offers_trench_and_fiducial(window):
    combo = window.salami_ui.setup_editor.step_combo
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "Mill Trenches",
        "Mill Fiducial",
    ]
    assert [combo.itemData(i) for i in range(combo.count())] == [
        TRENCH_KEY,
        FIDUCIAL_KEY,
    ]


def test_edits_reach_the_protocol_only_on_update(window, tmp_path):
    ui = window.salami_ui
    ui._adopt_experiment(SalamiExperiment.create(parent_dir=str(tmp_path), name="e"))
    editor = ui.setup_editor
    QApplication.processEvents()  # let the editor's load settle
    assert not editor.is_dirty

    editor.step_combo.setCurrentIndex(1)  # Mill Fiducial
    editor.skip_checkbox.setChecked(True)
    assert editor.is_dirty
    assert not ui.salamiExperimentConfig.setup.fiducial.skip

    ui._on_setup_update()
    assert not editor.is_dirty
    loaded = SalamiExperimentConfig.load(
        os.path.join(ui.salamiExperimentConfig.path, "experiment_config.yaml")
    )
    assert loaded.setup.fiducial.skip


def test_setup_sync_shows_the_fib_image(window, tmp_path):
    from fibsem import utils
    from fibsem.structures import BeamType

    ui = window.salami_ui
    ui._adopt_experiment(SalamiExperiment.create(parent_dir=str(tmp_path), name="e"))
    microscope, settings = utils.setup_session(manufacturer="Demo")
    ui.microscope, ui.settings = microscope, settings
    ui.setup_editor.set_microscope(microscope)

    ui._on_setup_sync(TRENCH_KEY)
    ui._sync_thread.join(timeout=30)
    QApplication.processEvents()

    image = ui.setup_editor._images[TRENCH_KEY]
    assert image.metadata.image_settings.beam_type is BeamType.ION
    assert image.metadata.image_settings.hfw == pytest.approx(
        ui.salamiExperimentConfig.setup.trench.milling.field_of_view
    )
