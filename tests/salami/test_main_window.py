"""The shell window and its pages.

The shell owns a tab per page and reparents the protocol-editing widgets out of
the Microscope page's tab strip. What is worth pinning is that the move happened
and that nothing was left behind or duplicated — the handlers all still point at
the same objects, so a widget landing in the wrong place is the failure mode.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.applications.salami.experiment import SalamiExperiment


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def microscope():
    from fibsem import utils

    scope, _settings = utils.setup_session(manufacturer="Demo")
    return scope


@pytest.fixture
def window(qapp):
    from fibsem.applications.salami.ui.main_window import SalamiMainWindow

    return SalamiMainWindow()


@pytest.fixture
def connected(window, microscope):
    """A window with a microscope and an experiment, as an operator would have."""
    from fibsem import utils

    ui = window.salami_ui
    _scope, settings = utils.setup_session(manufacturer="Demo")
    ui.microscope, ui.settings = microscope, settings
    ui.update_microscope_ui()
    ui.task_config_editor.set_microscope(microscope)
    return window


def _tab_titles(tab_widget):
    return [tab_widget.tabText(i) for i in range(tab_widget.count())]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def test_shell_has_a_page_per_area(window):
    assert _tab_titles(window.tab_widget) == ["Microscope", "Protocol"]


def test_protocol_page_is_disabled_until_there_is_an_experiment(window, tmp_path):
    index = window.tab_widget.indexOf(window.protocol_page)
    assert not window.tab_widget.isTabEnabled(index)
    assert window.tab_widget.tabToolTip(index)  # says why

    experiment = SalamiExperiment.create(parent_dir=str(tmp_path), name="e")
    window.salami_ui._adopt_experiment(experiment)

    assert window.tab_widget.isTabEnabled(index)
    assert not window.tab_widget.tabToolTip(index)


def test_window_title_names_the_open_experiment(window, tmp_path):
    assert "—" not in window.windowTitle()

    window.salami_ui._adopt_experiment(
        SalamiExperiment.create(parent_dir=str(tmp_path), name="my-sample")
    )
    assert "my-sample" in window.windowTitle()


def test_inner_menu_bar_is_hidden_and_actions_are_reused(window):
    """One definition of "New Experiment", surfaced by the shell."""
    assert not window.salami_ui.menuBar().isVisible()

    file_menu = window.menuBar().actions()[0].menu()
    actions = set(file_menu.actions())
    assert window.salami_ui.actionCreate_New_Experiment in actions
    assert window.salami_ui.actionLoad_Experiment in actions


# ---------------------------------------------------------------------------
# What moved where
# ---------------------------------------------------------------------------


def test_task_config_tab_is_gone_from_the_microscope_strip(window):
    ui = window.salami_ui
    assert "Task Config" not in _tab_titles(ui.tabWidget)
    assert ui.tabWidget.indexOf(ui.tabTaskConfig) == -1


def test_experiment_tab_keeps_only_status_and_controls(window):
    ui = window.salami_ui
    assert ui.groupBoxExpStatus.parent() is ui.tabExperiment
    assert ui.groupBox_ExperimentControls.parent() is ui.tabExperiment
    # Experiment info moved to the Protocol page.
    assert ui.groupBoxExpInfo.parent() is not ui.tabExperiment


def test_protocol_page_hosts_the_moved_widgets(window):
    ui = window.salami_ui
    page = window.protocol_page
    for widget in (ui.groupBoxExpInfo, ui.comboBoxTasks, ui.groupBoxTaskParameters):
        assert page.isAncestorOf(widget), f"{widget.objectName()} is not on the page"


def test_moved_widgets_still_drive_their_handlers(window, tmp_path):
    """Reparenting must not break the wiring — same objects, new parent."""
    ui = window.salami_ui
    ui._adopt_experiment(SalamiExperiment.create(parent_dir=str(tmp_path), name="e"))

    assert ui.comboBoxTasks.count() == 2
    ui.comboBoxTasks.setCurrentIndex(ui.comboBoxTasks.findData("Acquisition_Task"))
    assert ui.task_config_editor.task_config.name == "Acquisition_Task"


def test_standalone_ui_still_has_its_task_config_tab(qapp):
    """SalamiUI must remain usable on its own, not only inside the shell."""
    from fibsem.applications.salami.ui.SalamiUI import SalamiUI

    ui = SalamiUI()
    assert ui.tabWidget.indexOf(ui.tabTaskConfig) != -1
    assert ui.protocol_page is None


# ---------------------------------------------------------------------------
# The preview column
# ---------------------------------------------------------------------------


def test_preview_starts_empty(window):
    preview = window.protocol_page.preview
    assert preview.placeholder.isVisibleTo(preview)
    assert not preview.canvas.isVisibleTo(preview)


def test_sync_result_lands_on_the_preview(connected, tmp_path, microscope):
    """The sync buttons are on this page, so the image belongs here too."""
    from fibsem.applications.salami.tasks import sync_config_from_microscope

    ui = connected.salami_ui
    experiment = SalamiExperiment.create(
        parent_dir=str(tmp_path), name="e", microscope=microscope, parent_ui=ui
    )
    ui._adopt_experiment(experiment)

    result = sync_config_from_microscope(
        microscope, experiment.find_task("Acquisition_Task")
    )
    ui._on_sync_finished(result)

    preview = connected.protocol_page.preview
    assert preview.canvas.isVisibleTo(preview)
    assert not preview.placeholder.isVisibleTo(preview)
    assert preview.canvas.img_width is not None
    assert "SEM" in preview.caption_label.text()


def test_preview_caption_names_the_ion_column_for_a_milling_task(
    connected, tmp_path, microscope
):
    from fibsem.applications.salami.tasks import sync_config_from_microscope

    ui = connected.salami_ui
    experiment = SalamiExperiment.create(
        parent_dir=str(tmp_path), name="e", microscope=microscope, parent_ui=ui
    )
    ui._adopt_experiment(experiment)

    result = sync_config_from_microscope(microscope, experiment.find_task("Slice_Task"))
    ui._on_sync_finished(result)

    assert "FIB" in connected.protocol_page.preview.caption_label.text()


def test_opening_another_experiment_clears_a_stale_preview(
    connected, tmp_path, microscope
):
    from fibsem.applications.salami.tasks import sync_config_from_microscope

    ui = connected.salami_ui
    first = SalamiExperiment.create(
        parent_dir=str(tmp_path), name="first", microscope=microscope, parent_ui=ui
    )
    ui._adopt_experiment(first)
    ui._on_sync_finished(
        sync_config_from_microscope(microscope, first.find_task("Acquisition_Task"))
    )

    ui._adopt_experiment(
        SalamiExperiment.create(
            parent_dir=str(tmp_path), name="second", microscope=microscope, parent_ui=ui
        )
    )

    preview = connected.protocol_page.preview
    assert preview.placeholder.isVisibleTo(preview)
    assert not preview.canvas.isVisibleTo(preview)


# ---------------------------------------------------------------------------
# Tab bar: icons and the corner widget
# ---------------------------------------------------------------------------


def test_tabs_have_icons(window):
    for i in range(window.tab_widget.count()):
        assert not window.tab_widget.tabIcon(i).isNull(), (
            f"{window.tab_widget.tabText(i)} has no icon"
        )


def test_corner_widget_carries_the_global_controls(window):
    """Which experiment is open, and how to change it, apply to every page."""
    assert window.tab_widget.cornerWidget() is not None
    assert window.experiment_name_label.text() == "No Experiment"
    assert window.btn_create_experiment.text() == "Create Experiment"
    assert window.btn_load_experiment.text() == "Load Experiment"


def test_corner_buttons_are_usable_offline(window):
    """A Salami protocol can be prepared without a microscope, unlike autolamella."""
    assert window.salami_ui.microscope is None
    assert window.btn_create_experiment.isEnabled()
    assert window.btn_load_experiment.isEnabled()


def test_corner_buttons_trigger_the_file_menu_actions(window, monkeypatch):
    """One handler per operation, whichever control the operator used.

    Patches the directory chooser rather than the handler: the actions were
    connected to bound methods at setup, so replacing the method on the class
    afterwards would not intercept them — and the real handler would then open a
    modal dialog that never returns under the offscreen platform.
    """
    from PyQt5.QtWidgets import QFileDialog

    opened = []

    def cancel(*args, **kwargs):
        opened.append(args[1] if len(args) > 1 else "")
        return ""  # operator cancelled; the handler returns early

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(cancel))

    window.btn_create_experiment.click()
    window.btn_load_experiment.click()

    # Both reached a handler, and each asked for a directory with its own prompt.
    assert len(opened) == 2
    assert opened[0] != opened[1]


def test_corner_label_follows_the_open_experiment(window, tmp_path):
    window.salami_ui._adopt_experiment(
        SalamiExperiment.create(parent_dir=str(tmp_path), name="sample-42")
    )
    assert window.experiment_name_label.text() == "sample-42"


def test_bell_is_wired_to_the_toast_manager(window):
    assert window.salami_ui.toast_manager.notification_bell is window.notification_bell
    assert window.notification_bell.count == 0


def test_bell_records_workflow_events_not_chatter(window, tmp_path):
    """Runs last days, so a finished run has to still be visible on return.

    Transient validation messages go through notification_service.show_toast and
    are deliberately excluded from history; run outcomes go through show().
    """
    ui = window.salami_ui
    experiment = SalamiExperiment.create(parent_dir=str(tmp_path), name="e")
    ui._adopt_experiment(experiment)  # emits a transient toast
    assert window.notification_bell.count == 0

    ui._on_run_finished(None)
    assert window.notification_bell.count == 1

    ui._on_run_finished(RuntimeError("detector fault"))
    assert window.notification_bell.count == 2
