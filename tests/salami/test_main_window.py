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
    assert _tab_titles(window.tab_widget) == ["Microscope", "Protocol", "Workflow"]


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


def test_experiment_tab_is_replaced_by_the_workflow_page(window):
    ui = window.salami_ui
    assert ui.tabExperiment is None
    assert "Experiment" not in _tab_titles(ui.tabWidget)
    assert "Workflow" not in _tab_titles(ui.tabWidget)  # lifted into the shell
    page = window.workflow_page
    for button in (
        ui.pushButton_RunSetup,
        ui.pushButtonRunNextStep,
        ui.pushButtonRunExperiment,
        ui.pushButtonStopExperiment,
    ):
        assert page.isAncestorOf(button)


def test_task_list_is_edited_in_the_protocol_experiment_section(window):
    ui = window.salami_ui
    for button in (ui.pushButtonAddTask, ui.pushButtonRemoveTask):
        assert ui.groupBoxExpInfo.isAncestorOf(button)
        assert window.protocol_page.isAncestorOf(button)


def test_experiment_section_lists_task_and_step_counts(window, tmp_path):
    ui = window.salami_ui
    ui._adopt_experiment(SalamiExperiment.create(parent_dir=str(tmp_path), name="e"))
    config = ui.salamiExperimentConfig
    assert ui._exp_info_widgets["num_tasks"].value() == len(config.tasks)
    assert not ui._exp_info_widgets["num_tasks"].isEnabled()

    ui._exp_info_widgets["n_steps"].setValue(7)
    ui._apply_experiment_info()
    assert config.n_steps == 7
    assert window.workflow_page.label_n_steps.text() == "7"


def test_workflow_page_shows_the_experiment_read_only(window, tmp_path):
    ui = window.salami_ui
    page = window.workflow_page
    index = window.tab_widget.indexOf(page)
    assert not window.tab_widget.isTabEnabled(index)

    ui._adopt_experiment(SalamiExperiment.create(parent_dir=str(tmp_path), name="e"))
    assert window.tab_widget.isTabEnabled(index)
    assert page.label_name.text() == "e"
    assert page.label_setup.text() == "Not done"
    assert page.task_list.count() == len(ui.salamiExperimentConfig.tasks)
    # At rest every task is a circle.
    texts = [page.task_list.item(i).text() for i in range(page.task_list.count())]
    assert all(t.startswith("○") for t in texts)


def test_running_marks_done_current_and_pending_tasks(window, tmp_path):
    ui = window.salami_ui
    ui._adopt_experiment(SalamiExperiment.create(parent_dir=str(tmp_path), name="e"))
    tasks = ui.salamiExperimentConfig.tasks
    page = window.workflow_page
    page.show_tasks(tasks, step=0, next_index=1, running=True)
    glyphs = [page.task_list.item(i).text()[0] for i in range(len(tasks))]
    assert glyphs[:2] == ["✓", "▶"]
    assert all(g == "○" for g in glyphs[2:])


def test_setup_button_is_amber_until_setup_is_done(window, tmp_path):
    from fibsem.applications.salami.ui import workflow_page as module

    ui = window.salami_ui
    experiment = SalamiExperiment.create(parent_dir=str(tmp_path), name="e")
    ui._adopt_experiment(experiment)
    button = window.workflow_page.btn_setup
    assert button.styleSheet() == module.SETUP_PENDING_STYLESHEET

    experiment.status.setup_complete = True
    ui._update_experiment_status()
    assert button.styleSheet() == module.SETUP_DONE_STYLESHEET


def test_redoing_setup_asks_first(window, tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("fibsem.applications.salami.ui.SalamiUI")

    ui = window.salami_ui
    experiment = SalamiExperiment.create(parent_dir=str(tmp_path), name="e")
    experiment.status.setup_complete = True
    ui._adopt_experiment(experiment)

    asked = []
    monkeypatch.setattr(
        module.QMessageBox,
        "question",
        lambda *a, **k: asked.append(a) or module.QMessageBox.No,
    )
    started = []
    ui.setup_started.connect(lambda: started.append(1))
    ui._on_setup_clicked()
    assert asked and not started and ui._setup_thread is None


def test_a_prompt_offers_the_way_to_the_microscope_page(window, tmp_path):
    from fibsem.applications.salami.workflow_update_schema import (
        SalamiWorkflowUpdate,
    )

    # The page is disabled until there is an experiment, as during any prompt.
    window.salami_ui._adopt_experiment(
        SalamiExperiment.create(parent_dir=str(tmp_path), name="e")
    )
    page = window.workflow_page
    window.tab_widget.setCurrentWidget(page)
    page.show_update(SalamiWorkflowUpdate(msg="Check the trench", pos="Continue"))
    assert not page.btn_attention.isHidden()
    page.btn_attention.click()
    assert window.tab_widget.currentWidget() is window.salami_ui
    page.show_update(SalamiWorkflowUpdate())
    assert page.btn_attention.isHidden()


def test_protocol_page_hosts_the_moved_widgets(window):
    ui = window.salami_ui
    page = window.protocol_page
    for widget in (ui.groupBoxExpInfo, ui.comboBoxTasks, ui.groupBoxTaskParameters):
        assert page.isAncestorOf(widget), f"{widget.objectName()} is not on the page"


def test_moved_widgets_still_drive_their_handlers(window, tmp_path):
    """Reparenting must not break the wiring — same objects, new parent."""
    ui = window.salami_ui
    ui._adopt_experiment(SalamiExperiment.create(parent_dir=str(tmp_path), name="e"))

    assert ui.comboBoxTasks.count() == 3  # Setup, then the two tasks
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

    Patches the dialogs rather than the handler: the actions were connected to
    bound methods at setup, so replacing the method on the class afterwards would
    not intercept them — and the real handler would then open a modal dialog that
    never returns under the offscreen platform.
    """
    import importlib

    # The module, not the class `salami.ui` re-exports under the same name.
    salami_ui_module = importlib.import_module("fibsem.applications.salami.ui.SalamiUI")

    opened = []

    def cancel(name):
        def dialog(*args, **kwargs):
            opened.append(name)
            return None  # operator cancelled; the handler returns early

        return dialog

    monkeypatch.setattr(salami_ui_module, "create_experiment_dialog", cancel("create"))
    monkeypatch.setattr(salami_ui_module, "load_experiment_dialog", cancel("load"))

    window.btn_create_experiment.click()
    window.btn_load_experiment.click()

    # Each button reached its own dialog.
    assert opened == ["create", "load"]


def test_file_menu_mirrors_autolamella(window):
    """New / Load / Open Directory, then Load / Save Protocol, then Exit."""
    menu = window.menuBar().actions()[0].menu()
    labels = [a.text() or "---" for a in menu.actions()]
    assert labels == [
        "New Experiment",
        "Load Experiment",
        "Open Experiment Directory",
        "---",
        "Load Protocol",
        "Save Protocol",
        "---",
        "Exit",
    ]


def test_edit_menu_follows_file_and_holds_preferences(window):
    titles = [a.text() for a in window.menuBar().actions()]
    assert titles[:2] == ["File", "Edit"]
    edit_menu = window.menuBar().actions()[1].menu()
    assert edit_menu.actions() == [window.salami_ui.actionPreferences]


def test_view_and_imaging_menus_mirror_autolamella(window):
    menus = {a.text(): a.menu() for a in window.menuBar().actions()}
    assert list(menus)[:4] == ["File", "Edit", "View", "Imaging"]

    view = [a.text() for a in menus["View"].actions()]
    assert view == ["Full Screen", "Exit Full Screen", "Full Screen View"]
    imaging = [(a.text(), a.shortcut().toString()) for a in menus["Imaging"].actions()]
    assert imaging == [
        ("Acquire", "F2"),
        ("Live Acquisition", "F6"),
        ("Auto Contrast", "F9"),
        ("Auto Focus", "F11"),
    ]


def test_full_screen_actions_drive_the_quad_view(window):
    from fibsem.structures import BeamType

    ui = window.salami_ui
    ui._fullscreen_view_actions[1].trigger()  # FIB
    assert ui.view_controller.fullscreen is BeamType.ION
    ui._sync_view_menu()
    assert ui.action_toggle_fullscreen.isChecked()
    ui.action_exit_fullscreen.trigger()
    assert ui.view_controller.fullscreen is None


def test_imaging_hotkeys_do_nothing_without_a_microscope(window):
    for action in window.salami_ui._imaging_actions:
        action.trigger()  # must not raise


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


def test_tools_and_help_menus_mirror_autolamella(window):
    menus = {a.text(): a.menu() for a in window.menuBar().actions()}
    assert list(menus) == ["File", "Edit", "View", "Imaging", "Tools", "Help"]

    tools = [a.text() or "---" for a in menus["Tools"].actions()]
    assert tools == [
        "Scripts",
        "---",
        "Guided Setup...",
        "Plugins...",
        "Create Desktop Shortcut...",
    ]
    assert [a.text() for a in menus["Help"].actions()] == ["About"]


def test_scripts_need_an_experiment_and_an_idle_workflow(window, tmp_path):
    from fibsem.applications.salami.scripting import SalamiScriptContext

    ui = window.salami_ui
    context, reason = ui._script_context()
    assert context is None and "experiment" in reason

    ui._adopt_experiment(SalamiExperiment.create(parent_dir=str(tmp_path), name="s"))
    context, reason = ui._script_context()
    assert isinstance(context, SalamiScriptContext) and reason == ""
    assert context.path == tmp_path / "s"
    context.save()  # writes the protocol and status back without error


def test_salami_shortcut_points_at_salami(monkeypatch, tmp_path):
    from fibsem.applications.salami import desktop_shortcut

    assert desktop_shortcut.shortcut_path(tmp_path).stem == "Salami"
    looked_for = []
    monkeypatch.setattr(
        desktop_shortcut,
        "find_entry_point",
        lambda name: looked_for.append(name) or None,
    )
    with pytest.raises(FileNotFoundError):
        desktop_shortcut.create_desktop_shortcut(directory=tmp_path)
    assert looked_for == ["salami-ui"]
