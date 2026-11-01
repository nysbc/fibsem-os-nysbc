import logging
import os
import threading
from copy import deepcopy
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from fibsem import config as cfg
from fibsem.applications.salami.experiment import (
    CONFIG_FILENAME,
    SalamiExperiment,
    SalamiExperimentConfig,
)
from fibsem.applications.salami.preferences import (
    load_salami_preferences,
    save_salami_preferences,
)
from fibsem.applications.salami.recents import (
    SALAMI_LOG_DIR,
    last_experiment_directory,
    record_recent_experiment,
)
from fibsem.applications.salami.tasks import (
    MillingTaskConfig,
    apply_milling_region,
    ensure_task_directories,
    milling_region_for,
    sync_config_from_microscope,
    sync_config_to_microscope,
)
from fibsem.applications.salami.ui.add_task_dialog import AddTaskDialog
from fibsem.applications.salami.ui.experiment_dialogs import (
    create_experiment_dialog,
    load_experiment_dialog,
)
from fibsem.applications.salami.ui.preferences_dialog import SalamiPreferencesDialog
from fibsem.applications.salami.ui.protocol_page import RectangleSpec, preview_caption
from fibsem.applications.salami.ui.qtdesigner_files import SalamiUI as SalamiMainUI
from fibsem.applications.salami.ui.setup_editor import SetupConfigEditor
from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget
from fibsem.applications.salami.ui.workflow_page import SalamiWorkflowPage
from fibsem.applications.salami.workflow_update_schema import (
    SalamiWorkflowUpdate,
    format_duration,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import BeamType, FibsemRectangle, MicroscopeSettings
from fibsem.ui import (
    DETECTION_AVAILABLE,
    FibsemImageSettingsWidget,
    FibsemMovementWidget,
    FibsemSystemSetupWidget,
    MillingTaskViewerWidget,
    notification_service,
    stylesheets,
)
from fibsem.ui import utils as fui
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.stylesheets import GRAY_ICON_COLOR, NAPARI_STYLE
from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController
from fibsem.ui.widgets.notifications import ToastManager

# instructions
INSTRUCTIONS = {
    "NO_EXPERIMENT": "Please create or open an experiment (File -> New / Load Experiment)",
    "NOT_CONNECTED": "Please connect to the microscope (Connection -> Connect to Microscope).",
    "SETUP_EXPERIMENT": (
        "Click Setup Experiment to prepare the area and create a fiducial marker. "
        "For a protocol that needs no preparation, tick Setup Complete to enable Run."
    ),
    "INTERRUPTED": "A task was interrupted. Resolve it before running (File -> Load Experiment).",
}

# Auto-function regions drawn over the protocol preview: (config attribute,
# legend label, colour). Colours avoid the limegreen AlignmentAreaOverlay
# defaults to, which is reserved for the drift-correction area.
AUTO_FUNCTION_REGIONS = (
    ("autofocus", "Autofocus area", "deepskyblue"),
    ("autocontrast", "Autocontrast area", "orange"),
)

# The volume a milling task will collect. Yellow, distinct from the blue and
# orange auto-function areas.
MILLING_REGION_KEY = "milling_region"
MILLING_REGION_LABEL = "Volume to mill"
MILLING_REGION_COLOUR = "yellow"

NO_PREVIEW_FOR_REGION = "Sync from the microscope to place the volume on an image."


# The Setup entry in the Protocol page's task dropdown. Its data cannot clash
# with a task name, which is free text.
SETUP_TASK_KEY = "__setup__"
SETUP_TASK_LABEL = "Setup"


class SalamiUI(SalamiMainUI.Ui_MainWindow, QMainWindow):
    # The one channel from the experiment worker thread to the UI. Carries a
    # SalamiWorkflowUpdate: progress reports, user prompts, images and milling
    # configs all travel on it. Handled in the main thread by
    # `_on_workflow_update`, so Qt widget access stays thread-safe.
    workflow_update_signal = pyqtSignal(object)
    # A finished microscope sync: carries a MicroscopeSyncResult, or the Exception
    # that stopped it. Separate from workflow_update_signal, which has a
    # WAITING_FOR_UI_UPDATE handshake a sync has no use for.
    sync_finished_signal = pyqtSignal(object)
    # (step key, FibsemImage or Exception) from a setup-step sync.
    setup_sync_finished_signal = pyqtSignal(str, object)
    # A finished run: carries None, or the Exception that stopped it. The worker
    # cannot touch widgets, so re-enabling the controls happens on this slot.
    run_finished_signal = pyqtSignal(object)
    # Setup finished (True if complete); emitted from the setup worker thread.
    setup_finished_signal = pyqtSignal(bool)
    # Setup has started; the shell shows the Microscope page for its prompts.
    setup_started = pyqtSignal()
    # An experiment was created, opened or closed. The shell window listens so
    # it can enable the Protocol page and retitle itself.
    experiment_changed = pyqtSignal()

    def __init__(self, parent_ui: Optional["QWidget"] = None):
        super().__init__()
        self.setupUi(self)
        self.setStyleSheet(NAPARI_STYLE)
        # The image widget validates a dragged alignment area by writing to
        # `parent.label_instructions` and enabling `parent.pushButton_yes`.
        self.label_instructions = self.label_Instructions

        # Viewer-less: display goes through the Qt quad-view controller, which
        # sits in the left pane; the generated central widget (Salami controls)
        # becomes the right pane.
        self.viewer = None
        self.view_controller = MicroscopeViewController(parent=self)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.view_controller.widget)
        splitter.addWidget(self.centralwidget)
        splitter.setSizes([720, 460])
        self.setCentralWidget(splitter)

        # Toasts (notification_service) need a display sink or they go nowhere.
        # A bound method, not a lambda: the service outlives this window, and
        # PyQt only disconnects a slot on destruction when it belongs to the
        # destroyed object. A lambda would be called on a deleted window.
        self.toast_manager = ToastManager(self)
        notification_service._get_service().toast.connect(self._show_toast)

        # # Initialize UI components
        self.microscope: Optional[FibsemMicroscope] = None
        self.settings: Optional[MicroscopeSettings] = None

        # Worker <-> UI handshake state for ask_user / abort.
        self.WAITING_FOR_USER_INTERACTION: bool = (
            False  # signal for waiting for user interaction
        )
        self.WAITING_FOR_UI_UPDATE: bool = (
            False  # signal for waiting for UI to update with new workflow info
        )
        self.USER_RESPONSE: bool = False
        # An alignment area being edited on the FIB view for a prompt, and the
        # value the operator's click captured (see update_alignment_area_ui).
        self._editing_alignment_area: bool = False
        self.EDITED_ALIGNMENT_AREA: Optional[FibsemRectangle] = None
        self._workflow_stop_event = threading.Event()
        self._setup_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._run_thread: Optional[threading.Thread] = None

        self.workflow_update_signal.connect(self._on_workflow_update)
        self.sync_finished_signal.connect(self._on_sync_finished)
        self.run_finished_signal.connect(self._on_run_finished)
        self.setup_sync_finished_signal.connect(self._on_setup_sync_finished)
        self.setup_finished_signal.connect(self._setup_completed)

        self.system_widget = FibsemSystemSetupWidget(parent=self)
        self.image_widget: Optional[FibsemImageSettingsWidget] = None
        self.movement_widget: Optional[FibsemMovementWidget] = None
        self.milling_task_config_widget: Optional[MillingTaskViewerWidget] = None
        self.SenseAI_widget: Optional[QWidget] = None

        self.salamiExperimentConfig: Optional[SalamiExperimentConfig] = None
        self.salamiExperiment: Optional[SalamiExperiment] = None

        # Set by the shell window when this is embedded as the Microscope page.
        # None when running standalone, so every use is guarded.
        self.protocol_page = None

        # dynamic widgets for the Experiment Information groupbox, keyed by field name
        self._exp_info_widgets: dict = {}
        self._setup_experiment_info_layout()
        self._setup_workflow_page()
        self._setup_task_config_layout()

        self.tabWidget.insertTab(0, self.system_widget, "Connection")

        self.setup_connections()

    def setup_connections(self):

        self.system_widget.connected_signal.connect(self.connect_to_microscope)
        self.system_widget.disconnected_signal.connect(self.disconnect_from_microscope)

        self._setup_file_menu()
        self._setup_edit_menu()
        self._setup_view_menu()
        self._setup_imaging_menu()
        self._setup_tools_menu()
        self._setup_help_menu()

        # Setup runs setup_experiment in a worker thread so ask_user prompts
        # can be serviced by the main thread.
        self.pushButton_RunSetup.clicked.connect(self._on_setup_clicked)

        # The run loop goes on a worker thread for the same reason setup does:
        # it blocks for hours, and prompts have to be serviced by the main thread.
        self.pushButtonRunExperiment.clicked.connect(self._on_run_experiment_clicked)
        self.pushButtonRunNextStep.clicked.connect(self._on_run_next_step_clicked)
        self.pushButtonStopExperiment.clicked.connect(self._on_stop_experiment_clicked)

        # Resolve the ask_user busy-wait when the operator clicks yes/no.
        self.pushButton_yes.clicked.connect(self.push_interaction_button)
        self.pushButton_no.clicked.connect(self.push_interaction_button)

        # Task Config tab: pick a task, edit the parameters generated from its config.
        self.comboBoxTasks.currentIndexChanged.connect(self._on_selected_task_changed)

        # workflow info
        self.set_current_workflow_message(msg=None, show=False)
        self.label_Instructions.setWordWrap(True)
        self.label_workflowInformation.setWordWrap(True)

        self.update_ui()

    def _on_setup_clicked(self) -> None:
        """Run `setup_experiment` in a worker thread so ask_user prompts don't deadlock."""
        if self.salamiExperiment is None:
            QMessageBox.warning(
                self, "No Experiment", "Load or create an experiment first."
            )
            return
        if self._setup_thread is not None and self._setup_thread.is_alive():
            QMessageBox.information(
                self, "Setup Running", "Setup is already in progress."
            )
            return
        if self.salamiExperiment.status.setup_complete:
            # Redoing setup mills a new trench and fiducial, so ask first.
            reply = QMessageBox.question(
                self,
                "Run Setup Again",
                "Setup is already complete. Run it again?\n\n"
                "This mills the trench and fiducial again and replaces the "
                "alignment reference image.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self._workflow_stop_event.clear()
        self._setup_thread = threading.Thread(
            target=self._run_setup_experiment, daemon=True
        )
        self._setup_thread.start()
        self.setup_started.emit()  # type: ignore

    def _run_setup_experiment(self) -> None:

        self._task_worker_thread = threading.Thread(
            target=self._run_setup_worker, daemon=True
        )
        self._task_worker_thread.start()

    def _run_setup_worker(self) -> None:
        """Worker thread for running the experiment setup, which includes user prompts."""

        try:
            self.salamiExperiment.setup_experiment()
        except InterruptedError as e:
            logging.info(f"Setup aborted: {e}")
        except Exception:
            logging.exception("setup_experiment failed")

        finally:
            self._task_worker_thread = None
            success = (
                self.salamiExperiment is not None
                and self.salamiExperiment.status.setup_complete
            )
            # Through a signal: this is the worker thread, and the handler
            # touches widgets.
            self.setup_finished_signal.emit(success)  # type: ignore

    def _setup_completed(self, success: bool = True) -> None:

        if success:
            msg = "Experiment setup complete."
        else:
            msg = "Experiment setup incomplete. Please review the steps and try again."

        notification_service.show_toast(msg, "info")
        self._update_experiment_status()
        # Setup writes the trench and fiducial as milled back to the protocol.
        if self.salamiExperimentConfig is not None:
            self.setup_editor.set_setup(self.salamiExperimentConfig.setup)

        self.update_ui()

    @property
    def _setup_running(self) -> bool:
        return self._setup_thread is not None and self._setup_thread.is_alive()

    # ------------------------------------------------------------------
    # Setup steps on the Protocol page
    # ------------------------------------------------------------------

    def _on_setup_sync(self, key: str) -> None:
        """Acquire a FIB image at the step's field of view, on a worker thread."""
        if self.microscope is None or self.settings is None:
            return
        if self.salamiExperimentConfig is None:
            return
        if self._sync_thread is not None and self._sync_thread.is_alive():
            logging.info("A sync is already in progress.")
            return

        step = self.setup_editor.get_setup().step(key)
        image_settings = deepcopy(self.settings.image)
        image_settings.beam_type = BeamType.ION
        image_settings.hfw = step.milling.field_of_view
        image_settings.reduced_area = None
        image_settings.save = False

        self.setup_editor.set_busy(True, "Acquiring a FIB image...")
        microscope = self.microscope

        def worker():
            try:
                image = microscope.acquire_image(image_settings)
            except Exception as e:
                logging.exception("Setup sync failed")
                self.setup_sync_finished_signal.emit(key, e)  # type: ignore
                return
            self.setup_sync_finished_signal.emit(key, image)  # type: ignore

        self._sync_thread = threading.Thread(target=worker, daemon=True)
        self._sync_thread.start()

    def _on_setup_sync_finished(self, key: str, payload) -> None:
        self.setup_editor.set_busy(False)
        if isinstance(payload, Exception):
            self.setup_editor.sync_row.set_status(f"Sync failed: {payload}")
            notification_service.show(f"Microscope sync failed: {payload}", "error")
            return
        self.setup_editor.sync_row.set_status("")
        self.setup_editor.show_image(key, payload)
        self._display_sync_image(payload, BeamType.ION)

    def _on_setup_update(self) -> None:
        """Write the edited setup steps to the protocol and save it."""
        config = self.salamiExperimentConfig
        if config is None:
            return
        if self._setup_running:
            QMessageBox.information(
                self,
                "Setup Running",
                "Setup is running; update the protocol once it has finished.",
            )
            return
        config.setup = self.setup_editor.get_setup()
        self._save_config()
        self.setup_editor.mark_saved()
        notification_service.show_toast("Setup steps saved to the protocol.", "info")

    # ------------------------------------------------------------------
    # File menu
    # ------------------------------------------------------------------

    def _setup_file_menu(self) -> None:
        """Wire and label the File menu, in AutoLamella's order.

        The actions are the ones generated from the .ui file, plus two made here;
        the shell window adds these same objects to its own menu, so there is one
        handler per operation whichever menu or button reaches it.
        """
        self.actionOpen_Experiment_Directory = QAction(
            "Open Experiment Directory", self
        )
        self.actionExit = QAction("Exit", self)

        entries = (
            (
                self.actionCreate_New_Experiment,
                "New Experiment",
                "mdi:plus",
                self.create_new_experiment,
            ),
            (
                self.actionLoad_Experiment,
                "Load Experiment",
                "mdi:folder-open",
                self.open_experiment,
            ),
            (
                self.actionOpen_Experiment_Directory,
                "Open Experiment Directory",
                "mdi:folder",
                self.open_experiment_directory,
            ),
            (self.actionLoad_Config, "Load Protocol", None, self.load_protocol),
            (self.actionExport_Config, "Save Protocol", None, self.export_config),
            # The top-level window, which is the shell when embedded in one.
            (self.actionExit, "Exit", None, lambda: self.window().close()),
        )
        for action, text, icon, handler in entries:
            action.setText(text)
            if icon is not None:
                # setIconVisibleInMenu: Qt hides menu icons on macOS otherwise.
                action.setIcon(fibsem_icon(icon, color=GRAY_ICON_COLOR))
                action.setIconVisibleInMenu(True)
            action.triggered.connect(handler)

        # Standalone, this window's own File menu is the one shown. Rebuilt so it
        # matches the shell's rather than the order in the .ui file.
        self.menuFile.clear()
        self.menuFile.setTitle("File")
        self.add_file_menu_actions(self.menuFile)

    def add_file_menu_actions(self, menu) -> None:
        """Populate `menu` with the File actions, grouped as AutoLamella's are."""
        menu.addAction(self.actionCreate_New_Experiment)
        menu.addAction(self.actionLoad_Experiment)
        menu.addAction(self.actionOpen_Experiment_Directory)
        menu.addSeparator()
        menu.addAction(self.actionLoad_Config)
        menu.addAction(self.actionExport_Config)
        menu.addSeparator()
        menu.addAction(self.actionExit)

    # ------------------------------------------------------------------
    # Edit menu
    # ------------------------------------------------------------------

    def _setup_edit_menu(self) -> None:
        """Edit > Preferences, and the display preferences it governs."""
        self.actionPreferences = QAction("Preferences...", self)
        self.actionPreferences.setIcon(fibsem_icon("mdi:cog", color=GRAY_ICON_COLOR))
        self.actionPreferences.setIconVisibleInMenu(True)
        self.actionPreferences.triggered.connect(self.open_preferences)

        # Standalone, this window's own Edit menu is the one shown.
        self.menuEdit = self.menubar.addMenu("Edit")
        self.add_edit_menu_actions(self.menuEdit)

        self._prompt_beeped = False
        self._apply_display_preferences(cfg.load_user_preferences())

    def add_edit_menu_actions(self, menu) -> None:
        """Populate `menu` with the Edit actions."""
        menu.addAction(self.actionPreferences)

    def _apply_display_preferences(self, prefs: cfg.UserPreferences) -> None:
        self._sound_enabled = prefs.display.sound_enabled
        # Stored only: consumed once Salami has a workflow border to draw.
        self._border_enabled = prefs.display.border_enabled

    def open_preferences(self) -> None:
        """Edit the preferences Salami uses; save and apply them on OK.

        Movement needs no applying here: the stage widget reads it from the
        shared preferences on every move.
        """
        dialog = SalamiPreferencesDialog(
            cfg.load_user_preferences(), load_salami_preferences(), parent=self
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        prefs, salami_prefs = dialog.get_preferences()
        cfg.save_user_preferences(prefs)
        save_salami_preferences(salami_prefs)
        self._apply_display_preferences(prefs)

    # ------------------------------------------------------------------
    # View and Imaging menus
    # ------------------------------------------------------------------
    #
    # AutoLamella's, less its fluorescence entries: Salami has no FM. The F-key
    # shortcuts live on the QActions, scoped with WidgetWithChildrenShortcut and
    # added to this window, so they fire only when focus is on the Microscope
    # page (or anywhere in it, standalone) and not on the Protocol page.

    def _setup_view_menu(self) -> None:
        self.action_toggle_fullscreen = QAction("Full Screen", self)
        self.action_toggle_fullscreen.setCheckable(True)
        self.action_toggle_fullscreen.setShortcut(QKeySequence(Qt.Key_F5))
        self.action_toggle_fullscreen.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_toggle_fullscreen.triggered.connect(self._hotkey_toggle_fullscreen)

        self.action_exit_fullscreen = QAction("Exit Full Screen", self)
        self.action_exit_fullscreen.setShortcut(QKeySequence(Qt.Key_Escape))
        self.action_exit_fullscreen.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_exit_fullscreen.triggered.connect(self._hotkey_exit_fullscreen)

        self._fullscreen_view_actions = []
        for label, key in (("SEM", BeamType.ELECTRON), ("FIB", BeamType.ION)):
            action = QAction(label, self)
            action.triggered.connect(
                lambda _checked=False, k=key: self.view_controller.set_fullscreen(k)
            )
            self._fullscreen_view_actions.append(action)

        self.addAction(self.action_toggle_fullscreen)
        self.addAction(self.action_exit_fullscreen)

        self.menuView = self.menubar.addMenu("View")
        self.add_view_menu_actions(self.menuView)

    def add_view_menu_actions(self, menu) -> None:
        """Populate `menu` with the View actions, as AutoLamella's are."""
        menu.addAction(self.action_toggle_fullscreen)
        menu.addAction(self.action_exit_fullscreen)
        fullscreen_menu = menu.addMenu("Full Screen View")
        for action in self._fullscreen_view_actions:
            fullscreen_menu.addAction(action)
        # keep the checkable / enabled state honest each time the menu opens
        menu.aboutToShow.connect(self._sync_view_menu)

    def _sync_view_menu(self) -> None:
        """Reflect whether a view is full-screened (check + Exit enabled)."""
        fullscreen = self.view_controller.fullscreen is not None
        self.action_toggle_fullscreen.setChecked(fullscreen)
        self.action_exit_fullscreen.setEnabled(fullscreen)

    def _hotkey_toggle_fullscreen(self) -> None:
        """F5: toggle full screen for the selected view."""
        self.view_controller.toggle_fullscreen()

    def _hotkey_exit_fullscreen(self) -> None:
        """Esc: exit full screen (no-op when already showing the grid)."""
        self.view_controller.set_fullscreen(None)

    def _setup_imaging_menu(self) -> None:
        # Act on the selected view's beam: the view <-> radio sync keeps
        # image_widget.beam_type aligned with the selected view.
        self._imaging_actions = []
        for label, key, handler in (
            ("Acquire", Qt.Key_F2, self._hotkey_acquire),
            ("Live Acquisition", Qt.Key_F6, self._hotkey_toggle_live),
            ("Auto Contrast", Qt.Key_F9, self._hotkey_autocontrast),
            ("Auto Focus", Qt.Key_F11, self._hotkey_autofocus),
        ):
            action = QAction(label, self)
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            action.triggered.connect(handler)
            self.addAction(action)
            self._imaging_actions.append(action)

        self.menuImaging = self.menubar.addMenu("Imaging")
        self.add_imaging_menu_actions(self.menuImaging)

    def add_imaging_menu_actions(self, menu) -> None:
        """Populate `menu` with the Imaging actions."""
        for action in self._imaging_actions:
            menu.addAction(action)

    def _selected_image_widget(self):
        """The image widget when a SEM/FIB view is selected and connected, else None."""
        if self.view_controller.selected_view not in (BeamType.ELECTRON, BeamType.ION):
            return None
        return self.image_widget

    def _hotkey_acquire(self) -> None:
        """F2: acquire the selected view, unless an acquisition is already running."""
        image_widget = self._selected_image_widget()
        if image_widget is None:
            logging.info("F2 acquire: no SEM/FIB view selected, or not connected")
            return
        if getattr(image_widget, "is_acquiring", False):
            logging.info("F2 acquire: acquisition already in progress")
            return
        if self.view_controller.selected_view is BeamType.ELECTRON:
            image_widget.acquire_sem_image()
        else:
            image_widget.acquire_fib_image()

    def _hotkey_toggle_live(self) -> None:
        """F6: toggle live acquisition on the selected beam."""
        image_widget = self._selected_image_widget()
        if image_widget is None:
            logging.info("F6 live: no SEM/FIB view selected, or not connected")
            return
        image_widget.toggle_live_acquisition()

    def _hotkey_autocontrast(self) -> None:
        """F9: auto-contrast the selected beam."""
        image_widget = self._selected_image_widget()
        if image_widget is None:
            logging.info("F9 autocontrast: no SEM/FIB view selected, or not connected")
            return
        image_widget.run_autocontrast()

    def _hotkey_autofocus(self) -> None:
        """F11: auto-focus the selected beam."""
        image_widget = self._selected_image_widget()
        if image_widget is None:
            logging.info("F11 autofocus: no SEM/FIB view selected, or not connected")
            return
        image_widget.run_autofocus()

    # ------------------------------------------------------------------
    # Tools and Help menus
    # ------------------------------------------------------------------

    def _setup_tools_menu(self) -> None:
        """Scripts, then the install-level tools: AutoLamella's, less Reporting."""
        from fibsem.applications.salami.scripting import get_scripts_directory
        from fibsem.ui.widgets.script_menu import ScriptMenuController

        # A QMenu of its own so the standalone and shell Tools menus can both
        # hold it. The controller adds its two fixed entries and never touches
        # the scripts folder until the manager is opened.
        self.menuScripts = QMenu("Scripts", self)
        self.script_menu_controller = ScriptMenuController(
            menu=self.menuScripts,
            scripts_directory=get_scripts_directory,
            context_factory=self._script_context,
            notify=lambda message, level: self.toast_manager.show_toast(message, level),
            parent=self,
        )

        self.action_guided_setup = QAction("Guided Setup...", self)
        # NoRole: Qt's text heuristic moves actions starting with "setup",
        # "preference" and the like into the application menu (see AutoLamella).
        self.action_guided_setup.setMenuRole(QAction.NoRole)
        self.action_guided_setup.setToolTip(
            "A guided walkthrough that configures fibsemOS to work with your microscope"
        )
        self.action_guided_setup.triggered.connect(self._on_guided_setup)

        self.action_show_plugins = QAction("Plugins...", self)
        self.action_show_plugins.setToolTip(
            "Show every registered pattern, strategy and task, and where it came from"
        )
        self.action_show_plugins.triggered.connect(self._on_show_plugins)

        self.action_create_desktop_shortcut = QAction(
            "Create Desktop Shortcut...", self
        )
        self.action_create_desktop_shortcut.setToolTip(
            "Place a shortcut on your desktop that launches Salami from this "
            "environment"
        )
        self.action_create_desktop_shortcut.triggered.connect(
            self._on_create_desktop_shortcut
        )

        self.menuTools = self.menubar.addMenu("Tools")
        self.add_tools_menu_actions(self.menuTools)

    def add_tools_menu_actions(self, menu) -> None:
        """Populate `menu` with the Tools actions, grouped as AutoLamella's are."""
        menu.addMenu(self.menuScripts)
        # Below the separator because these act on the install, not the experiment.
        menu.addSeparator()
        menu.addAction(self.action_guided_setup)
        menu.addAction(self.action_show_plugins)
        menu.addAction(self.action_create_desktop_shortcut)

    def _setup_help_menu(self) -> None:
        self.action_about = QAction("About", self)
        self.action_about.triggered.connect(self.open_information_dialog)
        self.menuHelp = self.menubar.addMenu("Help")
        self.add_help_menu_actions(self.menuHelp)

    def add_help_menu_actions(self, menu) -> None:
        """Populate `menu` with the Help actions."""
        menu.addAction(self.action_about)

    def _script_context(self):
        """(context, reason) for a script run; a reason disables running."""
        from fibsem.applications.salami.scripting import SalamiScriptContext

        if self.salamiExperiment is None:
            return None, "Load an experiment to run scripts"
        if self._workflow_busy():
            # Tasks update the experiment on a worker thread, so a script
            # reading it mid-run would get a torn snapshot.
            return None, "Unavailable while a workflow is running"
        return (
            SalamiScriptContext(
                experiment=self.salamiExperiment,
                log=logging.info,
                microscope=self.microscope,
            ),
            "",
        )

    def _workflow_busy(self) -> bool:
        """Whether setup, a microscope sync or a run is in progress."""
        return any(
            thread is not None and thread.is_alive()
            for thread in (self._setup_thread, self._sync_thread, self._run_thread)
        )

    def _on_guided_setup(self) -> None:
        """Run the microscope guided setup, through the connection widget.

        Delegated so whatever the wizard saves is selected as the configuration
        the next connection uses, and the live microscope is handed over.
        """
        self.system_widget.run_guided_setup()

    def _on_show_plugins(self) -> None:
        """Open the read-only listing of registered extensions."""
        from fibsem.ui.widgets.plugins_dialog import PluginsDialog

        PluginsDialog(parent=self).exec_()

    def _on_create_desktop_shortcut(self) -> None:
        """Create a launcher shortcut for Salami, confirming overwrites."""
        from pathlib import Path

        from fibsem.applications.salami import desktop_shortcut

        chosen = QFileDialog.getExistingDirectory(
            self,
            "Choose Shortcut Location",
            str(desktop_shortcut.get_desktop_directory()),
        )
        if not chosen:
            return
        directory = Path(chosen)
        try:
            path = desktop_shortcut.create_desktop_shortcut(directory=directory)
        except FileExistsError as exc:
            reply = QMessageBox.question(
                self,
                "Create Desktop Shortcut",
                f"A shortcut already exists at:\n{exc}\n\nReplace it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            try:
                path = desktop_shortcut.create_desktop_shortcut(
                    overwrite=True, directory=directory
                )
            except Exception as exc2:
                self._show_shortcut_error(exc2)
                return
        except Exception as exc:
            self._show_shortcut_error(exc)
            return
        notification_service.show_toast(f"Shortcut created: {path}", "info")

    def _show_shortcut_error(self, exc: Exception) -> None:
        logging.error(f"Failed to create desktop shortcut: {exc}")
        QMessageBox.warning(
            self,
            "Create Desktop Shortcut",
            f"Could not create the desktop shortcut:\n{exc}",
        )

    def open_information_dialog(self) -> None:
        """Help > About. No connection guard: works before connecting too."""
        fui.open_information_dialog(self.microscope, self, application="Salami")

    def create_new_experiment(self):
        """Create a new experiment, from the default protocol or a saved one."""
        dialog = create_experiment_dialog(
            microscope=self.microscope, parent_ui=self, parent=self
        )
        if dialog is None:
            return
        experiment = dialog.get_experiment()
        self._adopt_experiment(
            experiment, source=dialog.source or f"new at {experiment.path}"
        )

    def open_experiment(self):
        """Choose an existing experiment and resume from its status."""
        directory = load_experiment_dialog(parent=self)
        if not directory:
            return

        try:
            experiment = SalamiExperiment.open(
                path=directory,
                microscope=self.microscope,
                parent_ui=self,
            )
        except FileNotFoundError as e:
            QMessageBox.warning(self, "Not an Experiment", str(e))
            return
        except Exception as e:
            logging.exception("Failed to open experiment")
            QMessageBox.critical(
                self, "Open Experiment Failed", f"Could not open experiment:\n{e}"
            )
            return

        self._adopt_experiment(experiment, source=f"opened from {directory}")
        self._check_interrupted_task()

    def _check_interrupted_task(self):
        """Ask the operator how to handle a task that never reported completion.

        There is no safe default: an interrupted mill has removed an unknown
        amount of material, so re-running it may double a slice and skipping it
        may leave one uncut.
        """
        experiment = self.salamiExperiment
        if experiment is None or not experiment.has_interrupted_task:
            return

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Interrupted Task")
        box.setText(
            f"Task '{experiment.status.in_flight_task}' was interrupted at slice "
            f"{experiment.status.current_step + 1}."
        )
        box.setInformativeText(
            "It may have partially completed. Re-running a milling task will "
            "remove more material; skipping it may leave a slice uncut.\n\n"
            "Inspect the sample before choosing."
        )
        retry_btn = box.addButton("Re-run Task", QMessageBox.AcceptRole)
        skip_btn = box.addButton("Skip Task", QMessageBox.DestructiveRole)
        box.addButton("Decide Later", QMessageBox.RejectRole)
        box.exec_()

        clicked = box.clickedButton()
        if clicked is retry_btn:
            experiment.resolve_interrupted_task("retry")
        elif clicked is skip_btn:
            experiment.resolve_interrupted_task("skip")
        else:
            return  # left unresolved; run() will refuse until it is

        self._update_experiment_status()
        self.update_ui()

    def open_experiment_directory(self):
        """Show the experiment's folder in the system file explorer."""
        config = self.salamiExperimentConfig
        if config is None or not config.path:
            notification_service.show_toast(
                "Create or open an experiment first.", "warning"
            )
            return
        if not os.path.isdir(config.path):
            notification_service.show_toast(
                f"Experiment directory not found: {config.path}", "error"
            )
            return
        if not fui.open_path_in_file_explorer(config.path):
            notification_service.show_toast(
                "Failed to open experiment directory.", "error"
            )

    def load_protocol(self):
        """Replace this experiment's tasks and slicing with a saved protocol.

        The experiment keeps its name, directory and run state; see
        `SalamiExperiment.replace_protocol` for when it refuses.
        """
        experiment = self.salamiExperiment
        if experiment is None:
            notification_service.show_toast(
                "Create or open an experiment first.", "warning"
            )
            return
        if self.is_running:
            # Same reason Add/Remove Task are locked: the run indexes into the list.
            QMessageBox.information(
                self, "Experiment Running", "Stop the run before loading a protocol."
            )
            return

        filename = fui.open_existing_file_dialog(
            msg="Select a protocol to load (*.yaml)",
            path=last_experiment_directory() or SALAMI_LOG_DIR,
            parent=self,
        )
        if not filename:
            return
        try:
            template = SalamiExperimentConfig.load(filename)
        except Exception as e:
            logging.exception("Failed to load protocol")
            QMessageBox.critical(
                self,
                "Invalid Protocol",
                f"The selected file is not a valid Salami protocol:\n\n{filename}\n\n{e}",
            )
            return

        confirm = QMessageBox.question(
            self,
            "Load Protocol",
            f"Replace the tasks and slice settings of {experiment.config.name!r} with "
            f"those from {template.name!r}?\n\n"
            "The experiment keeps its name, directory and progress "
            f"(slice {experiment.status.current_step} of {experiment.config.n_steps}).",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        try:
            experiment.replace_protocol(template)
        except RuntimeError as e:
            QMessageBox.warning(self, "Cannot Load Protocol", str(e))
            return
        except Exception as e:
            logging.exception("Failed to load protocol into the experiment")
            QMessageBox.critical(self, "Load Protocol Failed", str(e))
            return

        self._adopt_experiment(experiment, source=f"protocol {template.name!r} loaded")

    def export_config(self):
        """Save the current protocol for reuse on another sample."""
        if self.salamiExperimentConfig is None:
            QMessageBox.warning(
                self, "No Experiment", "Create or open an experiment first."
            )
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Protocol",
            os.path.join(SALAMI_LOG_DIR, f"{self.salamiExperimentConfig.name}.yaml"),
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if not filename:
            return

        try:
            self.salamiExperimentConfig.export(filename)
        except Exception as e:
            logging.exception("Failed to export protocol")
            QMessageBox.critical(
                self, "Export Failed", f"Could not export protocol:\n{e}"
            )
            return

        notification_service.show_toast(f"Protocol saved to {filename}", "info")

    def _show_toast(self, message: str, notification_type: str, temporary: bool):
        self.toast_manager.show_toast(message, notification_type, temporary=temporary)

    def _adopt_experiment(self, experiment: SalamiExperiment, source: str = ""):
        """Make `experiment` the active one and refresh the UI."""
        self.salamiExperiment = experiment
        # Other UI methods read from salamiExperimentConfig directly; keep it in sync.
        self.salamiExperimentConfig = experiment.config
        # Every way in — created, opened, re-protocoled — lands here, so this is
        # the one place the Load dialog's recent list is kept up to date.
        record_recent_experiment(experiment.config.path)

        msg = f"Salami experiment ready: {experiment.config.name}" + (
            f" ({source})" if source else ""
        )
        logging.info(msg)
        notification_service.show_toast(msg, "info")
        self._populate_experiment_info()
        self._populate_task_combo()
        self._update_experiment_status()
        self.update_ui()
        # A fresh experiment means a fresh protocol; the old preview is stale.
        if self.protocol_page is not None:
            self.protocol_page.clear_preview()
        self.experiment_changed.emit()

    # Fields in for_ui_display_MAIN() that are derived (not directly settable on
    # the config). `path` is where the experiment actually lives on disk and is
    # resolved on load, so editing it here would only desynchronise the two.
    _EXP_INFO_DERIVED = {"num_tasks", "path"}

    def _setup_experiment_info_layout(self):
        """Set up groupBoxExpInfo with an inner grid (dynamic fields) + a permanent Update button."""
        outer_layout = self.groupBoxExpInfo.layout()
        if outer_layout is None:
            outer_layout = QVBoxLayout(self.groupBoxExpInfo)
            self.groupBoxExpInfo.setLayout(outer_layout)

        self._exp_info_inner = QWidget(self.groupBoxExpInfo)
        self._exp_info_layout = QGridLayout(self._exp_info_inner)
        self._exp_info_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self._exp_info_inner)

        # Tasks are protocol, so the list is edited here. Remove acts on the
        # task selected beside the task parameters.
        task_row = QHBoxLayout()
        self.pushButtonAddTask = QPushButton("Add Task", self.groupBoxExpInfo)
        self.pushButtonAddTask.clicked.connect(self._on_add_task_clicked)
        self.pushButtonRemoveTask = QPushButton("Remove Task", self.groupBoxExpInfo)
        self.pushButtonRemoveTask.setToolTip("Remove the task selected under Task")
        self.pushButtonRemoveTask.clicked.connect(self._on_remove_task_clicked)
        task_row.addWidget(self.pushButtonAddTask)
        task_row.addWidget(self.pushButtonRemoveTask)
        outer_layout.addLayout(task_row)

        self.pushButtonUpdateExperiment = QPushButton(
            "Update Experiment", self.groupBoxExpInfo
        )
        self.pushButtonUpdateExperiment.clicked.connect(self._apply_experiment_info)
        outer_layout.addWidget(self.pushButtonUpdateExperiment)

    def _clear_experiment_info(self):
        """Remove dynamic widgets only; the Update button persists."""
        layout = self._exp_info_layout
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._exp_info_widgets.clear()

    def _make_display_widget(self, value) -> QWidget:
        """Pick an editor widget based on Python type. Order matters: bool before int."""
        if isinstance(value, bool):
            w = QCheckBox()
            w.setChecked(value)
            return w
        if isinstance(value, int):
            w = QSpinBox()
            w.setRange(-2_147_483_648, 2_147_483_647)
            w.setKeyboardTracking(False)
            w.setValue(value)
            return w
        if isinstance(value, float):
            w = QDoubleSpinBox()
            w.setRange(-1e12, 1e12)
            w.setDecimals(4)
            w.setKeyboardTracking(False)
            w.setValue(value)
            return w
        # str + fallback
        w = QLineEdit()
        w.setText("" if value is None else str(value))
        return w

    @staticmethod
    def _read_widget_value(widget: QWidget):
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        if isinstance(widget, QLineEdit):
            return widget.text()
        return None

    _EXP_INFO_LABELS = {
        "num_tasks": "Number of Tasks",
        "n_steps": "Number of Steps",
    }

    @classmethod
    def _format_field_name(cls, name: str) -> str:
        return cls._EXP_INFO_LABELS.get(name) or name.replace("_", " ").title()

    def _populate_experiment_info(self):
        """Build widgets dynamically from `for_ui_display_MAIN()` so new fields show up automatically."""
        self._clear_experiment_info()
        if self.salamiExperimentConfig is None:
            return

        info = self.salamiExperimentConfig.for_ui_display_MAIN()
        for row, (key, value) in enumerate(info.items()):
            label = QLabel(self._format_field_name(key))
            widget = self._make_display_widget(value)
            # Derived fields (e.g. num_tasks) can't be written back, so keep them read-only.
            if key in self._EXP_INFO_DERIVED:
                widget.setEnabled(False)
            if key == "n_steps":
                widget.setMinimum(1)
            self._exp_info_layout.addWidget(label, row, 0)
            self._exp_info_layout.addWidget(widget, row, 1)
            self._exp_info_widgets[key] = widget

    def _apply_experiment_info(self):
        """Write values from the editable info widgets back onto the experiment config."""
        config = self.salamiExperimentConfig
        if config is None:
            QMessageBox.warning(
                self, "No Experiment", "Load an experiment config first."
            )
            return

        # Hardcoded routing: dict key -> setter on config (or nested object).
        # `for_ui_display_MAIN` is itself hardcoded, so we mirror it here.
        def set_top(name):
            return lambda v: setattr(config, name, v)

        def set_advance(name):
            return lambda v: setattr(config.advance, name, v)

        setters = {
            "name": set_top("name"),
            "description": set_top("description"),
            "n_steps": lambda v: setattr(config, "n_steps", max(1, int(v))),
            "advance_milling_pattern": set_advance("advance_milling_pattern"),
            "advance_beam_shift": set_advance("advance_beam_shift"),
        }

        for key, widget in self._exp_info_widgets.items():
            if key in self._EXP_INFO_DERIVED:
                continue
            setter = setters.get(key)
            if setter is None:
                logging.warning(
                    f"No setter wired for experiment info field '{key}'; skipping."
                )
                continue
            setter(self._read_widget_value(widget))

        if self.salamiExperiment is not None:
            self.salamiExperiment.config = config

        self._save_config()

        # Reflect any side-effects (e.g. n_steps used by status, setup_complete
        # flipping the Run/Stop buttons) back into the UI.
        self._update_experiment_status()
        self.update_ui()
        notification_service.show_toast("Experiment config updated.", "info")

    def _save_config(self):
        """Persist the protocol to the experiment directory."""
        config = self.salamiExperimentConfig
        if config is None or not config.path:
            return
        try:
            config.save()
        except Exception as e:
            logging.exception("Failed to save experiment config")
            QMessageBox.critical(self, "Save Failed", f"Could not save config:\n{e}")

    # ------------------------------------------------------------------
    # Task Config tab
    # ------------------------------------------------------------------

    def _setup_task_config_layout(self):
        """Host the generated task editor inside groupBoxTaskParameters.

        The group box has no layout in the .ui file, so create one. The editor
        goes in a scroll area because a full acquisition config is taller than
        the tab.
        """
        layout = self.groupBoxTaskParameters.layout()
        if layout is None:
            layout = QVBoxLayout(self.groupBoxTaskParameters)
            self.groupBoxTaskParameters.setLayout(layout)

        self.task_config_editor = TaskConfigEditorWidget(
            microscope=self.microscope, parent=self
        )
        self.task_config_editor.config_changed.connect(self._on_task_config_changed)
        self.task_config_editor.sync_from_microscope_requested.connect(
            self._on_sync_from_microscope
        )
        self.task_config_editor.sync_to_microscope_requested.connect(
            self._on_sync_to_microscope
        )
        self.task_config_editor.slice_panel.thickness_changed.connect(
            self._on_slice_thickness_changed
        )

        scroll = QScrollArea(self.groupBoxTaskParameters)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.task_config_editor)
        layout.addWidget(scroll)
        self._task_editor_scroll = scroll

        # Shown instead of the task editor when Task is "Setup".
        self.setup_editor = SetupConfigEditor(parent=self.groupBoxTaskParameters)
        self.setup_editor.setVisible(False)
        self.setup_editor.sync_from_requested.connect(self._on_setup_sync)
        self.setup_editor.update_requested.connect(self._on_setup_update)
        layout.addWidget(self.setup_editor)

    def _populate_task_combo(self):
        """Fill the task selector from the protocol, keeping the current choice."""
        previous = self.comboBoxTasks.currentData()
        self.comboBoxTasks.blockSignals(True)
        self.comboBoxTasks.clear()
        if self.salamiExperimentConfig is not None:
            self.comboBoxTasks.addItem(SETUP_TASK_LABEL, SETUP_TASK_KEY)
            for task in self.salamiExperimentConfig.tasks:
                self.comboBoxTasks.addItem(
                    f"{task.name}  ({task.TASK_TYPE})", task.name
                )
            self.setup_editor.set_setup(self.salamiExperimentConfig.setup)
        index = self.comboBoxTasks.findData(previous)
        if index == -1:
            # Open on the first task rather than Setup: tasks are what a
            # protocol is mostly edited for.
            index = 1 if self.comboBoxTasks.count() > 1 else 0
        self.comboBoxTasks.setCurrentIndex(index)
        self.comboBoxTasks.blockSignals(False)
        self._on_selected_task_changed()

    @property
    def _setup_selected(self) -> bool:
        return self.comboBoxTasks.currentData() == SETUP_TASK_KEY

    def _on_selected_task_changed(self, *_args):
        """Load the selected task's config into the editor."""
        experiment = self.salamiExperiment
        task_name = self.comboBoxTasks.currentData()
        is_setup = task_name == SETUP_TASK_KEY
        self._task_editor_scroll.setVisible(not is_setup)
        self.setup_editor.setVisible(is_setup)
        if self.protocol_page is not None:
            self.protocol_page.show_setup_preview(is_setup)
        if is_setup:
            task_name = None
        task = experiment.find_task(task_name) if experiment and task_name else None
        self.task_config_editor.set_task_config(task)
        self._refresh_protocol_rectangles(task)
        self.update_ui()

    def _on_task_config_changed(self):
        """An edit has already landed on the config object; persist the protocol.

        Saved on every change rather than behind an Apply button: a protocol can
        be edited mid-run, so a half-applied config is worse than an eagerly
        saved one.
        """
        self._save_config()
        self._sync_rectangles_from_config()
        if not self.is_running:
            self._update_experiment_status()

    # ------------------------------------------------------------------
    # Editable regions on the protocol preview
    # ------------------------------------------------------------------

    def _refresh_protocol_rectangles(self, task):
        """Draw the selected task's editable regions over the preview image.

        Regions are edited on the preview rather than the microscope page:
        they belong to the protocol, and the preview is an image taken at the
        protocol's own settings, so the box is drawn over what it will act on.
        """
        page = self.protocol_page
        if page is None:
            return

        page.clear_rectangles()
        if task is None:
            return

        if getattr(task, "TASK_TYPE", None) == MillingTaskConfig.TASK_TYPE:
            self._draw_milling_region(task)
            return

        for attr, label, colour in AUTO_FUNCTION_REGIONS:
            auto_config = getattr(task, attr, None)
            if auto_config is None:
                continue  # milling tasks have no auto-functions
            page.draw_rectangle(
                RectangleSpec(
                    key=attr,
                    label=label,
                    colour=colour,
                    area=auto_config.area_of_interest,
                    on_changed=self._make_region_handler(auto_config),
                )
            )
            # A disabled function does nothing, so its region is noise. The
            # rectangle is created either way and only hidden, so the stored
            # value survives being switched off and back on.
            page.set_rectangle_visible(attr, auto_config.enabled)

    def _make_region_handler(self, auto_config):
        """Write a dragged rectangle back to the config it belongs to.

        Bound to the config object rather than looked up by name on each call,
        so a rectangle can never write to a task the operator has since
        switched away from.
        """

        def handler(area):
            # Update the existing rectangle rather than assigning a new one.
            # DataclassEditorWidget binds each spinbox to the owner object it
            # found when the form was built, so replacing the FibsemRectangle
            # would orphan the four controls from the value they display.
            current = auto_config.area_of_interest
            current.left, current.top = area.left, area.top
            current.width, current.height = area.width, area.height

            self._save_config()
            # The same region is also four spinboxes in the form; keep them in
            # step. refresh() re-reads from the config with signals suppressed,
            # so this cannot bounce back as another edit.
            self.task_config_editor.refresh()

        return handler

    # -- the milling volume -------------------------------------------------

    def _draw_milling_region(self, task):
        """Draw the volume this task will collect, and show what it adds up to.

        The box is not stored anywhere: it is reconstructed from the pattern and
        the step count, so there is no second copy of the volume to fall out of
        step with the protocol.
        """
        page = self.protocol_page
        config = self.salamiExperimentConfig
        panel = self.task_config_editor.slice_panel
        if page is None or config is None:
            return

        panel.set_thickness(config.advance.step_size)

        geometry = page.image_geometry
        if geometry is None:
            # Fractions of a frame that isn't there cannot be turned into metres.
            panel.set_no_region(NO_PREVIEW_FOR_REGION)
            return

        region = milling_region_for(task, config.n_steps, config.advance, geometry)
        if region is None:
            panel.set_no_region("This task has no milling stages to place.")
            return

        page.draw_rectangle(
            RectangleSpec(
                key=MILLING_REGION_KEY,
                label=MILLING_REGION_LABEL,
                colour=MILLING_REGION_COLOUR,
                area=region,
                on_changed=self._make_milling_region_handler(task),
            )
        )
        panel.set_volume(config.n_steps, config.advance.step_size)

    def _make_milling_region_handler(self, task):
        """Turn a dragged volume into a pattern and a slice count."""

        def handler(region):
            page = self.protocol_page
            config = self.salamiExperimentConfig
            geometry = page.image_geometry if page is not None else None
            if config is None or geometry is None:
                return

            config.n_steps = apply_milling_region(
                task, region, config.advance, geometry
            )
            self._save_config()
            # The pattern and the step count both moved; the form and the
            # status panel are showing the old ones.
            self.task_config_editor.refresh()
            self.task_config_editor.slice_panel.set_volume(
                config.n_steps, config.advance.step_size
            )
            self._update_experiment_status()

        return handler

    def _on_slice_thickness_changed(self, thickness: float):
        """Re-derive the slice count for the volume currently drawn.

        The box is the volume the operator asked for, so it stays put and the
        count follows: halve the thickness and you get twice as many slices of
        the same block.
        """
        config = self.salamiExperimentConfig
        page = self.protocol_page
        task = self.task_config_editor.task_config
        if config is None or task is None or thickness <= 0:
            return

        config.advance.step_size = thickness

        region = page.rectangle_area(MILLING_REGION_KEY) if page else None
        geometry = page.image_geometry if page else None
        if region is not None and geometry is not None:
            config.n_steps = apply_milling_region(
                task, region, config.advance, geometry
            )

        self._save_config()
        self.task_config_editor.refresh()
        self.task_config_editor.slice_panel.set_volume(
            config.n_steps, config.advance.step_size
        )
        self._update_experiment_status()

    def _sync_rectangles_from_config(self):
        """Follow the form: reposition the rectangles, and show or hide them.

        Covers both the region spinboxes and the function's Enabled checkbox,
        since any edit routes through here.
        """
        page = self.protocol_page
        task = self.task_config_editor.task_config
        if page is None or task is None:
            return

        # Dispatch on the task type, as _refresh_protocol_rectangles does.
        # Keying off which rectangles happen to be drawn is not the same thing:
        # selecting a task seeds its beam defaults, and that emits
        # config_changed before the rectangles have been redrawn — so this runs
        # once with the new task and the previous task's rectangles.
        if getattr(task, "TASK_TYPE", None) == MillingTaskConfig.TASK_TYPE:
            # Editing the pattern's width or position in the form moves the volume.
            config = self.salamiExperimentConfig
            geometry = page.image_geometry
            if config is not None and geometry is not None:
                region = milling_region_for(
                    task, config.n_steps, config.advance, geometry
                )
                if region is not None:
                    page.set_rectangle_area(MILLING_REGION_KEY, region)
            return

        for attr, _label, _colour in AUTO_FUNCTION_REGIONS:
            auto_config = getattr(task, attr, None)
            if auto_config is not None:
                page.set_rectangle_area(attr, auto_config.area_of_interest)
                page.set_rectangle_visible(attr, auto_config.enabled)

    # ------------------------------------------------------------------
    # Running the experiment
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._run_thread is not None and self._run_thread.is_alive()

    def _on_run_experiment_clicked(self):
        self._start_run("run", "Running experiment")

    def _on_run_next_step_clicked(self):
        self._start_run("run_next_step", "Running one step")

    def _start_run(self, method_name: str, msg: str):
        """Run the experiment on a worker thread.

        `method_name` is resolved on the experiment rather than passed as a bound
        method, so the call always lands on whichever experiment is current.
        """
        experiment = self.salamiExperiment
        if experiment is None:
            QMessageBox.warning(
                self, "No Experiment", "Create or open an experiment first."
            )
            return
        if self.microscope is None:
            QMessageBox.warning(self, "Not Connected", "Connect to a microscope first.")
            return
        if self.is_running:
            QMessageBox.information(
                self, "Already Running", "The experiment is already running."
            )
            return
        if experiment.has_interrupted_task:
            # run() would refuse anyway; say why here rather than surfacing it
            # as a failed run.
            self._check_interrupted_task()
            if experiment.has_interrupted_task:
                return
        if experiment.is_complete:
            QMessageBox.information(
                self,
                "Experiment Complete",
                f"All {experiment.config.n_steps} steps have already run. "
                "Increase the step count to continue.",
            )
            return

        self._workflow_stop_event.clear()
        self._set_running_ui(True)
        logging.info(f"{msg}: {experiment.config.name}")

        def worker():
            try:
                getattr(experiment, method_name)()
            except Exception as e:
                logging.exception("Experiment run failed")
                self.run_finished_signal.emit(e)  # type: ignore
                return
            self.run_finished_signal.emit(None)  # type: ignore

        self._run_thread = threading.Thread(target=worker, daemon=True)
        self._run_thread.start()

    def _on_stop_experiment_clicked(self):
        """Ask the run to stop at the next task boundary.

        Deliberately not an abort: the loop checks between tasks, so an
        acquisition or a mill already in progress finishes first. Interrupting a
        mill part way would leave an unknown amount of material removed.
        """
        if not self.is_running:
            return
        self._workflow_stop_event.set()
        self.pushButtonStopExperiment.setEnabled(False)
        self.set_instructions_msg("Stopping after the current task...")
        notification_service.show_toast("Stopping after the current task...", "info")

    def _set_running_ui(self, running: bool):
        """Swap Run for Stop and lock everything a run must not race with."""
        self.workflow_page.set_running(running)
        self.pushButtonRunNextStep.setEnabled(not running)
        self.pushButton_RunSetup.setEnabled(not running)
        # Structural edits to the task list are the one thing the run loop
        # cannot tolerate mid-flight — see _on_add_task_clicked.
        self.pushButtonAddTask.setEnabled(not running)
        self.pushButtonRemoveTask.setEnabled(not running)

        # The run reads n_steps every iteration and deepcopies each task config
        # per step, so edits are picked up safely at a step boundary. Syncing,
        # though, drives the column out from under an acquisition.
        self.task_config_editor.sync_row.set_microscope_available(
            self.microscope is not None and not running
        )

    def _on_run_finished(self, error):
        """Main-thread handler for a finished, stopped or failed run."""
        self._run_thread = None
        self._set_running_ui(False)
        self._update_experiment_status()
        self.update_ui()

        if isinstance(error, Exception):
            self.set_instructions_msg(f"Run failed: {error}")
            # show(), not show_toast(): a run outcome is a workflow event and
            # belongs in the notification history. These runs last days, and the
            # operator is often not watching when one ends.
            notification_service.show(f"Experiment run failed: {error}", "error")
            return

        experiment = self.salamiExperiment
        if experiment is None:
            return

        state = experiment.status.run_state
        step, total = experiment.status.current_step, experiment.config.n_steps
        if state == "completed":
            msg = f"Experiment complete: {total} steps."
        elif state == "paused":
            msg = f"Experiment paused at step {step}/{total}."
        else:
            msg = f"Run finished ({state}) at step {step}/{total}."
        self.set_instructions_msg(msg)
        notification_service.show(msg, "info")

    # ------------------------------------------------------------------
    # Microscope sync
    # ------------------------------------------------------------------

    def _on_sync_from_microscope(self):
        self._start_sync(
            sync_config_from_microscope, "Reading settings from the microscope..."
        )

    def _on_sync_to_microscope(self):
        self._start_sync(
            sync_config_to_microscope, "Applying settings to the microscope..."
        )

    def _start_sync(self, sync_fn, msg: str):
        """Run a sync on a worker thread; the acquisition it does takes seconds.

        The whole point of the spinner is that the UI keeps painting while the
        column is scanning, which it cannot do if the acquisition blocks the main
        thread.
        """
        task_config = self.task_config_editor.task_config
        if self.microscope is None or task_config is None:
            return
        if self._sync_thread is not None and self._sync_thread.is_alive():
            logging.info("A sync is already in progress.")
            return

        self.task_config_editor.sync_row.set_busy(True, msg)
        # Lock the form for the duration, the way the image widget greys out its
        # settings while acquiring. The sync writes config values and then
        # refreshes the form, so an edit made in the meantime would be lost; and
        # switching task mid-sync would land the result under the wrong task.
        self.task_config_editor.set_editing_enabled(False)
        self.comboBoxTasks.setEnabled(False)

        def worker():
            try:
                result = sync_fn(self.microscope, task_config)
            except Exception as e:
                logging.exception("Microscope sync failed")
                self.sync_finished_signal.emit(e)  # type: ignore
                return
            self.sync_finished_signal.emit(result)  # type: ignore

        self._sync_thread = threading.Thread(target=worker, daemon=True)
        self._sync_thread.start()

    def _on_sync_finished(self, payload):
        """Main-thread handler for a completed sync.

        Everything that touches a widget happens here, never on the worker.
        """
        sync_row = self.task_config_editor.sync_row
        sync_row.set_busy(False)
        # Unlock before the early return, so a failed sync doesn't leave the
        # form permanently inert.
        self.task_config_editor.set_editing_enabled(True)
        self.comboBoxTasks.setEnabled(True)

        if isinstance(payload, Exception):
            sync_row.set_status(f"Sync failed: {payload}")
            notification_service.show(f"Microscope sync failed: {payload}", "error")
            return

        # A sync from the microscope rewrites config values, so the form has to
        # re-read them; persist afterwards, as with any other edit.
        if payload.direction == "from_microscope":
            self.task_config_editor.refresh()
            self._save_config()

        if payload.image is not None:
            # The Protocol page first: that is where the sync buttons are, so it
            # is where the operator is looking. The microscope-page panes are
            # updated too, so they don't go stale behind the scenes.
            if self.protocol_page is not None:
                self.protocol_page.show_preview(payload.image, preview_caption(payload))
            self._display_sync_image(payload.image, payload.image_settings.beam_type)

        sync_row.set_status(payload.summary)
        notification_service.show_toast(payload.summary, "info")

    def _display_sync_image(self, image, beam_type=None):
        """Show the image the sync acquired, without saving it anywhere.

        Routed by beam type: an acquisition task syncs against the SEM, a
        milling task against the ion column, and each belongs in its own pane.
        """
        if self.image_widget is None:
            logging.debug("No image widget available; dropping the sync image.")
            return
        if beam_type is BeamType.ION:
            self.image_widget.ib_image = image
        else:
            self.image_widget.eb_image = image
        self.image_widget._on_acquire(image)

    def _setup_workflow_page(self):
        """Replace the Experiment tab with the Workflow page.

        The page owns the run controls; they are aliased under their old names
        so every handler and enable rule keeps working against them. Experiment
        information moves to the Task Config tab, renamed Protocol. The shell
        window lifts both out of this strip into pages of their own.
        """
        page = self.workflow_page = SalamiWorkflowPage(parent=self)
        self.pushButton_RunSetup = page.btn_setup
        self.pushButtonRunNextStep = page.btn_run_next_step
        self.pushButtonRunExperiment = page.btn_run
        self.pushButtonStopExperiment = page.btn_stop

        self.gridLayout_4.addWidget(self.groupBoxExpInfo, 2, 0, 1, 2)
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tabTaskConfig), "Protocol"
        )

        index = self.tabWidget.indexOf(self.tabExperiment)
        self.tabWidget.removeTab(index)
        self.tabExperiment.deleteLater()
        self.tabExperiment = None
        self.tabWidget.insertTab(index, page, "Workflow")

    # ------------------------------------------------------------------
    # Adding and removing tasks
    # ------------------------------------------------------------------
    #
    # Both change the *length* of `config.tasks`, which `status.next_task_index`
    # indexes into, so neither is safe while a run is in progress — that is the
    # whole reason `enabled` and `skip_once` exist. Both are locked while running.

    def _on_add_task_clicked(self):
        experiment = self.salamiExperiment
        if experiment is None:
            QMessageBox.warning(
                self, "No Experiment", "Create or open an experiment first."
            )
            return

        dialog = AddTaskDialog(
            existing_names=[t.name for t in experiment.config.tasks], parent=self
        )
        if dialog.exec_() != QDialog.Accepted:
            return

        task = dialog.task_config()
        experiment.config.add_task(task)
        # The task writes its output here; created now rather than at run time so
        # a protocol that cannot be written to fails while the operator is
        # looking at it, not three hours into a run.
        ensure_task_directories(task, experiment.config.path)
        self._save_config()

        self._populate_task_combo()
        index = self.comboBoxTasks.findData(task.name)
        if index != -1:
            self.comboBoxTasks.setCurrentIndex(index)
        self._update_experiment_status()
        self.update_ui()

        logging.info(f"Added task {task.name!r} ({task.TASK_TYPE}) to the protocol.")
        notification_service.show_toast(f"Added task {task.name!r}", "info")

    def _on_remove_task_clicked(self):
        experiment = self.salamiExperiment
        if experiment is None:
            return

        task_name = self.comboBoxTasks.currentData()
        if not task_name or task_name == SETUP_TASK_KEY:
            QMessageBox.information(self, "No Task", "Select a task to remove.")
            return

        confirm = QMessageBox.question(
            self,
            "Remove Task",
            f"Remove {task_name!r} from the protocol?\n\n"
            "Its output directory and any images already acquired are left on disk.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        if not experiment.config.remove_task(task_name):
            return

        # A pending one-shot skip would otherwise name a task that no longer
        # exists, and `_require_task` raises KeyError on it.
        experiment.cancel_skip_once(task_name)
        self._save_config()

        self._populate_task_combo()
        self._update_experiment_status()
        self.update_ui()

        logging.info(f"Removed task {task_name!r} from the protocol.")
        notification_service.show_toast(f"Removed task {task_name!r}", "info")

    def _update_experiment_status(self):
        """Show the experiment at rest on the Workflow page."""
        self.workflow_page.show_experiment(
            self.salamiExperiment, running=self.is_running
        )
        widget = self._exp_info_widgets.get("num_tasks")
        if widget is not None and self.salamiExperimentConfig is not None:
            widget.setValue(len(self.salamiExperimentConfig.tasks))

    def connect_to_microscope(self):

        self.microscope = self.system_widget.microscope
        self.settings = self.system_widget.settings

        # An experiment can be created and edited offline, so one may already be
        # loaded; hand it the microscope rather than making the user reload.
        if self.salamiExperiment is not None:
            self.salamiExperiment.microscope = self.microscope

        # Lets the generated editors offer the values this column supports.
        self.task_config_editor.set_microscope(self.microscope)
        self.setup_editor.set_microscope(self.microscope)

        self.update_microscope_ui()
        self.update_ui()

    def disconnect_from_microscope(self):
        self.microscope = None
        self.settings = None
        if self.salamiExperiment is not None:
            self.salamiExperiment.microscope = None
        self.task_config_editor.set_microscope(None)
        self.setup_editor.set_microscope(None)
        self.update_microscope_ui()
        self.update_ui()

    def update_microscope_ui(self):
        """Update the ui based on the current state of the microscope."""

        if self.microscope is not None:
            # Image widget FIRST: FibsemMovementWidget requires parent.image_widget.
            self.image_widget = FibsemImageSettingsWidget(
                microscope=self.microscope,
                image_settings=self.settings.image,  # type: ignore
                parent=self,
            )
            self.movement_widget = FibsemMovementWidget(
                microscope=self.microscope,
                parent=self,
            )
            # Milling tab: the viewer widget composes MillingTaskConfigWidget2 +
            # FibsemMillingWidget2 and draws patterns on the FIB canvas via the
            # controller (discovered through the injected image_widget).
            self.milling_task_config_widget = MillingTaskViewerWidget(
                microscope=self.microscope,
                image_widget=self.image_widget,
                parent=self,
            )

            # add widgets to tabs
            self.tabWidget.addTab(self.image_widget, "Image")
            self.tabWidget.addTab(self.movement_widget, "Movement")
            # TODO: replace this with MillingTaskWidget to support multi-task configuration
            self.tabWidget.addTab(self.milling_task_config_widget, "Milling")
            # SenseAI is an optional integration, switched on in the microscope
            # configuration; most columns have no external scan generator.
            senseai = getattr(self.microscope.system, "senseai", None)
            if getattr(senseai, "enabled", False):
                from fibsem.ui.SenseAIControlWidget import SenseAIControlWidget

                self.SenseAI_widget = SenseAIControlWidget(
                    microscope=self.microscope,
                    parent=self,
                )
                self.tabWidget.addTab(self.SenseAI_widget, "SenseAI")

            # self.image_widget.acquisition_progress_signal.connect(self.handle_acquisition_update)

        else:
            if self.image_widget is None:
                return

            # remove tabs
            if self.SenseAI_widget is not None:
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.SenseAI_widget))
                self.SenseAI_widget.deleteLater()
                self.SenseAI_widget = None
            if self.milling_task_config_widget is not None:
                self.tabWidget.removeTab(
                    self.tabWidget.indexOf(self.milling_task_config_widget)
                )
                self.milling_task_config_widget.deleteLater()
                self.milling_task_config_widget = None
            if self.movement_widget is not None:
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.movement_widget))
                self.movement_widget.deleteLater()
                self.movement_widget = None
            if self.image_widget is not None:
                self.tabWidget.removeTab(self.tabWidget.indexOf(self.image_widget))
                # self.image_widget.acquisition_progress_signal.disconnect(self.handle_acquisition_update)
                self.image_widget.deleteLater()
                self.image_widget = None
            # Reset the shared canvases now that the image widget is gone.
            self.view_controller.clear()

    def update_ui(self):
        """Update the ui based on the current state of the application."""

        # state flags

        is_microscope_connected = bool(self.microscope is not None)
        is_experiment_loaded = bool(self.salamiExperiment is not None)
        is_setup_complete = bool(
            is_experiment_loaded and self.salamiExperiment.status.setup_complete
        )
        has_interrupted_task = bool(
            is_experiment_loaded and self.salamiExperiment.has_interrupted_task
        )

        # Both work offline so a protocol can be prepared away from the
        # microscope. Task Config used to be gated on a connection because it
        # was going to use the live beam widgets; the generated editors read and
        # write the config only, so the gate is no longer needed.
        workflow_index = self.tabWidget.indexOf(self.workflow_page)
        if workflow_index != -1:
            self.tabWidget.setTabVisible(workflow_index, is_experiment_loaded)
        # Task Config is only in this strip when running standalone; the shell
        # window moves it onto the Protocol page, and indexOf returns -1 then.
        task_config_index = self.tabWidget.indexOf(self.tabTaskConfig)
        if task_config_index != -1:
            self.tabWidget.setTabVisible(task_config_index, is_experiment_loaded)

        # Setup and running need hardware; an unresolved interrupted task blocks
        # the run until the operator has decided what to do about it.
        running = self.is_running
        self.pushButton_RunSetup.setEnabled(
            is_experiment_loaded and is_microscope_connected and not running
        )
        can_run = (
            is_experiment_loaded
            and is_microscope_connected
            and is_setup_complete
            and not has_interrupted_task
        )
        # Stop is the inverse of the others: it is the only one that means
        # anything while a run is in flight.
        self.pushButtonRunExperiment.setEnabled(can_run and not running)
        self.pushButtonRunNextStep.setEnabled(can_run and not running)
        self.workflow_page.set_running(running)

        # Adding or removing a task changes the length of `config.tasks`, which
        # `status.next_task_index` indexes into while `_run_step` iterates. Both
        # need no microscope, so a protocol can still be assembled offline.
        self.pushButtonAddTask.setEnabled(is_experiment_loaded and not running)
        self.pushButtonRemoveTask.setEnabled(
            is_experiment_loaded
            and not running
            and self.comboBoxTasks.currentData() not in (None, SETUP_TASK_KEY)
        )
        # Setup reads the protocol's setup section as it runs.
        self.setup_editor.set_update_enabled(
            is_experiment_loaded and not self._setup_running
        )

        if running:
            return  # the run's own updates own the instruction line

        if not is_experiment_loaded:
            self.set_instructions_msg(INSTRUCTIONS["NO_EXPERIMENT"])
        elif not is_microscope_connected:
            self.set_instructions_msg(INSTRUCTIONS["NOT_CONNECTED"])
        elif has_interrupted_task:
            self.set_instructions_msg(INSTRUCTIONS["INTERRUPTED"])
        elif not is_setup_complete:
            self.set_instructions_msg(INSTRUCTIONS["SETUP_EXPERIMENT"])

    def set_instructions_msg(
        self,
        msg: str = "",
        pos: Optional[str] = None,
        neg: Optional[str] = None,
    ) -> None:
        """Set the instructions message, and user interaction buttons.
        Args:
            msg: The message to display.
            pos: The positive button text.
            neg: The negative button text.
        """
        self.label_Instructions.setText(msg)
        self.pushButton_yes.setText(pos)
        self.pushButton_no.setText(neg)

        # enable buttons
        self.pushButton_yes.setEnabled(pos is not None)
        self.pushButton_yes.setVisible(pos is not None)
        self.pushButton_no.setEnabled(neg is not None)
        self.pushButton_no.setVisible(neg is not None)

        if pos == "Run Milling":
            self.pushButton_yes.setStyleSheet(
                stylesheets.RUN_WORKFLOW_BUTTON_STYLESHEET
            )
        else:
            self.pushButton_yes.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)

    def set_current_workflow_message(
        self, msg: Optional[str] = None, show: bool = True
    ):
        """Set the current workflow information message"""
        if msg is not None:
            self.label_workflowInformation.setText(msg)
        self.label_workflowInformation.setVisible(show)

    def push_interaction_button(self):
        """Handle the user interaction with the workflow."""

        # positve / negative response
        self.USER_RESPONSE = bool(self.sender() == self.pushButton_yes)
        # Read the edited area here, on the main thread, before releasing the
        # worker that is waiting for it; then take the box down.
        if self._editing_alignment_area:
            self._editing_alignment_area = False
            if self.image_widget is not None:
                self.EDITED_ALIGNMENT_AREA = deepcopy(
                    self.image_widget.get_alignment_area()
                )
                self.image_widget.clear_alignment_area()
        self.WAITING_FOR_USER_INTERACTION = False

    def _on_workflow_update(self, update: SalamiWorkflowUpdate) -> None:
        """Apply a worker -> UI update on the main thread.

        Handles both progress reports and UI actions the worker has asked for.
        Any failure still has to clear WAITING_FOR_UI_UPDATE, otherwise the
        worker thread busy-waits on it forever.
        """
        try:
            self._apply_workflow_update(update)
        except Exception:
            logging.exception("Failed to apply workflow update")
        finally:
            self.WAITING_FOR_UI_UPDATE = False

    def _apply_workflow_update(self, update: SalamiWorkflowUpdate) -> None:
        # Update the image viewer. _on_acquire pushes to the shared canvas via
        # the view controller (controller.set_image), so no explicit refresh.
        if update.sem_image is not None or update.fib_image is not None:
            if self.image_widget is None:
                logging.warning("No image widget available; dropping image update.")
            else:
                if update.sem_image is not None:
                    self.image_widget.eb_image = update.sem_image
                    self.image_widget._on_acquire(update.sem_image)
                if update.fib_image is not None:
                    self.image_widget.ib_image = update.fib_image
                    self.image_widget._on_acquire(update.fib_image)

        # Push (or clear) the milling pattern the worker asked us to display.
        if self.milling_task_config_widget is not None:
            if update.milling_config is not None:
                self.tabWidget.setCurrentWidget(self.milling_task_config_widget)
                self.milling_task_config_widget.set_config(update.milling_config)
            elif update.clear_milling_config:
                self.milling_task_config_widget.set_config(FibsemMillingTaskConfig())

        self.set_instructions_msg(msg=update.msg, pos=update.pos, neg=update.neg)

        if update.alignment_area is not None:
            if self.image_widget is None:
                logging.warning("No image widget; cannot show the alignment area.")
            else:
                self.tabWidget.setCurrentWidget(self.image_widget)
                self.image_widget.toggle_alignment_area(update.alignment_area)
                self._editing_alignment_area = True

        # Beep once on entering a prompt, as AutoLamella does when it starts
        # waiting for the user; re-armed when the prompt clears.
        if update.pos is None:
            self._prompt_beeped = False
        elif self._sound_enabled and not self._prompt_beeped:
            QApplication.beep()
            self._prompt_beeped = True

        self.set_current_workflow_message(
            msg=update.workflow_msg,
            show=bool(update.workflow_msg),
        )

        self._update_progress_from(update)

    def _update_progress_from(self, update: SalamiWorkflowUpdate) -> None:
        """Drive the Workflow page from a live setup or run.

        Every update can raise or clear the prompt indicator; only progress-
        bearing ones move the progress and the task marker.
        """
        page = self.workflow_page
        page.show_update(update)
        config = self.salamiExperimentConfig
        if update.total_steps > 0 and config is not None:
            page.show_tasks(
                config.tasks,
                step=update.current_step,
                next_index=update.current_task_index,
                running=True,
            )


def main():
    """Launch the full application: the shell window with all its pages.

    Kept here as well as in `main_window` because the `salami-ui` console script
    is generated against this path at install time, so pointing it elsewhere
    would need a reinstall to take effect.
    """
    from fibsem.applications.salami.ui.main_window import main as _main

    _main()


if __name__ == "__main__":
    main()
