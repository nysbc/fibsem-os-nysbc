import logging
import os
import threading
from typing import Optional

from fibsem.applications.salami.experiment import (
    CONFIG_FILENAME,
    SalamiExperiment,
    SalamiExperimentConfig,
    default_experiment_name,
)
from fibsem.applications.salami.workflow_update_schema import (
    SalamiWorkflowUpdate,
    format_duration,
)

# Default home for new Salami experiments: <repo>/fibsem/applications/salami/log/
SALAMI_LOG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "log")
)


from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
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
from fibsem.applications.salami.ui.protocol_page import RectangleSpec, preview_caption
from fibsem.applications.salami.ui.qtdesigner_files import SalamiUI as SalamiMainUI
from fibsem.applications.salami.ui.task_editor import TaskConfigEditorWidget
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import BeamType, MicroscopeSettings
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
from fibsem.ui.stylesheets import NAPARI_STYLE
from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController
from fibsem.ui.widgets.notifications import ToastManager

# instructions
INSTRUCTIONS = {
    "NO_EXPERIMENT": "Please create or open an experiment (File -> New / Open Experiment)",
    "NOT_CONNECTED": "Please connect to the microscope (Connection -> Connect to Microscope).",
    "SETUP_EXPERIMENT": (
        "Click Setup Experiment to prepare the area and create a fiducial marker. "
        "For a protocol that needs no preparation, tick Setup Complete to enable Run."
    ),
    "INTERRUPTED": "A task was interrupted. Resolve it before running (File -> Open Experiment).",
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
    # A finished run: carries None, or the Exception that stopped it. The worker
    # cannot touch widgets, so re-enabling the controls happens on this slot.
    run_finished_signal = pyqtSignal(object)
    # An experiment was created, opened or closed. The shell window listens so
    # it can enable the Protocol page and retitle itself.
    experiment_changed = pyqtSignal()

    def __init__(self, parent_ui: Optional["QWidget"] = None):
        super().__init__()
        self.setupUi(self)
        self.setStyleSheet(NAPARI_STYLE)

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
        self._workflow_stop_event = threading.Event()
        self._setup_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._run_thread: Optional[threading.Thread] = None

        self.workflow_update_signal.connect(self._on_workflow_update)
        self.sync_finished_signal.connect(self._on_sync_finished)
        self.run_finished_signal.connect(self._on_run_finished)

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
        self._setup_experiment_status_layout()
        self._setup_task_config_layout()

        self.tabWidget.insertTab(0, self.system_widget, "Connection")

        self.setup_connections()

    def setup_connections(self):

        self.system_widget.connected_signal.connect(self.connect_to_microscope)
        self.system_widget.disconnected_signal.connect(self.disconnect_from_microscope)

        # Three distinct operations, deliberately separated: "New From Protocol"
        # re-homes a copy rather than adopting the source config in place, which
        # is what used to overwrite the original experiment's status file.
        self.actionLoad_Config.triggered.connect(self.new_experiment_from_template)
        self.actionExport_Config.triggered.connect(self.export_config)

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

        self.actionCreate_New_Experiment.triggered.connect(self.create_new_experiment)

        self.actionLoad_Experiment.triggered.connect(self.open_experiment)

        # Task Config tab: pick a task, edit the parameters generated from its config.
        self.comboBoxTasks.currentIndexChanged.connect(self._on_selected_task_changed)

        # Menu labels, so the three operations read distinctly.
        self.actionCreate_New_Experiment.setText("New Experiment")
        self.actionLoad_Experiment.setText("Open Experiment")
        self.actionLoad_Config.setText("New Experiment From Protocol")
        self.actionExport_Config.setText("Export Protocol")

        # Run/Stop button styling.
        self.pushButtonRunExperiment.setStyleSheet(
            stylesheets.RUN_WORKFLOW_BUTTON_STYLESHEET
        )
        self.pushButtonStopExperiment.setStyleSheet(
            stylesheets.STOP_WORKFLOW_BUTTON_STYLESHEET
        )
        self.pushButtonStopExperiment.setVisible(
            False
        )  # only show Stop when an experiment is running

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

        self._workflow_stop_event.clear()
        self._setup_thread = threading.Thread(
            target=self._run_setup_experiment, daemon=True
        )
        self._setup_thread.start()

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
            self._setup_completed(success=success)

    def _setup_completed(self, success: bool = True) -> None:

        if success:
            msg = "Experiment setup complete."
        else:
            msg = "Experiment setup incomplete. Please review the steps and try again."

        notification_service.show_toast(msg, "info")
        # self._update_experiment_status()

        self.update_ui()

    def _choose_new_experiment_location(self, title: str):
        """Prompt for a parent directory and experiment name. Returns (dir, name) or None."""
        os.makedirs(SALAMI_LOG_DIR, exist_ok=True)
        parent_dir = QFileDialog.getExistingDirectory(self, title, SALAMI_LOG_DIR)
        if not parent_dir:
            return None

        name, ok = QInputDialog.getText(
            self, "Experiment Name", "Name:", text=default_experiment_name()
        )
        if not ok or not name.strip():
            return None
        return parent_dir, name.strip()

    def create_new_experiment(self):
        """Create a new experiment from the default protocol."""
        choice = self._choose_new_experiment_location(
            "Choose Parent Directory for New Experiment"
        )
        if choice is None:
            return
        parent_dir, name = choice
        self._create_experiment(parent_dir, name, template=None)

    def new_experiment_from_template(self):
        """Reuse an existing protocol on a new sample.

        The template is copied and re-homed into a new directory; the file it
        came from is never written to.
        """
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Protocol to Reuse",
            SALAMI_LOG_DIR,
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if not filename:
            return

        try:
            template = SalamiExperimentConfig.load(filename)
        except Exception as e:
            logging.exception("Failed to load protocol")
            QMessageBox.critical(
                self,
                "Load Protocol Failed",
                f"Could not load protocol:\n{filename}\n\n{e}",
            )
            return

        choice = self._choose_new_experiment_location(
            "Choose Parent Directory for the New Experiment"
        )
        if choice is None:
            return
        parent_dir, name = choice
        self._create_experiment(
            parent_dir, name, template=template, source=f"template {template.name!r}"
        )

    def _create_experiment(self, parent_dir, name, template, source: str = ""):
        try:
            experiment = SalamiExperiment.create(
                parent_dir=parent_dir,
                name=name,
                template=template,
                microscope=self.microscope,
                parent_ui=self,
            )
        except FileExistsError as e:
            QMessageBox.warning(self, "Experiment Exists", str(e))
            return
        except Exception as e:
            logging.exception("Failed to create experiment")
            QMessageBox.critical(
                self, "Create Experiment Failed", f"Could not create experiment:\n{e}"
            )
            return

        self._adopt_experiment(experiment, source=source or f"new at {experiment.path}")

    def open_experiment(self):
        """Open an existing experiment directory and resume from its status."""
        directory = QFileDialog.getExistingDirectory(
            self, "Choose Experiment Directory", SALAMI_LOG_DIR
        )
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

    def export_config(self):
        """Write the current protocol out for reuse on another sample."""
        if self.salamiExperimentConfig is None:
            QMessageBox.warning(
                self, "No Experiment", "Create or open an experiment first."
            )
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Protocol",
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

        notification_service.show_toast(f"Protocol exported to {filename}", "info")

    def _show_toast(self, message: str, notification_type: str, temporary: bool):
        self.toast_manager.show_toast(message, notification_type, temporary=temporary)

    def _adopt_experiment(self, experiment: SalamiExperiment, source: str = ""):
        """Make `experiment` the active one and refresh the UI."""
        self.salamiExperiment = experiment
        # Other UI methods read from salamiExperimentConfig directly; keep it in sync.
        self.salamiExperimentConfig = experiment.config

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

    @staticmethod
    def _format_field_name(name: str) -> str:
        return name.replace("_", " ").title()

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

    def _populate_task_combo(self):
        """Fill the task selector from the protocol, keeping the current choice."""
        previous = self.comboBoxTasks.currentData()
        self.comboBoxTasks.blockSignals(True)
        self.comboBoxTasks.clear()
        if self.salamiExperimentConfig is not None:
            for task in self.salamiExperimentConfig.tasks:
                self.comboBoxTasks.addItem(
                    f"{task.name}  ({task.TASK_TYPE})", task.name
                )
        index = self.comboBoxTasks.findData(previous)
        self.comboBoxTasks.setCurrentIndex(max(index, 0))
        self.comboBoxTasks.blockSignals(False)
        self._on_selected_task_changed()

    def _on_selected_task_changed(self, *_args):
        """Load the selected task's config into the editor."""
        experiment = self.salamiExperiment
        task_name = self.comboBoxTasks.currentData()
        task = experiment.find_task(task_name) if experiment and task_name else None
        self.task_config_editor.set_task_config(task)
        self._refresh_protocol_rectangles(task)

    def _on_task_config_changed(self):
        """An edit has already landed on the config object; persist the protocol.

        Saved on every change rather than behind an Apply button: a protocol can
        be edited mid-run, so a half-applied config is worse than an eagerly
        saved one.
        """
        self._save_config()
        self._sync_rectangles_from_config()

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
        self.pushButtonRunExperiment.setVisible(not running)
        self.pushButtonStopExperiment.setVisible(running)
        self.pushButtonStopExperiment.setEnabled(running)
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

    def _setup_experiment_status_layout(self):
        """Add editable n_steps + setup_complete fields to the Experiment Status groupbox."""
        layout = self.groupBoxExpStatus.layout()
        is_grid = isinstance(layout, QGridLayout)
        next_row = layout.rowCount() if is_grid else 1

        self.label_n_steps = QLabel("Number of Steps", self.groupBoxExpStatus)
        self.spinBoxNSteps = QSpinBox(self.groupBoxExpStatus)
        self.spinBoxNSteps.setRange(1, 1_000_000)
        self.spinBoxNSteps.setKeyboardTracking(False)
        self.spinBoxNSteps.setValue(1)
        self.spinBoxNSteps.valueChanged.connect(self._on_n_steps_changed)

        self.label_setup_complete = QLabel("Setup Complete", self.groupBoxExpStatus)
        self.checkBoxSetupComplete = QCheckBox(self.groupBoxExpStatus)
        self.checkBoxSetupComplete.toggled.connect(self._on_setup_complete_toggled)

        # Tasks are protocol, like n_steps above, so they are edited here rather
        # than on the Protocol page — which selects a task to configure but does
        # not own the list.
        self.pushButtonAddTask = QPushButton("Add Task", self.groupBoxExpStatus)
        self.pushButtonAddTask.clicked.connect(self._on_add_task_clicked)
        self.pushButtonRemoveTask = QPushButton("Remove Task", self.groupBoxExpStatus)
        self.pushButtonRemoveTask.clicked.connect(self._on_remove_task_clicked)

        if is_grid:
            layout.addWidget(self.label_n_steps, next_row, 0)
            layout.addWidget(self.spinBoxNSteps, next_row, 1)
            layout.addWidget(self.label_setup_complete, next_row + 1, 0)
            layout.addWidget(self.checkBoxSetupComplete, next_row + 1, 1)
            layout.addWidget(self.pushButtonAddTask, next_row + 2, 0)
            layout.addWidget(self.pushButtonRemoveTask, next_row + 2, 1)
        else:
            layout.addWidget(self.label_n_steps)
            layout.addWidget(self.spinBoxNSteps)
            layout.addWidget(self.label_setup_complete)
            layout.addWidget(self.checkBoxSetupComplete)
            layout.addWidget(self.pushButtonAddTask)
            layout.addWidget(self.pushButtonRemoveTask)

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
        self.update_ui()

        logging.info(f"Added task {task.name!r} ({task.TASK_TYPE}) to the protocol.")
        notification_service.show_toast(f"Added task {task.name!r}", "info")

    def _on_remove_task_clicked(self):
        experiment = self.salamiExperiment
        if experiment is None:
            return

        task_name = self.comboBoxTasks.currentData()
        if not task_name:
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
        self.update_ui()

        logging.info(f"Removed task {task_name!r} from the protocol.")
        notification_service.show_toast(f"Removed task {task_name!r}", "info")

    def _on_n_steps_changed(self, value: int):
        config = self.salamiExperimentConfig
        if config is None:
            return
        config.n_steps = int(value)
        self._save_config()
        self._update_experiment_status()

    def _on_setup_complete_toggled(self, checked: bool):
        if self.salamiExperiment is None:
            return
        self.salamiExperiment.status.setup_complete = bool(checked)
        # Persist so manual toggles survive a reload.
        self.salamiExperiment.status.save(self.salamiExperiment.config.path)
        # setup_complete gates the Run/Stop buttons.
        self.update_ui()

    def _update_experiment_status(self):
        """Drive the Experiment Status progress bar from current_step / n_steps."""
        experiment = self.salamiExperiment
        if experiment is None:
            self.progressBarExpProgress.setValue(0)
            self.label_2.setText("Progress")
            self.checkBoxSetupComplete.blockSignals(True)
            self.checkBoxSetupComplete.setChecked(False)
            self.checkBoxSetupComplete.blockSignals(False)
            return

        status = experiment.for_ui_display_STATUS()
        current = int(status.get("current_step", 0))
        total = int(status.get("n_steps", 0))
        pct = int((current / total) * 100) if total > 0 else 0
        self.progressBarExpProgress.setValue(pct)
        self.label_2.setText(f"Progress: {current} / {total}")

        depth_um = status.get("volume_depth_um", 0.0)
        step_nm = status.get("advance_step_size_nm", 0.0)
        self.label_CurrentTaskInfo.setText(
            f"{status.get('run_state', 'created')} | "
            f"{step_nm:.1f} nm slices | {depth_um:.1f} um through the block"
        )

        # Sync the n_steps spinbox without retriggering its valueChanged handler.
        self.spinBoxNSteps.blockSignals(True)
        self.spinBoxNSteps.setValue(max(1, total))
        self.spinBoxNSteps.blockSignals(False)

        # Sync the setup_complete checkbox without retriggering its toggled handler.
        self.checkBoxSetupComplete.blockSignals(True)
        self.checkBoxSetupComplete.setChecked(bool(status.get("setup_complete", False)))
        self.checkBoxSetupComplete.blockSignals(False)

    def connect_to_microscope(self):

        self.microscope = self.system_widget.microscope
        self.settings = self.system_widget.settings

        # An experiment can be created and edited offline, so one may already be
        # loaded; hand it the microscope rather than making the user reload.
        if self.salamiExperiment is not None:
            self.salamiExperiment.microscope = self.microscope

        # Lets the generated editors offer the values this column supports.
        self.task_config_editor.set_microscope(self.microscope)

        self.update_microscope_ui()
        self.update_ui()

    def disconnect_from_microscope(self):
        self.microscope = None
        self.settings = None
        if self.salamiExperiment is not None:
            self.salamiExperiment.microscope = None
        self.task_config_editor.set_microscope(None)
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
        self.tabWidget.setTabVisible(
            self.tabWidget.indexOf(self.tabExperiment),
            is_experiment_loaded,
        )
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
        self.pushButtonStopExperiment.setEnabled(running)
        self.pushButtonRunExperiment.setVisible(not running)
        self.pushButtonStopExperiment.setVisible(running)

        # Adding or removing a task changes the length of `config.tasks`, which
        # `status.next_task_index` indexes into while `_run_step` iterates. Both
        # need no microscope, so a protocol can still be assembled offline.
        self.pushButtonAddTask.setEnabled(is_experiment_loaded and not running)
        self.pushButtonRemoveTask.setEnabled(
            is_experiment_loaded and not running and self.comboBoxTasks.count() > 0
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

        self.set_current_workflow_message(
            msg=update.workflow_msg,
            show=bool(update.workflow_msg),
        )

        self._update_progress_from(update)

    def _update_progress_from(self, update: SalamiWorkflowUpdate) -> None:
        """Drive the status panel from a live run.

        Only progress-bearing updates qualify; a bare prompt or image push
        leaves the panel showing whatever the run last reported.
        """
        if update.total_steps <= 0:
            return

        self.progressBarExpProgress.setValue(int(update.overall_progress))
        self.label_2.setText(f"Progress: {update.current_step} / {update.total_steps}")

        parts = [update.task_name or update.task]
        if update.total_tasks > 0:
            parts.append(f"task {update.current_task_index + 1}/{update.total_tasks}")
        if update.elapsed_time > 0:
            parts.append(format_duration(update.elapsed_time))
        self.label_CurrentTaskInfo.setText(" | ".join(p for p in parts if p))


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
