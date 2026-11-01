"""Create and Load Experiment dialogs.

Same shape and look as AutoLamella's (`autolamella_create_experiment_widget.py`,
`autolamella_load_experiment_widget.py`), adapted to what a Salami experiment is:

- **Create** asks for a name, description and parent directory, and which protocol
  to start from — the default, or any saved one. That last choice is what used to
  be a separate "New Experiment From Protocol" action.
- **Load** shows recently used experiments on the left and what the selected one
  holds on the right. A single click previews, a double click opens.

Previewing never opens the experiment. `SalamiExperiment.open` recreates missing
task directories, so it runs only when the operator commits; until then the two
YAML files are read and nothing is written.

The recent list is Salami's own (`salami/recents.py`), not AutoLamella's.
"""

import logging
import os
from datetime import datetime
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from fibsem.applications.autolamella.ui.autolamella_load_experiment_widget import (
    RECENT_ALERT_ICON,
    RECENT_COLUMN_WIDTH,
    RECENT_FOLDER_ICON,
    RECENT_ICON_COLOR,
    RECENT_LIST_STYLESHEET,
    RECENT_NAME_COLOR,
    RECENT_PILL_TEXT_COLOR,
    RECENT_UNAVAILABLE_COLOR,
)
from fibsem.applications.salami.experiment import (
    CONFIG_FILENAME,
    STATUS_FILENAME,
    SalamiExperiment,
    SalamiExperimentConfig,
    SalamiExperimentStatus,
    default_experiment_name,
)
from fibsem.applications.salami.recents import (
    SALAMI_LOG_DIR,
    SalamiExperimentSummary,
    default_parent_directory,
    get_recent_experiments,
    last_experiment_directory,
)
from fibsem.ui import utils as fui
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.stylesheets import (
    PRIMARY_BUTTON_STYLESHEET,
    ROW_ALT_COLOR,
    SECONDARY_BUTTON_STYLESHEET,
    TEXT_MUTED_COLOR,
)
from fibsem.ui.widgets.custom_widgets import (
    ElidedLabel,
    QDirectoryLineEdit,
    TitledPanel,
)

# The pill on a recent row shows slice progress where AutoLamella's shows a
# lamella count.
RECENT_SLICES_ICON = "mdi:layers-triple-outline"

NO_EXPERIMENT = "No experiment selected"
NO_PROTOCOL = "No protocol loaded"

WARNING_STYLE = "color: orange; font-style: italic;"
NOTE_STYLE = "color: gray; font-style: italic; font-size: 10px;"


def _read_only_line(placeholder: str) -> QtWidgets.QLineEdit:
    line = QtWidgets.QLineEdit()
    line.setEnabled(False)
    line.setPlaceholderText(placeholder)
    return line


def _set_line(line: QtWidgets.QLineEdit, text: str) -> None:
    line.setText(text)
    line.setCursorPosition(0)  # show the start of a long path, not its tail


def format_volume(config: SalamiExperimentConfig) -> str:
    """'50 slices × 100.0 nm = 5.00 µm'."""
    step_nm = config.advance.step_size * 1e9
    depth_um = config.volume_depth * 1e6
    return f"{config.n_steps} slices × {step_nm:.1f} nm = {depth_um:.2f} µm"


def format_tasks(config: SalamiExperimentConfig) -> str:
    return ", ".join(task.name for task in config.tasks) or "none"


class ProtocolInfoForm(QtWidgets.QWidget):
    """Read-only summary of a protocol, shared by both dialogs."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.lineEdit_name = _read_only_line(NO_PROTOCOL)
        self.lineEdit_path = _read_only_line(NO_PROTOCOL)
        self.lineEdit_tasks = _read_only_line(NO_PROTOCOL)
        self.lineEdit_volume = _read_only_line(NO_PROTOCOL)
        form.addRow("Name", self.lineEdit_name)
        form.addRow("Path", self.lineEdit_path)
        form.addRow("Tasks", self.lineEdit_tasks)
        form.addRow("Volume", self.lineEdit_volume)

    def show_config(self, config: Optional[SalamiExperimentConfig], path: str = ""):
        if config is None:
            for line in (
                self.lineEdit_name,
                self.lineEdit_path,
                self.lineEdit_tasks,
                self.lineEdit_volume,
            ):
                line.clear()
            return
        _set_line(self.lineEdit_name, config.name or "")
        _set_line(self.lineEdit_path, path)
        _set_line(self.lineEdit_tasks, format_tasks(config))
        _set_line(self.lineEdit_volume, format_volume(config))


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


class SalamiCreateExperimentDialog(QtWidgets.QDialog):
    """Name a new experiment, choose where it goes and the protocol it starts from.

    Creates the experiment on accept; `get_experiment()` returns it. Unlike
    AutoLamella's dialog this never overwrites: an existing experiment holds a run
    state, and replacing it would lose where a run had got to. The operator is
    pointed at Load Experiment instead.
    """

    def __init__(
        self,
        microscope=None,
        parent_ui=None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.microscope = microscope
        self.parent_ui = parent_ui
        self.experiment: Optional[SalamiExperiment] = None
        self.template: Optional[SalamiExperimentConfig] = None
        self.template_path: str = ""

        self.setWindowTitle("Create New Experiment")
        self.setMinimumWidth(600)

        self._setup_ui()
        self._connect_signals()
        self._use_default_protocol()
        self._validate()

    def _setup_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)

        # --- Experiment Information ---
        exp_content = QtWidgets.QWidget()
        exp_layout = QtWidgets.QVBoxLayout(exp_content)
        exp_layout.setContentsMargins(0, 0, 0, 0)

        self.lineEdit_experiment_name = QtWidgets.QLineEdit(default_experiment_name())
        self.lineEdit_experiment_description = QtWidgets.QLineEdit()
        self.lineEdit_experiment_description.setPlaceholderText(
            "Optional description of the experiment..."
        )
        self.lineEdit_experiment_directory = QDirectoryLineEdit()
        self.lineEdit_experiment_directory.setText(default_parent_directory())

        form = QtWidgets.QFormLayout()
        form.addRow("Name", self.lineEdit_experiment_name)
        form.addRow("Description", self.lineEdit_experiment_description)
        form.addRow("Directory", self.lineEdit_experiment_directory)
        exp_layout.addLayout(form)

        self.label_validation_warning = QtWidgets.QLabel("")
        self.label_validation_warning.setStyleSheet(WARNING_STYLE)
        self.label_validation_warning.setWordWrap(True)
        exp_layout.addWidget(self.label_validation_warning)

        main_layout.addWidget(
            TitledPanel(
                "Experiment Information", content=exp_content, collapsible=False
            )
        )

        # --- Protocol Information ---
        protocol_content = QtWidgets.QWidget()
        protocol_layout = QtWidgets.QVBoxLayout(protocol_content)
        protocol_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_default_protocol = QtWidgets.QPushButton("Default")
        self.btn_default_protocol.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.btn_default_protocol.setToolTip("Start from the default Salami protocol")
        self.btn_select_protocol = QtWidgets.QPushButton("Select")
        self.btn_select_protocol.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.btn_select_protocol.setToolTip(
            "Start from a saved protocol, or another experiment's "
            f"{CONFIG_FILENAME}. It is copied; the file itself is never changed."
        )

        self.protocol_form = ProtocolInfoForm()
        protocol_layout.addWidget(self.protocol_form)

        note = QtWidgets.QLabel(
            "Note: You will be able to edit the protocol after creating the experiment."
        )
        note.setStyleSheet(NOTE_STYLE)
        note.setWordWrap(True)
        protocol_layout.addWidget(note)

        protocol_panel = TitledPanel(
            "Protocol Information", content=protocol_content, collapsible=False
        )
        protocol_panel.add_header_widget(self.btn_default_protocol)
        protocol_panel.add_header_widget(self.btn_select_protocol)
        main_layout.addWidget(protocol_panel)

        # --- Buttons ---
        button_box = QtWidgets.QDialogButtonBox()
        self.btn_ok = QtWidgets.QPushButton("Create Experiment")
        self.btn_ok.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.btn_ok.setDefault(True)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        button_box.addButton(self.btn_ok, QtWidgets.QDialogButtonBox.AcceptRole)
        button_box.addButton(self.btn_cancel, QtWidgets.QDialogButtonBox.RejectRole)
        main_layout.addWidget(button_box)

    def _connect_signals(self) -> None:
        self.lineEdit_experiment_name.textChanged.connect(self._validate)
        self.lineEdit_experiment_directory.textChanged.connect(self._validate)
        self.btn_default_protocol.clicked.connect(self._use_default_protocol)
        self.btn_select_protocol.clicked.connect(self._select_protocol)
        self.btn_ok.clicked.connect(self._on_ok_clicked)
        self.btn_cancel.clicked.connect(self.reject)

    @property
    def experiment_name(self) -> str:
        return self.lineEdit_experiment_name.text().strip()

    @property
    def parent_directory(self) -> str:
        return self.lineEdit_experiment_directory.text().strip()

    def _problem(self) -> str:
        """Why Create cannot go ahead, or "" if it can."""
        if not self.experiment_name:
            return "Enter an experiment name."
        if not self.parent_directory or not os.path.isdir(self.parent_directory):
            return "Choose a directory that exists."
        target = os.path.join(self.parent_directory, self.experiment_name)
        if os.path.exists(os.path.join(target, CONFIG_FILENAME)):
            return (
                f"An experiment named '{self.experiment_name}' already exists in this "
                "directory. Choose another name, or open it with Load Experiment."
            )
        if self.template is None:
            return "Choose a protocol to start from."
        return ""

    def _validate(self, *_args) -> None:
        problem = self._problem()
        self.label_validation_warning.setText(f"⚠ {problem}" if problem else "")
        self.btn_ok.setEnabled(not problem)

    def _show_template(self) -> None:
        self.protocol_form.show_config(self.template, self.template_path)
        self._validate()

    def _use_default_protocol(self) -> None:
        try:
            self.template = SalamiExperiment.default_template()
            self.template_path = "Default protocol"
        except Exception as e:
            logging.exception("Failed to load the default Salami protocol")
            self.template, self.template_path = None, ""
            QtWidgets.QMessageBox.warning(
                self,
                "Protocol Load Error",
                f"The default protocol could not be loaded:\n\n{e}\n\n"
                "Select a protocol file instead.",
            )
        self._show_template()

    def _select_protocol(self) -> None:
        start = (
            os.path.dirname(self.template_path)
            if os.path.isfile(self.template_path)
            else (last_experiment_directory() or SALAMI_LOG_DIR)
        )
        filename = fui.open_existing_file_dialog(
            msg="Select a protocol to start from (*.yaml)",
            path=start,
            parent=self,
        )
        if not filename:
            return
        try:
            self.template = SalamiExperimentConfig.load(filename)
            self.template_path = filename
        except Exception as e:
            logging.exception("Failed to load protocol")
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid Protocol",
                f"The selected file is not a valid Salami protocol:\n\n{filename}\n\n{e}",
            )
            return
        self._show_template()

    def _on_ok_clicked(self) -> None:
        problem = self._problem()
        if problem:
            QtWidgets.QMessageBox.warning(self, "Cannot Create Experiment", problem)
            return
        try:
            experiment = SalamiExperiment.create(
                parent_dir=self.parent_directory,
                name=self.experiment_name,
                template=self.template,
                microscope=self.microscope,
                parent_ui=self.parent_ui,
            )
            description = self.lineEdit_experiment_description.text().strip()
            if description:
                experiment.config.description = description
                experiment.config.save()
        except Exception as e:
            logging.exception("Failed to create experiment")
            QtWidgets.QMessageBox.critical(
                self, "Create Experiment Failed", f"Could not create experiment:\n\n{e}"
            )
            return

        self.experiment = experiment
        self.accept()

    def get_experiment(self) -> Optional[SalamiExperiment]:
        return self.experiment

    @property
    def source(self) -> str:
        """How the protocol was chosen, for the log and the toast."""
        if os.path.isfile(self.template_path):
            return f"from protocol {self.template.name!r}"
        return ""


def create_experiment_dialog(
    microscope=None, parent_ui=None, parent: Optional[QtWidgets.QWidget] = None
) -> Optional[SalamiCreateExperimentDialog]:
    """Run the Create dialog. Returns the accepted dialog, or None if cancelled."""
    dialog = SalamiCreateExperimentDialog(
        microscope=microscope, parent_ui=parent_ui, parent=parent
    )
    if dialog.exec_() != QtWidgets.QDialog.Accepted or dialog.experiment is None:
        logging.info("Experiment creation cancelled")
        return None
    return dialog


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


class SalamiLoadExperimentDialog(QtWidgets.QDialog):
    """Pick an experiment to open: from the recent list, or by browsing.

    Returns the chosen directory via `get_experiment_path()`. Opening it is left
    to the caller, which owns the microscope and the interrupted-task prompt.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.experiment_path: Optional[str] = None

        self.setWindowTitle("Load Existing Experiment")
        self.setMinimumWidth(860)

        self._setup_ui()
        self._connect_signals()
        self._populate_recent_experiments()

    def _setup_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)

        # --- Experiment Information (right) ---
        exp_content = QtWidgets.QWidget()
        exp_form = QtWidgets.QFormLayout(exp_content)
        exp_form.setContentsMargins(0, 0, 0, 0)
        self.lineEdit_experiment_name = _read_only_line(NO_EXPERIMENT)
        self.lineEdit_experiment_description = _read_only_line(NO_EXPERIMENT)
        self.lineEdit_experiment_directory = _read_only_line(NO_EXPERIMENT)
        self.lineEdit_experiment_progress = _read_only_line(NO_EXPERIMENT)
        self.lineEdit_experiment_setup = _read_only_line(NO_EXPERIMENT)
        exp_form.addRow("Name", self.lineEdit_experiment_name)
        exp_form.addRow("Description", self.lineEdit_experiment_description)
        exp_form.addRow("Directory", self.lineEdit_experiment_directory)
        exp_form.addRow("Progress", self.lineEdit_experiment_progress)
        exp_form.addRow("Setup", self.lineEdit_experiment_setup)

        self.label_warning = QtWidgets.QLabel("")
        self.label_warning.setStyleSheet(WARNING_STYLE)
        self.label_warning.setWordWrap(True)
        exp_form.addRow(self.label_warning)

        exp_panel = TitledPanel(
            "Experiment Information", content=exp_content, collapsible=False
        )

        # --- Protocol Information (right) ---
        protocol_content = QtWidgets.QWidget()
        protocol_layout = QtWidgets.QVBoxLayout(protocol_content)
        protocol_layout.setContentsMargins(0, 0, 0, 0)
        self.protocol_form = ProtocolInfoForm()
        protocol_layout.addWidget(self.protocol_form)
        note = QtWidgets.QLabel(
            "Note: You will be able to edit the protocol after loading the experiment."
        )
        note.setStyleSheet(NOTE_STYLE)
        note.setWordWrap(True)
        protocol_layout.addWidget(note)
        protocol_panel = TitledPanel(
            "Protocol Information", content=protocol_content, collapsible=False
        )

        # --- Recent Experiments (left) ---
        recent_content = QtWidgets.QWidget()
        recent_layout = QtWidgets.QVBoxLayout(recent_content)
        recent_layout.setContentsMargins(0, 0, 0, 0)
        self.list_recent_experiments = QtWidgets.QListWidget()
        self.list_recent_experiments.setStyleSheet(RECENT_LIST_STYLESHEET)
        self.list_recent_experiments.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        self.list_recent_experiments.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection
        )
        recent_layout.addWidget(self.list_recent_experiments)
        self.btn_select_experiment = QtWidgets.QPushButton("Browse…")
        self.btn_select_experiment.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        recent_layout.addWidget(self.btn_select_experiment)
        recent_panel = TitledPanel(
            "Recent Experiments", content=recent_content, collapsible=False
        )
        recent_panel.setFixedWidth(RECENT_COLUMN_WIDTH)

        right_column = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(exp_panel)
        right_layout.addWidget(protocol_panel)
        right_layout.addStretch()

        columns = QtWidgets.QHBoxLayout()
        columns.addWidget(recent_panel)
        columns.addWidget(right_column, stretch=1)
        main_layout.addLayout(columns)

        # --- Buttons ---
        button_box = QtWidgets.QDialogButtonBox()
        self.btn_ok = QtWidgets.QPushButton("Load Experiment")
        self.btn_ok.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.btn_ok.setDefault(True)
        self.btn_ok.setEnabled(False)  # until an experiment is selected
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        button_box.addButton(self.btn_ok, QtWidgets.QDialogButtonBox.AcceptRole)
        button_box.addButton(self.btn_cancel, QtWidgets.QDialogButtonBox.RejectRole)
        main_layout.addWidget(button_box)

    def _connect_signals(self) -> None:
        self.btn_select_experiment.clicked.connect(self._select_experiment)
        self.btn_ok.clicked.connect(self._on_ok_clicked)
        self.btn_cancel.clicked.connect(self.reject)
        self.list_recent_experiments.itemClicked.connect(self._on_recent_clicked)
        self.list_recent_experiments.itemDoubleClicked.connect(
            self._on_recent_double_clicked
        )

    # -- selecting ----------------------------------------------------------

    def _select_experiment(self) -> None:
        directory = fui.open_existing_directory_dialog(
            msg="Select an experiment folder",
            path=last_experiment_directory() or SALAMI_LOG_DIR,
            parent=self,
        )
        if not directory:
            return
        if not os.path.exists(os.path.join(directory, CONFIG_FILENAME)):
            QtWidgets.QMessageBox.critical(
                self,
                "Not an Experiment",
                f"No {CONFIG_FILENAME} found in:\n\n{directory}\n\n"
                "Select the experiment directory itself.",
            )
            self._clear_display()
            return
        self.list_recent_experiments.clearSelection()
        self._preview(directory)

    def _preview(self, directory: str) -> bool:
        """Show what `directory` holds. Reads only; returns False if unreadable."""
        try:
            config = SalamiExperimentConfig.load(
                os.path.join(directory, CONFIG_FILENAME)
            )
            status_path = os.path.join(directory, STATUS_FILENAME)
            status = (
                SalamiExperimentStatus.load(status_path)
                if os.path.exists(status_path)
                else SalamiExperimentStatus()
            )
        except Exception as e:
            logging.exception(f"Failed to read experiment in {directory}")
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid Experiment",
                f"The experiment in\n\n{directory}\n\ncould not be read:\n\n{e}",
            )
            self._clear_display()
            return False

        self.experiment_path = directory
        _set_line(self.lineEdit_experiment_name, config.name or "")
        _set_line(self.lineEdit_experiment_description, config.description or "")
        _set_line(self.lineEdit_experiment_directory, config.path)
        _set_line(
            self.lineEdit_experiment_progress,
            f"{status.current_step} / {config.n_steps} slices · {status.run_state}",
        )
        _set_line(
            self.lineEdit_experiment_setup,
            "Complete" if status.setup_complete else "Not done",
        )
        # Opening asks what to do about it; say so before the operator commits.
        self.label_warning.setText(
            f"⚠ Task '{status.in_flight_task}' was interrupted. You will be asked "
            "how to resolve it after loading."
            if status.in_flight_task
            else ""
        )
        self.protocol_form.show_config(config, os.path.join(directory, CONFIG_FILENAME))
        self.btn_ok.setEnabled(True)
        return True

    def _clear_display(self) -> None:
        self.experiment_path = None
        for line in (
            self.lineEdit_experiment_name,
            self.lineEdit_experiment_description,
            self.lineEdit_experiment_directory,
            self.lineEdit_experiment_progress,
            self.lineEdit_experiment_setup,
        ):
            line.clear()
        self.label_warning.clear()
        self.protocol_form.show_config(None)
        self.btn_ok.setEnabled(False)

    def _on_ok_clicked(self) -> None:
        if self.experiment_path is None:
            QtWidgets.QMessageBox.warning(
                self, "No Experiment", "Select an experiment to load."
            )
            return
        self.accept()

    def get_experiment_path(self) -> Optional[str]:
        return self.experiment_path

    # -- recent list ----------------------------------------------------------

    def _on_recent_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        path = item.data(QtCore.Qt.UserRole)
        if path:
            self._preview(path)

    def _on_recent_double_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        path = item.data(QtCore.Qt.UserRole)
        if path and self._preview(path):
            self._on_ok_clicked()

    def _populate_recent_experiments(self) -> None:
        self.list_recent_experiments.clear()
        recents = get_recent_experiments()
        if not recents:
            placeholder = QtWidgets.QListWidgetItem("No recent experiments")
            placeholder.setFlags(QtCore.Qt.NoItemFlags)
            placeholder.setForeground(QtCore.Qt.gray)
            self.list_recent_experiments.addItem(placeholder)
            return

        for info in recents:
            item = QtWidgets.QListWidgetItem(self.list_recent_experiments)
            item.setData(QtCore.Qt.UserRole, info.path)
            if not info.available:
                item.setFlags(QtCore.Qt.NoItemFlags)
                item.setToolTip(f"{info.path}\n\nThis experiment could not be read.")
            else:
                item.setToolTip(info.path)
            row = self._make_recent_row_widget(info)
            # Width 0 lets the row track the viewport, so long names elide.
            item.setSizeHint(QtCore.QSize(0, row.sizeHint().height()))
            self.list_recent_experiments.addItem(item)
            self.list_recent_experiments.setItemWidget(item, row)

    def _make_recent_row_widget(
        self, info: SalamiExperimentSummary
    ) -> QtWidgets.QWidget:
        """folder icon | name + date | slice-progress pill — AutoLamella's row."""
        available = info.available
        icon_color = RECENT_ICON_COLOR if available else RECENT_UNAVAILABLE_COLOR
        name_color = RECENT_NAME_COLOR if available else RECENT_UNAVAILABLE_COLOR

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(10, 11, 8, 11)
        layout.setSpacing(9)

        icon_label = QtWidgets.QLabel()
        icon_label.setPixmap(
            fibsem_icon(
                RECENT_FOLDER_ICON if available else RECENT_ALERT_ICON, color=icon_color
            ).pixmap(18, 18)
        )
        icon_label.setFixedWidth(18)
        layout.addWidget(icon_label, alignment=QtCore.Qt.AlignVCenter)

        text_layout = QtWidgets.QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(5)

        name_label = ElidedLabel(info.name, mode=QtCore.Qt.ElideMiddle)
        name_font = name_label.font()
        name_font.setPixelSize(13)
        name_font.setWeight(QtGui.QFont.DemiBold)
        name_label.setFont(name_font)
        name_label.setStyleSheet(f"color: {name_color}; background: transparent;")

        if not available:
            date_str = "unavailable"
        elif info.created_at:
            date_str = datetime.fromtimestamp(info.created_at).strftime("%Y-%m-%d")
        else:
            date_str = "unknown date"
        date_label = QtWidgets.QLabel(date_str)
        date_color = TEXT_MUTED_COLOR if available else RECENT_UNAVAILABLE_COLOR
        date_label.setStyleSheet(
            f"color: {date_color}; font-size: 11px; background: transparent;"
        )

        text_layout.addWidget(name_label)
        text_layout.addWidget(date_label)
        layout.addLayout(text_layout, stretch=1)

        if available:
            layout.addWidget(
                self._make_progress_pill(info), alignment=QtCore.Qt.AlignVCenter
            )
        return widget

    def _make_progress_pill(self, info: SalamiExperimentSummary) -> QtWidgets.QWidget:
        pill = QtWidgets.QFrame()
        pill.setObjectName("slicePill")
        pill.setStyleSheet(
            f"#slicePill {{ background: {ROW_ALT_COLOR}; border-radius: 9px; }}"
        )
        pill.setToolTip(
            f"{info.current_step} of {info.n_steps} slices · {info.run_state}"
        )

        pill_layout = QtWidgets.QHBoxLayout(pill)
        pill_layout.setContentsMargins(7, 2, 8, 2)
        pill_layout.setSpacing(3)

        icon_label = QtWidgets.QLabel()
        icon_label.setPixmap(
            fibsem_icon(RECENT_SLICES_ICON, color=RECENT_PILL_TEXT_COLOR).pixmap(12, 12)
        )
        count_label = QtWidgets.QLabel(f"{info.current_step}/{info.n_steps}")
        count_label.setStyleSheet(
            f"color: {RECENT_PILL_TEXT_COLOR}; font-size: 11px; "
            "font-weight: 600; background: transparent;"
        )
        pill_layout.addWidget(icon_label)
        pill_layout.addWidget(count_label)
        return pill


def load_experiment_dialog(parent: Optional[QtWidgets.QWidget] = None) -> Optional[str]:
    """Run the Load dialog. Returns the chosen experiment directory, or None."""
    dialog = SalamiLoadExperimentDialog(parent)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        logging.info("Experiment loading cancelled")
        return None
    return dialog.get_experiment_path()
