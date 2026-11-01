"""Edit > Preferences for Salami.

AutoLamella's preferences dialog (`fibsem/ui/widgets/preferences_dialog.py`),
trimmed to what Salami uses: Display (sound, border), Experiment (default
directory and protocol) and Movement. Same layout, same labels.

Two stores behind one dialog:

- Sound, border and movement are per-user and stay in the shared
  `UserPreferences`, so they are the same setting in AutoLamella. Every section
  this dialog does not show is carried over untouched, so saving here never
  resets anything AutoLamella owns.
- The experiment defaults are Salami's own (`salami/preferences.py`): a Salami
  protocol in AutoLamella's `default_protocol_path` would break its create dialog.
"""

import dataclasses
import os
from typing import Tuple

from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QListWidget,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.salami.experiment import (
    CONFIG_FILENAME,
    SalamiExperimentConfig,
)
from fibsem.applications.salami.preferences import SalamiPreferences
from fibsem.config import (
    DisplayPreferences,
    MovementPreferences,
    UserPreferences,
)
from fibsem.ui.widgets.custom_widgets import QDirectoryLineEdit, QFileLineEdit
from fibsem.ui.widgets.preferences_dialog import (
    _LBL_ACQ_FIB,
    _LBL_ACQ_SEM,
    _LBL_BORDER,
    _LBL_EXP_DIR,
    _LBL_EXP_PROTOCOL,
    _LBL_SOUND,
    _TIP_ACQ_FIB,
    _TIP_ACQ_SEM,
    _TIP_BORDER,
    _TIP_EXP_DIR,
    _TIP_SOUND,
)

_TIP_BORDER_SALAMI = (
    f"{_TIP_BORDER} Shared with AutoLamella; has no effect in Salami yet."
)
_TIP_EXP_PROTOCOL_SALAMI = (
    f"Salami protocol ({CONFIG_FILENAME} or a saved protocol) the Default button "
    "starts new experiments from. Leave empty for the built-in protocol."
)


class SalamiPreferencesDialog(QDialog):
    """Dialog for editing the preferences Salami uses."""

    def __init__(
        self,
        preferences: UserPreferences,
        salami_preferences: SalamiPreferences,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(500)
        self._preferences = preferences
        self._setup_ui()
        self._load(preferences, salami_preferences)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Sidebar + stack
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)

        self._sidebar = QListWidget()
        self._sidebar.setFixedWidth(120)
        self._sidebar.addItems(["Display", "Experiment", "Movement"])
        self._sidebar.setCurrentRow(0)

        self._stack = QStackedWidget()
        body_layout.addWidget(self._sidebar)
        body_layout.addWidget(self._stack)
        layout.addWidget(body)

        # --- Display ---
        display_page = QWidget()
        display_form = QFormLayout(display_page)
        self._chk_sound = QCheckBox()
        self._chk_sound.setToolTip(_TIP_SOUND)
        self._chk_border = QCheckBox()
        self._chk_border.setToolTip(_TIP_BORDER_SALAMI)
        display_form.addRow(_LBL_SOUND, self._chk_sound)
        display_form.addRow(_LBL_BORDER, self._chk_border)
        self._stack.addWidget(display_page)

        # --- Experiment Defaults ---
        experiment_page = QWidget()
        experiment_form = QFormLayout(experiment_page)
        self._dir_experiment = QDirectoryLineEdit()
        self._dir_experiment.setToolTip(_TIP_EXP_DIR)
        self._dir_protocol = QFileLineEdit()
        self._dir_protocol.setToolTip(_TIP_EXP_PROTOCOL_SALAMI)
        experiment_form.addRow(_LBL_EXP_DIR, self._dir_experiment)
        experiment_form.addRow(_LBL_EXP_PROTOCOL, self._dir_protocol)
        self._stack.addWidget(experiment_page)

        # --- Movement ---
        movement_page = QWidget()
        movement_form = QFormLayout(movement_page)
        self._chk_acquire_sem = QCheckBox()
        self._chk_acquire_sem.setToolTip(_TIP_ACQ_SEM)
        self._chk_acquire_fib = QCheckBox()
        self._chk_acquire_fib.setToolTip(_TIP_ACQ_FIB)
        movement_form.addRow(_LBL_ACQ_SEM, self._chk_acquire_sem)
        movement_form.addRow(_LBL_ACQ_FIB, self._chk_acquire_fib)
        self._stack.addWidget(movement_page)

        self._sidebar.currentRowChanged.connect(self._stack.setCurrentIndex)

        # Buttons
        btn_layout = QHBoxLayout()
        self._btn_restore = QPushButton("Restore Defaults")
        self._btn_restore.clicked.connect(self._on_restore_defaults)
        btn_layout.addWidget(self._btn_restore)
        btn_layout.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        btn_layout.addWidget(buttons)
        layout.addLayout(btn_layout)

    def _load(self, prefs: UserPreferences, salami: SalamiPreferences):
        """Populate widgets from both stores."""
        self._chk_sound.setChecked(prefs.display.sound_enabled)
        self._chk_border.setChecked(prefs.display.border_enabled)
        self._dir_experiment.setText(salami.default_experiment_directory)
        self._dir_protocol.setText(salami.default_protocol_path)
        self._chk_acquire_sem.setChecked(
            prefs.movement.acquire_sem_after_stage_movement
        )
        self._chk_acquire_fib.setChecked(
            prefs.movement.acquire_fib_after_stage_movement
        )

    def _on_restore_defaults(self):
        reply = QMessageBox.question(
            self,
            "Restore Defaults",
            "Reset all preferences to their default values?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply == QMessageBox.Yes:
            self._load(UserPreferences(), SalamiPreferences())

    def _protocol_problem(self) -> str:
        """Why the protocol field cannot be saved, or "" if it can."""
        path = self._dir_protocol.text().strip()
        if not path:
            return ""
        if not os.path.isfile(path):
            return f"The protocol file does not exist:\n\n{path}"
        try:
            SalamiExperimentConfig.load(path)
        except Exception as e:
            return f"The file is not a valid Salami protocol:\n\n{path}\n\n{e}"
        return ""

    def _on_ok(self):
        problem = self._protocol_problem()
        if problem:
            QMessageBox.warning(self, "Invalid Default Protocol", problem)
            return
        self.accept()

    def get_preferences(self) -> Tuple[UserPreferences, SalamiPreferences]:
        """Both stores, from the current widget state.

        The shared result starts from the preferences the dialog was given, so
        features, agent, reporting, hooks, AutoLamella's experiment defaults,
        card layout and dev mode all come through unchanged.
        """
        prefs = self._preferences
        display: DisplayPreferences = dataclasses.replace(
            prefs.display,
            sound_enabled=self._chk_sound.isChecked(),
            border_enabled=self._chk_border.isChecked(),
        )
        movement = MovementPreferences(
            acquire_sem_after_stage_movement=self._chk_acquire_sem.isChecked(),
            acquire_fib_after_stage_movement=self._chk_acquire_fib.isChecked(),
        )
        shared = dataclasses.replace(prefs, display=display, movement=movement)
        salami = SalamiPreferences(
            default_experiment_directory=self._dir_experiment.text().strip(),
            default_protocol_path=self._dir_protocol.text().strip(),
        )
        return shared, salami
