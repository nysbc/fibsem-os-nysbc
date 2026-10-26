"""Pick a task type and name it, for adding a task to a protocol.

The type list comes from the task registry, so a newly registered task type
appears here without this file being touched.
"""

import logging
import re
from typing import List, Optional

from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.salami.tasks import (
    TaskConfig,
    available_task_types,
    create_task_config,
    suggest_task_name,
    task_type_label,
)
from fibsem.ui import stylesheets

# A task name becomes a directory name — `BaseTask.output_dir` joins it onto the
# experiment path — and part of every image filename under it. Anything that is
# not a word character, space, dash or dot would either create a subdirectory or
# fail to write at all, and on Windows it fails at the point of saving the first
# slice rather than here.
VALID_NAME = re.compile(r"^[\w \-.]+$")


class AddTaskDialog(QDialog):
    """Choose a task type and name. Returns a config; adds nothing itself.

    Building the config and putting it into a protocol are kept apart, the same
    split `SalamiExperiment.create` uses: the caller decides where it goes and
    when it is written.
    """

    def __init__(self, existing_names: List[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Add Task")
        self.setModal(True)
        self.existing_names = list(existing_names)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.comboBoxType = QComboBox(self)
        for task_type in available_task_types():
            self.comboBoxType.addItem(task_type_label(task_type), task_type)

        self.lineEditName = QLineEdit(self)
        form.addRow("Task Type", self.comboBoxType)
        form.addRow("Task Name", self.lineEditName)
        layout.addLayout(form)

        self.label_warning = QLabel(self)
        self.label_warning.setWordWrap(True)
        self.label_warning.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        layout.addWidget(self.label_warning)

        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)

        self.comboBoxType.currentIndexChanged.connect(self._on_type_changed)
        self.lineEditName.textChanged.connect(self._validate)

        self._on_type_changed()

    @property
    def task_type(self) -> str:
        return self.comboBoxType.currentData()

    @property
    def task_name(self) -> str:
        return self.lineEditName.text().strip()

    def _on_type_changed(self, *_args) -> None:
        """Suggest a name for the newly chosen type.

        Only when the operator has not typed one of their own, so switching type
        to compare the options cannot silently discard a name already entered.
        """
        suggested = suggest_task_name(self.task_type, self.existing_names)
        current = self.task_name
        if not current or current in self._suggestions():
            self.lineEditName.setText(suggested)
        self._validate()

    def _suggestions(self) -> List[str]:
        return [
            suggest_task_name(t, self.existing_names) for t in available_task_types()
        ]

    def _validate(self, *_args) -> None:
        name = self.task_name
        problem = ""
        if not name:
            problem = "Give the task a name."
        elif name in self.existing_names:
            problem = f"A task named {name!r} already exists in this protocol."
        elif not VALID_NAME.match(name):
            problem = (
                "The name is used as a folder name: letters, numbers, spaces, "
                "dashes, dots and underscores only."
            )

        self.label_warning.setText(f"⚠ {problem}" if problem else "")
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(not problem)

    def task_config(self) -> TaskConfig:
        """The config for the chosen type and name."""
        config = create_task_config(self.task_type, self.task_name)
        logging.info(
            f"Created {self.task_type!r} task config named {self.task_name!r}."
        )
        return config
