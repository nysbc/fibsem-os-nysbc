"""Editor for the protocol's setup section, shown when Task is "Setup".

    Step  [Mill Trenches ▾]
    [ ] Skip this step
    ┌ milling config (AutoLamella's milling editor) ┐
    │ parameters · alignment · acquisition · stages │
    └───────────────────────────────────────────────┘
    [Sync from Microscope]            [Update Protocol]

The preview — the FIB image with this step's patterns drawn on it — is a
canvas of its own (`preview_widget`), which the Protocol page shows in place of
the task preview while Setup is selected.

Unlike the task editor, edits are not saved as they are made. They collect in a
working copy and Update Protocol writes them to the experiment: a milling
config is edited in several moves (add a stage, move it, change its current)
and a half-made trench should not land in the protocol on the way.
"""

from copy import deepcopy
from typing import Dict, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.salami.setup_config import (
    SETUP_STEP_LABELS,
    TRENCH_KEY,
    SalamiSetupConfig,
)
from fibsem.applications.salami.ui.task_editor import MicroscopeSyncRow
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import BeamType, FibsemImage
from fibsem.ui import stylesheets
from fibsem.ui.widgets.canvas.quad_view import (
    LamellaEditorView,
    MicroscopeViewController,
)
from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget

UPDATE_PROTOCOL_LABEL = "Update Protocol"
UNSAVED_TEXT = "Unsaved changes"

_offline_microscope = None


def _offline_milling_microscope():
    """An unconnected Demo microscope, for editing milling while offline.

    The milling editor reads the selectable currents and application files
    from a microscope as it is built. Offline there is none, so these come from
    the simulator's lists until the real one connects and the editor is rebuilt
    against it. Never connected, so nothing can be milled through it.
    """
    global _offline_microscope
    if _offline_microscope is None:
        from fibsem import utils
        from fibsem.microscopes.simulator import DemoMicroscope

        settings = utils.load_microscope_configuration(None)
        _offline_microscope = DemoMicroscope(settings.system)
    return _offline_microscope


class SetupConfigEditor(QWidget):
    """Edits a working copy of `SalamiSetupConfig`, one step at a time."""

    # Acquire a FIB image for the step being edited (its key).
    sync_from_requested = pyqtSignal(str)
    # Write the working copy to the protocol.
    update_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup = SalamiSetupConfig()
        self._saved = self._setup.to_dict()
        self._key = TRENCH_KEY
        # The last image synced per step: steps differ in field of view.
        self._images: Dict[str, FibsemImage] = {}
        self._loading = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self.step_combo = QComboBox()
        for key, label in SETUP_STEP_LABELS.items():
            self.step_combo.addItem(label, key)
        form.addRow("Step", self.step_combo)
        self.skip_checkbox = QCheckBox("Skip this step")
        self.skip_checkbox.setToolTip(
            "Leave this step out when setup runs, e.g. when re-running setup on "
            "a sample where it is already done."
        )
        form.addRow("", self.skip_checkbox)
        layout.addLayout(form)

        # The preview canvas, with this step's patterns drawn on the FIB image.
        # Parented by whoever shows it (the Protocol page).
        self.view_controller = MicroscopeViewController(view=LamellaEditorView())
        self.preview_widget = self.view_controller.widget

        self._layout = layout
        self._microscope = None
        self.milling_editor: Optional[MillingTaskViewerWidget] = None
        self._build_milling_editor(_offline_milling_microscope())

        # Only "from": a setup step has no settings to push to the column.
        self.sync_row = MicroscopeSyncRow(parent=self)
        self.sync_row.button_to.setVisible(False)
        self.sync_row.button_from.setToolTip(
            "Acquire a FIB image at this step's field of view, to place the "
            "patterns on."
        )

        self.button_update = QPushButton(UPDATE_PROTOCOL_LABEL)
        self.button_update.setToolTip("Save these setup steps to the protocol")
        self.label_unsaved = QLabel(UNSAVED_TEXT)
        self.label_unsaved.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.addWidget(self.sync_row, stretch=1)
        bottom.addWidget(self.label_unsaved)
        bottom.addWidget(self.button_update)
        layout.addLayout(bottom)

        self.step_combo.currentIndexChanged.connect(self._on_step_changed)
        self.skip_checkbox.toggled.connect(self._on_skip_toggled)
        self.sync_row.sync_from_requested.connect(
            lambda: self.sync_from_requested.emit(self._key)
        )
        self.button_update.clicked.connect(self.update_requested)

        self._load_step()

    # ── public API ────────────────────────────────────────────────────────

    @property
    def current_step(self) -> str:
        return self._key

    def set_setup(self, setup: SalamiSetupConfig) -> None:
        """Load the protocol's setup section, discarding unsaved edits."""
        self._setup = deepcopy(setup)
        self._saved = self._setup.to_dict()
        self._load_step()

    def get_setup(self) -> SalamiSetupConfig:
        """The working copy, edits included."""
        return deepcopy(self._setup)

    def mark_saved(self) -> None:
        self._saved = self._setup.to_dict()
        self._refresh_dirty()

    @property
    def is_dirty(self) -> bool:
        return self._setup.to_dict() != self._saved

    def set_microscope(self, microscope) -> None:
        """Rebuild the milling editor against `microscope`'s currents."""
        if microscope is not self._microscope:
            self._microscope = microscope
            self._build_milling_editor(microscope or _offline_milling_microscope())
            self._load_step()
        self.sync_row.set_microscope_available(microscope is not None)
        if microscope is not None:
            self.sync_row.set_status("")

    def set_busy(self, busy: bool, msg: str = "") -> None:
        self.sync_row.set_busy(busy, msg)
        self.step_combo.setEnabled(not busy)

    def set_update_enabled(self, enabled: bool) -> None:
        self.button_update.setEnabled(enabled)

    def show_image(self, key: str, image: FibsemImage) -> None:
        """Keep `image` as step `key`'s preview, and show it if that step is open."""
        self._images[key] = image
        if key == self._key:
            self._show_current_image()

    # ── internals ─────────────────────────────────────────────────────────

    def _build_milling_editor(self, microscope) -> None:
        """(Re)create the milling editor; it reads currents only when built."""
        old = self.milling_editor
        editor = MillingTaskViewerWidget(
            microscope=microscope, milling_enabled=False, parent=self
        )
        editor.set_controller(self.view_controller)
        editor.set_alignment_area_visible(False)
        editor.settings_changed.connect(self._on_milling_changed)
        if old is None:
            # Between the skip checkbox and the sync row.
            self._layout.insertWidget(1, editor, stretch=1)
        else:
            old.settings_changed.disconnect(self._on_milling_changed)
            self._layout.replaceWidget(old, editor)
            old.deleteLater()
        self.milling_editor = editor

    def _step_config(self) -> FibsemMillingTaskConfig:
        return self._setup.step(self._key).milling

    def _load_step(self) -> None:
        self._loading = True
        try:
            step = self._setup.step(self._key)
            self.skip_checkbox.setChecked(step.skip)
            self.milling_editor.set_config(deepcopy(step.milling))
        finally:
            self._loading = False
        self._show_current_image()
        self._refresh_dirty()

    def _show_current_image(self) -> None:
        image = self._images.get(self._key)
        if image is None:
            self.view_controller.clear()
            return
        self.view_controller.set_image(BeamType.ION, image)
        self.milling_editor.set_fib_image(image)

    def _on_step_changed(self, *_args) -> None:
        self._key = self.step_combo.currentData() or TRENCH_KEY
        self._load_step()

    def _on_skip_toggled(self, checked: bool) -> None:
        if self._loading:
            return
        self._setup.step(self._key).skip = bool(checked)
        self._refresh_dirty()

    def _on_milling_changed(self, config: FibsemMillingTaskConfig) -> None:
        # The editor reports its config on load too (debounced, so after
        # `_loading` has cleared); comparing against the saved copy is what
        # keeps a plain load from reading as an edit.
        self._setup.step(self._key).milling = deepcopy(config)
        self._refresh_dirty()

    def _refresh_dirty(self) -> None:
        dirty = self.is_dirty
        self.label_unsaved.setVisible(dirty)
        self.button_update.setStyleSheet(
            stylesheets.PRIMARY_BUTTON_STYLESHEET
            if dirty
            else stylesheets.SECONDARY_BUTTON_STYLESHEET
        )
