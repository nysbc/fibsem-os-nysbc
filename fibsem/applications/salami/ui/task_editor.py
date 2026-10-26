"""Task Config tab: pick a task, edit its parameters.

The editors themselves are generated from the config dataclasses by
`config_editor.DataclassEditorWidget`. What lives here is the per-task-type
*view spec* — which fields to hide, which to show but lock, and which sections
open by default. That is the only thing a new field can't infer for itself, and
it is the reason the specs are declarative: adding a field to a task config makes
an editor appear automatically, and it only needs an entry here if it should
*not* be freely editable.
"""

import logging
from typing import Any, Dict, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.salami.tasks import (
    SENSEAI_SCAN_PATTERNS,
    AcquisitionTaskConfig,
    MillingTaskConfig,
    SenseAIAcquisitionTaskConfig,
    imaging_view,
    resolve_beam_defaults,
)
from fibsem.applications.salami.ui.config_editor import (
    FIELD_HINTS,
    DataclassEditorWidget,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamSettings, BeamType, field_meta
from fibsem.ui import stylesheets
from fibsem.ui.utils import install_wheel_blocker
from fibsem.ui.widgets.custom_widgets import ValueComboBox, _SpinnerLabel
from fibsem.utils import format_value

# A task's name is its output directory (`BaseTask.output_dir`) and the key used
# by `find_task` / `skip_once`. Renaming it mid-experiment would orphan the
# existing output directory and silently drop a pending skip.
NAME_IS_FIXED = (
    "The task name is its output directory and the key used to enable or skip "
    "it, so it is fixed once the experiment exists."
)

# The sections the sync buttons sit above, per task type. Each is the same path
# that spec puts its divider on, so the buttons land immediately below the rule,
# heading the group of settings the microscope and the protocol share.
SYNC_ROW_ANCHORS = ("state.electron_beam", "state.ion_beam")

# Button labels. The busy label matches FibsemImageSettingsWidget, which renames
# its acquire buttons while the column is scanning.
SYNC_FROM_LABEL = "\u2193 Sync from Microscope"
SYNC_TO_LABEL = "\u2191 Sync to Microscope"
SYNC_BUSY_LABEL = "Acquiring..."


ACQUISITION_SPEC: Dict[str, Any] = {
    "exclude": {
        # --- run state and sample state, which are not protocol ---
        "state.timestamp",
        # Sample-specific: belongs in status.sample_state. It is also the field
        # that makes set_microscope_state drive the stage, so it must never be
        # editable as part of a reusable protocol.
        "state.stage_position",
        "state.objective_position",
        # The SEM images the block face; the ion column is the miller.
        "state.ion_beam",
        "state.electron_detector",
        "state.ion_detector",
        # --- owned elsewhere ---
        # Fixed by which slot the beam block sits in.
        "state.electron_beam.beam_type",
        # BeamSettings also carries these three, but ImageSettings is what
        # AcquisitionTask.run hands to acquire_image, so it owns them. Editing
        # them in two places would leave the operator guessing which wins.
        "state.electron_beam.hfw",
        "state.electron_beam.resolution",
        "state.electron_beam.dwell_time",
        # Rewritten every step by AcquisitionTask.apply_advance to track the
        # receding face — an operator value here would be overwritten anyway.
        "state.electron_beam.shift",
        # Fine column tuning: live work, done on the Image tab against a picture.
        "state.electron_beam.stigmation",
        "state.electron_beam.scan_rotation",
        "state.electron_beam.preset",
        # --- derived at run time, not protocol ---
        "image.beam_type",
        "image.save",  # forced True in AcquisitionTask.run
        "image.filename",  # set from the step index in run()
        "image.path",  # set per step by AcquisitionTask.configure()
        "image.reduced_area",
        # Both are forced False in AcquisitionTask.run alongside image.save, so
        # showing them would offer a choice the run ignores. Per-slice contrast
        # is the `autocontrast` AutoFunctionConfig below, which has its own
        # cadence; a custom implementation is planned to replace it.
        "image.autocontrast",
        "image.autogamma",
        # Reserved for fiducial drift correction, which does not exist yet. The
        # auto-functions each have their own `area_of_interest` now, so showing
        # this would offer a region nothing reads.
        "alignment_area",
    },
    "read_only": {"name": NAME_IS_FIXED},
    # `state` holds only a beam block once the above are excluded, so render it
    # inline rather than as a section wrapping a section.
    "flatten": {"state"},
    "expanded": {"schedule", "state.electron_beam"},
    # Everything above the rule is about *when and whether* the task runs;
    # everything below is how the SEM images the face. Two different jobs, so
    # they read as two groups.
    "dividers": {"state.electron_beam"},
    "hints": {
        **FIELD_HINTS,
        "electron_beam": field_meta(label="Electron Beam (SEM)"),
        "area_of_interest": field_meta(
            label="Area of Interest",
            tooltip=(
                "Region of the frame this function uses, as fractions of the "
                "frame. Also editable by dragging the box on the preview."
            ),
        ),
        "enabled": field_meta(
            label="Task Enabled",
            tooltip=(
                "Turn the task off for the rest of the run. Persisted in the "
                "protocol, and reset when the protocol is reused on a new sample."
            ),
        ),
    },
}


# The working distance is focus. It is set by focusing the column, not by typing
# a number into a protocol, so it is shown for checking and moved by the sync.
WORKING_DISTANCE_IS_READ_ONLY = (
    "Set the working distance by focusing the column, then use Sync from "
    "Microscope to capture it."
)

# A milling task images the ion column only to see where to mill, so the beam
# block holds viewing parameters. Voltage and current are shown without an unset
# checkbox for the same reason as the acquisition task: the control is the value.
MILLING_HINTS: Dict[str, dict] = {
    **FIELD_HINTS,
    "ion_beam": field_meta(label="Ion Beam (FIB) — Imaging"),
    "voltage": {**FIELD_HINTS["voltage"], "always_set": True},
    "beam_current": {
        **FIELD_HINTS["beam_current"],
        "always_set": True,
        "tooltip": (
            "Imaging current for viewing the face. The milling current is set "
            "per stage on the Milling tab and is not affected by this."
        ),
    },
    # On a BeamSettings these are Optional, but here they are the task's viewing
    # parameters and always hold a value — `resolve_beam_defaults` seeds them
    # from the column. No unset state, exactly as on the acquisition task, where
    # they sit on a non-Optional ImageSettings.
    "hfw": {**FIELD_HINTS["hfw"], "always_set": True},
    "resolution": {**FIELD_HINTS["resolution"], "always_set": True},
    "dwell_time": {**FIELD_HINTS["dwell_time"], "always_set": True},
    "enabled": ACQUISITION_SPEC["hints"]["enabled"],
}


MILLING_SPEC: Dict[str, Any] = {
    "exclude": {
        # --- run state and sample state, which are not protocol ---
        "state.timestamp",
        "state.stage_position",
        "state.objective_position",
        # The ion column both images and mills here; the SEM is not involved.
        "state.electron_beam",
        "state.electron_detector",
        "state.ion_detector",
        # --- fixed or owned elsewhere ---
        # Always the ion beam for a milling task, so there is nothing to choose.
        "state.ion_beam.beam_type",
        "state.ion_beam.preset",
        # Live column tuning, done on the Image tab against a picture.
        "state.ion_beam.stigmation",
        "state.ion_beam.shift",
        "state.ion_beam.scan_rotation",
        # Pattern geometry, milling current, alignment and per-task acquisition
        # all belong to the milling engine and are edited on the Milling tab —
        # keeping them off this form is what stops two widgets disagreeing about
        # one FIB canvas.
        "milling",
    },
    "read_only": {
        "name": NAME_IS_FIXED,
        "state.ion_beam.working_distance": WORKING_DISTANCE_IS_READ_ONLY,
    },
    "flatten": {"state"},
    "expanded": {"schedule", "state.ion_beam"},
    # Same split as the acquisition task: above the rule is when and whether the
    # task runs, below is how the ion column views the face.
    "dividers": {"state.ion_beam"},
    "hints": MILLING_HINTS,
}


# The SenseAI task has no ImageSettings: the external scan generator owns
# resolution and dwell time, so they are edited in the Scan section and the beam
# block holds only HFW alongside voltage, current and working distance.
SENSEAI_HINTS: Dict[str, dict] = {
    **FIELD_HINTS,
    "electron_beam": field_meta(label="Electron Beam (SEM)"),
    "hfw": {**FIELD_HINTS["hfw"], "always_set": True},
    "voltage": {**FIELD_HINTS["voltage"], "always_set": True},
    "beam_current": {**FIELD_HINTS["beam_current"], "always_set": True},
    "scan": field_meta(label="Scan Generator (SenseAI)"),
    # The same choices as an ordinary task's resolution, because it is the same
    # quantity — the generator owns it while the column is in external scan.
    # Without `items` the editor has no control for a tuple, and shows it
    # read-only.
    "scan_size": field_meta(
        FIELD_HINTS["resolution"],
        label="Scan Size",
        tooltip="Pixels the generator scans, width x height. This is the resolution.",
    ),
    "pattern": field_meta(
        label="Scan Pattern",
        items=list(SENSEAI_SCAN_PATTERNS),
        tooltip=(
            "How the generator walks the frame while visiting the sampled "
            "pixels. Must be one the scan generator supports."
        ),
    ),
    "sampling": field_meta(
        label="Sampling",
        tooltip=(
            "Fraction of pixels actually measured. The reconstruction fills in "
            "the rest, so a lower value is a faster slice and a noisier image."
        ),
    ),
    "reconstruction": field_meta(label="Reconstruction"),
    "iterations": field_meta(
        label="Iterations",
        tooltip=(
            "50 is converged. Reconstruction quality is tuned in the SenseAI "
            "desktop app; the Input and Mask layers are saved for every slice, "
            "so any slice can be reconstructed again afterwards."
        ),
    ),
    "enabled": ACQUISITION_SPEC["hints"]["enabled"],
}


SENSEAI_SPEC: Dict[str, Any] = {
    "exclude": {
        # --- run state and sample state, which are not protocol ---
        "state.timestamp",
        "state.stage_position",
        "state.objective_position",
        # The SEM images the block face through the external generator.
        "state.ion_beam",
        "state.electron_detector",
        "state.ion_detector",
        # --- fixed or owned elsewhere ---
        "state.electron_beam.beam_type",
        "state.electron_beam.preset",
        # Owned by the scan generator, in the Scan section below. A BeamSettings
        # carries the fields, but nothing reads them for this task.
        "state.electron_beam.resolution",
        "state.electron_beam.dwell_time",
        # Rewritten every step to track the receding face.
        "state.electron_beam.shift",
        # Live column tuning, done on the Image tab against a picture.
        "state.electron_beam.stigmation",
        "state.electron_beam.scan_rotation",
        # Set per step by configure(); machine-specific, never protocol.
        "output_path",
    },
    "read_only": {
        "name": NAME_IS_FIXED,
        "state.electron_beam.working_distance": WORKING_DISTANCE_IS_READ_ONLY,
    },
    "flatten": {"state"},
    "expanded": {"schedule", "state.electron_beam", "scan"},
    # Same split as the other tasks: when and whether the task runs, then how
    # the face is imaged.
    "dividers": {"state.electron_beam"},
    "hints": SENSEAI_HINTS,
}


TASK_SPECS: Dict[str, Dict[str, Any]] = {
    AcquisitionTaskConfig.TASK_TYPE: ACQUISITION_SPEC,
    MillingTaskConfig.TASK_TYPE: MILLING_SPEC,
    SenseAIAcquisitionTaskConfig.TASK_TYPE: SENSEAI_SPEC,
}


class MicroscopeSyncRow(QWidget):
    """The two sync buttons, a spinner and a status line.

    Purely a control surface: it emits a request and waits to be told what
    happened. The work runs on a background thread owned by the main window,
    because an acquisition takes seconds and Qt only lets the main thread touch
    widgets — a spinner driven from the acquiring thread would not even animate.
    """

    sync_from_requested = pyqtSignal()
    sync_to_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(4)

        # Styled as primary actions, like the acquire buttons on
        # FibsemImageSettingsWidget: these are the main action of the panel, and
        # they acquire an image too.
        self.button_from = QPushButton(SYNC_FROM_LABEL)
        self.button_from.setToolTip(
            "Read the column's current imaging and beam settings into this task, "
            "then acquire an image at those settings."
        )
        self.button_to = QPushButton(SYNC_TO_LABEL)
        self.button_to.setToolTip(
            "Apply this task's imaging and beam settings to the column, then "
            "acquire an image at those settings. Never moves the stage."
        )

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(4)
        buttons.addWidget(self.button_from)
        buttons.addWidget(self.button_to)

        self.spinner = _SpinnerLabel(size=16, step_deg=30, interval_ms=40)
        self.spinner.setVisible(False)
        buttons.addWidget(self.spinner)
        layout.addLayout(buttons)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        self.button_from.clicked.connect(self.sync_from_requested)
        self.button_to.clicked.connect(self.sync_to_requested)

        self._has_microscope = False
        self._busy = False
        self._apply_button_state()

    def set_microscope_available(self, available: bool) -> None:
        self._has_microscope = available
        if not available:
            self.set_status("Connect to the microscope to sync settings.")
        self._apply_button_state()

    def set_busy(self, busy: bool, msg: str = "") -> None:
        """Show the acquisition spinner and lock the buttons while syncing."""
        self._busy = busy
        self.spinner.setVisible(busy)
        if busy:
            self.spinner.start()
            if msg:
                self.set_status(msg)
        else:
            self.spinner.stop()
        self._apply_button_state()

    def set_status(self, msg: str) -> None:
        self.status_label.setText(msg)
        self.status_label.setVisible(bool(msg))

    def _apply_button_state(self) -> None:
        """Style and label the buttons for the current state.

        Follows FibsemImageSettingsWidget: primary while idle, and while imaging
        they drop to secondary and rename to "Acquiring...". No style is cleared
        when disabled — both stylesheets carry their own `:disabled` rule.
        """
        enabled = self._has_microscope and not self._busy
        for button, label in (
            (self.button_from, SYNC_FROM_LABEL),
            (self.button_to, SYNC_TO_LABEL),
        ):
            button.setEnabled(enabled)
            if self._busy:
                button.setText(SYNC_BUSY_LABEL)
                button.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
            else:
                button.setText(label)
                button.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)


class SliceVolumePanel(QWidget):
    """Slice thickness for a milling task, and the volume it adds up to.

    Thickness is `advance.step_size`, which is experiment-level rather than the
    task's own — there is one volume, so one thickness, shared with the
    acquisition task's beam-shift advance. It is shown here because this is the
    task that does the slicing, and it is the control an operator reaches for.

    The slice *count* is derived, never typed: it is the height of the drawn
    region divided by the thickness. That height is the distance sliced through
    the block — not a stage's milling depth, which is how far each cut reaches
    into the face and is fixed per stage.
    """

    thickness_changed = pyqtSignal(float)  # metres

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 6)
        layout.setSpacing(4)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Slice Thickness"))
        self.thickness_spinbox = QDoubleSpinBox()
        self.thickness_spinbox.setSuffix(" nm")
        self.thickness_spinbox.setDecimals(2)
        self.thickness_spinbox.setRange(0.01, 1e6)
        self.thickness_spinbox.setKeyboardTracking(False)
        self.thickness_spinbox.setToolTip(
            "Height of the slab removed each slice, and the distance the pattern "
            "advances afterwards. Changing it re-derives the slice count for the "
            "same volume."
        )
        install_wheel_blocker(self.thickness_spinbox)
        row.addWidget(self.thickness_spinbox, stretch=1)
        layout.addLayout(row)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        layout.addWidget(self.summary_label)

        self.thickness_spinbox.valueChanged.connect(
            lambda nm: self.thickness_changed.emit(nm * 1e-9)
        )

    def set_thickness(self, metres: float) -> None:
        """Show `metres` without re-emitting; used when the config changes elsewhere."""
        self.thickness_spinbox.blockSignals(True)
        self.thickness_spinbox.setValue(metres * 1e9)
        self.thickness_spinbox.blockSignals(False)

    def set_volume(self, n_steps: int, thickness: float) -> None:
        extent = n_steps * thickness
        self.summary_label.setText(
            f"{n_steps} slices  ·  "
            f"{format_value(extent, 'm', precision=2)} through the block"
        )

    def set_no_region(self, msg: str) -> None:
        self.summary_label.setText(msg)


class MillingStagePanel(QWidget):
    """Per-stage milling settings: how much current, and how deep each cut goes.

    One row per stage rather than one control, because a slice task can carry a
    coarse and a fine stage cutting the same plane and they share neither
    setting. With the default one-stage slice it reads as a single pair.

    **Milling depth is not the height of the volume box.** It is how far each cut
    reaches *into* the block face, a fixed property of the stage that does not
    change when the region is resized. The box height is the distance sliced
    through, which is what sets the slice count. Two different axes that both
    get called depth, which is why they are labelled apart here.

    Currents come from the ion column when one is connected, so only values the
    microscope offers can be picked; offline both fall back to numeric entry, so
    a protocol can still be prepared away from the microscope.

    These are the only milling parameters shown on this form. Pattern geometry
    stays on the Milling tab, so two widgets never draw on one FIB canvas —
    neither of these has any geometry to draw.
    """

    changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 6)
        self._layout.setSpacing(4)
        # (stage, control) pairs, kept so tests and callers can reach the
        # controls without walking the layout.
        self._rows: list = []
        self._depth_rows: list = []

    def _clear(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())
        self._rows = []
        self._depth_rows = []

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

    def set_stages(self, stages: list, available: Optional[list] = None) -> None:
        """Build the rows for `stages`. `available` is the column's current list."""
        self._clear()

        for stage in stages:
            prefix = f"{stage.name} — " if len(stages) > 1 else ""
            self._add_current_row(stage, prefix, available)
            # A trench states its extent as upper/lower heights and a line has
            # none; only add the control for patterns that actually carry depth.
            if hasattr(stage.pattern, "depth"):
                self._add_depth_row(stage, prefix)

    def _add_current_row(self, stage, prefix: str, available: Optional[list]) -> None:
        stored = stage.milling.milling_current
        if available:
            control = ValueComboBox(items=list(available), value=stored, unit="A")
            control.setToolTip(
                "Milling current, from the values the ion column offers."
            )
            control.currentIndexChanged.connect(
                lambda _idx, s=stage, c=control: self._apply_current(s, c.value())
            )
        else:
            control = QDoubleSpinBox()
            control.setSuffix(" nA")
            control.setDecimals(3)
            control.setRange(0.0, 1e6)
            control.setKeyboardTracking(False)
            control.setValue((stored or 0.0) * 1e9)
            control.setToolTip(
                "Milling current. Connect to the microscope to choose from the "
                "values the ion column supports."
            )
            install_wheel_blocker(control)
            control.valueChanged.connect(
                lambda nA, s=stage: self._apply_current(s, nA * 1e-9)
            )

        self._add_row(f"{prefix}Milling Current", control)
        self._rows.append((stage, control))

    def _add_depth_row(self, stage, prefix: str) -> None:
        control = QDoubleSpinBox()
        control.setSuffix(" µm")
        control.setDecimals(2)
        control.setRange(0.01, 1000.0)
        control.setKeyboardTracking(False)
        control.setValue((stage.pattern.depth or 0.0) * 1e6)
        control.setToolTip(
            "How far each cut reaches into the block face. Fixed per stage — it "
            "does not change with the size of the region, and it is not the "
            "height of the volume box, which is the distance sliced through."
        )
        install_wheel_blocker(control)
        control.valueChanged.connect(
            lambda um, s=stage: self._apply_depth(s, um * 1e-6)
        )

        self._add_row(f"{prefix}Milling Depth", control)
        self._depth_rows.append((stage, control))

    def _add_row(self, label: str, control) -> None:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(label))
        row.addWidget(control, stretch=1)
        self._layout.addLayout(row)

    def _apply_current(self, stage, value: Optional[float]) -> None:
        if value is None or value == stage.milling.milling_current:
            return
        stage.milling.milling_current = value
        self.changed.emit()

    def _apply_depth(self, stage, value: float) -> None:
        if value == stage.pattern.depth:
            return
        stage.pattern.depth = value
        self.changed.emit()


class TaskConfigEditorWidget(QWidget):
    """Parameter editor for whichever task is selected in the Task Config tab.

    Rebuilds itself from the selected task's config each time, so switching
    tasks — or loading a protocol whose configs differ — needs no bookkeeping.
    Edits mutate the config in place; `config_changed` asks the owner to persist.
    """

    config_changed = pyqtSignal()
    sync_from_microscope_requested = pyqtSignal()
    sync_to_microscope_requested = pyqtSignal()

    def __init__(
        self,
        microscope: Optional[FibsemMicroscope] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.task_config: Optional[Any] = None

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)

        # Owned here, injected into the generated form so it lands directly below
        # the divider — at the head of the settings the microscope shares.
        self.sync_row = MicroscopeSyncRow(parent=self)
        self.sync_row.sync_from_requested.connect(self.sync_from_microscope_requested)
        self.sync_row.sync_to_requested.connect(self.sync_to_microscope_requested)
        self.sync_row.set_microscope_available(microscope is not None)

        # Slicing is what a milling task is for, so it heads the form rather
        # than sitting below the schedule and beam settings.
        self.slice_panel = SliceVolumePanel(parent=self)
        self.slice_panel.setVisible(False)
        self._layout.addWidget(self.slice_panel)

        # Beside the thickness: the two things that decide what one slice costs.
        self.milling_stage_panel = MillingStagePanel(parent=self)
        self.milling_stage_panel.setVisible(False)
        self.milling_stage_panel.changed.connect(self._on_editor_changed)
        self._layout.addWidget(self.milling_stage_panel)

        # Injected against both anchors: only one exists in any given form, so
        # the row lands under whichever beam section that task type shows.
        self.editor = DataclassEditorWidget(
            microscope=microscope,
            inject={anchor: self.sync_row for anchor in SYNC_ROW_ANCHORS},
        )
        self.editor.config_changed.connect(self._on_editor_changed)
        self._layout.addWidget(self.editor)

        # Derived readouts. Voxel isotropy is the thing a slice-and-view operator
        # most needs to see while setting imaging parameters, and it is not a
        # field on anything — in-plane pixel size should roughly match the slice
        # thickness.
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        self._layout.addWidget(self.info_label)

        self.placeholder = QLabel()
        self.placeholder.setWordWrap(True)
        self.placeholder.setVisible(False)
        self._layout.addWidget(self.placeholder)

    @property
    def microscope(self) -> Optional[FibsemMicroscope]:
        return self.editor.microscope

    def set_microscope(self, microscope: Optional[FibsemMicroscope]) -> None:
        """Point the editor at a microscope so it can read available values.

        Connecting turns voltage and current into dropdowns of what the column
        actually supports; disconnecting falls back to numeric entry so a protocol
        can still be prepared offline.
        """
        self.sync_row.set_microscope_available(microscope is not None)
        if microscope is not None:
            self.sync_row.set_status("")
            # Seed before the rebuild, so the dropdowns show stored values.
            self._seed_beam_defaults(microscope)
        self.editor.set_microscope(microscope)

    def _seed_beam_defaults(self, microscope: FibsemMicroscope) -> None:
        """Give unset beam fields a real value, and persist if anything changed.

        The dropdowns have no "unset" entry, so an unset field would display the
        first supported value while the config still held None — the form would be
        telling the operator something untrue.
        """
        config = self.task_config
        if config is None or getattr(config, "TASK_TYPE", None) not in TASK_SPECS:
            return
        if resolve_beam_defaults(microscope, config):
            self.config_changed.emit()

    def set_task_config(self, task_config: Optional[Any]) -> None:
        """Show an editor for `task_config`, or clear if None."""
        self.task_config = task_config

        if task_config is None:
            self.editor.set_object(None)
            self.info_label.clear()
            self._set_placeholder("Select a task to edit its parameters.")
            return

        task_type = getattr(task_config, "TASK_TYPE", None)
        spec = TASK_SPECS.get(task_type)
        if spec is None:
            self.editor.set_object(None)
            self.info_label.clear()
            self._set_placeholder(
                f"No editor is defined for task type {task_type!r} yet."
            )
            return

        self._ensure_editable(task_config)

        # `inject` stays constant so `_clear` always reparents the sync row out
        # instead of destroying it. Both task types sync — the acquisition task
        # against the SEM, the milling task against the ion column.
        self.sync_row.setVisible(True)
        is_milling = task_type == MillingTaskConfig.TASK_TYPE
        self.slice_panel.setVisible(is_milling)
        self.milling_stage_panel.setVisible(is_milling)
        if is_milling:
            self.milling_stage_panel.set_stages(
                task_config.milling.stages, self._available_milling_currents()
            )
        if self.microscope is not None:
            self._seed_beam_defaults(self.microscope)

        self.editor.set_object(None)  # drop the previous form before rebuilding
        self.editor.exclude = set(spec.get("exclude", ()))
        self.editor.read_only = dict(spec.get("read_only", {}))
        self.editor.flatten = set(spec.get("flatten", ()))
        self.editor.expanded = set(spec.get("expanded", ()))
        self.editor.dividers = set(spec.get("dividers", ()))
        self.editor.hints = spec.get("hints", FIELD_HINTS)
        self.editor.set_object(task_config)

        # Milling patterns are a list of dataclasses, which the generated editor
        # does not render, and the roadmap keeps pattern geometry on the Milling
        # tab so two widgets never draw on one FIB canvas. Say so rather than
        # quietly showing a milling task with no patterns in it.
        if task_type == MillingTaskConfig.TASK_TYPE:
            self._set_placeholder(
                "Slice pattern geometry is edited on the Milling tab — it is not "
                "shown here."
            )
        else:
            self.placeholder.setVisible(False)

        self._update_info()

    @staticmethod
    def _ensure_editable(task_config: Any) -> None:
        """Fill in optional sub-configs the editor needs to bind to.

        `MicroscopeState.electron_beam` is Optional, and a YAML file with a null
        beam block loads as None — there would be nothing to render or write to.
        """
        state = getattr(task_config, "state", None)
        if state is None:
            return
        if getattr(state, "electron_beam", None) is None:
            state.electron_beam = BeamSettings(beam_type=BeamType.ELECTRON)
            logging.debug("Created a default electron beam block for editing.")

    def refresh(self) -> None:
        """Re-read every control from the config, after something else changed it.

        Used by a sync from the microscope, which rewrites config values behind
        the form's back.
        """
        self.editor.refresh()
        self._update_info()

    def set_editing_enabled(self, enabled: bool) -> None:
        """Lock the parameter controls while a sync is running.

        The sync row keeps its own state — its buttons are handled by
        `MicroscopeSyncRow.set_busy`, and its spinner has to keep animating.
        """
        self.editor.set_fields_enabled(enabled)

    def _available_milling_currents(self) -> Optional[list]:
        """Currents the ion column offers, or None when there is no microscope.

        Cached on the microscope, so rebuilding the form as tasks are switched
        does not re-query the column each time.
        """
        microscope = self.microscope
        if microscope is None:
            return None
        try:
            return list(
                microscope.get_available_values_cached("current", BeamType.ION) or []
            )
        except Exception:
            logging.debug(
                "Could not read the available milling currents", exc_info=True
            )
            return None

    def _set_placeholder(self, msg: str) -> None:
        self.placeholder.setText(msg)
        self.placeholder.setVisible(True)

    def _on_editor_changed(self) -> None:
        self._update_info()
        self.config_changed.emit()

    def _update_info(self) -> None:
        """Show in-plane pixel size and per-image acquisition time.

        Goes through the imaging view so it reads the right place for either
        task type — a milling task has no `image` of its own.
        """
        if self.task_config is None:
            self.info_label.clear()
            return
        try:
            image = imaging_view(self.task_config).to_image_settings()
        except Exception:
            logging.debug("Could not locate imaging settings", exc_info=True)
            self.info_label.clear()
            return

        parts = []
        try:
            if image.resolution and image.resolution[0]:
                parts.append(
                    f"Pixel size: {format_value(image.hfw / image.resolution[0], 'm', precision=2)}"
                )
            parts.append(
                f"Acquisition: {format_value(image.estimated_time, 's', precision=2)} per image"
            )
        except Exception:
            logging.debug("Could not compute acquisition readouts", exc_info=True)
            self.info_label.clear()
            return

        self.info_label.setText("   |   ".join(parts))
