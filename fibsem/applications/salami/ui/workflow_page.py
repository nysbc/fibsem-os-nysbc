"""The Workflow page: running the experiment and watching it run.

The counterpart to the Protocol page. Protocol is *what the experiment will
do* and is edited; Workflow is *what it is doing* and is read-only, apart from
the controls that start and stop it. Shaped after AutoLamella's Workflow tab:

    ┌ Experiment ──────────┐ ┌ Task Sequence ───────────────────────┐
    │ name        [state]  │ │ ✓ Acquisition                        │
    │ Setup     Complete   │ │ ▶ Milling                            │
    │ Steps            50  │ │ ○ SenseAI                            │
    │ Slices  100 nm · 5µm │ │ ○ Overview (off)                     │
    │ [█████░░░] 12 / 50   │ │                                      │
    ├ Current Task ────────┤ │                                      │
    │ Milling              │ │                                      │
    │ task 2/4 · 00:12:30  │ │                                      │
    │ [Response needed →]  │ │                                      │
    └──────────────────────┘ └──────────────────────────────────────┘
    [⚙ Setup] [⏭ Next Step]                      [▶ Run Experiment]

Ticks and the triangle appear only while something is running; at rest every
task is a circle. Setup is amber until it has been done and green after; it
stays clickable, so setup can be redone.

The page owns its widgets; `SalamiUI` drives them, the same way it drives the
Protocol page. It knows how to *show* an experiment, not how to run one.
"""

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.salami.workflow_update_schema import format_duration
from fibsem.ui import stylesheets
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    DISABLED_BG_COLOR,
    DISABLED_TEXT_COLOR,
    GRAY_ICON_COLOR,
    OK_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
    WARN_COLOR,
)
from fibsem.ui.widgets.custom_widgets import TitledPanel

NO_EXPERIMENT_TEXT = "No experiment loaded"
IDLE_TEXT = "Idle"

# run_state -> pill colour. Anything unlisted is drawn muted.
_STATE_COLOURS = {
    "running": ACCENT_COLOR,
    "completed": OK_COLOR,
    "paused": WARN_COLOR,
    "error": "#d04040",
}

# Task row glyphs, for the slice in progress.
GLYPH_DONE = "✓"
GLYPH_CURRENT = "▶"
GLYPH_PENDING = "○"

_VALUE_STYLE = f"color: {TEXT_STRONG_COLOR};"
_MUTED_STYLE = f"color: {TEXT_MUTED_COLOR}; font-size: 11px;"


def _solid_button_style(
    background: str, hover: str, pressed: str, text: str = "white"
) -> str:
    """A filled button in the shape of the shared run/stop buttons."""
    return f"""
        QPushButton {{
            background-color: {background};
            color: {text};
            border: none;
            padding: 5px 14px;
            border-radius: 3px;
            font-weight: bold;
        }}
        QPushButton:hover {{ background-color: {hover}; }}
        QPushButton:pressed {{ background-color: {pressed}; }}
        QPushButton:disabled {{
            background-color: {DISABLED_BG_COLOR};
            color: {DISABLED_TEXT_COLOR};
        }}
    """


# Setup: amber until it has been done, green once it has.
SETUP_PENDING_STYLESHEET = _solid_button_style(
    WARN_COLOR, "#ebb24a", "#c88a20", text="#1e2124"
)
SETUP_DONE_STYLESHEET = _solid_button_style(OK_COLOR, "#388e3c", "#2e7d32")


def _pill_style(colour: str) -> str:
    return (
        f"color: {colour}; border: 1px solid {colour}; border-radius: 8px;"
        " padding: 1px 8px; font-size: 11px; font-weight: 600;"
    )


class SalamiWorkflowPage(QWidget):
    """Status, current task, run controls and the task sequence. Read-only."""

    # The operator asked to go and answer a prompt (on the Microscope page).
    attention_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(8)
        left_layout.addWidget(self._build_status_panel())
        left_layout.addWidget(self._build_current_task_panel())
        left_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(320)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.addWidget(self._build_sequence_panel())

        splitter = QSplitter(Qt.Horizontal)  # type: ignore
        splitter.addWidget(left_scroll)
        splitter.addWidget(right)
        splitter.setSizes([380, 900])
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, stretch=1)
        layout.addWidget(self._build_controls_bar())

        self.show_experiment(None)

    # ── construction ──────────────────────────────────────────────────────

    def _build_status_panel(self) -> TitledPanel:
        content = QWidget()
        outer = QVBoxLayout(content)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        header = QHBoxLayout()
        self.label_name = QLabel(NO_EXPERIMENT_TEXT)
        self.label_name.setStyleSheet(
            f"color: {TEXT_STRONG_COLOR}; font-size: 14px; font-weight: 600;"
        )
        self.label_state = QLabel("")
        header.addWidget(self.label_name, stretch=1)
        header.addWidget(self.label_state)
        outer.addLayout(header)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self.label_setup = QLabel("—")
        self.label_n_steps = QLabel("—")
        self.label_n_steps.setToolTip("Set on the Protocol page")
        self.label_slices = QLabel("—")
        for value in (self.label_setup, self.label_n_steps, self.label_slices):
            value.setStyleSheet(_VALUE_STYLE)
        form.addRow("Setup", self.label_setup)
        form.addRow("Number of Steps", self.label_n_steps)
        form.addRow("Slices", self.label_slices)
        outer.addLayout(form)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet(stylesheets.PROGRESS_BAR_STYLESHEET)
        self.label_progress = QLabel("")
        self.label_progress.setStyleSheet(_MUTED_STYLE)
        outer.addWidget(self.progress_bar)
        outer.addWidget(self.label_progress)

        return TitledPanel("Experiment", content=content, collapsible=False)

    def _build_current_task_panel(self) -> TitledPanel:
        content = QWidget()
        outer = QVBoxLayout(content)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(4)

        self.label_current_task = QLabel(IDLE_TEXT)
        self.label_current_task.setStyleSheet(
            f"color: {TEXT_STRONG_COLOR}; font-size: 14px; font-weight: 600;"
        )
        self.label_current_task_detail = QLabel("")
        self.label_current_task_detail.setStyleSheet(_MUTED_STYLE)
        self.label_current_task_detail.setWordWrap(True)
        outer.addWidget(self.label_current_task)
        outer.addWidget(self.label_current_task_detail)

        # Prompts are answered on the Microscope page, beside the images they
        # are about. This is the way there when the operator is watching here.
        self.btn_attention = QPushButton("Response needed — go to Microscope")
        self.btn_attention.setStyleSheet(stylesheets.USER_ATTENTION_BUTTON_STYLESHEET)
        self.btn_attention.clicked.connect(self.attention_requested)
        self.btn_attention.setVisible(False)
        outer.addWidget(self.btn_attention)

        return TitledPanel("Current Task", content=content, collapsible=False)

    def _build_controls_bar(self) -> QWidget:
        """Setup and Next Step bottom left, Run bottom right, as AutoLamella's."""
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(6)

        self.btn_setup = QPushButton("Setup Experiment")
        self.btn_setup.setIcon(fibsem_icon("mdi:cog", color="#1e2124"))
        self.btn_run_next_step = QPushButton("Run Next Step")
        self.btn_run_next_step.setIcon(
            fibsem_icon("mdi:skip-next", color=GRAY_ICON_COLOR)
        )
        self.btn_run_next_step.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.btn_run_next_step.setToolTip("Run one slice: every task once, in order")
        self.btn_run = QPushButton("Run Experiment")
        self.btn_run.setIcon(fibsem_icon("mdi:play-circle", color="white"))
        self.btn_run.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.btn_run.setToolTip("Run every remaining slice")
        self.btn_stop = QPushButton("Stop Experiment")
        self.btn_stop.setIcon(fibsem_icon("mdi:stop-circle", color="white"))
        self.btn_stop.setStyleSheet(stylesheets.STOP_WORKFLOW_BUTTON_STYLESHEET)
        self.btn_stop.setToolTip("Stops after the task in progress finishes")
        self.btn_stop.setVisible(False)
        for button in (
            self.btn_setup,
            self.btn_run_next_step,
            self.btn_run,
            self.btn_stop,
        ):
            button.setMinimumHeight(32)

        row.addWidget(self.btn_setup)
        row.addWidget(self.btn_run_next_step)
        row.addStretch()
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_stop)

        self.set_setup_complete(False)
        return bar

    def set_setup_complete(self, complete: bool) -> None:
        """Amber until setup has been done, green after. Clickable either way."""
        self.btn_setup.setStyleSheet(
            SETUP_DONE_STYLESHEET if complete else SETUP_PENDING_STYLESHEET
        )
        self.btn_setup.setIcon(
            fibsem_icon(
                "mdi:check-circle" if complete else "mdi:cog",
                color=("white" if complete else "#1e2124"),
            )
        )
        self.btn_setup.setToolTip(
            "Setup is complete. Click to run it again."
            if complete
            else "Mill the trench and fiducial, and take the reference image"
        )

    def _build_sequence_panel(self) -> TitledPanel:
        self.task_list = QListWidget()
        self.task_list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        # Read-only: the order and the tasks are edited on the Protocol page.
        self.task_list.setSelectionMode(QListWidget.NoSelection)
        self.task_list.setFocusPolicy(Qt.NoFocus)  # type: ignore
        return TitledPanel("Task Sequence", content=self.task_list, collapsible=False)

    # ── display ───────────────────────────────────────────────────────────

    def show_experiment(self, experiment, running: bool = False) -> None:
        """Show what `experiment` says about itself at rest. None clears the page."""
        if experiment is None:
            self.label_name.setText(NO_EXPERIMENT_TEXT)
            self._set_state("")
            for value in (self.label_setup, self.label_n_steps, self.label_slices):
                value.setText("—")
            self.progress_bar.setValue(0)
            self.label_progress.setText("")
            self.task_list.clear()
            self.set_setup_complete(False)
            self.show_idle()
            return

        config, status = experiment.config, experiment.status
        self.label_name.setText(config.name or "")
        self._set_state(status.run_state or "created")
        self.label_setup.setText("Complete" if status.setup_complete else "Not done")
        self.set_setup_complete(status.setup_complete)
        self.label_setup.setStyleSheet(
            f"color: {OK_COLOR if status.setup_complete else WARN_COLOR};"
        )
        self.label_n_steps.setText(str(config.n_steps))
        self.label_slices.setText(
            f"{config.advance.step_size * 1e9:.1f} nm · "
            f"{config.volume_depth * 1e6:.2f} µm through the block"
        )
        self.set_progress(status.current_step, config.n_steps)
        self.show_tasks(
            config.tasks,
            step=status.current_step,
            next_index=status.next_task_index,
            skip_once=status.skip_once,
            running=running,
        )
        if not running:
            self.show_idle(status.in_flight_task)

    def set_progress(self, current: int, total: int, percent: Optional[int] = None):
        if percent is None:
            percent = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        self.label_progress.setText(f"Slice {current} / {total}" if total else "")

    def show_tasks(
        self,
        tasks,
        step: int,
        next_index: int,
        skip_once=(),
        running: bool = False,
    ) -> None:
        """The protocol's tasks in run order, marked for the slice in progress."""
        self.task_list.clear()
        for index, task in enumerate(tasks):
            notes = []
            glyph, colour = GLYPH_PENDING, TEXT_STRONG_COLOR
            if running and index < next_index:
                glyph, colour = GLYPH_DONE, OK_COLOR
            elif running and index == next_index:
                glyph, colour = GLYPH_CURRENT, ACCENT_COLOR
            if not task.enabled:
                colour = TEXT_MUTED_COLOR
                notes.append("off")
            if task.enabled and not task.should_run(step):
                notes.append("not this slice")
                colour = TEXT_MUTED_COLOR
            if task.name in skip_once:
                notes.append("skip once")
            suffix = f"  ({', '.join(notes)})" if notes else ""
            item = QListWidgetItem(f"{glyph}  {task.name}{suffix}")
            item.setForeground(_qcolor(colour))
            item.setToolTip(f"{task.DISPLAY_NAME} task")
            self.task_list.addItem(item)

    def show_idle(self, interrupted_task: Optional[str] = None) -> None:
        if interrupted_task:
            self.label_current_task.setText(interrupted_task)
            self.label_current_task_detail.setText(
                "Interrupted — you will be asked how to resolve it before running."
            )
        else:
            self.label_current_task.setText(IDLE_TEXT)
            self.label_current_task_detail.setText("")
        self.btn_attention.setVisible(False)

    def show_update(self, update) -> None:
        """Reflect a live `SalamiWorkflowUpdate` from a setup or run."""
        if update.total_steps > 0:
            self.set_progress(
                update.current_step,
                update.total_steps,
                percent=int(update.overall_progress),
            )
            self._set_state("running")
            name = update.task_name or update.task
            if name:
                self.label_current_task.setText(name)
            parts = []
            if update.total_tasks > 0:
                parts.append(
                    f"task {update.current_task_index + 1}/{update.total_tasks}"
                )
            parts.append(f"slice {update.current_step}/{update.total_steps}")
            if update.elapsed_time > 0:
                parts.append(format_duration(update.elapsed_time))
            self.label_current_task_detail.setText(" · ".join(parts))
        # A prompt is waiting while `pos` is set.
        self.btn_attention.setVisible(update.pos is not None)

    def set_running(self, running: bool) -> None:
        """Swap Run for Stop; everything else is gated by the caller."""
        self.btn_run.setVisible(not running)
        self.btn_stop.setVisible(running)
        self.btn_stop.setEnabled(running)

    def _set_state(self, state: str) -> None:
        self.label_state.setText(state)
        self.label_state.setVisible(bool(state))
        self.label_state.setStyleSheet(
            _pill_style(_STATE_COLOURS.get(state, TEXT_MUTED_COLOR))
        )


def _qcolor(colour: str):
    from PyQt5.QtGui import QColor

    return QColor(colour)
