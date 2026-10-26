"""
Workflow Update Schema for Salami Experiments.

The single payload type carried by `SalamiUI.workflow_update_signal`. It covers
both directions of the worker <-> UI conversation:

- progress reporting (step counters, timings, results)
- UI actions the worker needs performed on its behalf (show these buttons,
  display this image, load this milling config)

Both used to travel on the same signal as two incompatible shapes — a dataclass
from the tasks and plain dicts from the interaction helpers — which meant the
slot crashed on whichever one it wasn't written for.
"""

from dataclasses import dataclass, fields
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from fibsem.milling.tasks import FibsemMillingTaskConfig
    from fibsem.structures import FibsemImage

# Fields carrying live objects or one-shot UI intent. Excluded from to_dict()
# so serialisation stays cheap and YAML-safe — asdict() would otherwise try to
# deep-copy whole images.
TRANSIENT_FIELDS = frozenset(
    {
        "pos",
        "neg",
        "milling_enabled",
        "sem_image",
        "fib_image",
        "milling_config",
        "clear_milling_config",
    }
)


def format_duration(seconds: float) -> str:
    """Seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class SalamiWorkflowUpdate:
    # Primary identification. Defaulted so UI-only updates stay terse.
    task: str = "ui"  # e.g. "milling", "acquisition", "alignment", "advance", "setup", "complete"
    msg: str = ""

    # Outer loop progress (volumetric steps)
    current_step: int = 0
    total_steps: int = 0

    # Progress within the current step (task list position)
    current_task_index: int = 0
    total_tasks: int = 0
    task_name: str = ""

    # Status and stage
    status: Literal["running", "paused", "completed", "error", "idle"] = "running"
    stage: Literal[
        "setup",
        "milling",
        "acquisition",
        "alignment",
        "advance",
        "saving",
        "skipped",
        "complete",
    ] = "setup"

    # Timing
    timestamp: Optional[datetime] = None
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0

    # Error
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Acquisition payload
    image_filename: Optional[str] = None
    image_shape: Optional[tuple[int, int]] = None
    image_metadata: Optional[dict] = None

    # Milling payload
    milling_time: Optional[float] = None
    milling_depth: Optional[float] = None
    estimated_milling_time: Optional[float] = None
    remaining_milling_time: Optional[float] = None

    # Arbitrary extras
    details: Optional[dict] = None

    # ---- UI actions -------------------------------------------------------
    # Heading shown above the instruction text ("Setup", "Mill Trenches", ...).
    workflow_msg: Optional[str] = None
    # Button labels for a user decision. `pos` set means the worker is blocked
    # waiting on a click; `neg` None shows only the positive button.
    pos: Optional[str] = None
    neg: Optional[str] = None
    milling_enabled: Optional[bool] = None
    # Live objects for the UI to display. Never serialised.
    sem_image: Optional["FibsemImage"] = None
    fib_image: Optional["FibsemImage"] = None
    milling_config: Optional["FibsemMillingTaskConfig"] = None
    clear_milling_config: bool = False

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @property
    def requests_user_input(self) -> bool:
        return self.pos is not None

    @property
    def step_progress(self) -> float:
        if self.total_steps > 0:
            return (self.current_step / self.total_steps) * 100.0
        return 0.0

    @property
    def task_progress(self) -> float:
        if self.total_tasks > 0:
            return (self.current_task_index / self.total_tasks) * 100.0
        return 0.0

    @property
    def overall_progress(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        step_fraction = self.current_step / self.total_steps
        if self.total_tasks > 0:
            task_fraction = (
                self.current_task_index / self.total_tasks
            ) / self.total_steps
            return (step_fraction + task_fraction) * 100.0
        return step_fraction * 100.0

    def to_dict(self) -> dict:
        """Serialisable view: progress fields only, no live objects."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in TRANSIENT_FIELDS
        }

    def format(self) -> str:
        lines = [self.msg]
        if self.total_steps > 0:
            lines.append(f"Step: {self.current_step + 1}/{self.total_steps}")
        if self.total_tasks > 0:
            task_str = f"Task: {self.current_task_index + 1}/{self.total_tasks}"
            if self.task_name:
                task_str += f" ({self.task_name})"
            lines.append(task_str)
        if self.overall_progress > 0:
            lines.append(f"Overall: {self.overall_progress:.1f}%")
        if self.elapsed_time > 0:
            lines.append(f"Elapsed: {format_duration(self.elapsed_time)}")
        if self.estimated_remaining > 0:
            h = int(self.estimated_remaining // 3600)
            m = int((self.estimated_remaining % 3600) // 60)
            lines.append(f"Remaining: ~{h:02d}:{m:02d}")
        if self.status == "error" and self.error:
            lines.append(f"ERROR: {self.error}")
        return "\n".join(lines)


def format_workflow_update(update: SalamiWorkflowUpdate) -> str:
    return update.format()
