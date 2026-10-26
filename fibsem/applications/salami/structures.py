import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from fibsem.structures import FibsemRectangle

if TYPE_CHECKING:
    from fibsem.applications.salami.workflow_update_schema import SalamiWorkflowUpdate


@dataclass
class RunSchedule:
    """Controls when a task (or sub-function) fires during the experiment loop.

    A task with the default schedule runs on every step. `every_n_steps=10` +
    `start_step=0` fires on steps 0, 10, 20, ... `end_step` is exclusive.
    """

    every_n_steps: int = 1
    start_step: int = 0
    end_step: int = -1  # -1 means "no upper bound"

    def should_run(self, step: int) -> bool:
        if self.every_n_steps <= 0:
            return False
        if step < self.start_step:
            return False
        if self.end_step >= 0 and step >= self.end_step:
            return False
        return (step - self.start_step) % self.every_n_steps == 0

    def to_dict(self) -> dict:
        return {
            "every_n_steps": self.every_n_steps,
            "start_step": self.start_step,
            "end_step": self.end_step,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunSchedule":
        return cls(
            every_n_steps=data.get("every_n_steps", 1),
            start_step=data.get("start_step", 0),
            end_step=data.get("end_step", -1),
        )


def centred_area(fraction: float) -> FibsemRectangle:
    """A box covering `fraction` of the frame in both axes, centred on it."""
    offset = (1.0 - fraction) / 2.0
    return FibsemRectangle(left=offset, top=offset, width=fraction, height=fraction)


@dataclass
class AutoFunctionConfig:
    """Wraps an auto-function (auto-focus, autocontrast) with its own cadence
    and its own region of the frame.

    Each function gets its own `area_of_interest` rather than sharing one: focus
    wants a feature with detail in it, while contrast wants a region
    representative of the whole face, and those are rarely the same box. The
    default is the full frame, which is what the functions did before the field
    existed.
    """

    enabled: bool = True
    schedule: RunSchedule = field(default_factory=RunSchedule)
    # Normalised to the frame (left/top/width/height in 0-1), so it survives a
    # change of resolution or HFW.
    area_of_interest: FibsemRectangle = field(default_factory=FibsemRectangle)

    def should_run(self, step: int) -> bool:
        return self.enabled and self.schedule.should_run(step)

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "schedule": self.schedule.to_dict(),
            "area_of_interest": self.area_of_interest.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, data: dict, default_area: Optional[FibsemRectangle] = None
    ) -> "AutoFunctionConfig":
        """`default_area` is what a protocol written before the field gets.

        Passed in by the caller because the sensible default differs per
        function — focus and contrast want different-sized boxes — and this
        class is used for both.
        """
        # .get, not []: protocols predating this field must still load.
        area = data.get("area_of_interest")
        if area:
            area_of_interest = FibsemRectangle(**area)
        else:
            area_of_interest = default_area or FibsemRectangle()
        return cls(
            enabled=data.get("enabled", True),
            schedule=RunSchedule.from_dict(data.get("schedule", {})),
            area_of_interest=area_of_interest,
        )


@dataclass
class RunContext:
    """Ambient state passed into every task's `run()`.

    Decouples a task from the experiment class: a task gets everything it
    needs (path, timing, step info, UI hooks) without importing `SalamiExperiment`.
    """

    experiment_path: str
    current_step: int = 0
    total_steps: int = 0
    task_index: int = 0
    total_tasks: int = 0
    start_time: float = 0.0
    parent_ui: Optional[Any] = None
    # The SenseAI scan generator, when one is in use. Handed in rather than
    # created by the task, because the DLL holds a single hardware connection:
    # building a second one does not fail cleanly, it leaves the generator
    # waiting for a frame that never arrives. One connection is set up once —
    # by the SenseAI tab's Initialise button — and every slice reuses it.
    senseai_module: Optional[Any] = None

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def emit(self, update: "SalamiWorkflowUpdate") -> None:
        if self.parent_ui is not None:
            self.parent_ui.workflow_update_signal.emit(update)
        logging.info(f"Salami update: {update.task} - {update.msg}")

    def stop_requested(self) -> bool:
        if self.parent_ui is None:
            return False
        return self.parent_ui._workflow_stop_event.is_set()


# How far the volume advances per step, and therefore how thick a slab each
# slice removes. One constant because they are one quantity: the milling
# pattern's height is the slice thickness. A pattern shorter than the advance
# leaves unmilled material between slices; taller, and it re-cuts what the
# previous step already removed.
DEFAULT_SLICE_THICKNESS = 100e-9  # 100 nm


@dataclass
class VolumeAdvanceConfig:
    """Describes the volumetric advance applied between every outer step.

    Lives at the experiment level, NOT on any single task — multiple milling
    or acquisition tasks in the same experiment share one advance.
    """

    step_size: float = DEFAULT_SLICE_THICKNESS
    advance_milling_pattern: bool = True  # shifts first stage pattern.point.y
    advance_beam_shift: bool = True  # shifts electron_beam.shift.y for acqs

    def to_dict(self) -> dict:
        return {
            "step_size": self.step_size,
            "advance_milling_pattern": self.advance_milling_pattern,
            "advance_beam_shift": self.advance_beam_shift,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VolumeAdvanceConfig":
        return cls(
            step_size=data.get("step_size", 100e-9),
            advance_milling_pattern=data.get("advance_milling_pattern", True),
            advance_beam_shift=data.get("advance_beam_shift", True),
        )
