import logging
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

import yaml

from fibsem.acquire import acquire_channels
from fibsem.applications.salami._default_settings import DEFAULT_TRENCH
from fibsem.applications.salami.structures import (
    RunContext,
    VolumeAdvanceConfig,
)
from fibsem.applications.salami.tasks import (
    TaskConfig,
    ensure_task_directories,
    prepare_task_for_reuse,
    run_task,
    runtime_task_config,
    task_from_dict,
)
from fibsem.applications.salami.workflow_ui import (
    _check_for_abort,
    ask_user,
    set_images_ui,
)
from fibsem.applications.salami.workflow_update_schema import SalamiWorkflowUpdate
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import MicroscopeState

if TYPE_CHECKING:
    from fibsem.applications.salami.ui.SalamiUI import SalamiUI

# An experiment is a directory containing exactly these two files.
CONFIG_FILENAME = "experiment_config.yaml"
STATUS_FILENAME = "Experiment_Status.yaml"
DEFAULT_STATUS_NAME = "Experiment_Status"

ALIGNMENT_IMAGE_NAME = "alignment_ref_image.tif"

# The protocol new experiments start from when no template is supplied.
DEFAULT_TEMPLATE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "configs", CONFIG_FILENAME)
)

# Lifecycle of a single run. Distinct from `setup_complete`, which records
# whether the one-off preparation (trench, fiducial) has been done.
RUN_STATES = ("created", "setup", "running", "paused", "completed", "failed")


def default_experiment_name() -> str:
    return f"Salami-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"


@dataclass
class SalamiExperimentConfig:
    """The protocol: what to do, independent of sample or machine.

    Everything here should survive being reused on a different sample. Run
    state lives in `SalamiExperimentStatus`; output paths are derived at run
    time and never persisted into the task configs.
    """

    name: str
    description: str = ""
    path: str = ""
    n_steps: int = 50
    advance: VolumeAdvanceConfig = field(default_factory=VolumeAdvanceConfig)
    tasks: List[TaskConfig] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "n_steps": self.n_steps,
            "advance": self.advance.to_dict(),
            "tasks": [t.to_dict() for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SalamiExperimentConfig":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            path=data.get("path", "") or "",
            n_steps=data.get("n_steps", 50),
            advance=VolumeAdvanceConfig.from_dict(data.get("advance", {})),
            tasks=[task_from_dict(t) for t in data.get("tasks", [])],
        )

    @property
    def volume_depth(self) -> float:
        """Total depth sliced through, in metres: n_steps x slice thickness."""
        return self.n_steps * self.advance.step_size

    def for_ui_display_MAIN(self) -> dict:
        """Returns a dict of display-friendly values for the main UI."""
        return {
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "num_tasks": len(self.tasks),
            "advance_milling_pattern": self.advance.advance_milling_pattern,
            "advance_beam_shift": self.advance.advance_beam_shift,
        }

    def save(self, path: Optional[str] = None) -> str:
        target_dir = path or self.path
        if not target_dir:
            raise ValueError(
                "Cannot save experiment config: no path set. "
                "Create the experiment through SalamiExperiment.create()."
            )
        os.makedirs(target_dir, exist_ok=True)
        filename = os.path.join(target_dir, CONFIG_FILENAME)
        with open(filename, "w") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)
        return filename

    def export(self, filename: str) -> str:
        """Write a reusable protocol to `filename`, stripped of sample state."""
        template = self.as_template(name=self.name)
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        with open(filename, "w") as f:
            yaml.dump(template.to_dict(), f, sort_keys=False)
        return filename

    def add_task(self, task: TaskConfig) -> None:
        self.tasks.append(task)

    def remove_task(self, task_name: str) -> bool:
        for i, task in enumerate(self.tasks):
            if task.name == task_name:
                del self.tasks[i]
                return True
        return False

    @classmethod
    def load(cls, filename: str) -> "SalamiExperimentConfig":
        with open(filename, "r") as f:
            data = yaml.safe_load(f) or {}
        config = cls.from_dict(data)
        # The stored `path` is provenance only. An experiment lives wherever
        # its file actually is, so a moved or copied folder still loads, and a
        # config written on another machine never points us at their disk.
        config.path = os.path.dirname(os.path.abspath(filename))
        return config

    def as_template(self, name: str, path: str = "") -> "SalamiExperimentConfig":
        """Copy this protocol for reuse on a new sample.

        Drops sample-specific state (stage positions) and machine-specific
        state (output paths) so the result is safe to run anywhere.
        """
        template = deepcopy(self)
        template.name = name
        template.path = path
        for task in template.tasks:
            prepare_task_for_reuse(task)
        return template


@dataclass
class SalamiExperimentStatus:
    """Where a run has got to. Written after every task so a crash is recoverable.

    `current_step` + `next_task_index` locate us precisely within the run:
    resuming re-enters the step at the task that hadn't run yet, rather than
    restarting the step and re-milling material that is already gone.
    """

    current_step: int = 0
    next_task_index: int = 0
    setup_complete: bool = False
    run_state: str = "created"
    # Name of a task that started but never reported completion. Set before the
    # task runs and cleared after, so an interrupted destructive operation is
    # visible on the next open.
    in_flight_task: Optional[str] = None
    # Tasks to skip the *next* time they would run, then resume as normal.
    # Run state rather than protocol: a one-shot is a decision about this run,
    # but it still has to survive a crash, or the operator's choice is lost.
    skip_once: List[str] = field(default_factory=list)
    started_at: Optional[float] = None
    last_updated: Optional[float] = None
    name: Optional[str] = None
    # The eucentric / sample position captured during setup. Sample-specific,
    # which is why it lives here and not in the protocol.
    sample_state: Optional[MicroscopeState] = None

    def __post_init__(self):
        if self.name is None:
            self.name = DEFAULT_STATUS_NAME

    def to_dict(self) -> dict:
        return {
            "current_step": self.current_step,
            "next_task_index": self.next_task_index,
            "setup_complete": self.setup_complete,
            "run_state": self.run_state,
            "in_flight_task": self.in_flight_task,
            "skip_once": list(self.skip_once),
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "name": self.name,
            "sample_state": self.sample_state.to_dict() if self.sample_state else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SalamiExperimentStatus":
        # `state` was the pre-rename key for `sample_state`; accept both so
        # status files written by earlier versions still load.
        sample_state = data.get("sample_state", data.get("state"))
        return cls(
            current_step=data.get("current_step", 0),
            next_task_index=data.get("next_task_index", 0),
            setup_complete=data.get("setup_complete", False),
            run_state=data.get("run_state", "created"),
            in_flight_task=data.get("in_flight_task", None),
            skip_once=list(data.get("skip_once") or []),
            started_at=data.get("started_at", None),
            last_updated=data.get("last_updated", None),
            name=data.get("name", None),
            sample_state=MicroscopeState.from_dict(sample_state)
            if sample_state
            else None,
        )

    def save(self, path: str) -> str:
        os.makedirs(path, exist_ok=True)
        self.last_updated = time.time()
        filename = os.path.join(path, STATUS_FILENAME)
        with open(filename, "w") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)
        return filename

    @classmethod
    def load(cls, filename: str) -> "SalamiExperimentStatus":
        """Load a status YAML by full path."""
        with open(filename, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def for_ui_display_STATUS(self) -> dict:
        return {
            "name": self.name,
            "current_step": self.current_step,
            "setup_complete": self.setup_complete,
            "run_state": self.run_state,
        }


class SalamiExperiment:
    """Runs an ordered list of tasks across `n_steps`.

    Task dispatch goes through the registry in `tasks.py` — the experiment
    class never type-checks individual task configs. Adding a new task type
    only requires defining a (Config, Task) pair and `@register_task`.

    Constructing an experiment does not touch the disk. Use `create()` or
    `open()`, which decide the target directory first; `initialise()` is what
    actually writes.
    """

    def __init__(
        self,
        config: SalamiExperimentConfig,
        microscope: Optional[FibsemMicroscope] = None,
        parent_ui: Optional["SalamiUI"] = None,
        status: Optional[SalamiExperimentStatus] = None,
    ):
        self.microscope = microscope
        self.config = config
        self.parent_ui = parent_ui
        self.status = status if status is not None else SalamiExperimentStatus()
        self._start_time = 0.0
        self.setup_path: Optional[str] = None
        # Trench config as adjusted by the operator during setup. Instance-only
        # for now; TODO: promote to the protocol so it survives a reload.
        self.trench_config: Optional[FibsemMillingTaskConfig] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def path(self) -> str:
        return self.config.path

    @property
    def has_interrupted_task(self) -> bool:
        """True if a task started but never reported completion."""
        return self.status.in_flight_task is not None

    @staticmethod
    def default_template() -> SalamiExperimentConfig:
        """The protocol new experiments start from."""
        return SalamiExperimentConfig.load(DEFAULT_TEMPLATE_PATH)

    def initialise(self) -> None:
        """Create the directory tree and write config + status."""
        os.makedirs(self.config.path, exist_ok=True)
        for task_cfg in self.config.tasks:
            ensure_task_directories(task_cfg, self.config.path)
        self.config.save()
        self.status.save(self.config.path)

    @classmethod
    def create(
        cls,
        parent_dir: str,
        name: Optional[str] = None,
        template: Optional[SalamiExperimentConfig] = None,
        microscope: Optional[FibsemMicroscope] = None,
        parent_ui: Optional["SalamiUI"] = None,
    ) -> "SalamiExperiment":
        """Start a new experiment in `parent_dir/name`.

        `template` is an existing protocol to reuse (scenario: same experiment,
        different sample). It is copied and re-homed — the template's own
        directory is never written to.
        """
        name = name or default_experiment_name()
        path = os.path.join(parent_dir, name)

        if os.path.exists(os.path.join(path, CONFIG_FILENAME)):
            raise FileExistsError(
                f"An experiment already exists at {path}. "
                "Choose a different name or open the existing experiment."
            )

        if template is None:
            template = cls.default_template()

        config = template.as_template(name=name, path=path)
        experiment = cls(
            config=config,
            microscope=microscope,
            parent_ui=parent_ui,
            status=SalamiExperimentStatus(run_state="created"),
        )
        experiment.initialise()
        logging.info(f"Created Salami experiment {name!r} at {path}")
        return experiment

    @classmethod
    def open(
        cls,
        path: str,
        microscope: Optional[FibsemMicroscope] = None,
        parent_ui: Optional["SalamiUI"] = None,
    ) -> "SalamiExperiment":
        """Open an existing experiment for resuming.

        `path` may be the experiment directory or any file inside it. Opening
        reads only: the stored protocol and status are left untouched until
        the run actually continues.
        """
        directory = (
            path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
        )
        config_path = os.path.join(directory, CONFIG_FILENAME)
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"No {CONFIG_FILENAME} found in {directory}. "
                "Select the experiment directory itself."
            )

        config = SalamiExperimentConfig.load(config_path)

        status_path = os.path.join(directory, STATUS_FILENAME)
        if os.path.exists(status_path):
            status = SalamiExperimentStatus.load(status_path)
        else:
            logging.warning(
                f"No {STATUS_FILENAME} in {directory}; starting from step 0."
            )
            status = SalamiExperimentStatus()

        experiment = cls(
            config=config, microscope=microscope, parent_ui=parent_ui, status=status
        )
        # Recreate any missing output directories, without modifying the protocol.
        for task_cfg in config.tasks:
            ensure_task_directories(task_cfg, config.path)

        logging.info(
            f"Opened Salami experiment {config.name!r} at {config.path} "
            f"(step {status.current_step}/{config.n_steps}, state={status.run_state})"
        )
        return experiment

    def resolve_interrupted_task(self, action: str) -> None:
        """Clear an interrupted task before the run can continue.

        `action` is 'retry' (run it again) or 'skip' (treat it as done). There
        is no safe automatic choice: an interrupted mill has removed an unknown
        amount of material, so the operator has to decide.
        """
        if self.status.in_flight_task is None:
            return
        if action == "skip":
            self.status.next_task_index = min(
                self.status.next_task_index + 1, len(self.config.tasks)
            )
        elif action != "retry":
            raise ValueError(f"Unknown action {action!r}; expected 'retry' or 'skip'.")
        logging.info(
            f"Interrupted task {self.status.in_flight_task!r} resolved with {action!r}."
        )
        self.status.in_flight_task = None
        self.status.save(self.config.path)

    # ------------------------------------------------------------------
    # Task control (safe to call mid-run)
    # ------------------------------------------------------------------
    #
    # Both of these are safe while a run is in progress: neither changes the
    # length of `config.tasks`, so `status.next_task_index` stays valid. That
    # is the whole reason to disable rather than remove a task mid-run.

    def find_task(self, task_name: str) -> Optional[TaskConfig]:
        for task in self.config.tasks:
            if task.name == task_name:
                return task
        return None

    def _require_task(self, task_name: str) -> TaskConfig:
        task = self.find_task(task_name)
        if task is None:
            known = [t.name for t in self.config.tasks]
            raise KeyError(f"No task named {task_name!r}. Known tasks: {known}")
        return task

    def set_task_enabled(self, task_name: str, enabled: bool) -> None:
        """Turn a task on or off for the rest of the run.

        Persisted in the protocol, so it survives a reload and resume. Reset
        by `as_template()` so it doesn't follow onto the next sample.
        """
        task = self._require_task(task_name)
        task.enabled = enabled
        self.config.save()
        logging.info(f"Task {task_name!r} {'enabled' if enabled else 'disabled'}.")

    def skip_task_once(self, task_name: str) -> None:
        """Skip `task_name` the next time it would run, then resume as normal."""
        self._require_task(task_name)
        if task_name not in self.status.skip_once:
            self.status.skip_once.append(task_name)
            self.status.save(self.config.path)
            logging.info(f"Task {task_name!r} will be skipped once.")

    def cancel_skip_once(self, task_name: str) -> None:
        """Undo a pending one-shot skip."""
        if task_name in self.status.skip_once:
            self.status.skip_once.remove(task_name)
            self.status.save(self.config.path)

    def task_states(self) -> List[dict]:
        """Per-task enabled / pending-skip summary, for the UI."""
        return [
            {
                "name": task.name,
                "type": task.TASK_TYPE,
                "enabled": task.enabled,
                "skip_next": task.name in self.status.skip_once,
                "runs_this_step": task.should_run(self.status.current_step),
            }
            for task in self.config.tasks
        ]

    def for_ui_display_STATUS(self) -> dict:
        """Merged config + status fields for the status panel."""
        return {
            **self.status.for_ui_display_STATUS(),
            "n_steps": self.config.n_steps,
            "advance_step_size_nm": self.config.advance.step_size * 1e9,
            "volume_depth_um": self.config.volume_depth * 1e6,
        }

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def _make_context(self, task_index: int = 0) -> RunContext:
        return RunContext(
            experiment_path=self.config.path,
            current_step=self.status.current_step,
            total_steps=self.config.n_steps,
            task_index=task_index,
            total_tasks=len(self.config.tasks),
            start_time=self._start_time,
            parent_ui=self.parent_ui,
            senseai_module=self.senseai_module,
        )

    @property
    def senseai_module(self):
        """The SenseAI scan generator the UI initialised, or None.

        Read from the widget on every context rather than captured once, because
        the operator can initialise the scan generator after the experiment is
        open — and because it must stay the *same* object for the whole run. The
        DLL holds one hardware connection; a second one does not fail cleanly,
        it leaves the generator waiting for a frame that never arrives.
        """
        widget = getattr(self.parent_ui, "SenseAI_widget", None)
        return getattr(widget, "senseAI", None)

    @property
    def is_complete(self) -> bool:
        return self.status.current_step >= self.config.n_steps

    def _begin_run(self) -> None:
        """Checks and bookkeeping shared by `run` and `run_next_step`.

        Both go through here so stepping by hand and letting the loop run can be
        interleaved freely without the two disagreeing about where the run is.
        """
        if self.microscope is None:
            raise RuntimeError("Cannot run experiment: no microscope connected.")
        if self.has_interrupted_task:
            raise RuntimeError(
                f"Task {self.status.in_flight_task!r} was interrupted at slice "
                f"{self.status.current_step}. Resolve it before running."
            )

        self._start_time = time.time()
        if self.status.started_at is None:
            self.status.started_at = self._start_time
        self.status.run_state = "running"
        self.status.save(self.config.path)

    def _complete(self) -> None:
        self.status.run_state = "completed"
        self.status.save(self.config.path)

        ctx = self._make_context()
        ctx.emit(
            SalamiWorkflowUpdate(
                task="complete",
                msg=f"Salami experiment completed: {self.config.n_steps} steps",
                current_step=self.config.n_steps,
                total_steps=self.config.n_steps,
                total_tasks=len(self.config.tasks),
                status="completed",
                stage="complete",
                elapsed_time=ctx.elapsed(),
            )
        )

    def run(self) -> None:
        """Run every remaining step, or until stopped."""
        self._begin_run()

        ctx = self._make_context()
        ctx.emit(
            SalamiWorkflowUpdate(
                task="setup",
                msg=f"Starting Salami experiment: {self.config.name}",
                current_step=self.status.current_step,
                total_steps=self.config.n_steps,
                total_tasks=len(self.config.tasks),
                stage="setup",
            )
        )

        try:
            while not self.is_complete:
                if not self._execute_step():
                    self._pause()
                    return
        except Exception:
            self.status.run_state = "failed"
            self.status.save(self.config.path)
            raise

        self._complete()

    def run_next_step(self) -> bool:
        """Run a single slice and stop. Returns False if nothing ran.

        The unit of work is the same one `run` uses, so a run can be advanced by
        hand and then handed to the continuous loop, or the reverse.
        """
        if self.is_complete:
            logging.info(
                f"Experiment already complete: {self.status.current_step}"
                f"/{self.config.n_steps} steps."
            )
            return False

        self._begin_run()
        try:
            advanced = self._execute_step()
        except Exception:
            self.status.run_state = "failed"
            self.status.save(self.config.path)
            raise

        if not advanced:
            self._pause()
            return False

        if self.is_complete:
            self._complete()
        else:
            self.status.run_state = "paused"
            self.status.save(self.config.path)
        return True

    def _execute_step(self) -> bool:
        """Run one slice. Returns False if the run was stopped part-way."""
        step = self.status.current_step
        logging.info("-" * 80)
        logging.info(f"Step {step + 1}/{self.config.n_steps}")

        if self._make_context().stop_requested():
            logging.info("Stop requested. Exiting Salami experiment.")
            return False

        if not self._run_step(step):
            return False

        self._emit_advance(step)

        self.status.current_step += 1
        self.status.next_task_index = 0
        self.status.save(self.config.path)
        return True

    def _run_step(self, step: int) -> bool:
        """Run the scheduled tasks for `step`. Returns False if stopped.

        Starts at `next_task_index` rather than 0, so resuming mid-step picks
        up where it left off instead of repeating completed tasks.
        """
        tasks = self.config.tasks
        for i in range(self.status.next_task_index, len(tasks)):
            task_cfg = tasks[i]
            ctx = self._make_context(task_index=i)
            if ctx.stop_requested():
                return False

            if not task_cfg.should_run(step):
                reason = "disabled" if not task_cfg.enabled else "schedule"
                logging.debug(
                    f"Skipping task {task_cfg.name!r} at step {step} ({reason})."
                )
                self._mark_task_complete(i)
                continue

            # Checked after the schedule so "skip the next one" means the next
            # time it would actually have run, not the next step in the loop.
            if task_cfg.name in self.status.skip_once:
                self._consume_skip_once(task_cfg, ctx, step, i)
                continue

            # Derived, not accumulated: the stored protocol stays at step 0 and
            # the runtime copy carries this step's geometry and output paths.
            cfg = runtime_task_config(
                task_cfg, self.config.path, step, self.config.advance
            )

            # Record the task as in-flight *before* running it. Milling removes
            # material and cannot be safely repeated, so a crash here has to be
            # visible on resume rather than silently re-run.
            self.status.in_flight_task = task_cfg.name
            self.status.save(self.config.path)

            run_task(self.microscope, cfg, ctx)

            self.status.in_flight_task = None
            self._mark_task_complete(i)

        return True

    def _consume_skip_once(
        self, task_cfg: TaskConfig, ctx: RunContext, step: int, index: int
    ) -> None:
        """Honour a one-shot skip and clear it, so the task runs again next time."""
        self.status.skip_once.remove(task_cfg.name)
        logging.info(f"Skipping task {task_cfg.name!r} once at step {step}.")
        ctx.emit(
            SalamiWorkflowUpdate(
                task="skipped",
                msg=f"Skipped {task_cfg.name!r} at step {step + 1} (one-shot skip)",
                current_step=step,
                total_steps=self.config.n_steps,
                current_task_index=index,
                total_tasks=len(self.config.tasks),
                task_name=task_cfg.name,
                stage="skipped",
            )
        )
        # _mark_task_complete saves, which also commits the consumed skip —
        # so a crash right here doesn't skip the task a second time.
        self._mark_task_complete(index)

    def _mark_task_complete(self, index: int) -> None:
        self.status.next_task_index = index + 1
        self.status.save(self.config.path)

    def _emit_advance(self, step: int) -> None:
        """Report the volumetric advance. The geometry itself is derived per
        step in `_run_step`, so there is nothing to mutate here."""
        delta = self.config.advance.step_size
        if delta == 0.0:
            return
        ctx = self._make_context()
        ctx.emit(
            SalamiWorkflowUpdate(
                task="advance",
                msg=f"Advancing volume by {delta * 1e9:.1f} nm",
                workflow_msg="Advance",
                current_step=step,
                total_steps=self.config.n_steps,
                total_tasks=len(self.config.tasks),
                stage="advance",
                elapsed_time=ctx.elapsed(),
            )
        )

    def _pause(self) -> None:
        self.status.run_state = "paused"
        self.status.save(self.config.path)
        logging.info(
            f"Salami experiment paused at step {self.status.current_step}, "
            f"task index {self.status.next_task_index}."
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_experiment(self) -> None:
        """Prepare the sample for slicing.

        Includes:
        - Milling trenches
        - Fiducial creation and alignment area ref image
        """
        self.setup_path = os.path.join(self.config.path, "Setup")
        os.makedirs(self.setup_path, exist_ok=True)

        self.status.run_state = "setup"
        self.status.save(self.config.path)

        # Dialogue to confirm eucentric positioning and sample tilt
        if not self._confirm_position():
            raise InterruptedError("Experiment setup aborted by user.")

        # dig trench and polish
        self._prep_area()

        # create fiducial and alignment reference image
        self._create_fiducial()
        self._confirm_fiducial()

        self.status.setup_complete = True
        self.status.run_state = "paused"
        self.status.save(self.config.path)

    def _image_settings(self):
        """A private copy of the UI's image settings, or None when headless.

        `acquire_channels` mutates `beam_type` on whatever it is handed, so the
        shared settings object must never be passed in directly.
        """
        settings = getattr(self.parent_ui, "settings", None) if self.parent_ui else None
        if settings is None:
            logging.warning(
                "No image settings available (headless); skipping acquisition."
            )
            return None
        return deepcopy(settings.image)

    def _acquire_and_display(self, filename_prefix: Optional[str] = None):
        """Acquire both channels, push them to the UI, optionally save to Setup."""
        image_settings = self._image_settings()
        if image_settings is None:
            return None, None

        eb, ib = acquire_channels(self.microscope, image_settings=image_settings)
        set_images_ui(parent_ui=self.parent_ui, eb_image=eb, ib_image=ib)

        if filename_prefix and self.setup_path:
            if eb is not None:
                eb.save(os.path.join(self.setup_path, f"{filename_prefix}_eb.tif"))
            if ib is not None:
                ib.save(os.path.join(self.setup_path, f"{filename_prefix}_ib.tif"))
        return eb, ib

    def _confirm_position(self) -> bool:
        self._acquire_and_display()

        if self.status.sample_state is not None:
            move_to_state = ask_user(
                parent_ui=self.parent_ui,
                workflow_msg="Setup",
                msg="Existing experiment state detected. Move to saved position?",
                pos="Move",
                neg="No, start from current position",
            )
            if move_to_state:
                self.microscope.set_microscope_state(self.status.sample_state)
                return True

        ret = ask_user(
            parent_ui=self.parent_ui,
            workflow_msg="Setup",
            msg=(
                "Please confirm that the sample is eucentric and tilted to the "
                "correct angle for milling."
            ),
            pos="Confirm",
            neg="Exit",
        )
        if not ret:
            return False

        # Only the position matters, but save the whole state.
        # TODO: when restoring, make sure unnecessary fields are not set.
        self.status.sample_state = self.microscope.get_microscope_state()
        self.status.save(self.config.path)

        self._acquire_and_display(filename_prefix="initial_position")
        return True

    def _prep_area(self) -> None:
        """Mill trenches and polish the area."""
        milling_config = deepcopy(DEFAULT_TRENCH)

        if self.parent_ui is None:
            logging.warning("Headless mode: skipping interactive trench milling.")
            self.trench_config = milling_config
            return

        self._set_milling_config_ui(milling_config)

        pos, neg = "Run Milling", "Continue"
        msg = (
            "Click Run Milling to mill the trenches and polish the area. "
            "After milling, click Continue to proceed to fiducial creation."
        )

        response = ask_user(
            parent_ui=self.parent_ui,
            workflow_msg="Mill Trenches and Polish to Prep Area",
            msg=msg,
            pos=pos,
            neg=neg,
        )

        milling_widget = self.parent_ui.milling_task_config_widget.milling_widget
        while response:
            milling_widget.start_milling_signal.emit()

            wait_for_milling_timeout = 5  # seconds
            start_wait = time.time()
            while not milling_widget.is_milling:
                if time.time() - start_wait > wait_for_milling_timeout:
                    logging.warning(
                        f"Timed out waiting for milling to start after "
                        f"{wait_for_milling_timeout}s."
                    )
                    break
                _check_for_abort(self.parent_ui)
                time.sleep(0.1)

            logging.info("WAITING FOR MILLING TO FINISH... ")
            while milling_widget.is_milling:
                _check_for_abort(self.parent_ui)
                time.sleep(1)

            response = ask_user(
                parent_ui=self.parent_ui,
                workflow_msg="Mill Trenches and Polish to Prep Area",
                msg=msg,
                pos=pos,
                neg=neg,
            )

        # Keep whatever the operator adjusted in the widget, rather than
        # discarding it and reverting to the default trench.
        self.trench_config = deepcopy(
            self.parent_ui.milling_task_config_widget.get_config()
        )
        self.clear_milling_config_ui()

        self._acquire_and_display(filename_prefix="Post_Trench_Milling")

    def _create_fiducial(self) -> None:
        pass

    def _confirm_fiducial(self) -> None:
        pass

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _set_milling_config_ui(self, milling_config: FibsemMillingTaskConfig):
        """Set the milling config in the milling widget."""
        if self.parent_ui is None:
            return

        _check_for_abort(self.parent_ui)

        self.parent_ui.WAITING_FOR_UI_UPDATE = True
        self.parent_ui.workflow_update_signal.emit(
            SalamiWorkflowUpdate(  # type: ignore
                task="milling",
                msg="Updating Milling Config",
                milling_config=deepcopy(milling_config),
            )
        )
        while self.parent_ui.WAITING_FOR_UI_UPDATE:
            time.sleep(0.5)

    def clear_milling_config_ui(self):
        """Clear the milling config from the milling widget."""
        if self.parent_ui is None:
            return

        self.parent_ui.WAITING_FOR_UI_UPDATE = True
        self.parent_ui.workflow_update_signal.emit(
            SalamiWorkflowUpdate(  # type: ignore
                task="milling",
                msg="Clearing Milling Config",
                clear_milling_config=True,
            )
        )
        while self.parent_ui.WAITING_FOR_UI_UPDATE:
            time.sleep(0.5)


def run_salami_experiment(
    microscope: FibsemMicroscope,
    config: SalamiExperimentConfig,
    parent_ui: Optional["SalamiUI"] = None,
) -> None:
    SalamiExperiment(config=config, microscope=microscope, parent_ui=parent_ui).run()


def main():
    pass


if __name__ == "__main__":
    main()
