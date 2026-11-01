"""Salami's binding for user scripts (Tools > Scripts).

The discovery, running and menu are application-agnostic (`fibsem.scripting`,
`fibsem.ui.widgets.script_menu`). This supplies what is specific to Salami:
where the scripts live, and what a script is handed when it runs.

A folder of its own rather than AutoLamella's: a script written there expects
`ctx.experiment` to be an AutoLamella experiment, with lamellae, and would fail
against a Salami one.
"""

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from fibsem.cancellation import raise_if_cancelled

if TYPE_CHECKING:
    from fibsem.applications.salami.experiment import SalamiExperiment
    from fibsem.microscope import FibsemMicroscope

SCRIPTS_DIR_ENV_VAR = "SALAMI_SCRIPTS_DIR"
DEFAULT_SCRIPTS_DIR = Path.home() / ".salami" / "scripts"


def get_scripts_directory() -> Path:
    """Resolve the user scripts directory. Does not create it."""
    override = os.environ.get(SCRIPTS_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return DEFAULT_SCRIPTS_DIR


@dataclass
class SalamiScriptContext:
    """What a script is handed when it runs: `run(ctx)`.

    The same surface as AutoLamella's `ScriptContext`, with a Salami experiment:
    `ctx.experiment.config` is the protocol, `ctx.experiment.status` the run state.
    `microscope` is None when no microscope is connected.
    """

    experiment: "SalamiExperiment"
    log: Callable[[str], None] = logging.info
    microscope: Optional["FibsemMicroscope"] = None
    # Set by the runner for scripts that run off the GUI thread. Stopping one is
    # cooperative: the runner sets this, and the script has to look at it.
    stop_event: Optional[threading.Event] = None

    @property
    def path(self) -> Path:
        """The experiment directory — the default place to write output."""
        return Path(self.experiment.path)

    def save(self) -> None:
        """Persist the experiment. Called by the runner only for `writes` scripts."""
        self.experiment.config.save()
        self.experiment.status.save(self.experiment.path)

    @property
    def cancelled(self) -> bool:
        """Whether the user has asked this script to stop."""
        return self.stop_event is not None and self.stop_event.is_set()

    def raise_if_cancelled(self, msg: str = "Script cancelled by user.") -> None:
        """Abort the script if the user pressed Stop. Call between steps."""
        raise_if_cancelled(self.stop_event, msg)
