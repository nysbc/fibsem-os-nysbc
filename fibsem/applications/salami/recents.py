"""Salami's recently-used experiments, for the Load Experiment dialog.

Kept apart from AutoLamella's list in user preferences on purpose. That list holds
paths to `experiment.yaml` files and AutoLamella's load dialog offers every entry
in it, so a Salami experiment recorded there would show up in AutoLamella as one
it cannot open. Salami records experiment *directories* in its own file instead.

Machine-local, like the user preferences beside it, and gitignored with them.
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import yaml

from fibsem.applications.salami.experiment import (
    CONFIG_FILENAME,
    STATUS_FILENAME,
)
from fibsem.config import CONFIG_PATH, MAX_RECENT_EXPERIMENTS

RECENTS_PATH = os.path.join(CONFIG_PATH, "salami-recent-experiments.yaml")

# Default home for new Salami experiments: <repo>/fibsem/applications/salami/log/
SALAMI_LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "log"))


@dataclass
class SalamiExperimentSummary:
    """What an experiment directory says about itself, read without opening it.

    `SalamiExperiment.open` recreates missing task directories, so it is not
    something to call just because the operator clicked a row in a list. This
    reads the two YAML files and nothing else.
    """

    path: str  # the experiment directory
    name: str
    created_at: float = 0.0
    n_steps: int = 0
    current_step: int = 0
    run_state: str = ""
    num_tasks: int = 0
    exists: bool = True
    available: bool = True  # False if the protocol is missing or unreadable


def peek_experiment(directory: str) -> SalamiExperimentSummary:
    """Summarise the experiment in `directory`. Never raises."""
    directory = os.path.normpath(directory)
    fallback_name = os.path.basename(directory) or directory
    config_path = os.path.join(directory, CONFIG_FILENAME)

    if not os.path.isdir(directory):
        return SalamiExperimentSummary(
            path=directory, name=fallback_name, exists=False, available=False
        )
    if not os.path.exists(config_path):
        return SalamiExperimentSummary(
            path=directory, name=fallback_name, available=False
        )

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        status = {}
        status_path = os.path.join(directory, STATUS_FILENAME)
        if os.path.exists(status_path):
            with open(status_path, "r") as f:
                status = yaml.safe_load(f) or {}
        return SalamiExperimentSummary(
            path=directory,
            name=config.get("name") or fallback_name,
            # The protocol file is written once when the experiment is created,
            # so its creation time is the experiment's.
            created_at=os.path.getctime(config_path),
            n_steps=int(config.get("n_steps") or 0),
            current_step=int(status.get("current_step") or 0),
            run_state=status.get("run_state") or "created",
            num_tasks=len(config.get("tasks") or []),
        )
    except Exception as e:
        logging.warning(f"Failed to read Salami experiment in {directory}: {e}")
        return SalamiExperimentSummary(
            path=directory, name=fallback_name, available=False
        )


def _load_paths() -> List[str]:
    if not os.path.exists(RECENTS_PATH):
        return []
    try:
        with open(RECENTS_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
        return [str(p) for p in data.get("recent_experiments") or []]
    except Exception as e:
        logging.warning(f"Failed to read {RECENTS_PATH}: {e}")
        return []


def _save_paths(paths: List[str]) -> None:
    try:
        os.makedirs(os.path.dirname(RECENTS_PATH), exist_ok=True)
        with open(RECENTS_PATH, "w") as f:
            yaml.safe_dump({"recent_experiments": paths}, f)
    except Exception as e:
        logging.warning(f"Failed to save {RECENTS_PATH}: {e}")


def record_recent_experiment(directory: str) -> None:
    """Move `directory` to the front of the list, de-duplicated and truncated."""
    if not directory:
        return
    path = os.path.normpath(str(directory))
    paths = [p for p in _load_paths() if os.path.normpath(p) != path]
    paths.insert(0, path)
    _save_paths(paths[:MAX_RECENT_EXPERIMENTS])


def get_recent_experiments(prune_missing: bool = True) -> List[SalamiExperimentSummary]:
    """Recent experiments, most recent first.

    Directories that no longer exist are dropped and the list saved without them.
    One that exists but cannot be read is kept and marked unavailable, so it is
    shown greyed out rather than silently vanishing.
    """
    paths = _load_paths()
    infos = [peek_experiment(p) for p in paths]
    if prune_missing:
        kept = [info for info in infos if info.exists]
        if len(kept) != len(infos):
            _save_paths([info.path for info in kept])
        return kept
    return infos


def default_parent_directory() -> str:
    """Where a new experiment goes unless the operator says otherwise.

    Beside the most recent experiment, since a run of samples tends to share a
    parent folder; otherwise the application's own log directory.
    """
    for path in _load_paths():
        parent = os.path.dirname(os.path.normpath(path))
        if parent and os.path.isdir(parent):
            return parent
    os.makedirs(SALAMI_LOG_DIR, exist_ok=True)
    return SALAMI_LOG_DIR


def last_experiment_directory() -> Optional[str]:
    """The most recently used experiment directory still on disk, or None."""
    for path in _load_paths():
        if os.path.isdir(path):
            return path
    return None
