"""Salami's own preferences: where new experiments go and what they start from.

Kept apart from AutoLamella's `ExperimentPreferences` on purpose. AutoLamella's
create dialog loads `default_protocol_path` as an AutoLamella protocol, so a
Salami protocol stored there would break it, and the reverse. The directory is
split for the same reason: the two applications keep their experiments apart.

Everything else Salami's Preferences dialog shows (sound, border, movement) is
per-user and lives in the shared user preferences.

Machine-local, like the user preferences beside it, and gitignored with them.
"""

import logging
import os
from dataclasses import asdict, dataclass, fields

import yaml

from fibsem.config import CONFIG_PATH

PREFERENCES_PATH = os.path.join(CONFIG_PATH, "salami-preferences.yaml")


@dataclass
class SalamiPreferences:
    # Empty means "not set": beside the last experiment, else the log directory.
    default_experiment_directory: str = ""
    # Empty means the built-in default protocol.
    default_protocol_path: str = ""


def load_salami_preferences() -> SalamiPreferences:
    """Read the preferences file, or the defaults if it is missing. Never raises."""
    if not os.path.exists(PREFERENCES_PATH):
        return SalamiPreferences()
    try:
        with open(PREFERENCES_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
        known = {f.name for f in fields(SalamiPreferences)}
        return SalamiPreferences(
            **{k: str(v or "") for k, v in data.items() if k in known}
        )
    except Exception as e:
        logging.warning(f"Failed to read {PREFERENCES_PATH}: {e}")
        return SalamiPreferences()


def save_salami_preferences(preferences: SalamiPreferences) -> None:
    """Write the preferences file. Never raises."""
    try:
        os.makedirs(os.path.dirname(PREFERENCES_PATH), exist_ok=True)
        with open(PREFERENCES_PATH, "w") as f:
            yaml.safe_dump(asdict(preferences), f)
    except Exception as e:
        logging.warning(f"Failed to save {PREFERENCES_PATH}: {e}")
