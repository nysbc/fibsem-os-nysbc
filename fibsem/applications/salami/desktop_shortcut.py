"""Create a desktop shortcut that launches the Salami UI.

`fibsem.tools.desktop_shortcut` does this for AutoLamella, with the entry point
and names fixed. This reuses its pieces (finding the entry point and the
Desktop, writing the script) and supplies Salami's names instead.

Pure Python, no Qt.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fibsem.tools.desktop_shortcut import (
    _posix_quote,
    _write_script,
    find_entry_point,
    get_desktop_directory,  # noqa: F401 - re-exported for the UI
)

ENTRY_POINT_NAME = "salami-ui"
SHORTCUT_BASENAME = "Salami"


def shortcut_path(directory: Optional[Path] = None) -> Path:
    """The path the shortcut will be written to on this platform."""
    if directory is None:
        directory = get_desktop_directory()
    if os.name == "nt":
        return directory / (SHORTCUT_BASENAME + ".lnk")
    if sys.platform == "darwin":
        return directory / (SHORTCUT_BASENAME + ".command")
    return directory / (SHORTCUT_BASENAME + ".desktop")


def create_desktop_shortcut(
    overwrite: bool = False, directory: Optional[Path] = None
) -> Path:
    """Create a shortcut launching the Salami UI; return its path.

    Raises like the AutoLamella version: FileNotFoundError when the entry point
    cannot be located, FileExistsError when a shortcut already exists and
    `overwrite` is False, OSError / CalledProcessError when writing fails.
    """
    target = find_entry_point(ENTRY_POINT_NAME)
    if target is None:
        raise FileNotFoundError(
            f"Could not locate the '{ENTRY_POINT_NAME}' executable in this environment."
        )

    destination = shortcut_path(directory)
    if destination.exists() and not overwrite:
        raise FileExistsError(str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        _write_windows_lnk(destination, target)
    elif sys.platform == "darwin":
        _write_script(
            destination, "#!/bin/bash\nexec '{}'\n".format(_posix_quote(target))
        )
    else:
        _write_script(
            destination,
            "[Desktop Entry]\n"
            "Type=Application\n"
            "Name=Salami\n"
            "Comment=Launch the Salami UI\n"
            'Exec="{}"\n'
            "Terminal=false\n".format(str(target)),
        )
    return destination


def _write_windows_lnk(destination: Path, target: Path) -> None:
    """A real .lnk via WScript.Shell, as core writes AutoLamella's."""

    def q(p: Path) -> str:
        return "'" + str(p).replace("'", "''") + "'"

    script = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut({dest}); "
        "$s.TargetPath = {target}; "
        "$s.WorkingDirectory = {workdir}; "
        "$s.Description = 'Launch the Salami UI'; "
        "$s.Save()"
    ).format(dest=q(destination), target=q(target), workdir=q(Path.home()))
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
