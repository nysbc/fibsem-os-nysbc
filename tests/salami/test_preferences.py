"""Edit > Preferences: Salami's own store, the shared one, and the Create dialog."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem import config
from fibsem.applications.salami import preferences
from fibsem.applications.salami.experiment import SalamiExperiment
from fibsem.applications.salami.preferences import (
    SalamiPreferences,
    load_salami_preferences,
    save_salami_preferences,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def saved_protocol(tmp_path):
    """A loadable Salami protocol file on disk, named apart from the default."""
    template = SalamiExperiment.default_template()
    template.name = "preferred-protocol"
    return template.export(str(tmp_path / "preferred.yaml"))


def test_salami_preferences_round_trip(tmp_path):
    prefs = SalamiPreferences(
        default_experiment_directory=str(tmp_path),
        default_protocol_path=str(tmp_path / "p.yaml"),
    )
    save_salami_preferences(prefs)
    assert os.path.exists(preferences.PREFERENCES_PATH)
    assert load_salami_preferences() == prefs


def test_missing_preferences_file_gives_defaults():
    assert not os.path.exists(preferences.PREFERENCES_PATH)
    assert load_salami_preferences() == SalamiPreferences()


def test_dialog_keeps_what_it_does_not_show(qapp, tmp_path):
    from fibsem.applications.salami.ui.preferences_dialog import (
        SalamiPreferencesDialog,
    )

    shared = config.UserPreferences()
    shared.display.dev_mode = True
    shared.display.lamella_card_mode = "compact"
    shared.features.grid_workflow = True
    shared.features.agent_server_enabled = True
    shared.experiment.default_protocol_path = "autolamella-protocol.yaml"
    shared.experiment.user = "someone"

    dialog = SalamiPreferencesDialog(shared, SalamiPreferences())
    dialog._chk_sound.setChecked(True)
    dialog._chk_acquire_fib.setChecked(False)
    dialog._dir_experiment.setText(str(tmp_path))
    result, salami = dialog.get_preferences()

    # What the dialog shows changed...
    assert result.display.sound_enabled is True
    assert result.movement.acquire_fib_after_stage_movement is False
    assert salami.default_experiment_directory == str(tmp_path)
    # ...and AutoLamella's settings did not.
    assert result.display.dev_mode is True
    assert result.display.lamella_card_mode == "compact"
    assert result.features == shared.features
    assert result.experiment == shared.experiment


def test_dialog_refuses_an_invalid_protocol(qapp, tmp_path, monkeypatch):
    from fibsem.applications.salami.ui import preferences_dialog as module

    bad = tmp_path / "bad.yaml"
    bad.write_text("not: a protocol\n")
    warnings = []
    monkeypatch.setattr(
        module.QMessageBox, "warning", lambda *a, **k: warnings.append(a)
    )

    dialog = module.SalamiPreferencesDialog(
        config.UserPreferences(), SalamiPreferences()
    )
    dialog._dir_protocol.setText(str(bad))
    dialog._on_ok()
    assert warnings and dialog.result() != module.QDialog.Accepted


def test_create_dialog_uses_preferred_directory_and_protocol(
    qapp, tmp_path, saved_protocol
):
    from fibsem.applications.salami.ui.experiment_dialogs import (
        SalamiCreateExperimentDialog,
    )

    target = tmp_path / "experiments"
    target.mkdir()
    save_salami_preferences(
        SalamiPreferences(
            default_experiment_directory=str(target),
            default_protocol_path=saved_protocol,
        )
    )

    dialog = SalamiCreateExperimentDialog()
    assert dialog.lineEdit_experiment_directory.text() == str(target)
    assert dialog.template.name == "preferred-protocol"
    assert dialog.template_path == saved_protocol


def test_create_dialog_falls_back_when_preferred_protocol_is_missing(qapp, tmp_path):
    from fibsem.applications.salami.ui.experiment_dialogs import (
        SalamiCreateExperimentDialog,
    )

    save_salami_preferences(
        SalamiPreferences(
            default_experiment_directory=str(tmp_path / "gone"),
            default_protocol_path=str(tmp_path / "gone.yaml"),
        )
    )

    dialog = SalamiCreateExperimentDialog()
    assert dialog.template is not None
    assert dialog.template_path == "Default protocol"
    assert dialog.lineEdit_experiment_directory.text() != str(tmp_path / "gone")
