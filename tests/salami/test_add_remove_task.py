"""Adding and removing tasks from the protocol through the UI.

The dialog is exercised directly rather than through the button, because what
matters is what it produces and what it refuses: a task name becomes a directory
name and a key `find_task` looks up, so a duplicate or a path separator is a
real fault rather than a cosmetic one.
"""

import os

import pytest
from PyQt5.QtWidgets import QApplication, QDialogButtonBox

from fibsem.applications.salami.experiment import (
    SalamiExperiment,
    SalamiExperimentConfig,
)
from fibsem.applications.salami.structures import VolumeAdvanceConfig
from fibsem.applications.salami.tasks import (
    AcquisitionTaskConfig,
    MillingTaskConfig,
    SenseAIAcquisitionTaskConfig,
    available_task_types,
    create_task_config,
    suggest_task_name,
)
from fibsem.applications.salami.ui.add_task_dialog import AddTaskDialog


@pytest.fixture
def qapp():
    """A QApplication for the dialog. pytest-qt is not installed; this is the
    same fixture the other salami UI tests use."""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def experiment(tmp_path) -> SalamiExperiment:
    protocol = SalamiExperimentConfig(
        name="Add Task",
        n_steps=5,
        advance=VolumeAdvanceConfig(step_size=10e-9),
        tasks=[MillingTaskConfig(name="Slice_Task")],
    )
    return SalamiExperiment.create(
        parent_dir=str(tmp_path), name="add", template=protocol
    )


# ---------------------------------------------------------------------------
# The registry drives the choices
# ---------------------------------------------------------------------------


def test_every_registered_task_type_can_be_created():
    """The dropdown is built from the registry, so this is what it offers."""
    for task_type in available_task_types():
        config = create_task_config(task_type, "Some_Task")
        assert config.name == "Some_Task"
        assert config.TASK_TYPE == task_type


def test_unknown_task_type_is_refused():
    with pytest.raises(ValueError):
        create_task_config("not_a_task", "Some_Task")


def test_suggested_names_avoid_collisions():
    existing = [suggest_task_name(AcquisitionTaskConfig.TASK_TYPE, [])]
    second = suggest_task_name(AcquisitionTaskConfig.TASK_TYPE, existing)
    assert second not in existing


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------


def _ok_enabled(dialog: AddTaskDialog) -> bool:
    return dialog.buttonBox.button(QDialogButtonBox.Ok).isEnabled()


def test_dialog_offers_every_task_type(qapp):
    dialog = AddTaskDialog(existing_names=[])
    offered = {
        dialog.comboBoxType.itemData(i) for i in range(dialog.comboBoxType.count())
    }
    assert offered == set(available_task_types())


def test_dialog_suggests_a_usable_name(qapp):
    dialog = AddTaskDialog(existing_names=[])
    assert dialog.task_name
    assert _ok_enabled(dialog)


@pytest.mark.parametrize(
    "name, reason",
    [
        ("", "empty"),
        ("Slice_Task", "duplicate"),
        ("Acq/2", "path separator would create a subdirectory"),
        ("Acq:2", "invalid on windows"),
    ],
)
def test_dialog_refuses_unusable_names(qapp, name, reason):
    dialog = AddTaskDialog(existing_names=["Slice_Task"])
    dialog.lineEditName.setText(name)
    assert not _ok_enabled(dialog), reason
    assert dialog.label_warning.text()


def test_a_typed_name_survives_switching_type(qapp):
    """Switching type to compare options must not discard a name already typed."""
    dialog = AddTaskDialog(existing_names=[])
    dialog.lineEditName.setText("Charge_Settling")

    for index in range(dialog.comboBoxType.count()):
        dialog.comboBoxType.setCurrentIndex(index)

    assert dialog.task_name == "Charge_Settling"


def test_dialog_builds_the_chosen_task(qapp):
    dialog = AddTaskDialog(existing_names=[])
    index = dialog.comboBoxType.findData(SenseAIAcquisitionTaskConfig.TASK_TYPE)
    dialog.comboBoxType.setCurrentIndex(index)
    dialog.lineEditName.setText("SenseAI_Real")

    config = dialog.task_config()

    assert isinstance(config, SenseAIAcquisitionTaskConfig)
    assert config.name == "SenseAI_Real"


# ---------------------------------------------------------------------------
# The protocol after a change
# ---------------------------------------------------------------------------


def test_added_tasks_survive_a_reload(experiment):
    """The workflow this is for: a settling acquisition, then the real one."""
    for name in ("Charge_Settling", "Real_Acquisition"):
        task = create_task_config(AcquisitionTaskConfig.TASK_TYPE, name)
        experiment.config.add_task(task)
    experiment.config.save()

    reopened = SalamiExperiment.open(experiment.config.path)
    names = [t.name for t in reopened.config.tasks]

    assert names == ["Slice_Task", "Charge_Settling", "Real_Acquisition"]
    # Order is the run order, so the settling acquisition precedes the real one.
    assert names.index("Charge_Settling") < names.index("Real_Acquisition")


def test_each_added_task_gets_its_own_output_directory(experiment):
    from fibsem.applications.salami.tasks import ensure_task_directories

    task = create_task_config(AcquisitionTaskConfig.TASK_TYPE, "Charge_Settling")
    experiment.config.add_task(task)
    ensure_task_directories(task, experiment.config.path)

    assert os.path.isdir(os.path.join(experiment.config.path, "Charge_Settling"))


def test_removing_a_task_drops_it_from_the_protocol(experiment):
    task = create_task_config(AcquisitionTaskConfig.TASK_TYPE, "Charge_Settling")
    experiment.config.add_task(task)
    experiment.config.save()

    assert experiment.config.remove_task("Charge_Settling") is True
    experiment.config.save()

    reopened = SalamiExperiment.open(experiment.config.path)
    assert [t.name for t in reopened.config.tasks] == ["Slice_Task"]


def test_removing_a_task_clears_a_pending_skip(experiment):
    """A skip naming a task that no longer exists makes _require_task raise."""
    task = create_task_config(AcquisitionTaskConfig.TASK_TYPE, "Charge_Settling")
    experiment.config.add_task(task)
    experiment.skip_task_once("Charge_Settling")

    experiment.config.remove_task("Charge_Settling")
    experiment.cancel_skip_once("Charge_Settling")

    assert "Charge_Settling" not in experiment.status.skip_once
