"""The recent-experiments list and loading a protocol into an experiment."""

import os

import pytest

from fibsem.applications.salami import recents
from fibsem.applications.salami.experiment import SalamiExperiment


def _create(tmp_path, name):
    return SalamiExperiment.create(parent_dir=str(tmp_path), name=name)


def test_recent_experiments_most_recent_first_without_duplicates(tmp_path):
    a = _create(tmp_path, "A")
    b = _create(tmp_path, "B")
    recents.record_recent_experiment(a.path)
    recents.record_recent_experiment(b.path)
    recents.record_recent_experiment(a.path)

    names = [info.name for info in recents.get_recent_experiments()]
    assert names == ["A", "B"]


def test_recent_experiments_drop_deleted_directories(tmp_path):
    a = _create(tmp_path, "A")
    recents.record_recent_experiment(a.path)
    recents.record_recent_experiment(str(tmp_path / "gone"))

    assert [info.name for info in recents.get_recent_experiments()] == ["A"]
    # ...and the pruned list is what was saved.
    assert [info.name for info in recents.get_recent_experiments()] == ["A"]


def test_peek_reports_progress_without_writing(tmp_path):
    experiment = _create(tmp_path, "A")
    experiment.status.current_step = 3
    experiment.status.save(experiment.path)
    before = sorted(os.listdir(experiment.path))

    info = recents.peek_experiment(experiment.path)

    assert info.available
    assert info.current_step == 3
    assert info.n_steps == experiment.config.n_steps
    assert sorted(os.listdir(experiment.path)) == before


def test_a_folder_without_a_protocol_is_unavailable(tmp_path):
    (tmp_path / "not-an-experiment").mkdir()
    info = recents.peek_experiment(str(tmp_path / "not-an-experiment"))
    assert info.exists and not info.available


def test_new_experiments_go_beside_the_last_one(tmp_path):
    a = _create(tmp_path, "A")
    recents.record_recent_experiment(a.path)
    assert os.path.normpath(recents.default_parent_directory()) == os.path.normpath(
        str(tmp_path)
    )


def test_replace_protocol_keeps_identity_and_progress(tmp_path):
    experiment = _create(tmp_path, "Target")
    experiment.config.description = "this sample"
    experiment.status.current_step = 7
    experiment.status.skip_once = ["Gone_Task"]

    template = _create(tmp_path, "Source").config
    template.n_steps = 123
    template.tasks = template.tasks[:1]

    experiment.replace_protocol(template)

    assert experiment.config.name == "Target"
    assert experiment.config.description == "this sample"
    assert os.path.normpath(experiment.config.path) == os.path.normpath(
        str(tmp_path / "Target")
    )
    assert experiment.config.n_steps == 123
    assert [t.name for t in experiment.config.tasks] == [template.tasks[0].name]
    assert experiment.status.current_step == 7
    assert experiment.status.skip_once == []
    # The source experiment's own protocol file is untouched.
    reopened = SalamiExperiment.open(str(tmp_path / "Source"))
    assert reopened.config.n_steps != 123


def test_replace_protocol_refused_part_way_through_a_slice(tmp_path):
    experiment = _create(tmp_path, "Target")
    experiment.status.next_task_index = 1
    with pytest.raises(RuntimeError, match="part-way"):
        experiment.replace_protocol(_create(tmp_path, "Source").config)
