"""Setup's fiducial step: the alignment area, where it is kept, and the prompt."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from fibsem.applications.salami._default_settings import DEFAULT_FIDUCIAL
from fibsem.applications.salami.experiment import (
    ALIGNMENT_REFERENCE_FILENAME,
    SalamiExperiment,
    SalamiExperimentStatus,
    fiducial_alignment_area,
)
from fibsem.structures import FibsemRectangle


def test_alignment_area_surrounds_the_fiducial():
    area = fiducial_alignment_area(DEFAULT_FIDUCIAL)
    assert area.is_valid_reduced_area
    # The default fiducial sits 25 um right of centre at an 80 um field of view.
    centre_x = area.left + area.width / 2
    assert centre_x == pytest.approx(0.5 + 25 / 80, abs=0.02)
    assert area.top < 0.5 < area.top + area.height


def test_alignment_area_survives_a_reload(tmp_path):
    status = SalamiExperimentStatus(
        alignment_area=FibsemRectangle(0.6, 0.4, 0.2, 0.2),
        alignment_field_of_view=80e-6,
    )
    path = status.save(str(tmp_path))
    loaded = SalamiExperimentStatus.load(path)
    assert loaded.alignment_area.to_dict() == status.alignment_area.to_dict()
    assert loaded.alignment_field_of_view == 80e-6


def test_old_status_files_load_without_an_alignment_area():
    status = SalamiExperimentStatus.from_dict({"current_step": 3})
    assert status.alignment_area is None
    assert status.alignment_field_of_view is None


def test_headless_setup_records_the_alignment_area(tmp_path):
    from fibsem import utils

    microscope, _settings = utils.setup_session(manufacturer="Demo")
    experiment = SalamiExperiment.create(
        parent_dir=str(tmp_path), name="e", microscope=microscope
    )
    experiment.setup_experiment()

    status = SalamiExperimentStatus.load(
        os.path.join(experiment.path, "Experiment_Status.yaml")
    )
    assert status.setup_complete
    assert status.alignment_area is not None
    assert status.alignment_field_of_view == DEFAULT_FIDUCIAL.field_of_view
    assert experiment.alignment_reference_path.endswith(
        os.path.join("Setup", ALIGNMENT_REFERENCE_FILENAME)
    )


class _FakeImageWidget:
    """Just the alignment-area surface of FibsemImageSettingsWidget."""

    def __init__(self):
        self.shown = None
        self.cleared = False

    def toggle_alignment_area(self, area):
        self.shown = area

    def get_alignment_area(self):
        return FibsemRectangle(0.1, 0.2, 0.3, 0.4)  # as if dragged

    def clear_alignment_area(self):
        self.cleared = True


def test_continue_captures_the_dragged_area(tmp_path):
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from fibsem.applications.salami.ui.SalamiUI import SalamiUI
    from fibsem.applications.salami.workflow_update_schema import (
        SalamiWorkflowUpdate,
    )

    ui = SalamiUI()
    fake = ui.image_widget = _FakeImageWidget()
    # setCurrentWidget needs a real page; the fake is not one.
    ui.tabWidget.setCurrentWidget = lambda _w: None
    initial = FibsemRectangle(0.6, 0.4, 0.2, 0.2)
    ui._apply_workflow_update(
        SalamiWorkflowUpdate(msg="edit", pos="Continue", alignment_area=initial)
    )
    assert fake.shown is initial

    ui.WAITING_FOR_USER_INTERACTION = True
    ui.pushButton_yes.click()
    assert not ui.WAITING_FOR_USER_INTERACTION
    assert ui.USER_RESPONSE
    assert ui.EDITED_ALIGNMENT_AREA.to_dict() == {
        "left": 0.1,
        "top": 0.2,
        "width": 0.3,
        "height": 0.4,
    }
    assert fake.cleared
    ui.image_widget = None
