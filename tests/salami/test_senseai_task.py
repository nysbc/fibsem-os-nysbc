"""The SenseAI acquisition task: scan mode, output, and reconstruction cadence.

No microscope and no SenseAI DLL. Both are faked, because what is under test is
the task's contract with them — that external scan is set before the frame is
taken and given back afterwards even on failure, that a slice lands on disk as a
four page stack, and that reconstruction honours its own schedule.
"""

import os

import numpy as np
import pytest
import tifffile

from fibsem.applications.salami import tasks as tasks_module
from fibsem.applications.salami.structures import (
    RunContext,
    RunSchedule,
    VolumeAdvanceConfig,
)
from fibsem.applications.salami.tasks import (
    SenseAIAcquisitionTaskConfig,
    run_task,
    runtime_task_config,
)
from fibsem.SenseAI_SG import (
    INPUT_PAGE,
    MASK_PAGE,
    RECONSTRUCTION_PAGE,
    REFERENCE_PAGE,
)

ROWS, COLS = 24, 32


class FakeMicroscope:
    """Records what the task did to the column, in order."""

    def __init__(self):
        self.calls = []

    def set(self, key, value=None, beam_type=None):
        self.calls.append(key)

    def set_full_frame_scanning_mode(self, beam_type):
        self.calls.append("full_frame")

    def set_field_of_view(self, hfw, beam_type):
        self.calls.append(f"hfw={hfw}")

    def auto_focus(self, beam_type, area):
        self.calls.append("auto_focus")

    def autocontrast(self, beam_type, area):
        self.calls.append("autocontrast")


class FakeModule:
    """Stands in for SenseAI_ModuleControl."""

    def __init__(self, fail: bool = False, problem: str = None):
        self.fail = fail
        self.problem = problem
        self.config = type("cfg", (), {"scanGen_config": {"Name": "WCScanGen"}})()
        self.scangen_updates = []
        self.recon_calls = 0

    # The task goes through require_senseai_module, which refuses anything that
    # reports a problem rather than acquiring against a dead generator.
    def readiness_problem(self):
        return self.problem

    @property
    def is_ready(self) -> bool:
        return self.problem is None

    def update_ScanGenerator(self, config):
        self.scangen_updates.append(config)

    def acquire_image(self):
        if self.fail:
            raise RuntimeError("scan generator exploded")
        rng = np.random.default_rng(0)
        image = rng.integers(-20000, 20000, (ROWS, COLS), dtype=np.int16)
        mask = np.zeros((ROWS, COLS), np.uint8)
        mask[::2] = 1
        image[mask == 0] = -32768  # the detector's unvisited sentinel
        return [image, mask]

    def reconstruct(self, stack, **kwargs):
        self.recon_calls += 1
        return np.full((ROWS, COLS), 0.5, dtype=np.float32)


@pytest.fixture
def module(monkeypatch):
    """Publish a fake as the process's module, as the SenseAI tab would.

    Patching the module global rather than the accessor, so the task goes
    through the real `require_senseai_module` gate.
    """
    fake = FakeModule()
    monkeypatch.setattr("fibsem.SenseAI_SG._MODULE", fake)
    return fake


@pytest.fixture
def context(tmp_path) -> RunContext:
    return RunContext(experiment_path=str(tmp_path), current_step=0, total_steps=5)


def _config(**kwargs) -> SenseAIAcquisitionTaskConfig:
    cfg = SenseAIAcquisitionTaskConfig(name="SenseAI_Task", **kwargs)
    # Auto-functions off by default here: they are exercised in their own test.
    cfg.autofocus.enabled = False
    cfg.autocontrast.enabled = False
    return cfg


def _run(cfg, microscope, context, step=0):
    context.current_step = step
    runtime = runtime_task_config(
        cfg, context.experiment_path, step, VolumeAdvanceConfig(step_size=10e-9)
    )
    run_task(microscope, runtime, context)
    return runtime


# ---------------------------------------------------------------------------
# Scan mode
# ---------------------------------------------------------------------------


def test_external_scan_is_set_for_the_frame_and_given_back(module, context):
    microscope = FakeMicroscope()
    _run(_config(), microscope, context)

    assert "scantype_external" in microscope.calls
    assert "full_frame" in microscope.calls
    assert microscope.calls.index("scantype_external") < microscope.calls.index(
        "full_frame"
    )


def test_the_column_is_left_able_to_image_after_a_failure(module, context):
    """A failed frame must not strand the column in external scan."""
    module.fail = True
    microscope = FakeMicroscope()

    with pytest.raises(RuntimeError):
        _run(_config(), microscope, context)

    assert microscope.calls[-1] == "full_frame"


def test_auto_functions_run_before_the_scan_is_handed_over(module, context):
    """They drive the column's own scan engine, so they cannot run in external mode."""
    cfg = _config()
    cfg.autofocus.enabled = True
    cfg.autocontrast.enabled = True
    microscope = FakeMicroscope()

    _run(cfg, microscope, context)

    for function in ("auto_focus", "autocontrast"):
        assert microscope.calls.index(function) < microscope.calls.index(
            "scantype_external"
        )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def test_a_slice_is_saved_as_a_four_page_stack(module, context):
    runtime = _run(_config(), FakeMicroscope(), context)

    saved = os.path.join(runtime.output_path, "SenseAI_Task-Image-000.tiff")
    assert os.path.exists(saved)

    with tifffile.TiffFile(saved) as tif:
        # Four separate pages, not one four-sample image: SenseAI reads the
        # latter as a 4-deep volume and assigns no layers.
        assert len(tif.pages) == 4
        assert tif.pages[0].tags["SamplesPerPixel"].value == 1

    stack = tifffile.imread(saved)
    assert stack.dtype == np.uint16
    assert np.array_equal(stack[REFERENCE_PAGE], stack[INPUT_PAGE])
    # The measurement is what has to survive: every non-zero input pixel is
    # inside the mask, and nothing outside it.
    measured = stack[MASK_PAGE] != 0
    assert not np.any((stack[INPUT_PAGE] != 0) & ~measured)


def test_the_reconstruction_lands_in_page_zero(module, context):
    _run(_config(), FakeMicroscope(), context)
    stack = tifffile.imread(
        os.path.join(
            context.experiment_path, "SenseAI_Task", "SenseAI_Task-Image-000.tiff"
        )
    )
    # Scaled, not truncated: a float 0..1 assigned straight into uint16 is zero.
    assert stack[RECONSTRUCTION_PAGE].any()


# ---------------------------------------------------------------------------
# Reconstruction cadence
# ---------------------------------------------------------------------------


def test_reconstruction_honours_its_own_schedule(module, context):
    """Input and mask are saved every slice, so reconstruction can be sparse."""
    cfg = _config()
    cfg.reconstruction.schedule = RunSchedule(every_n_steps=3)

    for step in range(6):
        _run(cfg, FakeMicroscope(), context, step=step)

    assert module.recon_calls == 2  # steps 0 and 3

    # ... but every step still produced a slice.
    written = os.listdir(os.path.join(context.experiment_path, "SenseAI_Task"))
    assert len(written) == 6


def test_reconstruction_can_be_turned_off_entirely(module, context):
    cfg = _config()
    cfg.reconstruction.enabled = False

    runtime = _run(cfg, FakeMicroscope(), context)

    assert module.recon_calls == 0
    stack = tifffile.imread(
        os.path.join(runtime.output_path, "SenseAI_Task-Image-000.tiff")
    )
    assert not stack[RECONSTRUCTION_PAGE].any()


# ---------------------------------------------------------------------------
# Scan settings and the protocol
# ---------------------------------------------------------------------------


def test_scan_settings_reach_the_generator_in_its_own_units(module, context):
    cfg = _config()
    cfg.scan.scan_size = (640, 480)
    cfg.scan.sampling = 0.25
    cfg.scan.dwell_time = 3e-6  # seconds

    _run(cfg, FakeMicroscope(), context)

    sent = module.scangen_updates[-1]
    assert sent["ScanSize"] == [640, 480]
    assert sent["Sampling"] == 0.25
    # SenseAI's workspace JSON is in milliseconds: 3 us -> 0.003.
    assert sent["DwellTime"] == pytest.approx(0.003)
    # Keys we do not model must survive, or the generator loses its name.
    assert sent["Name"] == "WCScanGen"


def test_the_protocol_round_trips(module):
    cfg = _config()
    cfg.scan.pattern = "Raster"
    cfg.reconstruction.iterations = 120
    cfg.output_path = "/tmp/should-not-persist"

    restored = SenseAIAcquisitionTaskConfig.from_dict(cfg.to_dict())

    assert restored.scan.pattern == "Raster"
    assert restored.reconstruction.iterations == 120
    assert restored.scan.dwell_time == cfg.scan.dwell_time


def test_reuse_strips_the_machine_specific_output_path(module):
    cfg = _config()
    cfg.output_path = "/some/machine/Experiment/SenseAI_Task"

    tasks_module.prepare_task_for_reuse(cfg)

    assert cfg.output_path is None


# ---------------------------------------------------------------------------
# The task never loads the DLL or initialises a generator
# ---------------------------------------------------------------------------
#
# Both happen once per process, from the SenseAI tab. The DLL's source registry
# outlives the Python object, so a second module inherits the first one's
# sources, collides on "source already exists", and yields a generator that
# never returns a frame. The task's whole job is per-slice: update the config,
# take the scan external, give it back.


def test_the_task_refuses_when_no_generator_is_available(monkeypatch, context):
    """Rather than building one, which is what produced a dead generator."""
    monkeypatch.setattr("fibsem.SenseAI_SG._MODULE", None)

    with pytest.raises(RuntimeError, match="No SenseAI scan generator is available"):
        _run(_config(), FakeMicroscope(), context)


def test_the_task_does_not_build_a_module(monkeypatch, context):
    import fibsem.SenseAI_SG as sg

    monkeypatch.setattr("fibsem.SenseAI_SG._MODULE", None)
    monkeypatch.setattr(
        sg, "SenseAI_ModuleControl", _fail_if_constructed("SenseAI_ModuleControl")
    )

    with pytest.raises(RuntimeError):
        _run(_config(), FakeMicroscope(), context)


def test_the_task_refuses_a_generator_that_was_never_initialised(monkeypatch, context):
    """`is_ready` is False here; acquiring anyway is what hangs on a frame."""
    monkeypatch.setattr(
        "fibsem.SenseAI_SG._MODULE",
        FakeModule(problem="scan generator 'WCScanGen' was never initialised"),
    )

    with pytest.raises(RuntimeError, match="never initialised"):
        _run(_config(), FakeMicroscope(), context)


def test_a_module_on_the_context_is_used_for_a_headless_run(context, monkeypatch):
    """Headless callers pass their own; nothing is published process-wide."""
    monkeypatch.setattr("fibsem.SenseAI_SG._MODULE", None)
    supplied = FakeModule()
    context.senseai_module = supplied

    _run(_config(), FakeMicroscope(), context)

    assert len(supplied.scangen_updates) == 1


def _fail_if_constructed(name):
    def _boom(*args, **kwargs):
        raise AssertionError(f"the task must not construct {name}")

    return _boom


# ---------------------------------------------------------------------------
# The preview goes to the UI thread, with metadata
# ---------------------------------------------------------------------------


def test_the_preview_rides_the_workflow_update_not_the_psygnal(module, context):
    """`microscope.sem_acquisition_signal` is a psygnal: its handlers run in the
    emitting thread, which would put Qt drawing on the run's worker thread."""
    microscope = FakeMicroscope()
    updates = []
    context.parent_ui = type(
        "UI",
        (),
        {
            "workflow_update_signal": type(
                "S", (), {"emit": staticmethod(updates.append)}
            )(),
            "_workflow_stop_event": None,
        },
    )()

    _run(_config(), FakeMicroscope(), context)

    with_image = [u for u in updates if getattr(u, "sem_image", None) is not None]
    assert len(with_image) == 1
    assert "sem_acquisition_signal" not in microscope.calls


def test_the_preview_carries_what_the_viewer_needs(module, context):
    """A bare FibsemImage raises "Image metadata is None" and is dropped."""
    from fibsem.applications.salami.tasks import senseai_preview_image
    from fibsem.structures import BeamType

    cfg = _config()
    image = senseai_preview_image(
        np.zeros((ROWS, COLS), np.uint16), cfg, cfg.state.electron_beam
    )

    assert image.metadata is not None
    assert image.metadata.beam_type is BeamType.ELECTRON  # chooses the pane
    assert image.metadata.pixel_size.x > 0  # drives the scale bar


# ---------------------------------------------------------------------------
# Which page reaches the viewer
# ---------------------------------------------------------------------------


def _preview_from(updates):
    """The image the task asked the UI to display."""
    with_image = [u for u in updates if getattr(u, "sem_image", None) is not None]
    assert len(with_image) == 1
    return with_image[0]


def _recording_ui():
    updates = []
    ui = type(
        "UI",
        (),
        {
            "workflow_update_signal": type(
                "S", (), {"emit": staticmethod(updates.append)}
            )(),
            "_workflow_stop_event": None,
        },
    )()
    return ui, updates


def test_the_reconstruction_is_what_gets_displayed(module, context):
    """The Input page at 30% sampling is a sparse scatter; the point of
    reconstructing during a run is to see the dense image."""
    ui, updates = _recording_ui()
    context.parent_ui = ui

    _run(_config(), FakeMicroscope(), context)

    update = _preview_from(updates)
    # FakeModule.reconstruct returns a constant, so the displayed frame is
    # uniform — the sparse measurement never is.
    assert len(np.unique(update.sem_image.data)) == 1
    assert "reconstruction" in update.msg


def test_the_measurement_is_displayed_when_reconstruction_is_off(module, context):
    """Reconstruction has its own cadence; the slices in between still show."""
    ui, updates = _recording_ui()
    context.parent_ui = ui
    cfg = _config()
    cfg.reconstruction.enabled = False

    _run(cfg, FakeMicroscope(), context)

    update = _preview_from(updates)
    assert len(np.unique(update.sem_image.data)) > 1  # the sparse measurement
    assert "measurement" in update.msg


def test_a_failed_reconstruction_falls_back_to_the_measurement(module, context):
    """Rather than displaying the empty reconstruction page."""
    module.reconstruct = lambda stack, **kwargs: None
    ui, updates = _recording_ui()
    context.parent_ui = ui

    _run(_config(), FakeMicroscope(), context)

    update = _preview_from(updates)
    assert update.sem_image.data.any()
    assert "measurement" in update.msg
