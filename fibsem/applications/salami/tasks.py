import logging
import os
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, List, Optional, Type, Union

from fibsem.acquire import acquire_image
from fibsem.applications.salami._default_settings import DEFAULT_SLICE
from fibsem.applications.salami.structures import (
    AutoFunctionConfig,
    RunContext,
    RunSchedule,
    VolumeAdvanceConfig,
    centred_area,
)
from fibsem.applications.salami.workflow_update_schema import SalamiWorkflowUpdate
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.tasks import FibsemMillingTaskConfig, run_milling_task
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemImage,
    FibsemImageMetadata,
    FibsemRectangle,
    ImageSettings,
    MicroscopeState,
    Point,
)
from fibsem.utils import format_value

# ---------------------------------------------------------------------------
# Task configs (pure data)
# ---------------------------------------------------------------------------

# Default region each auto-function works on, as a fraction of the frame,
# centred. Focus is given the larger box because it needs enough structure in
# view to score sharpness against; contrast only needs a patch representative of
# the face. Neither defaults to the full frame — the edges of a freshly milled
# face include the trench and the unmilled surface either side of it, which are
# not what either function should be judging.
AUTOFOCUS_AREA_FRACTION = 0.4
AUTOCONTRAST_AREA_FRACTION = 0.3


@dataclass
class AcquisitionTaskConfig:
    name: str = "Acquisition Task"
    description: str = ""
    enabled: bool = True
    schedule: RunSchedule = field(default_factory=RunSchedule)
    state: MicroscopeState = field(default_factory=MicroscopeState)
    image: ImageSettings = field(default_factory=ImageSettings)
    alignment_area: FibsemRectangle = field(default_factory=FibsemRectangle)
    autofocus: AutoFunctionConfig = field(
        default_factory=lambda: AutoFunctionConfig(
            area_of_interest=centred_area(AUTOFOCUS_AREA_FRACTION)
        )
    )
    autocontrast: AutoFunctionConfig = field(
        default_factory=lambda: AutoFunctionConfig(
            area_of_interest=centred_area(AUTOCONTRAST_AREA_FRACTION)
        )
    )

    TASK_TYPE: ClassVar[str] = "acquisition"
    DISPLAY_NAME: ClassVar[str] = "Acquisition"

    def should_run(self, step: int) -> bool:
        """`enabled` is the operator's on/off switch; `schedule` is the planned
        cadence. Kept separate so turning a task off doesn't destroy its
        schedule (mirrors AutoFunctionConfig)."""
        return self.enabled and self.schedule.should_run(step)

    def to_dict(self) -> dict:
        return {
            "type": self.TASK_TYPE,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "schedule": self.schedule.to_dict(),
            "state": self.state.to_dict(),
            "image": self.image.to_dict(),
            "alignment_area": self.alignment_area.to_dict(),
            "autofocus": self.autofocus.to_dict(),
            "autocontrast": self.autocontrast.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AcquisitionTaskConfig":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            schedule=RunSchedule.from_dict(data.get("schedule", {})),
            state=MicroscopeState.from_dict(data["state"]),
            image=ImageSettings.from_dict(data["image"]),
            alignment_area=FibsemRectangle(**data["alignment_area"]),
            autofocus=AutoFunctionConfig.from_dict(
                data.get("autofocus", {}),
                default_area=centred_area(AUTOFOCUS_AREA_FRACTION),
            ),
            autocontrast=AutoFunctionConfig.from_dict(
                data.get("autocontrast", {}),
                default_area=centred_area(AUTOCONTRAST_AREA_FRACTION),
            ),
        )


@dataclass
class MillingTaskConfig:
    name: str = "Milling Task"
    description: str = ""
    enabled: bool = True
    schedule: RunSchedule = field(default_factory=RunSchedule)
    state: MicroscopeState = field(default_factory=MicroscopeState)
    milling: FibsemMillingTaskConfig = field(default_factory=FibsemMillingTaskConfig)

    TASK_TYPE: ClassVar[str] = "milling"
    DISPLAY_NAME: ClassVar[str] = "Milling"

    def should_run(self, step: int) -> bool:
        """See AcquisitionTaskConfig.should_run."""
        return self.enabled and self.schedule.should_run(step)

    def __post_init__(self):
        if len(self.milling.stages) == 0:
            # deepcopy: DEFAULT_SLICE is a module-level singleton, and appending it
            # directly would have every default task share (and mutate) one stage.
            self.milling.stages.append(deepcopy(DEFAULT_SLICE))

    def to_dict(self) -> dict:
        return {
            "type": self.TASK_TYPE,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "schedule": self.schedule.to_dict(),
            "state": self.state.to_dict(),
            "milling": self.milling.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MillingTaskConfig":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            schedule=RunSchedule.from_dict(data.get("schedule", {})),
            state=MicroscopeState.from_dict(data["state"]),
            milling=FibsemMillingTaskConfig.from_dict(data["milling"]),
        )


# Scan patterns the external generator offers, as listed on the SenseAI tab.
SENSEAI_SCAN_PATTERNS = (
    "Raster",
    "Linehop",
    "Random",
    "Raster-Random",
    "Lane-Constrained Random",
    "Interleaved Linehop",
)


@dataclass
class SenseAIScanConfig:
    """What the external scan generator does: the sub-sampled scan itself.

    These replace resolution and dwell time rather than sitting alongside them.
    While the column is in external scan the generator decides how many pixels
    are visited and for how long, so an `ImageSettings` on this task would be a
    second home for values the run never reads.
    """

    # Pixels the generator scans, (width, height). SenseAI's own ScanSize is in
    # this order; note the simulator flips it to index a numpy array.
    scan_size: tuple = (1536, 1024)
    pattern: str = "Linehop"
    # Fraction of pixels actually measured. The point of the whole exercise:
    # 0.3 visits 30% of the frame and the reconstruction fills in the rest.
    sampling: float = 0.3
    # Seconds, as everywhere else in fibsem. SenseAI's workspace JSON stores
    # milliseconds, so the conversion happens in `apply_to`.
    dwell_time: float = 1e-6
    reset_buffer: bool = True

    def apply_to(self, scangen_config: Optional[dict]) -> dict:
        """Overlay these settings onto the generator's existing config.

        Copied rather than rebuilt: the workspace JSON carries keys we do not
        model (the generator's name, hardware-specific entries), and dropping
        them would leave the generator misconfigured.
        """
        config = dict(scangen_config or {})
        config["ScanSize"] = [int(self.scan_size[0]), int(self.scan_size[1])]
        config["Pattern"] = self.pattern
        config["Sampling"] = float(self.sampling)
        config["DwellTime"] = float(self.dwell_time) * 1e3  # s -> ms
        config["ResetBuffer"] = bool(self.reset_buffer)
        return config

    def to_dict(self) -> dict:
        return {
            "scan_size": list(self.scan_size),
            "pattern": self.pattern,
            "sampling": self.sampling,
            "dwell_time": self.dwell_time,
            "reset_buffer": self.reset_buffer,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SenseAIScanConfig":
        size = data.get("scan_size") or (1536, 1024)
        return cls(
            scan_size=tuple(size),
            pattern=data.get("pattern", "Linehop"),
            sampling=data.get("sampling", 0.3),
            dwell_time=data.get("dwell_time", 1e-6),
            reset_buffer=data.get("reset_buffer", True),
        )


@dataclass
class ReconstructionConfig:
    """Whether, and how often, to reconstruct during the run.

    Reconstruction is optional work: the Input and Mask layers are saved
    losslessly for every slice, so any slice can be reconstructed afterwards in
    the SenseAI desktop app. During a run it is really for watching progress,
    which is why it has its own cadence — reconstructing every tenth slice keeps
    an eye on the volume without paying for it 5000 times.

    The three parameters exist so a tuning arrived at in the SenseAI app can be
    applied here. Searching for one is not this application's job; SenseAI ships
    an Optuna study for that.
    """

    enabled: bool = True
    schedule: RunSchedule = field(default_factory=RunSchedule)
    patch_shape: tuple = (20, 20)
    dict_size: int = 36
    iterations: int = 50

    def should_run(self, step: int) -> bool:
        return self.enabled and self.schedule.should_run(step)

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "schedule": self.schedule.to_dict(),
            "patch_shape": list(self.patch_shape),
            "dict_size": self.dict_size,
            "iterations": self.iterations,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReconstructionConfig":
        patch = data.get("patch_shape") or (20, 20)
        return cls(
            enabled=data.get("enabled", True),
            schedule=RunSchedule.from_dict(data.get("schedule", {})),
            patch_shape=tuple(patch),
            dict_size=data.get("dict_size", 36),
            iterations=data.get("iterations", 50),
        )


@dataclass
class SenseAIAcquisitionTaskConfig:
    """Acquire a sub-sampled image through the SenseAI scan generator.

    Unlike `AcquisitionTaskConfig` there is no `ImageSettings`: the scan
    generator owns resolution and dwell time, and the remaining viewing
    parameter (HFW) sits on the beam block with voltage, current and working
    distance. That is the arrangement `TaskImagingView` already describes for
    the milling task, and it leaves exactly one home for each value.
    """

    name: str = "SenseAI Acquisition Task"
    description: str = ""
    enabled: bool = True
    schedule: RunSchedule = field(default_factory=RunSchedule)
    state: MicroscopeState = field(default_factory=MicroscopeState)
    scan: SenseAIScanConfig = field(default_factory=SenseAIScanConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    autofocus: AutoFunctionConfig = field(
        default_factory=lambda: AutoFunctionConfig(
            area_of_interest=centred_area(AUTOFOCUS_AREA_FRACTION)
        )
    )
    autocontrast: AutoFunctionConfig = field(
        default_factory=lambda: AutoFunctionConfig(
            area_of_interest=centred_area(AUTOCONTRAST_AREA_FRACTION)
        )
    )
    # Machine-specific, set per step by configure() on the runtime copy.
    output_path: Optional[str] = None

    TASK_TYPE: ClassVar[str] = "senseai_acquisition"
    DISPLAY_NAME: ClassVar[str] = "SenseAI Acquisition"

    def should_run(self, step: int) -> bool:
        """See AcquisitionTaskConfig.should_run."""
        return self.enabled and self.schedule.should_run(step)

    def to_dict(self) -> dict:
        return {
            "type": self.TASK_TYPE,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "schedule": self.schedule.to_dict(),
            "state": self.state.to_dict(),
            "scan": self.scan.to_dict(),
            "reconstruction": self.reconstruction.to_dict(),
            "autofocus": self.autofocus.to_dict(),
            "autocontrast": self.autocontrast.to_dict(),
            "output_path": self.output_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SenseAIAcquisitionTaskConfig":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            schedule=RunSchedule.from_dict(data.get("schedule", {})),
            state=MicroscopeState.from_dict(data["state"]),
            scan=SenseAIScanConfig.from_dict(data.get("scan", {})),
            reconstruction=ReconstructionConfig.from_dict(
                data.get("reconstruction", {})
            ),
            autofocus=AutoFunctionConfig.from_dict(
                data.get("autofocus", {}),
                default_area=centred_area(AUTOFOCUS_AREA_FRACTION),
            ),
            autocontrast=AutoFunctionConfig.from_dict(
                data.get("autocontrast", {}),
                default_area=centred_area(AUTOCONTRAST_AREA_FRACTION),
            ),
            output_path=data.get("output_path"),
        )


TaskConfig = Union[
    AcquisitionTaskConfig, MillingTaskConfig, SenseAIAcquisitionTaskConfig
]


# ---------------------------------------------------------------------------
# The milling region: a volume drawn on the FIB view
# ---------------------------------------------------------------------------
#
# The operator picks the volume by dragging a box over the preview. That box is
# not stored: it is reconstructed from the pattern and the step count, so it
# redraws itself correctly on reload and there is no second copy to fall out of
# step with the protocol.
#
#     box width  <-> pattern width          the slice width
#     box height <-> n_steps x step_size    how far through the block to cut
#
# 'Depth' is deliberately avoided for the box height: a pattern's own `depth`
# is how far each cut reaches *into* the face, a fixed per-stage setting that
# has nothing to do with how big the region is. The box height is the extent
# the slab walks through, which is what sets the slice count.
#     box position <-> pattern.point        where on the face
#
# The pattern itself is one thin slab, `step_size` tall; the box is the extent
# it walks through.


@dataclass
class ImageGeometry:
    """Enough of a preview image to convert between metres and frame fractions.

    Frame fractions have their origin at the top-left and y increasing
    downwards (the overlay's convention); pattern metres have their origin at
    the frame centre with y increasing *upwards* (`draw_rectangle_shape`
    computes `cy = icy - centre_y / pixelsize`). The y flip lives here so no
    caller has to remember it.
    """

    pixel_size: float
    width_px: int
    height_px: int

    @classmethod
    def from_image(cls, image: FibsemImage) -> "ImageGeometry":
        height_px, width_px = image.data.shape[:2]
        return cls(
            pixel_size=image.metadata.pixel_size.x,
            width_px=width_px,
            height_px=height_px,
        )

    @property
    def width_m(self) -> float:
        return self.width_px * self.pixel_size

    @property
    def height_m(self) -> float:
        return self.height_px * self.pixel_size

    def to_pattern_x(self, fraction: float) -> float:
        return (fraction - 0.5) * self.width_m

    def to_pattern_y(self, fraction: float) -> float:
        return (0.5 - fraction) * self.height_m

    def to_fraction_x(self, x_m: float) -> float:
        return x_m / self.width_m + 0.5

    def to_fraction_y(self, y_m: float) -> float:
        return 0.5 - y_m / self.height_m


def _slice_pattern(config: MillingTaskConfig):
    """The stage that defines the slice geometry. First stage wins."""
    return config.milling.stages[0].pattern if config.milling.stages else None


def milling_region_for(
    config: MillingTaskConfig,
    n_steps: int,
    advance: VolumeAdvanceConfig,
    geometry: ImageGeometry,
) -> Optional[FibsemRectangle]:
    """The volume `config` will collect, as a rectangle on the frame."""
    pattern = _slice_pattern(config)
    if pattern is None:
        return None

    thickness = advance.step_size
    width_m = getattr(pattern, "width", thickness)
    extent_m = max(n_steps, 1) * thickness

    # The slab sits at the bottom of the volume on step 0 and walks up, because
    # `apply_advance` adds to y and pattern y is positive up the frame.
    y_bottom = pattern.point.y - thickness / 2.0
    y_top = y_bottom + extent_m

    return FibsemRectangle(
        left=geometry.to_fraction_x(pattern.point.x - width_m / 2.0),
        top=geometry.to_fraction_y(y_top),
        width=width_m / geometry.width_m,
        height=extent_m / geometry.height_m,
    )


def apply_milling_region(
    config: MillingTaskConfig,
    region: FibsemRectangle,
    advance: VolumeAdvanceConfig,
    geometry: ImageGeometry,
) -> int:
    """Set the slice pattern from `region`; return the slice count it implies.

    Mutates `config` in place and returns `n_steps` rather than setting it,
    because the step count is experiment-level — one volume, one count — and
    this function is only handed the task.
    """
    pattern = _slice_pattern(config)
    thickness = advance.step_size
    if pattern is None or thickness <= 0:
        return 1  # a volume always has at least one slice

    width_m = region.width * geometry.width_m
    extent_m = region.height * geometry.height_m

    centre_x = geometry.to_pattern_x(region.left + region.width / 2.0)
    y_bottom = geometry.to_pattern_y(region.top) - extent_m

    for stage in config.milling.stages:
        # Every stage cuts the same plane, so they share the box.
        if hasattr(stage.pattern, "width"):
            stage.pattern.width = width_m
        # Mutate the Point rather than replacing it: the editor binds its
        # controls to the object it found when the form was built.
        stage.pattern.point.x = centre_x
        stage.pattern.point.y = y_bottom + thickness / 2.0

    return max(1, round(extent_m / thickness))


# ---------------------------------------------------------------------------
# Moving a milling pattern
# ---------------------------------------------------------------------------


def advance_electron_beam_shift(
    config: "TaskConfig", step: int, advance: VolumeAdvanceConfig
) -> None:
    """Shift the SEM in y to track the block face as it recedes.

    Shared by every task that images the face with the electron column, so the
    two acquisition tasks cannot drift apart on how the volume advance is
    applied. Absolute rather than cumulative, like every other advance here: the
    stored config is the step-0 baseline.

    NOTE: this writes to the config's beam block, and something still has to
    apply that block to the column — see `apply_beam_settings`. Until the task's
    `run` does so, the shift computed here is discarded with the runtime copy.
    """
    if not advance.advance_beam_shift or step == 0:
        return
    beam = getattr(config.state, "electron_beam", None)
    if beam is None:
        return
    # A baseline shift of None means "no shift configured"; treat it as the
    # origin so the advance still tracks the face rather than silently
    # doing nothing.
    baseline = beam.shift if beam.shift is not None else Point(0.0, 0.0)
    beam.shift = Point(baseline.x, baseline.y + step * advance.step_size)


def advance_pattern_y(pattern, delta: float) -> None:
    """Move a milling pattern `delta` metres along y, in place.

    Patterns disagree about where their position lives. `RectanglePattern`
    builds its shape from `point` (`centre_x=self.point.x`), but
    `LinePattern.define()` ignores `point` entirely and uses only its
    `start_/end_` coordinates. Shifting `point` alone therefore moves a
    rectangle and does *nothing* to a line — and the default slice pattern is a
    line, so the whole volume would be cut in one plane.

    Handled here rather than in `fibsem.milling.patterning`, because those
    pattern classes are shared with applications whose contracts this change is
    not in a position to re-verify.

    Note the sign convention: pattern y is positive *up* the image
    (`draw_rectangle_shape` computes `cy = icy - centre_y / pixelsize`), so a
    positive delta walks the pattern up the frame.
    """
    if delta == 0.0:
        return

    # Endpoint-defined patterns: move the geometry define() actually reads, and
    # leave `point` alone so a future fix upstream cannot double-count it.
    if hasattr(pattern, "start_y") and hasattr(pattern, "end_y"):
        pattern.start_y += delta
        pattern.end_y += delta
        return

    point = getattr(pattern, "point", None)
    if point is None:
        logging.warning(
            f"Cannot advance pattern {type(pattern).__name__}: no point and no "
            "start_y/end_y. The volume will not advance for this stage."
        )
        return
    point.y += delta


# ---------------------------------------------------------------------------
# Microscope <-> config sync
# ---------------------------------------------------------------------------
#
# Moving imaging parameters between the column and the protocol, in either
# direction, with an image taken at the resulting settings so the operator can
# see what they got. No Qt in here: the UI is one caller, a script or a test is
# another, and the interesting logic should not be trapped behind a button.

# The parameters the protocol owns and the sync moves.
#
# Deliberately narrow. `BeamSettings` also carries hfw / resolution / dwell_time,
# but `ImageSettings` is what `acquire_image` is handed, so it owns those three
# and they are never copied into the beam block. One home each, so there is no
# question about which value wins at run time.
SYNCED_IMAGE_FIELDS = ("resolution", "dwell_time", "hfw")
SYNCED_BEAM_FIELDS = ("voltage", "beam_current", "working_distance")

# Beam fields applied to the microscope, in the order they are applied.
#
# `shift` is here even though it is not operator-editable: it is written by
# AcquisitionTask.apply_advance, and applying it is what makes
# `advance_beam_shift` track the receding face at all.
#
# `stigmation` and `scan_rotation` are deliberately absent. The protocol neither
# edits nor reads them back, so the only value they can ever hold is the artifact
# `BeamSettings.from_dict` creates: a null `stigmation` in YAML loads as
# `Point(0, 0)`, not None. Applying that would silently zero the stigmator on a
# column the operator had just tuned. They belong to live work on the Image tab.
_BEAM_SETTERS = (
    ("voltage", "set_beam_voltage", "V"),
    ("beam_current", "set_beam_current", "A"),
    ("working_distance", "set_working_distance", "m"),
    ("shift", "set_beam_shift", ""),
)


@dataclass
class MicroscopeSyncResult:
    """What a sync did, and the image it took afterwards.

    `image` is deliberately not saved to disk — see `acquire_preview_image`.
    """

    direction: str  # "from_microscope" or "to_microscope"
    image_settings: ImageSettings
    beam_settings: Optional[BeamSettings] = None
    changes: List[str] = field(default_factory=list)
    image: Optional[FibsemImage] = None

    @property
    def summary(self) -> str:
        """Human-readable, with SI prefixes. For display."""
        verb = "Read from" if self.direction == "from_microscope" else "Applied to"
        if not self.changes:
            return f"{verb} microscope: already in sync."
        return f"{verb} microscope: " + ", ".join(self.changes)

    @property
    def log_summary(self) -> str:
        """`summary` with the micro sign spelled out, for logging.

        `format_value` emits U+03BC, and a logging handler on a cp1252 console
        raises UnicodeEncodeError on it. That is swallowed by logging's own error
        handling rather than crashing, but it corrupts the log — which matters
        here, because these runs last days and the log is the record of them.
        """
        return self.summary.replace("μ", "u")


def beam_block_for(
    config: "TaskConfig", beam_type: Optional[BeamType] = None
) -> BeamSettings:
    """The beam block for `beam_type` on this task, created if absent.

    `MicroscopeState` holds one block per column and both are Optional, so a
    protocol whose YAML has a null beam block would otherwise have nowhere to
    put the values read back from the microscope.
    """
    if beam_type is None:
        image = getattr(config, "image", None)
        beam_type = (
            image.beam_type if image is not None else None
        ) or BeamType.ELECTRON
    attr = "electron_beam" if beam_type is BeamType.ELECTRON else "ion_beam"
    beam = getattr(config.state, attr, None)
    if beam is None:
        beam = BeamSettings(beam_type=beam_type)
        setattr(config.state, attr, beam)
    return beam


@dataclass
class TaskImagingView:
    """Where a task keeps the imaging parameters a sync moves, whichever task it is.

    The two task types keep them in different places, and the sync should not
    have to know which it is looking at.

    An acquisition task owns an `ImageSettings` that `acquire_image` is handed
    directly, so resolution / dwell time / HFW live there.

    A milling task images the ion column only to see where to mill. Its
    `ImageSettings` blocks belong to the milling engine — alignment references
    and pre/post-task images — and are rewritten as the task runs, so they are
    the wrong place for a protocol to state its viewing parameters. It keeps
    them on the ion beam block instead, which carries the same three fields.
    """

    beam_type: BeamType
    beam: BeamSettings
    # Holds resolution / dwell_time / hfw. The beam block itself for a milling
    # task, a separate ImageSettings for an acquisition task.
    image_owner: Any

    @property
    def owns_imaging_on_beam(self) -> bool:
        return self.image_owner is self.beam

    def to_image_settings(self) -> ImageSettings:
        """The imaging parameters as an ImageSettings, wherever they are held."""
        if isinstance(self.image_owner, ImageSettings):
            settings = deepcopy(self.image_owner)
            settings.beam_type = self.beam_type
            return settings

        # A BeamSettings leaves these unset by default; fall back to the same
        # defaults an ImageSettings would have rather than passing None on.
        defaults = ImageSettings()
        return ImageSettings(
            beam_type=self.beam_type,
            resolution=tuple(self.image_owner.resolution or defaults.resolution),
            dwell_time=self.image_owner.dwell_time or defaults.dwell_time,
            hfw=self.image_owner.hfw or defaults.hfw,
        )


def imaging_view(config: "TaskConfig") -> TaskImagingView:
    """Locate the imaging parameters on `config`. See `TaskImagingView`."""
    if getattr(config, "TASK_TYPE", None) == MillingTaskConfig.TASK_TYPE:
        beam = beam_block_for(config, BeamType.ION)
        return TaskImagingView(beam_type=BeamType.ION, beam=beam, image_owner=beam)

    # A task with no ImageSettings keeps its viewing parameters on the beam
    # block, the same arrangement as the milling task. The SenseAI task does
    # this because the external scan generator owns resolution and dwell time.
    if getattr(config, "image", None) is None:
        beam = beam_block_for(config, BeamType.ELECTRON)
        return TaskImagingView(beam_type=BeamType.ELECTRON, beam=beam, image_owner=beam)

    beam_type = config.image.beam_type or BeamType.ELECTRON
    return TaskImagingView(
        beam_type=beam_type,
        beam=beam_block_for(config, beam_type),
        image_owner=config.image,
    )


def apply_beam_settings(
    microscope: FibsemMicroscope,
    beam_settings: Optional[BeamSettings],
    include_shift: bool = True,
) -> List[str]:
    """Apply the fields of `beam_settings` that are set, and nothing else.

    Returns a description of what was applied.

    `include_shift` is for the sync, which should not touch it. Beam shift is
    run state, written per step by `AcquisitionTask.apply_advance` to track the
    receding face — the protocol only ever holds the step-0 baseline. Worse, a
    null `shift` in YAML loads as `Point(0, 0)` rather than None, so applying it
    during a sync would silently re-centre a column the operator had just
    shifted onto the ROI. The run path wants it; moving settings around does not.

    Not `microscope.set_beam_settings`: that forwards every field straight into a
    setter including `None`, which on ThermoMicroscope assigns `None` into
    AutoScript, on the simulator silently corrupts the stored state, and on both
    raises once it reaches `set_scan_rotation(None)`. A Salami protocol leaves
    most beam fields unset by design — unset means "leave the column alone" —
    so almost every field would be `None` on a fresh protocol.

    Not `microscope.set_microscope_state` either: it calls
    `safe_absolute_stage_movement` whenever `state.stage_position` is not None,
    and `MicroscopeState.stage_position` defaults to a *non*-None
    `FibsemStagePosition` with all-None axes. That guard therefore always passes,
    and the all-None position reaches `movement.angle_difference`, which raises
    TypeError on `None % float`. Slice-and-view must never move the stage
    mid-volume in any case.
    """
    if beam_settings is None:
        return []

    beam_type = beam_settings.beam_type or BeamType.ELECTRON
    applied: List[str] = []
    for name, setter_name, unit in _BEAM_SETTERS:
        if name == "shift" and not include_shift:
            continue
        value = getattr(beam_settings, name, None)
        if value is None:
            continue
        try:
            getattr(microscope, setter_name)(value, beam_type)
        except Exception:
            logging.warning(f"Could not apply {name}={value!r}", exc_info=True)
            continue
        shown = (
            format_value(value, unit)
            if unit and isinstance(value, (int, float))
            else value
        )
        applied.append(f"{name.replace('_', ' ')}={shown}")
    return applied


def acquire_preview_image(
    microscope: FibsemMicroscope, image_settings: ImageSettings
) -> FibsemImage:
    """Take one image at `image_settings` without writing it to disk.

    A sync is a preview, not a slice. Saving it would drop an extra image into
    the task's output directory, where it would be indistinguishable from run
    data — and the protocol's `path` is only set per-step at run time anyway, so
    saving could also land it in the wrong place entirely.

    Goes through `fibsem.acquire.acquire_image`, so autocontrast and autogamma
    behave exactly as they do during a run.
    """
    settings = deepcopy(image_settings)
    settings.save = False
    settings.path = None
    return acquire_image(microscope, settings)


def _copy_fields(source, target, names, unit_by_field=None) -> List[str]:
    """Copy `names` from source to target, describing what actually changed."""
    unit_by_field = unit_by_field or {}
    changes: List[str] = []
    for name in names:
        new = getattr(source, name, None)
        if new is None:
            continue
        old = getattr(target, name, None)
        # Resolution is declared a tuple but arrives from YAML as a list.
        if isinstance(new, (list, tuple)) and isinstance(old, (list, tuple)):
            unchanged = tuple(new) == tuple(old)
            new = tuple(new)
        else:
            unchanged = old == new
        if unchanged:
            continue
        setattr(target, name, new)
        unit = unit_by_field.get(name)
        shown = (
            format_value(new, unit) if unit and isinstance(new, (int, float)) else new
        )
        changes.append(f"{name.replace('_', ' ')}={shown}")
    return changes


_IMAGE_UNITS = {"dwell_time": "s", "hfw": "m"}
_BEAM_UNITS = {"voltage": "V", "beam_current": "A", "working_distance": "m"}


def sync_config_from_microscope(
    microscope: FibsemMicroscope,
    config: "TaskConfig",
    acquire: bool = True,
) -> MicroscopeSyncResult:
    """Read the column's current imaging and beam settings into `config`.

    The operator's workflow: tune the microscope live until the image looks
    right, then capture those settings into the protocol. Works for either task
    type — see `TaskImagingView` for where each keeps them.

    Mutates `config` in place and reports what changed. Nothing is written to
    disk — persisting the protocol is the caller's decision.
    """
    view = imaging_view(config)

    live_image = microscope.get_imaging_settings(view.beam_type)
    live_beam = microscope.get_beam_settings(view.beam_type)

    changes = _copy_fields(
        live_image, view.image_owner, SYNCED_IMAGE_FIELDS, _IMAGE_UNITS
    )
    changes += _copy_fields(live_beam, view.beam, SYNCED_BEAM_FIELDS, _BEAM_UNITS)

    image_settings = view.to_image_settings()
    image = acquire_preview_image(microscope, image_settings) if acquire else None

    result = MicroscopeSyncResult(
        direction="from_microscope",
        image_settings=image_settings,
        beam_settings=deepcopy(view.beam),
        changes=changes,
        image=image,
    )
    logging.info(result.log_summary)
    return result


def sync_config_to_microscope(
    microscope: FibsemMicroscope,
    config: "TaskConfig",
    acquire: bool = True,
) -> MicroscopeSyncResult:
    """Apply `config`'s imaging and beam settings to the column.

    The other half of the operator's workflow: put the protocol back on the
    microscope to see what it will actually do. This is also exactly what the
    task needs to do at run time, which is why it lives here rather than in the UI.

    Only viewing parameters are applied. Milling current and pattern geometry
    are never touched — those belong to the milling stages, which the milling
    engine sets up when the task runs.
    """
    view = imaging_view(config)
    image_settings = view.to_image_settings()

    changes: List[str] = []
    microscope.set_imaging_settings(image_settings)
    changes.append(f"resolution={tuple(image_settings.resolution)}")
    changes.append(f"dwell time={format_value(image_settings.dwell_time, 's')}")
    changes.append(f"hfw={format_value(image_settings.hfw, 'm')}")
    changes += apply_beam_settings(microscope, view.beam, include_shift=False)

    image = acquire_preview_image(microscope, image_settings) if acquire else None

    result = MicroscopeSyncResult(
        direction="to_microscope",
        image_settings=image_settings,
        beam_settings=deepcopy(view.beam),
        changes=changes,
        image=image,
    )
    logging.info(result.log_summary)
    return result


def resolve_beam_defaults(
    microscope: FibsemMicroscope, config: "TaskConfig"
) -> List[str]:
    """Give unset imaging fields a concrete value the column supports.

    The editor shows these without an "unset" entry, so the config has to hold a
    real value for what is displayed to be true. Enumerable fields take the
    first value the column offers; the rest are seeded from what the column is
    currently doing.

    Returns the fields that were filled in, so the caller knows to persist.
    """
    view = imaging_view(config)
    beam = view.beam
    filled: List[str] = []

    for name, key in (("voltage", "voltage"), ("beam_current", "current")):
        if getattr(beam, name, None) is not None:
            continue
        try:
            values = microscope.get_available_values_cached(key, view.beam_type)
        except Exception:
            logging.debug(f"Could not read available {key!r} values", exc_info=True)
            continue
        if not values:
            continue
        setattr(beam, name, values[0])
        filled.append(name)

    # Working distance is not enumerable, so seed it from the column's current
    # value rather than inventing one.
    if beam.working_distance is None:
        try:
            beam.working_distance = microscope.get_working_distance(view.beam_type)
            filled.append("working_distance")
        except Exception:
            logging.debug("Could not read the current working distance", exc_info=True)

    # A milling task keeps its viewing parameters on the beam block, where they
    # start unset. An acquisition task's ImageSettings already has real defaults.
    if view.owns_imaging_on_beam:
        try:
            live = microscope.get_imaging_settings(view.beam_type)
        except Exception:
            logging.debug("Could not read the current imaging settings", exc_info=True)
        else:
            for name in SYNCED_IMAGE_FIELDS:
                if getattr(beam, name, None) is None:
                    setattr(beam, name, getattr(live, name))
                    filled.append(name)

    if filled:
        logging.info(f"Filled unset imaging settings from the microscope: {filled}")
    return filled


# ---------------------------------------------------------------------------
# Task ABC + registry
# ---------------------------------------------------------------------------


class BaseTask(ABC):
    """Base class for a task. Subclass it, set CONFIG_CLS, and add @register_task."""

    CONFIG_CLS: ClassVar[Type]

    def __init__(self, config: TaskConfig):
        self.config = config

    def output_dir(self, experiment_path: str) -> str:
        """Directory this task writes its output into."""
        return os.path.join(experiment_path, self.config.name)

    def configure(self, experiment_path: str) -> None:
        """Set runtime output paths on the config. Default no-op.

        Called on a per-step *runtime copy* of the config, never on the stored
        protocol — runtime paths are machine-specific and must not be persisted.
        """

    def prepare_for_reuse(self) -> None:
        """Strip everything tying this config to one sample or one machine.

        Called when a config is reused as a template for a new experiment.
        Stage position is sample-specific and lives in the run status, not the
        protocol; output paths are machine-specific and are set at run time.
        """
        state = getattr(self.config, "state", None)
        if state is not None:
            state.stage_position = None
        # An operator disabling a task mid-run is a decision about *that* run
        # (a failed detector, say). It should not silently follow the protocol
        # onto the next sample.
        self.config.enabled = True
        self._clear_runtime_paths()

    def _clear_runtime_paths(self) -> None:
        """Reset persisted output paths. Default no-op."""

    @abstractmethod
    def run(self, microscope: FibsemMicroscope, context: RunContext) -> None: ...

    def apply_slice_thickness(self, advance: VolumeAdvanceConfig) -> None:
        """Size this config's geometry to one slice. Default no-op."""

    def apply_advance(self, step: int, advance: VolumeAdvanceConfig) -> None:
        """Offset this config's geometry for `step`. Default no-op.

        Absolute, not cumulative: the stored config is always the step-0
        baseline and the offset is `step * advance.step_size`. That makes
        resuming at an arbitrary step land at the right depth, and avoids
        accumulating float error over thousands of slices.
        """


_TASK_CLASSES: dict = {}


def register_task(cls):
    """Class decorator: register a BaseTask subclass against its config's TASK_TYPE."""
    key = cls.CONFIG_CLS.TASK_TYPE
    _TASK_CLASSES[key] = cls
    return cls


def _task_for(task_config: TaskConfig) -> BaseTask:
    cls = _TASK_CLASSES.get(task_config.TASK_TYPE)
    if cls is None:
        raise ValueError(
            f"Unsupported task type: {task_config.TASK_TYPE!r}. "
            f"Known: {list(_TASK_CLASSES)}"
        )
    return cls(config=task_config)


def task_from_dict(data: dict) -> TaskConfig:
    t = data.get("type")
    cls = _TASK_CLASSES.get(t)
    if cls is None:
        raise ValueError(f"Unknown task type: {t!r}. Known: {list(_TASK_CLASSES)}")
    return cls.CONFIG_CLS.from_dict(data)


def available_task_types() -> List[str]:
    """Task types a protocol can contain, in registration order.

    The registry is the single source of truth, so a newly registered task type
    appears in the UI without anything else being edited.
    """
    return list(_TASK_CLASSES)


def task_type_label(task_type: str) -> str:
    """Human-readable name for a task type, for a dropdown."""
    cls = _TASK_CLASSES.get(task_type)
    if cls is None:
        return task_type
    return getattr(cls.CONFIG_CLS, "DISPLAY_NAME", task_type)


def create_task_config(task_type: str, name: str) -> TaskConfig:
    """A default config of `task_type`, named `name`.

    Nothing is written to disk and nothing is added to a protocol — the caller
    decides where it goes, the same split `SalamiExperiment.create` uses.
    """
    cls = _TASK_CLASSES.get(task_type)
    if cls is None:
        raise ValueError(
            f"Unknown task type: {task_type!r}. Known: {list(_TASK_CLASSES)}"
        )
    return cls.CONFIG_CLS(name=name)


def suggest_task_name(task_type: str, existing: List[str]) -> str:
    """A default name for a new task that does not collide with `existing`.

    Task names are directory names — `BaseTask.output_dir` joins the name onto
    the experiment path — so two tasks sharing one would write into the same
    folder and `find_task` could only ever return the first.
    """
    base = task_type_label(task_type).replace(" ", "_") + "_Task"
    if base not in existing:
        return base
    n = 2
    while f"{base}_{n}" in existing:
        n += 1
    return f"{base}_{n}"


def configure_task(task_config: TaskConfig, experiment_path: str) -> None:
    _task_for(task_config).configure(experiment_path)


def prepare_task_for_reuse(task_config: TaskConfig) -> None:
    """Strip sample- and machine-specific fields so `task_config` can be reused."""
    _task_for(task_config).prepare_for_reuse()


def ensure_task_directories(task_config: TaskConfig, experiment_path: str) -> str:
    """Create this task's output directory without touching its config."""
    path = _task_for(task_config).output_dir(experiment_path)
    os.makedirs(path, exist_ok=True)
    return path


def run_task(
    microscope: FibsemMicroscope,
    task_config: TaskConfig,
    context: RunContext,
) -> None:
    _task_for(task_config).run(microscope, context)


def apply_slice_thickness(
    task_config: TaskConfig,
    advance_config: VolumeAdvanceConfig,
) -> None:
    """Size `task_config`'s geometry to one slice, in place.

    Intended for a runtime copy — see `MillingTask.apply_slice_thickness`.
    """
    _task_for(task_config).apply_slice_thickness(advance_config)


def apply_advance(
    task_config: TaskConfig,
    step: int,
    advance_config: VolumeAdvanceConfig,
) -> None:
    """Offset `task_config` in place to the geometry for `step`.

    Intended for a runtime copy — see `BaseTask.apply_advance`.
    """
    _task_for(task_config).apply_advance(step, advance_config)


def runtime_task_config(
    task_config: TaskConfig,
    experiment_path: str,
    step: int,
    advance_config: VolumeAdvanceConfig,
) -> TaskConfig:
    """Build the config actually used to run `task_config` at `step`.

    The stored protocol stays pristine: this returns a copy carrying the
    machine-specific output paths and the derived slice geometry.
    """
    cfg = deepcopy(task_config)
    configure_task(cfg, experiment_path)
    apply_slice_thickness(cfg, advance_config)
    apply_advance(cfg, step, advance_config)
    return cfg


# ---------------------------------------------------------------------------
# Concrete tasks
# ---------------------------------------------------------------------------


@register_task
class AcquisitionTask(BaseTask):
    CONFIG_CLS = AcquisitionTaskConfig

    def configure(self, experiment_path: str) -> None:
        self.config.image.path = self.output_dir(experiment_path)
        os.makedirs(self.config.image.path, exist_ok=True)

    def _clear_runtime_paths(self) -> None:
        self.config.image.path = None

    def run(self, microscope: FibsemMicroscope, context: RunContext) -> None:
        cfg = self.config
        step = context.current_step

        # figure out the state stuff, lots of none setting

        # microscope.set_microscope_state(cfg.state)

        if cfg.autofocus.should_run(step) or cfg.autocontrast.should_run(step):
            context.emit(
                SalamiWorkflowUpdate(
                    task="alignment",
                    msg=f"Auto-functions for {cfg.name!r} (step {step + 1})",
                    current_step=step,
                    total_steps=context.total_steps,
                    current_task_index=context.task_index,
                    total_tasks=context.total_tasks,
                    task_name=cfg.name,
                    stage="alignment",
                    elapsed_time=context.elapsed(),
                )
            )
            # Each function uses its own region. `cfg.alignment_area` is reserved
            # for fiducial drift correction and is deliberately not used here.
            if cfg.autofocus.should_run(step):
                auto_focus_basic(
                    microscope, cfg.image.beam_type, cfg.autofocus.area_of_interest
                )
            if cfg.autocontrast.should_run(step):
                auto_contrast_basic(
                    microscope, cfg.image.beam_type, cfg.autocontrast.area_of_interest
                )

        context.emit(
            SalamiWorkflowUpdate(
                task="acquisition",
                msg=f"Acquiring {cfg.name!r} at step {step + 1}/{context.total_steps}",
                current_step=step,
                total_steps=context.total_steps,
                current_task_index=context.task_index,
                total_tasks=context.total_tasks,
                task_name=cfg.name,
                stage="acquisition",
                elapsed_time=context.elapsed(),
            )
        )

        cfg.image.save = True
        cfg.image.autocontrast = False
        cfg.image.autogamma = False
        cfg.image.filename = f"{cfg.name}-Image-{step:03d}"
        image = microscope.acquire_image(cfg.image)
        image.save(os.path.join(cfg.image.path, cfg.image.filename))  # type: ignore

        # cfg.state = microscope.get_microscope_state(cfg.image.beam_type)

        if context.parent_ui is not None:
            microscope.sem_acquisition_signal.emit(image)

        image_shape = (
            (image.data.shape[0], image.data.shape[1])
            if image.data is not None
            else None
        )
        context.emit(
            SalamiWorkflowUpdate(
                task="acquisition",
                msg=f"Completed {cfg.name!r} at step {step + 1}/{context.total_steps}",
                current_step=step,
                total_steps=context.total_steps,
                current_task_index=context.task_index,
                total_tasks=context.total_tasks,
                task_name=cfg.name,
                stage="acquisition",
                image_filename=cfg.image.filename,
                image_shape=image_shape,
                elapsed_time=context.elapsed(),
            )
        )

    def apply_advance(self, step: int, advance: VolumeAdvanceConfig) -> None:
        """Shift the SEM in y to track the block face as it recedes."""
        advance_electron_beam_shift(self.config, step, advance)


@register_task
class MillingTask(BaseTask):
    CONFIG_CLS = MillingTaskConfig

    def configure(self, experiment_path: str) -> None:
        self.config.milling.acquisition.imaging.path = self.output_dir(experiment_path)
        os.makedirs(self.config.milling.acquisition.imaging.path, exist_ok=True)

    def _clear_runtime_paths(self) -> None:
        self.config.milling.acquisition.imaging.path = None

    def run(self, microscope: FibsemMicroscope, context: RunContext) -> None:
        cfg = self.config
        step = context.current_step

        print(
            f"Running milling task {cfg.name!r} at step {step + 1}/{context.total_steps}"
        )

        context.emit(
            SalamiWorkflowUpdate(
                task="milling",
                msg=f"Milling {cfg.name!r} at step {step + 1}/{context.total_steps}",
                current_step=step,
                total_steps=context.total_steps,
                current_task_index=context.task_index,
                total_tasks=context.total_tasks,
                task_name=cfg.name,
                stage="milling",
                elapsed_time=context.elapsed(),
            )
        )

        # microscope.set_microscope_state(cfg.state)
        # NB: the output path is set by configure() on the runtime copy; don't
        # overwrite it with the experiment root here.

        t0 = time.time()
        run_milling_task(
            microscope=microscope,
            config=cfg.milling,
            parent_ui=context.parent_ui,
        )
        milling_time = time.time() - t0

        context.emit(
            SalamiWorkflowUpdate(
                task="milling",
                msg=f"Completed milling {cfg.name!r} at step {step + 1}/{context.total_steps}",
                current_step=step,
                total_steps=context.total_steps,
                current_task_index=context.task_index,
                total_tasks=context.total_tasks,
                task_name=cfg.name,
                stage="milling",
                milling_time=milling_time,
                elapsed_time=context.elapsed(),
            )
        )

    def apply_slice_thickness(self, advance: VolumeAdvanceConfig) -> None:
        """Make each stage remove a slab exactly one slice thick.

        The height of the slab cut and the distance advanced afterwards are the
        same quantity, so `advance.step_size` is the single source of truth and
        the pattern height follows it. Letting them drift apart would leave
        unmilled material between slices, or re-cut what the previous step had
        already removed — either way silently, and only visible once the stack
        is reconstructed.

        Patterns that express their extent some other way (a trench has upper
        and lower heights, a line has none at all) are left alone rather than
        guessed at.
        """
        for stage in self.config.milling.stages:
            if hasattr(stage.pattern, "height"):
                stage.pattern.height = advance.step_size

    def apply_advance(self, step: int, advance: VolumeAdvanceConfig) -> None:
        """Walk the slice pattern into the block.

        Every stage advances, not just the first: a slice task with a coarse
        and a fine stage cuts both at the same plane.
        """
        if not advance.advance_milling_pattern or step == 0:
            return
        for stage in self.config.milling.stages:
            advance_pattern_y(stage.pattern, step * advance.step_size)


def senseai_preview_image(
    frame: "np.ndarray", cfg: "SenseAIAcquisitionTaskConfig", beam
) -> FibsemImage:
    """The measured frame as a `FibsemImage` the viewer can actually display.

    Real metadata is required, not decoration: the image widget reads
    `metadata.beam_type` to choose which pane to put the frame in, and the
    canvas reads `pixel_size` for its scale bar. A bare `FibsemImage(array)`
    fails both — it raises "Image metadata is None" and the frame is dropped.

    The generator decides the geometry while the scan is external, so the
    resolution here is the scan size rather than anything on the column.
    """
    width, height = int(cfg.scan.scan_size[0]), int(cfg.scan.scan_size[1])
    hfw = getattr(beam, "hfw", None) if beam is not None else None

    image_settings = ImageSettings(
        beam_type=BeamType.ELECTRON,
        resolution=(width, height),
        dwell_time=cfg.scan.dwell_time,
        hfw=hfw if hfw is not None else ImageSettings().hfw,
        save=False,
    )
    pixel_size = image_settings.hfw / width if width else 0.0

    return FibsemImage(
        data=frame,
        metadata=FibsemImageMetadata(
            image_settings=image_settings,
            pixel_size=Point(pixel_size, pixel_size),
            microscope_state=MicroscopeState(),
        ),
    )


@register_task
class SenseAIAcquisitionTask(BaseTask):
    """Acquire a sub-sampled frame through the SenseAI scan generator, and
    optionally reconstruct it, as one step of the volume.

    The saved artifact is one 4-page TIFF per slice: Input and Mask are the
    measurement and are written losslessly, so any slice can be reconstructed
    again later even when the in-run reconstruction is scheduled off.
    """

    CONFIG_CLS = SenseAIAcquisitionTaskConfig

    def configure(self, experiment_path: str) -> None:
        self.config.output_path = self.output_dir(experiment_path)
        os.makedirs(self.config.output_path, exist_ok=True)

    def _clear_runtime_paths(self) -> None:
        self.config.output_path = None

    def run(self, microscope: FibsemMicroscope, context: RunContext) -> None:
        # Imported here rather than at module scope: SenseAI is one site's
        # hardware, and a Salami install without it should still be able to load
        # tasks.py and run a protocol that does not use this task.
        from fibsem.SenseAI_SG import (
            INPUT_PAGE,
            RECONSTRUCTION_PAGE,
            build_capture_stack,
            reconstruction_to_uint16,
            require_senseai_module,
            save_capture_stack,
            set_external_scan,
            set_internal_scan,
        )

        cfg = self.config
        step = context.current_step
        beam_type = BeamType.ELECTRON

        # The module the SenseAI tab initialised, carried on the context so every
        # slice reuses the one connection.
        #
        # This task never loads the DLL or initialises a generator. Both happen
        # once per process, from the SenseAI tab (or from whatever built the
        # module for a headless run). Everything the task is allowed to do is
        # per-slice: update the generator's config, take the scan external, and
        # give it back. Creating a second module here would collide with the
        # first one's sources in the DLL registry and leave a generator that
        # never returns a frame — so a missing one is an error, not a prompt to
        # build another.
        module = require_senseai_module(context.senseai_module)

        # Auto-functions first, while the column still owns its own scan. They
        # drive the internal scan engine, so running them in external mode would
        # have them fighting the generator for the beam.
        if cfg.autofocus.should_run(step) or cfg.autocontrast.should_run(step):
            context.emit(
                SalamiWorkflowUpdate(
                    task="alignment",
                    msg=f"Auto-functions for {cfg.name!r} (step {step + 1})",
                    current_step=step,
                    total_steps=context.total_steps,
                    current_task_index=context.task_index,
                    total_tasks=context.total_tasks,
                    task_name=cfg.name,
                    stage="alignment",
                    elapsed_time=context.elapsed(),
                )
            )
            if cfg.autofocus.should_run(step):
                auto_focus_basic(microscope, beam_type, cfg.autofocus.area_of_interest)
            if cfg.autocontrast.should_run(step):
                auto_contrast_basic(
                    microscope, beam_type, cfg.autocontrast.area_of_interest
                )

        # Beam block: voltage, current, working distance, and the beam shift
        # that tracks the receding face. HFW is not in `_BEAM_SETTERS`, so it is
        # applied separately.
        beam = getattr(cfg.state, "electron_beam", None)
        applied = apply_beam_settings(microscope, beam)
        if beam is not None and beam.hfw is not None:
            microscope.set_field_of_view(beam.hfw, beam_type)
            applied.append(f"hfw={format_value(beam.hfw, 'm')}")
        if applied:
            logging.info(f"Applied to the column: {', '.join(applied)}")

        context.emit(
            SalamiWorkflowUpdate(
                task="acquisition",
                msg=f"Acquiring {cfg.name!r} at step {step + 1}/{context.total_steps}",
                current_step=step,
                total_steps=context.total_steps,
                current_task_index=context.task_index,
                total_tasks=context.total_tasks,
                task_name=cfg.name,
                stage="acquisition",
                elapsed_time=context.elapsed(),
            )
        )

        module.update_ScanGenerator(cfg.scan.apply_to(module.config.scanGen_config))

        # Set external as late as possible and always give the scan back, so a
        # failure here cannot leave the column unable to image.
        set_external_scan(microscope, beam_type)
        try:
            output = module.acquire_image()
        finally:
            set_internal_scan(microscope, beam_type)

        if output is None:
            raise RuntimeError(
                f"{cfg.name!r}: the SenseAI scan generator returned no frame at "
                f"step {step + 1}. Check the scan generator is initialised."
            )

        stack = build_capture_stack(output[0], output[1])

        reconstructed = False
        if cfg.reconstruction.should_run(step):
            recon = module.reconstruct(
                stack,
                iterations=cfg.reconstruction.iterations,
                patch_shape=tuple(cfg.reconstruction.patch_shape) + (1, 1),
                dict_size=cfg.reconstruction.dict_size,
            )
            if recon is None:
                logging.warning(
                    f"{cfg.name!r}: reconstruction failed at step {step + 1}."
                )
            else:
                stack[RECONSTRUCTION_PAGE] = reconstruction_to_uint16(recon)
                reconstructed = True

        filename = f"{cfg.name}-Image-{step:03d}.tiff"
        save_capture_stack(os.path.join(cfg.output_path, filename), stack)

        # Show the reconstruction when there is one. The Input page is the raw
        # measurement — at 30% sampling that is a sparse scatter of pixels, which
        # is close to unreadable as a check that the slice came out. The
        # reconstruction is the dense image, and watching the volume build is
        # most of why it runs during an experiment at all.
        #
        # Reconstruction has its own cadence, so on the slices in between this
        # falls back to the measurement; the caption says which, otherwise a
        # sparse frame looks like something went wrong.
        preview_page = RECONSTRUCTION_PAGE if reconstructed else INPUT_PAGE
        shown = "reconstruction" if reconstructed else "measurement"

        context.emit(
            SalamiWorkflowUpdate(
                task="acquisition",
                msg=(
                    f"Completed {cfg.name!r} at step {step + 1}/{context.total_steps} "
                    f"(showing {shown})"
                ),
                current_step=step,
                total_steps=context.total_steps,
                current_task_index=context.task_index,
                total_tasks=context.total_tasks,
                task_name=cfg.name,
                stage="acquisition",
                image_filename=filename,
                image_shape=(stack.shape[1], stack.shape[2]),
                elapsed_time=context.elapsed(),
                # Carried on the workflow update rather than emitted on
                # `microscope.sem_acquisition_signal`. That signal is a psygnal,
                # so its handlers run in the emitting thread — and the handler
                # here is the image widget, which draws on the quad-view canvas.
                # Emitting it from the run's worker thread put Qt drawing on a
                # non-GUI thread. The workflow signal is queued to the main
                # thread, which is where widgets may be touched.
                sem_image=senseai_preview_image(stack[preview_page], cfg, beam),
            )
        )

    def apply_advance(self, step: int, advance: VolumeAdvanceConfig) -> None:
        """Shift the SEM in y to track the block face as it recedes."""
        advance_electron_beam_shift(self.config, step, advance)


def auto_focus_basic(
    microscope: FibsemMicroscope, beam_type: BeamType, alignment_area: FibsemRectangle
) -> None:
    ## basic auto-focus implementation
    microscope.auto_focus(beam_type, alignment_area)


def auto_contrast_basic(
    microscope: FibsemMicroscope, beam_type: BeamType, alignment_area: FibsemRectangle
) -> None:
    ### basic auto-contrast implementation
    microscope.autocontrast(beam_type, alignment_area)
