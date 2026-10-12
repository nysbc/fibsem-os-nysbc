######## NEEDLE LIFTOUT TASK DEFINITIONS ########

"""General, config-driven needle (manipulator) liftout task.

Rather than a new hard-coded task every time a needle workflow varies, this
expresses the shape they all share as data:

    move to position
      -> acquire reference images
      -> detect the target(s) ONCE, before the needle can occlude them
      -> insert needle
      -> for each HFW step:
             image at the step's field of view
             detect + move the needle until within tolerance   (optional)
             run the step's milling                             (optional)
             blind relative move                                (optional)
      -> retract needle                                         (optional)
      -> acquire reference images

Each iteration of the loop is a :class:`NeedleLoopStep`, defined by the field of
view it runs at. The task ships with one neutral step and no milling: the protocol
is built in the editor -- add steps, tick Enable Milling on the ones that mill --
rather than starting from a recipe for one particular sample.

A step's milling is an entry in ``config.milling`` keyed by the step
(:attr:`NeedleLoopStep.key`), so every step gets the full milling editor and canvas.
Ticking Enable Milling creates the entry (:meth:`NeedleLiftoutTaskConfig.sync_step_milling`);
nothing ever deletes one, so renaming or unticking a step cannot lose its patterns.

The needle mechanics -- axis conventions, the convergence loop -- are the free
functions in :mod:`fibsem.applications.autolamella.workflows.needle`. This module
is the step engine around them.
"""

import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np

from fibsem import config as fcfg
from fibsem import constants
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows.core import align_feature_coincident
from fibsem.applications.autolamella.workflows.needle import (
    PARK_POSITION,
    NeedleConvergence,
    detect_target,
    move_needle,
    move_needle_to_feature,
)
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
from fibsem.applications.autolamella.workflows.ui import ask_user, update_detection_ui
from fibsem.detection import detection
from fibsem.detection.detection import DetectedFeatures, Feature
from fibsem.milling import FibsemMillingStage
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import BeamType, FibsemImage, Point, field_meta

# ---------------------------------------------------------------------------
# Feature name helpers
# ---------------------------------------------------------------------------

#: Every ML feature the detection module knows about, for the editor's combos.
FEATURE_NAMES: Tuple[str, ...] = tuple(
    sorted(feature.__name__ for feature in detection.__FEATURES__)
)

#: "" means "not configured", so it is offered first.
FEATURE_CHOICES: Tuple[str, ...] = ("",) + FEATURE_NAMES

BEAM_CHOICES: Tuple[str, ...] = ("ELECTRON", "ION")

ORIENTATION_CHOICES: Tuple[str, ...] = ("FIB", "SEM", "MILLING")


def _feature_label(name: Any) -> str:
    """How a feature choice reads in a combo: the empty choice says what it means."""
    return str(name) if name else "(none)"


def resolve_feature_name(name: str, flip: bool = False) -> Optional[Feature]:
    """Resolve a feature by class name, optionally swapping Top <-> Bottom.

    The same physical edge appears at the top or the bottom of the image depending
    on scan rotation. Deriving the flipped name keeps the protocol flat -- one
    string per feature -- rather than needing a second field for the pair.

    Args:
        name: feature class name, e.g. ``"VolumeBlockBottomEdge"``. Empty returns
            None (not configured).
        flip: swap Top/Bottom in the name, when the flipped feature exists.

    Raises:
        ValueError: ``name`` is not a known feature.
    """
    if not name:
        return None

    if flip:
        if "Top" in name:
            flipped = name.replace("Top", "Bottom")
        elif "Bottom" in name:
            flipped = name.replace("Bottom", "Top")
        else:
            flipped = name  # no vertical component, nothing to flip
        if flipped in FEATURE_NAMES:
            name = flipped

    if name not in FEATURE_NAMES:
        raise ValueError(
            f"Unknown detection feature '{name}'. "
            f"Valid features are: {', '.join(FEATURE_NAMES)}"
        )
    return detection.get_feature(name)


def _distance_meta(label: str, tooltip: str, **kwargs: Any) -> Dict[str, Any]:
    """Metadata for a distance stored in metres and edited in micrometres."""
    return field_meta(label=label, tooltip=tooltip, unit="m", scale=1e6, **kwargs)


def _feature_meta(label: str, tooltip: str, **kwargs: Any) -> Dict[str, Any]:
    return field_meta(
        label=label,
        tooltip=tooltip,
        items=FEATURE_CHOICES,
        format_fn=_feature_label,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Loop step
# ---------------------------------------------------------------------------

# Step fields old protocol files may still carry, and what became of them. Logged
# at info rather than warned as unknown: they were removed on purpose.
_RETIRED_STEP_FIELDS = {
    "relative_move_dy": "never applied -- relative moves run in the FIB frame, "
    "where the needle's second axis is z (see corrected_needle_delta)",
    "align_feature": "moved to the task's Stage Align Feature",
}


@dataclass
class NeedleLoopStep:
    """One iteration of the needle loop, defined by its field of view.

    Every field is a scalar, bool or str, so the step list editor renders each one
    with the same form builder as task parameters, and a step round-trips through
    the protocol file as a plain dict. Fields marked ``advanced`` sit behind the
    step editor's Advanced panel.
    """

    ITEM_LABEL: ClassVar[str] = "HFW Step"
    # A step added by copying this one gets its own milling entry rather than
    # silently sharing this one's -- editing either would edit both.
    RESET_ON_COPY: ClassVar[Tuple[str, ...]] = ("milling_key",)

    name: str = field(
        default="Step 01",
        metadata=field_meta(
            label="Step Name",
            tooltip="Identifies the step, and names its milling entry unless "
            "Milling Key is set.",
        ),
    )
    enabled: bool = field(
        default=True,
        metadata=field_meta(
            label="Enabled", tooltip="Untick to skip this step without deleting it."
        ),
    )
    field_of_view: float = field(
        default=fcfg.REFERENCE_HFW_INIT_NEEDLE_VIEW_IB,
        metadata=_distance_meta(
            "Field of View",
            "The HFW this step runs at -- the defining property of the step.",
            minimum=0.0,
        ),
    )

    # -- detection -------------------------------------------------------

    from_feature_eb: str = field(
        default="NeedleTip",
        metadata=_feature_meta(
            "From Feature (EB)",
            "The feature that moves with the needle, as seen in the SEM view.",
        ),
    )
    to_feature_eb: str = field(
        default="",
        metadata=_feature_meta(
            "To Feature (EB)",
            "The stationary target, as seen in the SEM view. On the first step this "
            "is detected once, before the needle is inserted.",
        ),
    )
    from_feature_ib: str = field(
        default="NeedleTip",
        metadata=_feature_meta(
            "From Feature (IB)",
            "The feature that moves with the needle, as seen in the FIB view.",
        ),
    )
    to_feature_ib: str = field(
        default="",
        metadata=_feature_meta(
            "To Feature (IB)",
            "The stationary target, as seen in the FIB view. On the first step this "
            "is detected once, before the needle is inserted.",
        ),
    )

    # -- movement --------------------------------------------------------

    eb_tolerance: float = field(
        default=20e-6,
        metadata=_distance_meta(
            "EB Tolerance",
            "Stop moving once the needle is within this distance of the target in "
            "the SEM view.",
            minimum=0.0,
        ),
    )
    ib_tolerance: float = field(
        default=20e-6,
        metadata=_distance_meta(
            "IB Tolerance",
            "Stop moving once the needle is within this vertical distance of the "
            "target in the FIB view.",
            minimum=0.0,
        ),
    )
    movement_buffer: float = field(
        default=0.8,
        metadata=field_meta(
            label="Movement Buffer",
            tooltip="Fraction of the measured offset to actually move. Below 1.0 "
            "approaches gradually, which avoids overshooting onto the sample.",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
        ),
    )
    skip_needle_move: bool = field(
        default=False,
        metadata=field_meta(
            label="Skip Needle Movement",
            tooltip="Run only this step's milling and relative move, without "
            "aligning the needle. For cut-off and lift-off steps.",
        ),
    )
    milling_enabled: bool = field(
        default=False,
        metadata=field_meta(
            label="Enable Milling",
            tooltip="Run milling in this step. Ticking this adds a milling entry "
            "for the step, edited in the milling panel.",
        ),
    )

    # -- advanced: models ------------------------------------------------

    eb_model: str = field(
        default="",
        metadata=field_meta(
            label="EB Model",
            tooltip="Checkpoint for SEM detection in this step. Blank uses the "
            "task's Model Checkpoint.",
            filepath=True,
            advanced=True,
        ),
    )
    ib_model: str = field(
        default="",
        metadata=field_meta(
            label="IB Model",
            tooltip="Checkpoint for FIB detection in this step. Blank uses the "
            "task's Model Checkpoint.",
            filepath=True,
            advanced=True,
        ),
    )

    # -- advanced: target ------------------------------------------------

    target_offset_eb_x: float = field(
        default=0.0,
        metadata=_distance_meta(
            "Target Offset EB dX",
            "Added to the detected SEM target position.",
            advanced=True,
        ),
    )
    target_offset_eb_y: float = field(
        default=0.0,
        metadata=_distance_meta(
            "Target Offset EB dY",
            "Added to the detected SEM target position.",
            advanced=True,
        ),
    )
    target_offset_ib_x: float = field(
        default=0.0,
        metadata=_distance_meta(
            "Target Offset IB dX",
            "Added to the detected FIB target position.",
            advanced=True,
        ),
    )
    target_offset_ib_y: float = field(
        default=0.0,
        metadata=_distance_meta(
            "Target Offset IB dY",
            "Added to the detected FIB target position. Positive sits below the "
            "target, for welding.",
            advanced=True,
        ),
    )
    update_target: bool = field(
        default=False,
        metadata=field_meta(
            label="Update Target Position",
            tooltip="Re-detect the target at this step's HFW, using this step's "
            "To Features. Off by default: the target is detected once before "
            "insertion, so the needle cannot occlude it.",
            advanced=True,
        ),
    )

    # -- advanced: relative move -----------------------------------------

    relative_move_dx: float = field(
        default=0.0,
        metadata=_distance_meta(
            "Relative Move dX",
            "Blind needle move across the image, applied after milling.",
            advanced=True,
        ),
    )
    relative_move_dz: float = field(
        default=0.0,
        metadata=_distance_meta(
            "Relative Move dZ",
            "Blind needle move out of the sample, applied after milling. Negative "
            "lifts the needle away from the sample.",
            advanced=True,
        ),
    )
    relative_move_repeats: int = field(
        default=1,
        metadata=field_meta(
            label="Relative Move Repeats",
            tooltip="How many times to apply the relative move, imaging between each.",
            minimum=1,
            advanced=True,
        ),
    )

    # -- advanced: milling -----------------------------------------------

    milling_key: str = field(
        default="",
        metadata=field_meta(
            label="Milling Key",
            tooltip="Which milling entry this step runs. Blank uses the step name. "
            "Set it to keep a step pointed at its patterns across a rename.",
            advanced=True,
        ),
    )
    milling_anchor_feature: str = field(
        default="",
        metadata=_feature_meta(
            "Milling Anchor Feature",
            "Detect this feature and offset the milling patterns by its position, "
            "so patterns are placed relative to it. Blank mills at the configured "
            "pattern positions.",
            advanced=True,
        ),
    )
    milling_anchor_beam: str = field(
        default="ION",
        metadata=field_meta(
            label="Milling Anchor Beam",
            tooltip="Which beam to detect the milling anchor feature in.",
            items=BEAM_CHOICES,
            advanced=True,
        ),
    )

    @property
    def key(self) -> str:
        """The ``config.milling`` entry this step runs."""
        return self.milling_key or self.name

    @property
    def moves_needle(self) -> bool:
        return not self.skip_needle_move

    @property
    def has_relative_move(self) -> bool:
        return not (
            np.isclose(self.relative_move_dx, 0.0)
            and np.isclose(self.relative_move_dz, 0.0)
        )

    @property
    def anchor_beam_type(self) -> BeamType:
        return BeamType[self.milling_anchor_beam]

    @property
    def target_offset_eb(self) -> Point:
        return Point(x=self.target_offset_eb_x, y=self.target_offset_eb_y)

    @property
    def target_offset_ib(self) -> Point:
        return Point(x=self.target_offset_ib_x, y=self.target_offset_ib_y)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, ddict: Optional[Dict[str, Any]]) -> "NeedleLoopStep":
        known = {f.name for f in fields(cls)}
        kwargs = {}
        for key, value in (ddict or {}).items():
            if key in known:
                kwargs[key] = value
            elif key in _RETIRED_STEP_FIELDS:
                logging.info(
                    f"NeedleLoopStep '{key}' ignored: {_RETIRED_STEP_FIELDS[key]}."
                )
            else:
                logging.warning(f"Unknown NeedleLoopStep parameter '{key}'. Ignoring.")
        return cls(**kwargs)


def default_needle_loop_steps() -> List[NeedleLoopStep]:
    """One neutral approach step and no milling: a starting point, not a recipe."""
    return [NeedleLoopStep(name="Step 01")]


def blank_step_milling(step: NeedleLoopStep) -> FibsemMillingTaskConfig:
    """A milling entry for a step that has just had milling ticked.

    One default stage at the step's field of view: enough for the milling editor
    to show something to place, not something meant to be milled as it stands.
    """
    return FibsemMillingTaskConfig(
        name=step.name,
        field_of_view=step.field_of_view,
        stages=[FibsemMillingStage(name=f"{step.name} Milling")],
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class NeedleLiftoutTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the general needle liftout task.

    Only what is genuinely global to the task lives here. Everything per field of
    view -- features, models, tolerances, offsets, milling -- lives on the steps.
    """

    # Kept from the previous codebase: task_type is written into protocol files,
    # so this is what makes protocols built there load here.
    task_type: ClassVar[str] = "LIFTOUT_NEEDLE_GENERAL"
    display_name: ClassVar[str] = "Needle Liftout (General)"

    model_checkpoint: str = field(
        default="autolamella-waffle-20240107.pt",
        metadata=field_meta(
            tooltip="ML model checkpoint, used wherever a step does not name its own."
        ),
    )
    orientation: str = field(
        default="FIB",
        metadata=field_meta(
            tooltip="Stage orientation to run in. FIB and SEM move to that "
            "orientation; MILLING uses the lamella's stored milling pose.",
            items=ORIENTATION_CHOICES,
        ),
    )
    milling_angle: float = field(
        default=90.0,
        metadata=field_meta(
            tooltip="The angle between the FIB and the sample to tilt to. Only used "
            "when the orientation is not FIB.",
            unit=constants.DEGREE_SYMBOL,
        ),
    )
    align_feature: str = field(
        default="",
        metadata=_feature_meta(
            "Stage Align Feature",
            "Re-centre the stage on this feature after moving to the orientation "
            "involves a rotation. Blank skips it.",
        ),
    )
    initial_target_hfw: float = field(
        default=fcfg.REFERENCE_HFW_INIT_NEEDLE_VIEW_IB,
        metadata=_distance_meta(
            "Initial Target HFW",
            "The HFW to detect the target(s) at before inserting the needle, using "
            "the first enabled step's To Features.",
            minimum=0.0,
        ),
    )
    insert_needle: bool = field(
        default=True,
        metadata=field_meta(
            tooltip="Insert the manipulator to the park position before the steps run."
        ),
    )
    retract_needle: bool = field(
        default=False,
        metadata=field_meta(tooltip="Retract the manipulator after the last step."),
    )
    abort_if_not_converged: bool = field(
        default=True,
        metadata=field_meta(
            tooltip="Unsupervised, fail the task when the needle does not reach its "
            "target, rather than going on to mill with it out of position. "
            "Supervised, the operator decides.",
        ),
    )
    flip_features_with_scan_rotation: bool = field(
        default=True,
        metadata=field_meta(
            tooltip="At 180 degree scan rotation, swap Top/Bottom in feature names: "
            "a bottom edge appears at the top of the image.",
            advanced=True,
        ),
    )
    save_ml_training_images: bool = field(
        default=True,
        metadata=field_meta(
            tooltip="Save every detection image as ML training data.",
            advanced=True,
        ),
    )
    for_liftout: bool = field(
        default=True,
        metadata=field_meta(
            tooltip="Marks this task as part of a liftout protocol. Not enforced by "
            "the workflow yet: choose the lamella it runs on in the Workflow tab.",
            advanced=True,
        ),
    )

    # Declared last so it renders at the bottom of the parameter form.
    steps: List[NeedleLoopStep] = field(
        default_factory=default_needle_loop_steps,
        metadata=field_meta(
            label="HFW Steps",
            tooltip="The loop. One entry per field of view, run in order.",
        ),
    )

    def __post_init__(self) -> None:
        self.sync_step_milling()

    @property
    def enabled_steps(self) -> List[NeedleLoopStep]:
        return [step for step in self.steps if step.enabled]

    def sync_step_milling(self) -> List[str]:
        """Give every milling-enabled step a milling entry. Returns the keys added.

        Never removes one. Unticking a step's milling, renaming it or deleting it
        leaves its entry where it was, so a slip in the step editor cannot destroy
        patterns that took time to place. An orphaned entry is inert: milling runs
        only for steps that are enabled, milling-enabled, and keyed to it.

        Called by the editors after a parameter change, which is what makes ticking
        Enable Milling bring the entry up in the milling panel. Deliberately *not*
        called at run time: a blank entry created then would be milled unseen.
        """
        added = []
        for step in self.steps:
            if step.milling_enabled and step.key not in self.milling:
                self.milling[step.key] = blank_step_milling(step)
                added.append(step.key)
        return added

    def to_dict(self) -> Dict[str, Any]:
        ddict = super().to_dict()
        # The base stores parameters verbatim, and a step object is not YAML.
        ddict["parameters"]["steps"] = [step.to_dict() for step in self.steps]
        return ddict

    @classmethod
    def from_dict(cls, ddict: Dict[str, Any]) -> "NeedleLiftoutTaskConfig":
        ddict = deepcopy(ddict)
        parameters = ddict.get("parameters") or {}
        raw_steps = parameters.get("steps")
        if raw_steps is not None:
            # A protocol from the previous codebase kept the align feature on each
            # step, though only the first one's was ever read. Carry that across.
            if "align_feature" not in parameters:
                for raw in raw_steps:
                    if isinstance(raw, dict) and raw.get("enabled", True):
                        if raw.get("align_feature"):
                            parameters["align_feature"] = raw["align_feature"]
                        break
            parameters["steps"] = [
                step
                if isinstance(step, NeedleLoopStep)
                else NeedleLoopStep.from_dict(step)
                for step in raw_steps
            ]
            ddict["parameters"] = parameters
        return super().from_dict(ddict)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class NeedleLiftoutTask(AutoLamellaTask):
    """General, config-driven needle liftout task. See the module docstring."""

    config: NeedleLiftoutTaskConfig
    config_cls: ClassVar[Type[NeedleLiftoutTaskConfig]] = NeedleLiftoutTaskConfig

    def _run(self) -> None:
        self.image_settings = self.config.imaging
        self.image_settings.path = self.lamella.path
        self._target_eb: Optional[Point] = None
        self._target_ib: Optional[Point] = None

        steps = self.config.enabled_steps
        self._validate_config(steps)
        first_step = steps[0]

        # 1. into position
        self._move_to_position(first_step)

        # 2. reference images at the first step's field of view
        self._acquire_reference_image(
            self.image_settings, field_of_view=first_step.field_of_view
        )

        # 3. detect the target(s) BEFORE inserting, so the needle cannot occlude
        #    them on the way in
        self._resolve_targets(first_step, hfw=self.config.initial_target_hfw)

        # 4. insert the needle
        if self.config.insert_needle:
            self._insert_needle()

        # 5. the loop
        for index, step in enumerate(steps):
            self._check_for_abort()
            self._run_step(step, index, len(steps))

        # 6. retract the needle
        if self.config.retract_needle:
            self._retract_needle()

        # 7. final reference images
        self._acquire_set_of_reference_images(self.image_settings)
        if self.config.save_ml_training_images and self._last_fib_image is not None:
            self._save_images_for_ml_training(fib_image=self._last_fib_image)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_config(self, steps: List[NeedleLoopStep]) -> None:
        """Fail before anything moves, with a message that says what to fix."""
        if not steps:
            raise ValueError(
                f"{self.task_name}: no enabled HFW steps. Add one in the protocol "
                "editor, or enable an existing one."
            )
        if self.config.orientation not in ORIENTATION_CHOICES:
            raise ValueError(
                f"{self.task_name}: invalid orientation '{self.config.orientation}'. "
                f"Expected one of {ORIENTATION_CHOICES}."
            )
        try:
            resolve_feature_name(self.config.align_feature)
        except ValueError as e:
            raise ValueError(
                f"{self.task_name}: invalid Stage Align Feature: {e}"
            ) from e

        # Targets detected up front come from the first step's To Features. A
        # later step can only steer by those, or by its own if it re-detects.
        first = steps[0]
        cached = {"eb": bool(first.to_feature_eb), "ib": bool(first.to_feature_ib)}

        for step in steps:
            where = f"{self.task_name}: step '{step.name}'"
            for label, name in (
                ("From Feature (EB)", step.from_feature_eb),
                ("To Feature (EB)", step.to_feature_eb),
                ("From Feature (IB)", step.from_feature_ib),
                ("To Feature (IB)", step.to_feature_ib),
                ("Milling Anchor Feature", step.milling_anchor_feature),
            ):
                try:
                    resolve_feature_name(name)
                except ValueError as e:
                    raise ValueError(f"{where} has an invalid {label}: {e}") from e

            if step.milling_anchor_beam not in BEAM_CHOICES:
                raise ValueError(
                    f"{where} has an invalid Milling Anchor Beam "
                    f"'{step.milling_anchor_beam}'. Expected one of {BEAM_CHOICES}."
                )
            if step.field_of_view <= 0:
                raise ValueError(f"{where} has a non-positive field of view.")
            if step.milling_enabled and step.key not in self.config.milling:
                raise ValueError(
                    f"{where} has milling enabled but no milling entry named "
                    f"'{step.key}'. Open the task in the protocol editor to create "
                    "it, or untick Enable Milling."
                )
            if step.moves_needle:
                eb_ok = bool(step.from_feature_eb) and (
                    cached["eb"] or (step.update_target and bool(step.to_feature_eb))
                )
                ib_ok = bool(step.from_feature_ib) and (
                    cached["ib"] or (step.update_target and bool(step.to_feature_ib))
                )
                if not (eb_ok or ib_ok):
                    raise ValueError(
                        f"{where} moves the needle but has nothing to aim at in "
                        "either view. Set a From and To Feature on the first step, "
                        "or tick Skip Needle Movement."
                    )

    # ------------------------------------------------------------------
    # Feature and checkpoint resolution
    # ------------------------------------------------------------------

    @property
    def _should_flip(self) -> bool:
        """Whether feature names flip Top/Bottom at the current scan rotation."""
        if not self.config.flip_features_with_scan_rotation:
            return False
        scan_rotation = self.microscope.get_scan_rotation(beam_type=BeamType.ION)
        return not np.isclose(scan_rotation, 0)

    def _feature(self, name: str) -> Optional[Feature]:
        return resolve_feature_name(name, flip=self._should_flip)

    def _checkpoints(self, step: NeedleLoopStep) -> Dict[str, str]:
        """Per-beam checkpoints: the step's own, else the task's."""
        return {
            "eb": step.eb_model or self.config.model_checkpoint,
            "ib": step.ib_model or self.config.model_checkpoint,
        }

    # ------------------------------------------------------------------
    # Stage movement
    # ------------------------------------------------------------------

    def _move_to_position(self, first_step: NeedleLoopStep) -> None:
        """Move the stage into position for the needle workflow."""
        orientation = self.config.orientation

        if orientation == "MILLING":
            self.log_status_message("MOVE_TO_POSITION", "Moving to the milling pose...")
            self._move_to_milling_pose()
            return

        self.log_status_message(
            "MOVE_TO_POSITION", f"Moving to the {orientation} orientation..."
        )
        target_position = self._get_stage_position_for_orientation(
            self.lamella.stage_position, orientation
        )
        has_rotated = not np.isclose(
            target_position.r, self.lamella.stage_position.r, atol=1e-2
        )
        self.microscope.safe_absolute_stage_movement(target_position)

        align_feature = self._feature(self.config.align_feature)
        if has_rotated and align_feature is not None:
            logging.info(
                f"Rotated into position; re-centring on {align_feature.name} to "
                "correct the rotation misalignment."
            )
            align_feature_coincident(
                microscope=self.microscope,
                image_settings=deepcopy(self.image_settings),
                lamella=self.lamella,
                checkpoint=self._checkpoints(first_step)["ib"],
                parent_ui=self.parent_ui,
                validate=self.validate,
                feature=align_feature,
                hfw=first_step.field_of_view,
            )

        if orientation != "FIB":
            self._tilt_to_milling_angle()

    def _tilt_to_milling_angle(self) -> None:
        """Tilt to the configured milling angle, asking first when supervised."""
        milling_angle = self.config.milling_angle
        if self.microscope.is_close_to_milling_angle(milling_angle=milling_angle):
            return

        if self.validate:
            current = self.microscope.get_current_milling_angle()
            tilting = ask_user(
                parent_ui=self.parent_ui,
                msg=f"Tilt to the milling angle ({milling_angle:.1f}"
                f"{constants.DEGREE_SYMBOL})? The current milling angle is "
                f"{current:.1f}{constants.DEGREE_SYMBOL}.",
                pos="Tilt",
                neg="Skip",
            )
            if not tilting:
                return

        self.microscope.move_to_milling_angle(milling_angle=np.radians(milling_angle))

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    def _resolve_targets(self, step: NeedleLoopStep, hfw: float) -> None:
        """Detect the step's stationary target(s) and cache them.

        The needle is aimed at the cached positions, which is what lets an approach
        survive the needle occluding its target. A view whose To Feature is blank
        keeps the target it already had, rather than losing it.
        """
        self.log_status_message("DETECT_TARGET", "Detecting the target position(s)...")
        checkpoints = self._checkpoints(step)

        target = self._detect_target(
            BeamType.ELECTRON,
            step.to_feature_eb,
            checkpoints["eb"],
            step.target_offset_eb,
            hfw,
        )
        if target is not None:
            self._target_eb = target

        target = self._detect_target(
            BeamType.ION,
            step.to_feature_ib,
            checkpoints["ib"],
            step.target_offset_ib,
            hfw,
        )
        if target is not None:
            self._target_ib = target

    def _detect_target(
        self,
        beam_type: BeamType,
        feature_name: str,
        checkpoint: str,
        offset: Point,
        hfw: float,
    ) -> Optional[Point]:
        feature = self._feature(feature_name)
        if feature is None:
            return None
        return detect_target(
            microscope=self.microscope,
            image_settings=self.image_settings,
            feature=feature,
            beam_type=beam_type,
            hfw=hfw,
            checkpoint=checkpoint,
            offset=offset,
            parent_ui=self.parent_ui,
            validate=self.validate,
            msg=self.lamella.status_info,
            on_detection=self._on_detection,
        )

    # ------------------------------------------------------------------
    # Manipulator
    # ------------------------------------------------------------------

    # NOTE: the same as MoveNeedleTask's. That task exists to benchmark the
    # primitives, and once this one covers it, it and this duplicate can go.

    def _insert_needle(self) -> None:
        """Insert the manipulator to the park position, asking first when supervised."""
        if self.microscope.get_manipulator_state():
            logging.info("The manipulator is already inserted; skipping insertion.")
            return

        self.log_status_message(
            "INSERT_NEEDLE", "Inserting the needle to the park position..."
        )
        if self.validate:
            inserting = ask_user(
                parent_ui=self.parent_ui,
                msg="Insert the needle to the park position? It is currently "
                "retracted.",
                pos="Insert",
                neg="Skip",
            )
            if not inserting:
                logging.info("The operator skipped inserting the needle.")
                return

        self.microscope.insert_manipulator(name=PARK_POSITION)

    def _retract_needle(self) -> None:
        """Retract the manipulator, asking first when supervised."""
        if not self.microscope.get_manipulator_state():
            logging.info("The manipulator is already retracted; nothing to do.")
            return

        self.log_status_message("RETRACT_NEEDLE", "Retracting the needle...")
        if self.validate:
            retracting = ask_user(
                parent_ui=self.parent_ui,
                msg="Retract the needle? It is currently inserted.",
                pos="Retract",
                neg="Leave Inserted",
            )
            if not retracting:
                logging.info("The operator left the needle inserted.")
                return

        self.microscope.retract_manipulator()

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def _refresh(self, step: NeedleLoopStep) -> None:
        self._refresh_view(
            image_settings=self.image_settings, field_of_view=step.field_of_view
        )

    def _run_step(self, step: NeedleLoopStep, index: int, total: int) -> None:
        """One iteration of the loop."""
        self.log_status_message(
            f"STEP_{index + 1:02d}",
            f"{step.name} ({index + 1}/{total}) at "
            f"{step.field_of_view * 1e6:.0f} um HFW...",
        )
        self._refresh(step)

        if step.update_target:
            self._resolve_targets(step, hfw=step.field_of_view)

        if step.moves_needle:
            self._converge(step)

        if step.milling_enabled:
            self._mill_step(step)

        if step.has_relative_move:
            self._relative_move(step)

    def _converge(self, step: NeedleLoopStep) -> None:
        """Move the needle until it is within tolerance, in both views."""
        self.log_status_message(
            "MOVE_NEEDLE",
            f"Aligning the needle at {step.field_of_view * 1e6:.0f} um "
            f"(EB tol {step.eb_tolerance * 1e6:.1f} um, "
            f"IB tol {step.ib_tolerance * 1e6:.1f} um)...",
        )
        result = move_needle_to_feature(
            microscope=self.microscope,
            image_settings=self.image_settings,
            hfw=step.field_of_view,
            from_feature_eb=self._feature(step.from_feature_eb),
            to_feature_eb=self._feature(step.to_feature_eb),
            from_feature_ib=self._feature(step.from_feature_ib),
            to_feature_ib=self._feature(step.to_feature_ib),
            target_eb=deepcopy(self._target_eb),
            target_ib=deepcopy(self._target_ib),
            eb_tolerance=step.eb_tolerance,
            ib_tolerance=step.ib_tolerance,
            movement_buffer=step.movement_buffer,
            checkpoints=self._checkpoints(step),
            default_checkpoint=self.config.model_checkpoint,
            parent_ui=self.parent_ui,
            validate=self.validate,
            msg=self.lamella.status_info,
            on_detection=self._on_detection,
            on_refresh=lambda: self._refresh(step),
            stop_event=self._stop_event,
        )
        self._report(step, result)

        # Supervised, a non-converged result is the operator's call (they pressed
        # Skip). Unsupervised, nobody has looked -- and the next thing this step
        # may do is mill, with the needle somewhere it should not be.
        if (
            not result.converged
            and not self.validate
            and self.config.abort_if_not_converged
        ):
            raise RuntimeError(
                f"Step '{step.name}': the needle did not reach its target "
                f"({result.skipped_reason}). Stopping before anything else runs "
                "with it out of position."
            )

    def _mill_step(self, step: NeedleLoopStep) -> None:
        """Run the step's milling, optionally placed relative to a feature."""
        milling_task_config = self.config.milling[step.key]
        milling_task_config.field_of_view = step.field_of_view
        milling_task_config.acquisition.imaging.path = self.lamella.path
        if self.lamella.alignment_area is not None:
            milling_task_config.alignment.rect = deepcopy(self.lamella.alignment_area)

        offset = self._detect_milling_anchor(step)

        current_config = deepcopy(milling_task_config)
        if offset is not None:
            self._offset_patterns(current_config, offset, sign=1)

        self.log_status_message("MILL", f"Milling {step.name}...")
        updated_config = self.update_milling_config_ui(
            current_config,
            msg=f"Press Run Milling to run {step.name} for {self.lamella.name}. "
            "Press Continue when done.",
        )

        # Take the anchor offset back out before storing, so the protocol keeps
        # positions relative to the feature rather than accumulating the detected
        # offset on every run.
        if offset is not None:
            self._offset_patterns(updated_config, offset, sign=-1)
        self.config.milling[step.key] = deepcopy(updated_config)

    def _detect_milling_anchor(self, step: NeedleLoopStep) -> Optional[Point]:
        """Detect the step's anchor feature and return its position."""
        anchor = self._feature(step.milling_anchor_feature)
        if anchor is None:
            return None

        beam_type = step.anchor_beam_type
        image_settings = deepcopy(self.image_settings)
        image_settings.beam_type = beam_type
        image_settings.hfw = step.field_of_view

        checkpoint_key = "ib" if beam_type is BeamType.ION else "eb"
        det = update_detection_ui(
            microscope=self.microscope,
            image_settings=image_settings,
            checkpoint=self._checkpoints(step)[checkpoint_key],
            features=[anchor],
            parent_ui=self.parent_ui,
            validate=self.validate,
            msg=self.lamella.status_info,
        )
        self._on_detection(det, beam_type)

        # both views back after the single-beam detection
        self._refresh(step)

        point = det.features[0].feature_m
        logging.info(
            f"Milling anchor {anchor.name} detected at "
            f"({point.x * 1e6:.2f} um, {point.y * 1e6:.2f} um)."
        )
        return point

    @staticmethod
    def _offset_patterns(
        milling_task_config: FibsemMillingTaskConfig, offset: Point, sign: int = 1
    ) -> None:
        """Shift every stage's pattern by ``sign * offset``."""
        for stage in milling_task_config.stages:
            if not hasattr(stage.pattern, "point"):
                logging.warning(
                    f"Milling stage '{stage.name}' has no pattern point; it cannot "
                    "be anchored to a detected feature."
                )
                continue
            stage.pattern.point.x += sign * offset.x
            stage.pattern.point.y += sign * offset.y

    def _relative_move(self, step: NeedleLoopStep) -> None:
        """Apply the step's blind relative needle move, in the FIB frame."""
        if not self.microscope.get_manipulator_state():
            logging.warning(
                f"The manipulator is not inserted; skipping the relative move for "
                f"step '{step.name}'."
            )
            return

        repeats = max(1, step.relative_move_repeats)
        for i in range(repeats):
            self.log_status_message(
                "RELATIVE_MOVE",
                f"{step.name}: relative move {i + 1}/{repeats} "
                f"(dx {step.relative_move_dx * 1e6:.1f}, "
                f"dz {step.relative_move_dz * 1e6:.1f} um)...",
            )
            move_needle(
                self.microscope,
                dx=step.relative_move_dx,
                dz=step.relative_move_dz,
                beam_type=BeamType.ION,
                parent_ui=self.parent_ui,
                # only ask on the last one
                validate=self.validate and i == repeats - 1,
                on_refresh=lambda: self._refresh(step),
                stop_event=self._stop_event,
            )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _on_detection(self, det: DetectedFeatures, beam_type: BeamType) -> None:
        """Save a detection image as ML training data. Guarded: a training-data
        write that fails should not take a liftout down with it."""
        if not self.config.save_ml_training_images:
            return
        try:
            image = FibsemImage(data=det.image)
        except Exception as e:
            logging.debug(f"Could not wrap the detection image for ML training: {e}")
            return

        if beam_type is BeamType.ELECTRON:
            self._save_images_for_ml_training(sem_image=image)
        else:
            self._save_images_for_ml_training(fib_image=image)

    def _report(self, step: NeedleLoopStep, result: NeedleConvergence) -> None:
        logging.debug(
            {
                "msg": "needle_approach",
                "lamella": self.lamella.name,
                "task_name": self.task_name,
                "step": step.name,
                "converged": result.converged,
                "attempts": result.attempts,
                "eb_distance": result.eb_distance,
                "ib_distance": result.ib_distance,
                "skipped_reason": result.skipped_reason,
            }
        )
        if result.converged:
            self.log_status_message(
                "NEEDLE_CONVERGED",
                f"{step.name}: the needle reached its target in "
                f"{result.attempts} round(s).",
            )
        else:
            self.log_status_message(
                "NEEDLE_NOT_CONVERGED",
                f"{step.name}: the needle did not reach its target "
                f"({result.skipped_reason}).",
            )
