######## MOVE NEEDLE TASK DEFINITIONS ########

"""A minimal needle approach, for testing and benchmarking the primitives.

One target, one approach, no milling. It exists to exercise
:mod:`fibsem.applications.autolamella.workflows.needle` on real hardware -- the
axis conventions, the convergence loop, the insert/retract prompts -- without the
step engine, the per-step milling or the protocol editing that the general needle
liftout task needs on top.

The two features are deliberately hard-coded: the needle is found with
``NeedleTip`` and aimed at a ``LandingPost``. Making them configurable is the
general task's job; here they are constants so there is one less thing to get
wrong while benchmarking the movement itself.

The stage is not moved. Every other task opens by driving somewhere, but this one
is pointed at whatever is already in view -- which is what you want when you are
setting up a landing post by hand to test against.
"""

import logging
from dataclasses import dataclass, field
from typing import ClassVar, Type

from fibsem import config as fcfg
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows.needle import (
    PARK_POSITION,
    NeedleConvergence,
    detect_target,
    move_needle_to_feature,
)
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
from fibsem.applications.autolamella.workflows.ui import ask_user
from fibsem.detection.detection import DetectedFeatures, LandingPost, NeedleTip
from fibsem.structures import BeamType, FibsemImage, field_meta

#: The feature that moves with the needle. Hard-coded; see the module docstring.
NEEDLE_FEATURE = NeedleTip

#: The stationary feature the needle is aimed at. Hard-coded.
TARGET_FEATURE = LandingPost


@dataclass
class MoveNeedleTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MoveNeedleTask."""

    task_type: ClassVar[str] = "MOVE_NEEDLE"
    display_name: ClassVar[str] = "Move Needle"

    model_checkpoint: str = field(
        default="autolamella-waffle-20240107.pt",
        metadata=field_meta(
            tooltip="ML model checkpoint used to detect the needle and the target"
        ),
    )
    field_of_view: float = field(
        default=fcfg.REFERENCE_HFW_INIT_NEEDLE_VIEW_IB,
        metadata=field_meta(
            tooltip="The field of view to detect and approach at",
            unit="m",
            scale=1e6,
        ),
    )
    eb_tolerance: float = field(
        default=20e-6,
        metadata=field_meta(
            tooltip="Stop moving once the needle is within this distance of the "
            "target in the SEM view",
            unit="m",
            scale=1e6,
        ),
    )
    ib_tolerance: float = field(
        default=20e-6,
        metadata=field_meta(
            tooltip="Stop moving once the needle is within this distance of the "
            "target in the FIB view",
            unit="m",
            scale=1e6,
        ),
    )
    movement_buffer: float = field(
        default=0.8,
        metadata=field_meta(
            tooltip="Fraction of the measured offset to actually move. Below 1.0 "
            "approaches gradually, which avoids overshooting onto the sample",
        ),
    )
    retract_needle: bool = field(
        default=True,
        metadata=field_meta(
            tooltip="Offer to retract the manipulator once the approach is done"
        ),
    )
    save_ml_training_images: bool = field(
        default=True,
        metadata=field_meta(
            tooltip="Save every detection image as ML training data"
        ),
    )


class MoveNeedleTask(AutoLamellaTask):
    """Detect a target, insert the needle, and drive it to the target."""

    config: MoveNeedleTaskConfig
    config_cls: ClassVar[Type[MoveNeedleTaskConfig]] = MoveNeedleTaskConfig

    def _run(self) -> None:
        image_settings = self.config.imaging
        image_settings.path = self.lamella.path
        hfw = self.config.field_of_view
        checkpoint = self.config.model_checkpoint

        # 1. reference images at the field of view the approach will run at
        self._acquire_reference_image(image_settings, field_of_view=hfw)

        # 2. detect the target in both views BEFORE the needle is inserted. Once
        #    it is in, it occludes the target on the way in -- which is the whole
        #    reason the approach takes cached target positions.
        self.log_status_message(
            "DETECT_TARGET", f"Detecting the {TARGET_FEATURE().name}..."
        )
        targets = {}
        for beam_type in (BeamType.ELECTRON, BeamType.ION):
            targets[beam_type] = detect_target(
                microscope=self.microscope,
                image_settings=image_settings,
                feature=TARGET_FEATURE(),
                beam_type=beam_type,
                hfw=hfw,
                checkpoint=checkpoint,
                parent_ui=self.parent_ui,
                validate=self.validate,
                msg=self.lamella.status_info,
                on_detection=self._on_detection,
            )

        # 3. insert the needle
        self._insert_needle()

        # 4. detect the needle and drive it to the target. The "would you like to
        #    adjust?" prompt lives inside the approach, so the operator can nudge
        #    the needle by hand and have it re-detect and re-verify.
        self.log_status_message("MOVE_NEEDLE", "Moving the needle to the target...")
        result = move_needle_to_feature(
            microscope=self.microscope,
            image_settings=image_settings,
            hfw=hfw,
            from_feature_eb=NEEDLE_FEATURE(),
            from_feature_ib=NEEDLE_FEATURE(),
            target_eb=targets[BeamType.ELECTRON],
            target_ib=targets[BeamType.ION],
            eb_tolerance=self.config.eb_tolerance,
            ib_tolerance=self.config.ib_tolerance,
            movement_buffer=self.config.movement_buffer,
            default_checkpoint=checkpoint,
            parent_ui=self.parent_ui,
            validate=self.validate,
            msg=self.lamella.status_info,
            on_detection=self._on_detection,
            on_refresh=lambda: self._refresh_view(
                image_settings=image_settings, field_of_view=hfw
            ),
            stop_event=self._stop_event,
        )
        self._report(result)

        # 5. retract the needle
        if self.config.retract_needle:
            self._retract_needle()

        # 6. final reference images
        self._acquire_set_of_reference_images(image_settings)

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

    def _on_detection(self, det: DetectedFeatures, beam_type: BeamType) -> None:
        """Save a detection image as ML training data.

        Guarded: this task exists to benchmark the approach, and a training-data
        write that fails should not take the run down with it.
        """
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

    def _report(self, result: NeedleConvergence) -> None:
        """Log how the approach went, which is the point of this task."""
        logging.debug(
            {
                "msg": "needle_approach",
                "lamella": self.lamella.name,
                "task_name": self.task_name,
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
                f"The needle reached its target in {result.attempts} round(s).",
            )
            return
        # Not an exception: the operator may have skipped the approach on purpose,
        # and the task's remaining steps (retract, reference images) still apply.
        self.log_status_message(
            "NEEDLE_NOT_CONVERGED",
            f"The needle did not reach its target ({result.skipped_reason}).",
        )
