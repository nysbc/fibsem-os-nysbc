######## UNDERCUT TASK DEFINITIONS ########

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Literal, Optional, Tuple, Type

import numpy as np

from fibsem import config as fcfg
from fibsem import constants
from fibsem.applications.autolamella.protocol.constants import TRENCH_KEY, UNDERCUT_KEY
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows._default_milling_config import (
    DEFAULT_MILLING_CONFIG,
)
from fibsem.applications.autolamella.workflows.core import (
    align_feature_coincident,
    update_detection_ui,
)
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
from fibsem.applications.autolamella.workflows.tasks.trench import MillTrenchTaskConfig
from fibsem.applications.autolamella.workflows.ui import ask_user
from fibsem.autofunctions.charge_neutralisation import auto_charge_neutralisation
from fibsem.detection.detection import (
    Feature,
    LamellaBottomEdge,
    LamellaCentre,
    LamellaLeftEdge,
    LamellaRightEdge,
    LamellaTopEdge,
    VolumeBlockBottomEdge,
    VolumeBlockCentre,
    VolumeBlockTopEdge,
)
from fibsem.structures import BeamType, ImageSettings, field_meta

# The features the undercut can align to, by the name the protocol stores. One
# feature serves both alignments: the first, at Initial Feature Alignment HFW, and
# the re-centre after each tilt.
ALIGN_FEATURES: Dict[str, Type[Feature]] = {
    "LamellaCentre": LamellaCentre,
    "VolumeBlockCentre": VolumeBlockCentre,
}

# The edges an undercut pattern can be placed from, by stored name: (top, bottom).
# Which of the pair is used follows the scan rotation; see get_placement_edge.
PLACEMENT_EDGES: Dict[str, Tuple[Type[Feature], Type[Feature]]] = {
    "LamellaEdge": (LamellaTopEdge, LamellaBottomEdge),
    "VolumeBlockEdge": (VolumeBlockTopEdge, VolumeBlockBottomEdge),
}

# What Auto Abort and the trench-size gate have been developed for. Only these
# features carry pix_threshold, and only their detection ever reports a failure
# (detection.detect_lamella), so for any other feature both checks would silently
# do nothing. They are skipped instead, and the task says so.
LAMELLA_FEATURES: Tuple[Type[Feature], ...] = (
    LamellaCentre,
    LamellaLeftEdge,
    LamellaRightEdge,
    LamellaTopEdge,
    LamellaBottomEdge,
)


def get_placement_edge(name: str, scan_rotation: float) -> Feature:
    """The edge to place an undercut pattern from, for a Pattern Placement Feature.

    The top edge at 0 scan rotation; any other rotation flips the image, so the
    bottom edge. _run offsets the pattern by the same test (+height/2 at 0,
    -height/2 otherwise), so the edge and the offset cannot disagree. A fresh
    instance on every call, because detection writes its results onto the feature.

    Args:
        name: a key of PLACEMENT_EDGES.
        scan_rotation: the ion beam scan rotation, in radians.
    """
    top, bottom = PLACEMENT_EDGES[name]
    return top() if np.isclose(scan_rotation, 0) else bottom()


@dataclass
class MillUndercutTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MillUndercutTask."""

    orientation: Optional[Literal["SEM", "FIB", "MILLING"]] = field(
        default="SEM",
        metadata=field_meta(
            tooltip="The orientation to perform undercut milling in",
            items=("SEM", "FIB", "MILLING", None),
        ),
    )
    milling_angles: List[float] = field(
        default_factory=lambda: [25, 20],  # in degrees
        metadata=field_meta(
            tooltip="The angles to mill the undercuts at", unit=constants.DEGREE_SYMBOL
        ),
    )

    model_checkpoint: str = field(
        default="autolamella-waffle-20240107.pt",
        metadata=field_meta(tooltip="ML model checkpoint"),
    )

    align_feature: str = field(
        default="LamellaCentre",
        metadata=field_meta(
            label="Alignment Feature",
            tooltip="The feature to align to: first at Initial Feature Alignment HFW, "
            "then again after each tilt to an undercut angle.",
            items=tuple(ALIGN_FEATURES),
        ),
    )

    align_feature_hfw: float = field(
        default=fcfg.REFERENCE_HFW_HIGH,
        metadata=field_meta(
            label="Initial Feature Alignment HFW",
            tooltip="The field of view for the first alignment to the Alignment "
            "Feature. A larger target, such as a liftout block, needs a wider view.",
            unit="m",
            scale=1e6,
            minimum=0.0,
        ),
    )

    pattern_placement_feature: str = field(
        default="LamellaEdge",
        metadata=field_meta(
            label="Pattern Placement Feature",
            tooltip="The edge each undercut pattern is placed from: the top edge at "
            "0 degree scan rotation, the bottom edge otherwise.",
            items=tuple(PLACEMENT_EDGES),
        ),
    )

    auto_abort: bool = field(
        default=True,
        metadata=field_meta(
            tooltip="Auto abort the step and mark lamella as defect if ML detection for undercut fails. This allows overmilling to be avoided on lamellae where detections went wrong. "
            "Lamella features only: ignored when the Alignment Feature is not a lamella feature."
        ),
    )

    gate_det_based_on_trench_milling: bool = field(
        default=False,
        metadata=field_meta(
            tooltip="WARNING: Experimental feature: Add measure for ML detection based on Trench milling step, estimates lamella size of undercut from known geometry from trench milling. "
            + "Helps make ML detection more robust and makes sure bad detections do not ruin sample, Trench milling step must precede undercut. Built primarily for Waffle Method. "
            + "Lamella features only: ignored when the Alignment Feature is not a lamella feature."
        ),
    )

    task_type: ClassVar[str] = "MILL_UNDERCUT"
    display_name: ClassVar[str] = "Undercut Milling"

    def __post_init__(self):
        if self.milling == {}:
            self.milling = deepcopy(
                {UNDERCUT_KEY: DEFAULT_MILLING_CONFIG[UNDERCUT_KEY]}
            )


class MillUndercutTask(AutoLamellaTask):
    """Task to mill the undercut for a lamella."""

    config: MillUndercutTaskConfig
    config_cls: ClassVar[Type[MillUndercutTaskConfig]] = MillUndercutTaskConfig

    @property
    def _lamella_feature(self) -> bool:
        """Whether the Alignment Feature is a lamella feature, which Auto Abort and
        the trench-size gate need. See LAMELLA_FEATURES."""
        return issubclass(ALIGN_FEATURES[self.config.align_feature], LAMELLA_FEATURES)

    def _validate_config(self) -> None:
        """Fail before anything moves, with a message that says what to fix."""
        if self.config.align_feature not in ALIGN_FEATURES:
            raise ValueError(
                f"{self.task_name}: unknown Alignment Feature "
                f"'{self.config.align_feature}'. Expected one of "
                f"{tuple(ALIGN_FEATURES)}."
            )
        if self.config.pattern_placement_feature not in PLACEMENT_EDGES:
            raise ValueError(
                f"{self.task_name}: unknown Pattern Placement Feature "
                f"'{self.config.pattern_placement_feature}'. Expected one of "
                f"{tuple(PLACEMENT_EDGES)}."
            )
        if self.config.align_feature_hfw <= 0:
            raise ValueError(
                f"{self.task_name}: Initial Feature Alignment HFW must be positive."
            )

    def _run(self) -> None:
        self._validate_config()
        if not self._lamella_feature and (
            self.config.auto_abort or self.config.gate_det_based_on_trench_milling
        ):
            logging.warning(
                f"{self.task_name}: Auto Abort and the trench-size gate support "
                f"lamella features only; both are ignored for "
                f"{self.config.align_feature}."
            )

        # bookkeeping
        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        checkpoint = self.config.model_checkpoint

        # move to sem orientation
        self.log_status_message("MOVE_TO_UNDERCUT", "Moving to Undercut Position...")
        undercut_position = self._get_stage_position_for_orientation(
            self.lamella.stage_position, self.config.orientation
        )
        self.microscope.safe_absolute_stage_movement(undercut_position)
        # TODO: support compucentric offset
        #
        # change image hfw for aligning
        align_coincident_image_settings = deepcopy(image_settings)

        # align feature coincident
        feature = ALIGN_FEATURES[self.config.align_feature]()
        lamella = align_feature_coincident(
            microscope=self.microscope,
            image_settings=align_coincident_image_settings,
            lamella=self.lamella,
            checkpoint=checkpoint,
            parent_ui=self.parent_ui,
            validate=self.validate,
            feature=feature,
            hfw=self.config.align_feature_hfw,
        )

        self._save_images_for_ml_training(
            image_settings=align_coincident_image_settings,
            acquire_sem=True,
            acquire_fib=True,
        )

        # mill under cut
        milling_task_config = self.config.milling[UNDERCUT_KEY]
        undercut_milling_angles = self.config.milling_angles  # deg

        # TODO: support multiple undercuts?

        if len(milling_task_config.stages) != len(undercut_milling_angles):
            raise ValueError(
                f"Number of undercut milling angles ({len(undercut_milling_angles)}) "
                f"does not match number of undercut milling stages ({len(milling_task_config.stages)})"
            )

        baseline_point_y = [s.pattern.point.y for s in milling_task_config.stages]

        for i, undercut_milling_angle in enumerate(undercut_milling_angles):
            nid = f"{i + 1:02d}"  # helper

            # tilt down, align to trench
            self.log_status_message(
                f"TILT_UNDERCUT_{nid}", f"Tilting to Undercut Position {nid}..."
            )
            self.microscope.move_to_milling_angle(
                milling_angle=np.radians(undercut_milling_angle)
            )

            # detect
            self.log_status_message(
                f"ALIGN_UNDERCUT_{nid}", f"Aligning Undercut Position {nid}..."
            )
            # self._acquire_reference_image(image_settings,
            #                               filename=f"ref_{self.task_name}_align_ml_{nid}",
            #                               field_of_view=milling_task_config.field_of_view)

            # get pattern
            scan_rotation = self.microscope.get_scan_rotation(beam_type=BeamType.ION)

            # once tilted, realign to the alignment feature
            image_settings.hfw = self.config.milling[
                UNDERCUT_KEY
            ].field_of_view  # change hfw for detection to match milling fov
            image_settings.beam_type = BeamType.ION

            # fresh each tilt: detection writes its results onto the feature
            undercut_feature = ALIGN_FEATURES[self.config.align_feature]()

            # the size gate and the abort are lamella-only (see LAMELLA_FEATURES);
            # the alignment between them runs for every feature
            if self._lamella_feature and self.config.gate_det_based_on_trench_milling:
                lamella_pix = self._estimate_lamella_size(
                    image_settings=image_settings, tilt=undercut_milling_angle
                )
                if lamella_pix is not None:
                    undercut_feature.pix_threshold = lamella_pix
                    logging.info(
                        f"Setting Lamella Pixel Threshold for Undercut Detection to {lamella_pix} based on Trench Milling Step"
                    )

            detected_successfully = self._align_with_ml_locally(
                image_settings=image_settings,
                milling_key=UNDERCUT_KEY,
                feature=undercut_feature,
            )

            if (
                self._lamella_feature
                and not detected_successfully
                and self.config.auto_abort
            ):
                # need to mark lamella as failed?
                # should not continue with the rest of the step
                # Only valid when unsupervised, when supervised, user will make decision
                if self.validate:
                    should_abort = ask_user(
                        parent_ui=self.parent_ui,
                        msg="ML Model has failed to detect feature, Abort Undercut and follwing workflow for Lamella? "
                        + "Workflow will attempt to continue onto next lamella",
                        pos="Abort",
                        neg="Continue with Lamella",
                    )
                else:
                    should_abort = True

                if should_abort:
                    logging.info(
                        f"Aborting Undercut Step for Lamella {self.lamella.name} "
                    )
                    logging.info("Marking Lamella as Defect")
                    self.lamella.defect.set_defect(
                        description="Undercut ML detection Failed"
                    )
                    break

            features = [
                get_placement_edge(self.config.pattern_placement_feature, scan_rotation)
            ]

            det_Edge = update_detection_ui(
                microscope=self.microscope,
                image_settings=image_settings,
                checkpoint=checkpoint,
                features=features,
                parent_ui=self.parent_ui,
                validate=self.validate,
                msg=lamella.status_info,
            )

            self._save_images_for_ml_training(fib_image=self._last_fib_image)

            # only want to run specific stage
            current_milling_config = deepcopy(milling_task_config)
            current_milling_config.stages = [deepcopy(milling_task_config.stages[i])]

            # set pattern position
            offset = (
                current_milling_config.stages[0].pattern.height / 2
            )  # pattern height
            # offset += current_milling_config.stages[0].pattern.point.y # pattern y point
            # offset += lamella_mid_height # add offset to middle of lamella height

            # top/bottom edge detection y point
            det_y = float(det_Edge.features[0].feature_m.y)

            # the same test as get_placement_edge: top edge and +offset at 0
            det_y += offset if np.isclose(scan_rotation, 0) else -offset
            # current_milling_config.stages[0].pattern.point.y += det_y
            current_milling_config.stages[0].pattern.point.y = (
                deepcopy(milling_task_config.stages[i]).pattern.point.y + det_y
            )
            # current_milling_config.stages[0].pattern.point.y = baseline_point_y[i] + det_y

            # mill undercut
            self.log_status_message(f"MILL_UNDERCUT_{nid}")
            msg = f"Press Run Milling to mill the Undercut for {self.lamella.name}. Press Continue when done."
            current_milling_config = self.update_milling_config_ui(
                current_milling_config, msg=msg
            )

            # milling_task_config.stages[i] = current_milling_config.stages[0]

            ## charge neutralise (seems to build up a lot)

            self.log_status_message(
                "CHARGE_NEUTRALISATION", "Neutralising Sample Charge..."
            )
            image_settings.beam_type = BeamType.ELECTRON
            auto_charge_neutralisation(self.microscope, image_settings)

            self._acquire_reference_image(
                image_settings,
                field_of_view=self.config.milling[UNDERCUT_KEY].field_of_view,
                filename=f"ref_undercut_{nid}_final",
            )
        # log undercut stages
        self.config.milling[UNDERCUT_KEY] = deepcopy(milling_task_config)

        # write pose
        self.lamella.milling_pose = self.microscope.get_microscope_state()

        # acquire reference images
        self._acquire_set_of_reference_images(image_settings)

    def _estimate_lamella_size(self, image_settings: ImageSettings, tilt: int):
        """
        Based on Trench Milling Pattern, Lamella geometry can be calculated at different tilt angles.
        This gives an estimated pixel count for the lamella class for detection, which can help identify bad detections
        """

        # find trench key, if this returns error, return default

        trench_milling = None
        for task_config in self.lamella.task_config.values():
            if task_config.task_type == MillTrenchTaskConfig.task_type:
                trench_milling = task_config.milling.get(TRENCH_KEY)
        if trench_milling is None:
            logging.warning(
                "Trench milling configuration not found; "
                "cannot estimate lamella size for undercut detection."
            )
            return None

        try:
            lamella_height = trench_milling.stages[0].pattern.spacing
        except Exception as e:
            logging.warning(f"Error obtaining spacing from trench milling pattern: {e}")
            return None

        try:
            lamella_width = trench_milling.stages[0].pattern.width
        except Exception as e:
            logging.warning(f"Error obtaining width from trench milling pattern: {e}")
            return None

        # calculate px_size

        px_size = image_settings.hfw / image_settings.resolution[0]

        new_height = lamella_height * np.sin(np.radians(tilt))

        estimated_pix = new_height / px_size * lamella_width / px_size

        estimated_pix *= 0.8  # set lower to account for variability, will probably need to play with this number

        return int(estimated_pix)
