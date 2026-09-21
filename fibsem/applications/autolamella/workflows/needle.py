"""Needle (manipulator) movement primitives for the liftout workflows.

Free functions taking a microscope, rather than methods on the task base, for the
same reason :mod:`fibsem.applications.autolamella.poses` is laid out that way: the
mechanics are testable without a task, a lamella or a GUI, and the task base does
not grow a section only one task uses. The conventions follow
``core.align_feature_coincident``, the closest existing analogue -- a
detect-and-move loop as a free function that takes ``parent_ui`` and ``validate``
and drives the UI through the ``workflows.ui`` helpers.

The axis conventions are the part worth reading. Which way the needle moves for a
requested delta depends on the beam being watched *and* the scan rotation, and
getting it wrong drives the manipulator into the sample. They used to be four
inline sign flips in the middle of a 200-line function; here they are one pure
function, :func:`corrected_needle_delta`, which is the only place they live.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np

from fibsem.applications.autolamella.workflows.ui import ask_user, update_detection_ui
from fibsem.cancellation import raise_if_cancelled
from fibsem.structures import BeamType, ImageSettings, Point

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from threading import Event

    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
    from fibsem.detection.detection import DetectedFeatures, Feature
    from fibsem.microscope import FibsemMicroscope

#: The manipulator is always inserted to the park position.
PARK_POSITION = "PARK"

#: How many detect-and-move rounds before the loop stops and asks for help.
#: Reached with a supervisor present, the operator is asked to intervene and the
#: count restarts; unsupervised, the loop gives up rather than grinding on.
MAX_CONVERGENCE_ATTEMPTS = 5

__all__ = [
    "PARK_POSITION",
    "MAX_CONVERGENCE_ATTEMPTS",
    "NeedleConvergence",
    "corrected_needle_delta",
    "detect_target",
    "move_needle",
    "move_needle_to_feature",
]


@dataclass
class NeedleConvergence:
    """The outcome of a :func:`move_needle_to_feature` run.

    Returned rather than only logged, so a caller can record how hard the approach
    was and a test can assert on it without parsing the log.
    """

    converged: bool
    attempts: int
    eb_distance: Optional[float] = None
    ib_distance: Optional[float] = None
    skipped_reason: Optional[str] = None


def corrected_needle_delta(
    dx: float,
    dy: float,
    dz: float,
    beam_type: BeamType,
    scan_rotation_deg: float,
) -> Tuple[float, float]:
    """The (dx, dy) to hand to ``move_manipulator_corrected`` for a needle delta.

    The caller asks in needle-frame terms -- dx across, dy along, dz up out of the
    sample -- and this maps them onto the two axes the corrected move accepts, for
    the beam the movement is being judged in.

    Two things are going on:

    * **At 180 degree scan rotation the image is flipped**, so a move that should
      appear to go right appears to go left. dx and dy invert to compensate.
    * **The manipulator is mirrored in the ion view**: what reads as vertical there
      is the needle's z, not its y. So for ``BeamType.ION`` the second axis is
      driven from ``dz`` and ``dy`` is not used at all. At 0 degree scan rotation
      that vertical axis also points the other way, so dz inverts.

    Which means ``dy`` only ever applies to ELECTRON and ``dz`` only to ION. That
    is not obvious at a call site, and is why this is one named function.

    Args:
        dx: needle-frame movement across the image, in metres.
        dy: needle-frame movement along the image. ELECTRON only.
        dz: needle-frame movement out of the sample. ION only; negative lifts the
            needle away from the sample.
        beam_type: the beam the movement is being judged in.
        scan_rotation_deg: the ion beam scan rotation, in **degrees**.

    Returns:
        ``(dx, dy)`` in the frame ``move_manipulator_corrected`` expects.
    """
    at_zero = bool(np.isclose(scan_rotation_deg, 0))
    at_flipped = bool(np.isclose(scan_rotation_deg, 180))

    if at_flipped:
        dx = -dx
        dy = -dy
    if at_zero:
        dz = -dz
    if beam_type is BeamType.ION:
        dy = dz

    return dx, dy


def move_needle(
    microscope: "FibsemMicroscope",
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    beam_type: BeamType = BeamType.ION,
    parent_ui: Optional["AutoLamellaUI"] = None,
    validate: bool = False,
    on_refresh: Optional[Callable[[], None]] = None,
    stop_event: Optional["Event"] = None,
) -> bool:
    """Move the needle by a delta, corrected for the beam and the scan rotation.

    Args:
        microscope: the instrument.
        dx, dy, dz: the needle-frame delta in metres. See
            :func:`corrected_needle_delta` for which apply to which beam.
        beam_type: the beam the movement is being judged in.
        parent_ui: the supervising window, or None when headless.
        validate: ask the operator to confirm the needle went where it should.
            Only meaningful with a ``parent_ui``.
        on_refresh: called after the move to re-image and update the view. The
            task passes its own ``_refresh_view``, so there is one implementation
            of that rather than two.
        stop_event: checked before the move, which is the irreversible part.

    Returns:
        Whether the move was applied. A manipulator move that raises is logged and
        reported rather than raised: an approach loop decides what to do about a
        refused move, and it has better options than unwinding the whole task.
    """
    raise_if_cancelled(stop_event, "Needle movement cancelled by user.")

    scan_rotation_deg = float(
        np.rad2deg(microscope.get_scan_rotation(beam_type=BeamType.ION))
    )
    corrected_dx, corrected_dy = corrected_needle_delta(
        dx=dx, dy=dy, dz=dz, beam_type=beam_type, scan_rotation_deg=scan_rotation_deg
    )

    logging.debug(
        {
            "msg": "move_needle",
            "requested": {"dx": dx, "dy": dy, "dz": dz},
            "corrected": {"dx": corrected_dx, "dy": corrected_dy},
            "beam_type": beam_type.name,
            "scan_rotation_deg": scan_rotation_deg,
        }
    )

    try:
        microscope.move_manipulator_corrected(
            dx=corrected_dx, dy=corrected_dy, beam_type=beam_type
        )
    except Exception as e:
        logging.error(f"Error moving the manipulator (relative): {e}", exc_info=True)
        return False

    if on_refresh is not None:
        on_refresh()

    if validate and parent_ui is not None:
        ask_user(
            parent_ui=parent_ui,
            msg="Has the needle moved as expected? If not, move it manually to the "
            "correct position. Press Continue when done.",
            pos="Continue",
        )

    return True


def _distance(a: Point, b: Point) -> float:
    """Straight-line distance between two points, in metres."""
    return float(np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2))


def _detect(
    microscope: "FibsemMicroscope",
    image_settings: ImageSettings,
    checkpoint: str,
    features: List["Feature"],
    parent_ui: Optional["AutoLamellaUI"],
    validate: bool,
    msg: str,
    on_detection: Optional[Callable[["DetectedFeatures", BeamType], None]],
) -> "DetectedFeatures":
    """One detection pass, handing the result to ``on_detection``.

    NOTE: the old flow passed ``tolerance_point`` / ``tolerance_radius`` through to
    the detection widget, which drew a circle showing whether the needle was close
    enough. ``update_detection_ui`` has no such parameters here yet; restoring the
    overlay means adding the two fields to ``ConfirmDetection`` and drawing them in
    the widget. The approach loop is correct without it -- the operator just has to
    read the distance off the log rather than see it.
    """
    det = update_detection_ui(
        microscope=microscope,
        image_settings=image_settings,
        checkpoint=checkpoint,
        features=features,
        parent_ui=parent_ui,
        validate=validate,
        msg=msg,
    )
    if on_detection is not None:
        on_detection(det, image_settings.beam_type)
    return det


def detect_target(
    microscope: "FibsemMicroscope",
    image_settings: ImageSettings,
    feature: "Feature",
    beam_type: BeamType,
    hfw: float,
    checkpoint: str = "",
    offset: Optional[Point] = None,
    parent_ui: Optional["AutoLamellaUI"] = None,
    validate: bool = False,
    msg: str = "Target",
    on_detection: Optional[Callable[["DetectedFeatures", BeamType], None]] = None,
) -> Point:
    """Detect a stationary target and return where the needle should end up.

    Called *before* the needle is inserted. The needle occludes the target as it
    approaches, so a detection that succeeds on the first round of an approach can
    fail on the last; detecting once up front and caching the answer is what makes
    the approach survive that. See :func:`move_needle_to_feature`.

    Args:
        microscope: the instrument.
        image_settings: base settings to image with; copied, not mutated.
        feature: the stationary feature to detect.
        beam_type: which view to detect it in.
        hfw: the field of view to detect at.
        checkpoint: the model checkpoint to detect with.
        offset: added to the detected position. This is how a workflow aims at a
            point *near* a feature rather than at the feature itself -- sitting
            just below a target to weld to it, for instance.
        parent_ui: the supervising window, or None when headless.
        validate: whether the detection is confirmed by the operator.
        msg: status prefix for the detection prompt.
        on_detection: called with the detection and its beam, for saving ML
            training images.

    Returns:
        The target position in microscope image coordinates, offset included.
    """
    settings = deepcopy(image_settings)
    settings.beam_type = beam_type
    settings.hfw = hfw

    det = _detect(
        microscope,
        settings,
        checkpoint,
        [feature],
        parent_ui,
        validate,
        msg,
        on_detection,
    )

    detected = det.features[0].feature_m
    offset = offset or Point(x=0.0, y=0.0)
    target = Point(x=detected.x + offset.x, y=detected.y + offset.y)

    logging.info(
        f"Target {feature.name} ({beam_type.name}) at "
        f"({target.x * 1e6:.2f} um, {target.y * 1e6:.2f} um), including offset "
        f"({offset.x * 1e6:.2f}, {offset.y * 1e6:.2f}) um."
    )
    return target


def move_needle_to_feature(
    microscope: "FibsemMicroscope",
    image_settings: ImageSettings,
    hfw: float,
    from_feature_eb: Optional["Feature"] = None,
    to_feature_eb: Optional["Feature"] = None,
    from_feature_ib: Optional["Feature"] = None,
    to_feature_ib: Optional["Feature"] = None,
    target_eb: Optional[Point] = None,
    target_ib: Optional[Point] = None,
    eb_tolerance: float = 20e-6,
    ib_tolerance: float = 20e-6,
    movement_buffer: float = 1.0,
    checkpoints: Optional[Dict[str, str]] = None,
    default_checkpoint: str = "",
    parent_ui: Optional["AutoLamellaUI"] = None,
    validate: bool = False,
    msg: str = "Needle",
    on_detection: Optional[Callable[["DetectedFeatures", BeamType], None]] = None,
    on_refresh: Optional[Callable[[], None]] = None,
    stop_event: Optional["Event"] = None,
) -> NeedleConvergence:
    """Move the needle until it is within tolerance of its target, in both views.

    Each round detects the feature that moves with the needle (the "from" feature)
    and, unless a target position is supplied, the stationary one it is heading for
    (the "to" feature), then moves by the offset between them scaled by
    ``movement_buffer``. It repeats until both views are inside tolerance.

    **A supplied target is used in preference to detecting the "to" feature**, and
    that is the point of caching one: as the needle approaches it occludes the
    target, so a detection that worked on the first round fails on the last. The
    caller detects the target once, before the needle is inserted, and passes it
    here. With a target given, only the "from" feature is detected.

    The two views are not treated symmetrically, deliberately:

    * **SEM** is judged on straight-line distance, and corrected in x and y.
    * **FIB** is judged on the *signed vertical* offset alone, and corrected in z.
      In that view the vertical axis is the needle's z (see
      :func:`corrected_needle_delta`), which is the axis that decides whether the
      needle is about to touch down.

    Args:
        microscope: the instrument.
        image_settings: base settings to image with; copied per beam, not mutated.
        hfw: the field of view to run this approach at.
        from_feature_eb, to_feature_eb: the moving and stationary features in the
            SEM view. A view with no "from" feature is skipped entirely, as is one
            with neither a "to" feature nor a cached target.
        from_feature_ib, to_feature_ib: the same for the FIB view.
        target_eb, target_ib: pre-detected target positions. See above.
        eb_tolerance, ib_tolerance: stop once inside these, in metres.
        movement_buffer: fraction of the measured offset to actually move. Below
            1.0 approaches gradually, which avoids overshooting onto the sample.
        checkpoints: per-beam model checkpoints, keyed ``"eb"`` / ``"ib"``. A
            missing *or empty* value falls back to ``default_checkpoint`` -- ``or``
            rather than ``dict.get``'s default, so an unset model path in the UI
            falls back too rather than being passed through as "".
        default_checkpoint: the task-level checkpoint.
        parent_ui: the supervising window, or None when headless.
        validate: whether detections are confirmed by the operator.
        msg: status prefix for the detection prompts.
        on_detection: called with each detection and its beam, for saving ML
            training images.
        on_refresh: called to re-image and update the view after a move.
        stop_event: polled every round and before every move.

    Returns:
        A :class:`NeedleConvergence` describing how it went. Failing to converge is
        reported, not raised: the caller may reasonably carry on regardless.
    """
    if not microscope.get_manipulator_state():
        reason = "the manipulator is not inserted"
        logging.warning(f"Skipping the needle approach: {reason}.")
        return NeedleConvergence(converged=False, attempts=0, skipped_reason=reason)

    checkpoints = checkpoints or {}
    eb_checkpoint = checkpoints.get("eb") or default_checkpoint
    ib_checkpoint = checkpoints.get("ib") or default_checkpoint

    # A view is driven only when it has something to track *and* something to aim
    # at. Without the second half, det.features[1] below would not exist.
    use_eb = from_feature_eb is not None and (
        target_eb is not None or to_feature_eb is not None
    )
    use_ib = from_feature_ib is not None and (
        target_ib is not None or to_feature_ib is not None
    )
    if not use_eb and not use_ib:
        reason = "neither view has both a from-feature and a target"
        logging.warning(f"Skipping the needle approach: {reason}.")
        return NeedleConvergence(converged=False, attempts=0, skipped_reason=reason)

    eb_settings = deepcopy(image_settings)
    eb_settings.beam_type = BeamType.ELECTRON
    eb_settings.hfw = hfw

    ib_settings = deepcopy(image_settings)
    ib_settings.beam_type = BeamType.ION
    ib_settings.hfw = hfw

    # With a cached target there is nothing to detect but the needle itself.
    eb_features: List["Feature"] = [from_feature_eb]
    if target_eb is None and to_feature_eb is not None:
        eb_features.append(to_feature_eb)

    ib_features: List["Feature"] = [from_feature_ib]
    if target_ib is None and to_feature_ib is not None:
        ib_features.append(to_feature_ib)

    eb_distance: Optional[float] = None
    ib_distance: Optional[float] = None
    # Two counters on purpose. `rounds_since_prompt` drives the escape hatch and
    # restarts when the operator intervenes; `attempts` is the honest total, which
    # is what the caller records. Collapsing them would under-report every
    # approach that needed help.
    rounds_since_prompt = 0
    attempts = 0

    while True:
        raise_if_cancelled(stop_event, "Needle approach cancelled by user.")

        eb_ok = not use_eb
        ib_ok = not use_ib

        if use_eb:
            det = _detect(
                microscope,
                eb_settings,
                eb_checkpoint,
                eb_features,
                parent_ui,
                validate,
                msg,
                on_detection,
            )
            needle_eb = det.features[0].feature_m
            to_eb = target_eb if target_eb is not None else det.features[1].feature_m
            eb_distance = _distance(to_eb, needle_eb)
            eb_ok = eb_distance <= eb_tolerance
            logging.info(
                f"Needle approach (SEM): {eb_distance * 1e6:.2f} um from target, "
                f"tolerance {eb_tolerance * 1e6:.2f} um."
            )
            if not eb_ok:
                move_needle(
                    microscope,
                    dx=movement_buffer * (to_eb.x - needle_eb.x),
                    beam_type=BeamType.ELECTRON,
                    parent_ui=parent_ui,
                    on_refresh=on_refresh,
                    stop_event=stop_event,
                )
                move_needle(
                    microscope,
                    dy=movement_buffer * (to_eb.y - needle_eb.y),
                    beam_type=BeamType.ELECTRON,
                    parent_ui=parent_ui,
                    on_refresh=on_refresh,
                    stop_event=stop_event,
                )

        if use_ib:
            det = _detect(
                microscope,
                ib_settings,
                ib_checkpoint,
                ib_features,
                parent_ui,
                validate,
                msg,
                on_detection,
            )
            needle_ib = det.features[0].feature_m
            to_ib = target_ib if target_ib is not None else det.features[1].feature_m
            # Signed vertical offset, not a distance: see the note above.
            ib_distance = float(to_ib.y - needle_ib.y)
            ib_ok = abs(ib_distance) <= ib_tolerance
            logging.info(
                f"Needle approach (FIB): {ib_distance * 1e6:+.2f} um vertical "
                f"offset, tolerance {ib_tolerance * 1e6:.2f} um."
            )
            if not ib_ok:
                move_needle(
                    microscope,
                    dz=movement_buffer * ib_distance,
                    beam_type=BeamType.ION,
                    parent_ui=parent_ui,
                    on_refresh=on_refresh,
                    stop_event=stop_event,
                )

        attempts += 1
        rounds_since_prompt += 1

        if eb_ok and ib_ok:
            logging.info(f"The needle is within tolerance after {attempts} round(s).")
            if on_refresh is not None:
                on_refresh()
            if validate and parent_ui is not None:
                accepted = ask_user(
                    parent_ui=parent_ui,
                    msg="The needle is within tolerance of its target. Press "
                    "Continue to accept this position, or Adjust to move the "
                    "needle by hand and run the approach again.",
                    pos="Continue",
                    neg="Adjust",
                )
                if not accepted:
                    # The operator moves the needle themselves; the next round
                    # re-detects from wherever it now is and corrects the rest.
                    # Looping rather than returning is what lets them adjust as
                    # many times as they like without the task moving on.
                    logging.info(
                        "Operator asked to adjust the needle; re-running the "
                        "approach from its new position."
                    )
                    rounds_since_prompt = 0
                    continue
            return NeedleConvergence(
                converged=True,
                attempts=attempts,
                eb_distance=eb_distance,
                ib_distance=ib_distance,
            )

        if rounds_since_prompt >= MAX_CONVERGENCE_ATTEMPTS:
            logging.warning(
                f"The needle is still outside tolerance after {attempts} round(s) "
                f"(SEM {eb_distance}, FIB {ib_distance})."
            )
            if not (validate and parent_ui is not None):
                # Unsupervised: stop rather than grind on against a detection that
                # is not improving.
                return NeedleConvergence(
                    converged=False,
                    attempts=attempts,
                    eb_distance=eb_distance,
                    ib_distance=ib_distance,
                    skipped_reason="did not converge",
                )
            keep_going = ask_user(
                parent_ui=parent_ui,
                msg=f"The needle has been moved {attempts} times and is still "
                "outside tolerance. Move it manually to the correct position, then "
                "press Retry to continue the approach, or Skip to carry on with "
                "the workflow as it stands.",
                pos="Retry",
                neg="Skip",
            )
            if not keep_going:
                return NeedleConvergence(
                    converged=False,
                    attempts=attempts,
                    eb_distance=eb_distance,
                    ib_distance=ib_distance,
                    skipped_reason="skipped by the operator",
                )
            rounds_since_prompt = 0
