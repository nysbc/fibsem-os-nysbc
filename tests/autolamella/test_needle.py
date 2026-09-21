"""Tests for the needle (manipulator) movement primitives.

The axis conventions in :func:`corrected_needle_delta` are the reason this file
exists. Which way the needle actually moves for a requested delta depends on the
beam being watched and on the scan rotation, and getting a sign wrong drives the
manipulator into the sample -- a failure with a physical cost, not a red test.

They were previously four inline sign flips in the middle of a 200-line routine,
where nothing could reach them. The table below is transcribed from that routine's
behaviour, so a change to the rules has to be a deliberate edit here rather than a
silent one there.
"""

import pytest

from fibsem.applications.autolamella.workflows.needle import (
    NeedleConvergence,
    corrected_needle_delta,
    move_needle_to_feature,
)
from fibsem.structures import BeamType, ImageSettings

# (scan_rotation_deg, beam, dx, dy, dz) -> (corrected_dx, corrected_dy)
#
# The two rules, visible in the table:
#   * at 180 degrees the image is flipped, so dx and dy inverted;
#   * in the ION view the second axis is the needle's z, not its y -- so dy is
#     ignored there and dz drives it, inverted at 0 degrees.
AXIS_CASES = [
    # SEM: dx/dy pass through at 0, both invert at 180, dz is never used.
    (0.0, BeamType.ELECTRON, 5e-6, 7e-6, -10e-6, 5e-6, 7e-6),
    (180.0, BeamType.ELECTRON, 5e-6, 7e-6, -10e-6, -5e-6, -7e-6),
    (0.0, BeamType.ELECTRON, 5e-6, 7e-6, 0.0, 5e-6, 7e-6),
    # FIB: dy is ignored entirely; dz drives the second axis, inverted at 0.
    (0.0, BeamType.ION, 5e-6, 7e-6, -10e-6, 5e-6, 10e-6),
    (180.0, BeamType.ION, 5e-6, 7e-6, -10e-6, -5e-6, -10e-6),
    # ...which means changing dy alone is a no-op in the ion view.
    (0.0, BeamType.ION, 5e-6, 0.0, -10e-6, 5e-6, 10e-6),
    (0.0, BeamType.ION, 5e-6, 99e-6, -10e-6, 5e-6, 10e-6),
    # An off-axis scan rotation matches neither rule, so nothing inverts; dz
    # still drives the ion view's second axis.
    (90.0, BeamType.ELECTRON, 5e-6, 7e-6, -10e-6, 5e-6, 7e-6),
    (90.0, BeamType.ION, 5e-6, 7e-6, -10e-6, 5e-6, -10e-6),
]


@pytest.mark.parametrize(
    "scan_rotation_deg,beam_type,dx,dy,dz,expected_dx,expected_dy", AXIS_CASES
)
def test_corrected_needle_delta(
    scan_rotation_deg, beam_type, dx, dy, dz, expected_dx, expected_dy
):
    """Each rule, at each scan rotation, in each beam."""
    result = corrected_needle_delta(
        dx=dx, dy=dy, dz=dz, beam_type=beam_type, scan_rotation_deg=scan_rotation_deg
    )
    assert result == pytest.approx((expected_dx, expected_dy))


def test_ion_view_ignores_dy_entirely():
    """The mirrored-manipulator rule, stated on its own.

    Worth its own test rather than only a table row: a caller that reasonably
    expects dy to do something in the ion view gets no movement at all, and that
    is the single most surprising thing about this function.
    """
    moved = corrected_needle_delta(
        dx=0.0, dy=50e-6, dz=0.0, beam_type=BeamType.ION, scan_rotation_deg=0.0
    )
    assert moved == (0.0, 0.0)


def test_zero_delta_is_zero_in_every_frame():
    """No rule turns a request to stay put into a movement."""
    for rotation in (0.0, 90.0, 180.0):
        for beam in (BeamType.ELECTRON, BeamType.ION):
            assert corrected_needle_delta(
                dx=0.0, dy=0.0, dz=0.0, beam_type=beam, scan_rotation_deg=rotation
            ) == (0.0, 0.0)


class _RetractedMicroscope:
    """The one question the approach asks before it does anything."""

    def get_manipulator_state(self) -> bool:
        return False


def test_approach_is_skipped_when_the_manipulator_is_retracted():
    """A retracted manipulator is reported, not raised.

    The task carries on to its milling and relative-move steps, which is the
    behaviour a workflow depends on -- so this must not become an exception.
    """
    result = move_needle_to_feature(
        microscope=_RetractedMicroscope(),
        image_settings=ImageSettings(),
        hfw=100e-6,
    )
    assert isinstance(result, NeedleConvergence)
    assert result.converged is False
    assert result.attempts == 0
    assert result.skipped_reason is not None


class _InsertedMicroscope:
    def get_manipulator_state(self) -> bool:
        return True


def test_approach_is_skipped_when_no_features_are_configured():
    """A view needs both something to track and something to aim at.

    Without the second half the loop would index a detection result that has only
    one feature in it, so this is refused up front rather than part way through.
    """
    result = move_needle_to_feature(
        microscope=_InsertedMicroscope(),
        image_settings=ImageSettings(),
        hfw=100e-6,
        from_feature_eb=None,
        to_feature_eb=None,
    )
    assert result.converged is False
    assert result.attempts == 0
    assert result.skipped_reason is not None
