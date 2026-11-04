"""When the acquire buttons ask before imaging: ImagingPreferences.exceeds_limit.

No Qt here on purpose, so this runs on every CI job rather than only the ui-tests
one. The dialog that asks is in FibsemImageSettingsWidget; this is the decision it
asks on.
"""

import pytest

from fibsem.config import ImagingPreferences, UserPreferences


def _limits(electron=None, ion=None) -> ImagingPreferences:
    return ImagingPreferences(
        max_imaging_current_electron=electron, max_imaging_current_ion=ion
    )


def test_no_limit_never_asks():
    limits = _limits()

    assert not limits.exceeds_limit("ION", 65e-9)
    assert not limits.exceeds_limit("ELECTRON", None)


@pytest.mark.parametrize(
    "current, asks",
    [(30e-12, False), (50e-12, False), (0.5e-9, True), (65e-9, True)],
)
def test_asks_only_above_the_limit(current, asks):
    """At the limit is within it: the limit is one of the instrument's currents,
    and imaging at it is what choosing it means."""
    limits = _limits(ion=50e-12)

    assert limits.exceeds_limit("ION", current) is asks


def test_an_unreadable_current_asks():
    """The check is for the current nobody noticed, so not knowing it must not wave
    the acquisition through."""
    assert _limits(ion=50e-12).exceeds_limit("ION", None)


def test_each_beam_has_its_own_limit():
    """SEM and FIB imaging currents differ by orders of magnitude."""
    limits = _limits(electron=1.6e-9, ion=50e-12)

    assert not limits.exceeds_limit("ELECTRON", 0.4e-9)
    assert limits.exceeds_limit("ION", 0.4e-9)


def test_a_preferences_file_without_the_section_gets_the_defaults():
    prefs = UserPreferences.from_dict({"movement": {}})

    assert prefs.imaging == ImagingPreferences()
