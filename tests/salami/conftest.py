import pytest


@pytest.fixture(autouse=True)
def isolated_recent_experiments(tmp_path, monkeypatch):
    """Keep tests out of the operator's recent-experiments list.

    Every experiment the UI adopts is recorded there, and the tests create a lot
    of them in temporary directories that would otherwise fill the real Load
    dialog with entries that vanish.
    """
    from fibsem import config
    from fibsem.applications.salami import preferences, recents

    path = tmp_path / "salami-recent-experiments.yaml"
    monkeypatch.setattr(recents, "RECENTS_PATH", str(path))
    # Edit > Preferences writes both of these; keep the real ones untouched.
    monkeypatch.setattr(
        preferences, "PREFERENCES_PATH", str(tmp_path / "salami-preferences.yaml")
    )
    monkeypatch.setattr(
        config, "USER_PREFERENCES_PATH", str(tmp_path / "user-preferences.yaml")
    )
    return path
