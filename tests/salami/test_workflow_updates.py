"""Every worker -> UI message travels as one type: SalamiWorkflowUpdate.

Progress reports and UI actions used to share the signal as two incompatible
shapes (a dataclass from tasks, plain dicts from the interaction helpers), so
the slot crashed on whichever one it wasn't written for. These tests pin the
payload type at each emitter.
"""

import threading

import pytest
import yaml

from fibsem.applications.salami.structures import RunContext
from fibsem.applications.salami.workflow_ui import ask_user, set_images_ui
from fibsem.applications.salami.workflow_update_schema import (
    TRANSIENT_FIELDS,
    SalamiWorkflowUpdate,
    format_duration,
)


class FakeSignal:
    """Stands in for a pyqtSignal, recording what was emitted."""

    def __init__(self, on_emit=None):
        self.emitted = []
        self._on_emit = on_emit

    def emit(self, payload):
        self.emitted.append(payload)
        if self._on_emit is not None:
            self._on_emit(payload)


class FakeUI:
    """Minimal stand-in for SalamiUI's worker-facing surface."""

    def __init__(self, user_response=True):
        self.WAITING_FOR_USER_INTERACTION = False
        self.WAITING_FOR_UI_UPDATE = False
        self.USER_RESPONSE = user_response
        self._workflow_stop_event = threading.Event()
        # Answer the prompt as soon as it arrives, as a click would.
        self.workflow_update_signal = FakeSignal(on_emit=self._respond)

    def _respond(self, payload):
        self.WAITING_FOR_USER_INTERACTION = False
        self.WAITING_FOR_UI_UPDATE = False


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_to_dict_omits_live_objects():
    """asdict() would try to deep-copy whole images; to_dict must not."""
    update = SalamiWorkflowUpdate(
        task="acquisition",
        msg="hello",
        current_step=3,
        sem_image=object(),
        milling_config=object(),
        pos="Continue",
        clear_milling_config=True,
    )

    data = update.to_dict()

    assert data["task"] == "acquisition"
    assert data["current_step"] == 3
    for name in TRANSIENT_FIELDS:
        assert name not in data


def test_to_dict_is_yaml_safe():
    update = SalamiWorkflowUpdate(
        task="milling", msg="m", current_step=1, total_steps=10
    )
    # timestamp is a datetime; yaml handles it, images would not
    assert yaml.safe_load(yaml.safe_dump(update.to_dict())) is not None


def test_task_and_msg_are_optional():
    """UI-only updates shouldn't have to invent a task name."""
    update = SalamiWorkflowUpdate()
    assert update.task == "ui"
    assert update.msg == ""


def test_requests_user_input_tracks_pos():
    assert SalamiWorkflowUpdate(pos="Continue").requests_user_input
    assert not SalamiWorkflowUpdate(msg="just progress").requests_user_input


def test_overall_progress_accounts_for_task_position():
    update = SalamiWorkflowUpdate(
        current_step=5, total_steps=10, current_task_index=1, total_tasks=2
    )
    # half way through the steps, plus half of one step's tasks
    assert update.overall_progress == pytest.approx(55.0)


@pytest.mark.parametrize(
    "seconds,expected",
    [(0.4, "00:00:00"), (61, "00:01:01"), (3661, "01:01:01"), (172800, "48:00:00")],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------


def test_ask_user_emits_the_dataclass_with_buttons():
    ui = FakeUI(user_response=True)

    result = ask_user(
        parent_ui=ui, workflow_msg="Setup", msg="Confirm?", pos="Yes", neg="No"
    )

    assert result is True
    prompt = ui.workflow_update_signal.emitted[0]
    assert isinstance(prompt, SalamiWorkflowUpdate)
    assert prompt.msg == "Confirm?"
    assert prompt.workflow_msg == "Setup"
    assert (prompt.pos, prompt.neg) == ("Yes", "No")
    assert prompt.requests_user_input


def test_ask_user_clears_the_prompt_afterwards():
    ui = FakeUI()
    ask_user(parent_ui=ui, workflow_msg="Setup", msg="Confirm?", pos="Yes", neg="No")

    clear = ui.workflow_update_signal.emitted[-1]
    assert isinstance(clear, SalamiWorkflowUpdate)
    assert clear.msg == ""
    assert not clear.requests_user_input  # buttons hidden again


def test_ask_user_returns_the_negative_response():
    ui = FakeUI(user_response=False)
    assert (
        ask_user(parent_ui=ui, workflow_msg="Setup", msg="?", pos="Yes", neg="No")
        is False
    )


def test_ask_user_auto_confirms_headless():
    assert ask_user(parent_ui=None, workflow_msg="Setup", msg="?", pos="Yes") is True


def test_set_images_ui_emits_the_dataclass():
    ui = FakeUI()
    sem, fib = _fake_image(), _fake_image()

    set_images_ui(parent_ui=ui, eb_image=sem, ib_image=fib)

    update = ui.workflow_update_signal.emitted[-1]
    assert isinstance(update, SalamiWorkflowUpdate)
    assert update.sem_image is sem
    assert update.fib_image is fib


def test_run_context_emits_the_dataclass():
    ui = FakeUI()
    ctx = RunContext(experiment_path="", parent_ui=ui)

    ctx.emit(SalamiWorkflowUpdate(task="milling", msg="milling slice 1"))

    assert isinstance(ui.workflow_update_signal.emitted[-1], SalamiWorkflowUpdate)


def test_run_context_emit_is_a_noop_headless():
    RunContext(experiment_path="", parent_ui=None).emit(
        SalamiWorkflowUpdate(task="milling", msg="no ui")
    )  # must not raise


def _fake_image():
    """A stand-in for FibsemImage: set_images_ui only touches this attribute."""

    class _Settings:
        save = True

    class _Meta:
        image_settings = _Settings()

    class _Image:
        metadata = _Meta()

    return _Image()
