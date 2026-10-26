"""User-interaction helpers for Salami workflows.

`ask_user` is called from the experiment worker thread to request a yes/no
decision from the operator. It emits `user_prompt_signal` on the parent UI
(handled in the main thread) and busy-waits on a flag the slot clears when
the user clicks a button. Headless mode (no parent UI) auto-confirms.
"""

import logging
import time
from typing import TYPE_CHECKING, Any, Optional

from fibsem.applications.salami.workflow_update_schema import SalamiWorkflowUpdate
from fibsem.structures import FibsemImage

if TYPE_CHECKING:
    from fibsem.applications.salami.ui.SalamiUI import SalamiUI


def _check_for_abort(
    parent_ui: Optional[Any], msg: str = "Workflow aborted by user."
) -> None:
    """Raise InterruptedError if the parent UI's stop event has been set."""
    if parent_ui is None:
        return
    stop_event = getattr(parent_ui, "_workflow_stop_event", None)
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError(msg)


def ask_user(
    parent_ui: Optional[Any],
    workflow_msg: str,
    msg: str,
    pos: str,
    neg: Optional[str] = None,
    mill: Optional[bool] = None,
) -> bool:
    """Block the worker until the operator clicks `pos` (True) or `neg` (False).

    In headless mode (`parent_ui is None`) we log and return True so scripted
    runs aren't gated on a UI that doesn't exist.
    """
    if parent_ui is None:
        logging.warning(
            f"User input requested in headless mode: {msg!r}; auto-confirming."
        )
        return True

    parent_ui.WAITING_FOR_USER_INTERACTION = True
    parent_ui.workflow_update_signal.emit(
        SalamiWorkflowUpdate(
            task="user_input",
            msg=msg,
            workflow_msg=workflow_msg,
            pos=pos,
            neg=neg,
            milling_enabled=mill,
            status="paused",
        )
    )

    logging.info("WAITING_FOR_USER_INTERACTION...")
    while parent_ui.WAITING_FOR_USER_INTERACTION:
        _check_for_abort(parent_ui=parent_ui)
        time.sleep(1)

    # Clear the prompt: no pos/neg means the buttons are hidden again.
    parent_ui.workflow_update_signal.emit(
        SalamiWorkflowUpdate(task="user_input", msg="")
    )

    return parent_ui.USER_RESPONSE


def ask_user_continue_workflow(
    parent_ui: Optional[Any],
    workflow_msg: str,
    msg: str = "Continue with the next stage?",
    validate: bool = True,
) -> bool:
    """Ask whether to proceed; returns True immediately when validation is off."""
    if not validate:
        return True
    return ask_user(
        parent_ui=parent_ui,
        workflow_msg=workflow_msg,
        msg=msg,
        pos="Continue",
        neg="Exit",
    )


def set_images_ui(
    parent_ui: Optional["SalamiUI"],
    eb_image: Optional[FibsemImage] = None,
    ib_image: Optional[FibsemImage] = None,
):

    if parent_ui is None:
        logging.info("No parent UI, skipping image update.")
        return

    _check_for_abort(parent_ui)

    # TMP: prevent milling images overwriting existing
    if eb_image is not None:
        eb_image.metadata.image_settings.save = False
    if ib_image is not None:
        ib_image.metadata.image_settings.save = False

    parent_ui.WAITING_FOR_UI_UPDATE = True
    parent_ui.workflow_update_signal.emit(
        SalamiWorkflowUpdate(
            task="images",
            msg="Updating Images",
            sem_image=eb_image,
            fib_image=ib_image,
        )
    )

    while parent_ui.WAITING_FOR_UI_UPDATE:
        time.sleep(0.5)
