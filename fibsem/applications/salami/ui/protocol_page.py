"""The Protocol page: everything about *what the experiment will do*.

Three columns, following the same shape as
`fibsem/ui/widgets/autolamella_task_config_editor.py`:

    experiment information │ task parameters │ preview image

The page does not build the first two columns. It is handed widgets the main
window already owns — the experiment info group box and the task editor — and
arranges them. That keeps every existing handler working against the same
objects, and means this file is layout plus the preview canvas, nothing else.

The preview is the reason the page exists rather than the settings staying in
the right-hand tab strip: the sync buttons live in column 2, and an operator
pressing "Sync from Microscope" wants to see the resulting image without
changing pages to find it.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.salami.tasks import ImageGeometry
from fibsem.structures import BeamType, FibsemImage, FibsemRectangle
from fibsem.ui import stylesheets
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.overlays.alignment_overlay import AlignmentAreaOverlay

NO_IMAGE_TEXT = (
    "No preview yet.\n\n"
    "Sync from or to the microscope to acquire an image at these settings."
)


@dataclass
class RectangleSpec:
    """A named region drawn over the preview, bound to a config field.

    `on_changed` receives the new `FibsemRectangle` when the operator finishes a
    drag or resize. It is not called for programmatic updates, so writing a value
    back into the form cannot loop.
    """

    key: str
    label: str
    colour: str
    area: FibsemRectangle
    editable: bool = True
    on_changed: Optional[Callable[[FibsemRectangle], None]] = None


def _in_scroll_area(content: QWidget) -> QScrollArea:
    """Wrap a column so a long form scrolls rather than squashing its neighbours."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(content)
    return scroll


class SyncPreviewPanel(QWidget):
    """Column 3: the image the last sync acquired.

    Deliberately a plain viewer. The image is a preview, never saved (see
    `tasks.acquire_preview_image`), so there is nothing here to configure — it
    answers "what does the protocol actually look like on this sample right
    now", and that is all.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.title_label = QLabel("Preview")
        self.title_label.setStyleSheet("font-weight: bold;")
        header.addWidget(self.title_label)
        header.addStretch()
        layout.addLayout(header)

        self.caption_label = QLabel("")
        self.caption_label.setWordWrap(True)
        self.caption_label.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        layout.addWidget(self.caption_label)

        self.canvas = FibsemImageCanvas(parent=self)
        self.canvas.setVisible(False)
        layout.addWidget(self.canvas, stretch=1)

        self.placeholder = QLabel(NO_IMAGE_TEXT)
        self.placeholder.setWordWrap(True)
        self.placeholder.setAlignment(Qt.AlignCenter)  # type: ignore
        self.placeholder.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        layout.addWidget(self.placeholder, stretch=1)

        # Colour key for the rectangles currently drawn. Without it the boxes are
        # anonymous, and two of them are the norm.
        self.legend_row = QHBoxLayout()
        self.legend_row.setContentsMargins(0, 0, 0, 0)
        self.legend_row.setSpacing(10)
        layout.addLayout(self.legend_row)

        self._rectangles: Dict[str, AlignmentAreaOverlay] = {}
        self._legend_labels: Dict[str, QLabel] = {}
        self._geometry: Optional[ImageGeometry] = None

    @property
    def image_geometry(self) -> Optional[ImageGeometry]:
        """Scale of the displayed image, for rectangles that mean metres.

        None until something has been shown — a region drawn over no image has
        no size, so callers that convert have to check.
        """
        return self._geometry

    def show_image(self, image: FibsemImage, caption: str = "") -> None:
        """Display `image`, swapping out the placeholder on first use."""
        try:
            self.canvas.set_image(image)
        except Exception:
            logging.warning("Could not display the sync preview image", exc_info=True)
            return
        try:
            self._geometry = ImageGeometry.from_image(image)
        except Exception:
            logging.debug("Preview image carries no pixel size", exc_info=True)
            self._geometry = None
        self.canvas.setVisible(True)
        self.placeholder.setVisible(False)
        self.caption_label.setText(caption)

    def clear_image(self) -> None:
        self.canvas.clear()
        self.canvas.setVisible(False)
        self.placeholder.setVisible(True)
        self.caption_label.clear()
        self._geometry = None

    # ── rectangles ────────────────────────────────────────────────────────
    #
    # Thin management over `AlignmentAreaOverlay`, which already does the work:
    # drag, corner-resize, clamping, and normalised FibsemRectangle in/out. The
    # canvas holds a list of overlays, so several can coexist — what is added
    # here is naming them, colour-keying them, and binding each to a config
    # field.

    def draw_rectangle(self, spec: RectangleSpec) -> AlignmentAreaOverlay:
        """Draw (or update) the named region. Returns the overlay."""
        overlay = self._rectangles.get(spec.key)
        if overlay is None:
            overlay = AlignmentAreaOverlay(color=spec.colour, editable=spec.editable)
            self.canvas.add_overlay(overlay)
            self._rectangles[spec.key] = overlay
            self._add_legend_entry(spec)
        else:
            overlay.set_editable(spec.editable)

        # Reconnect each time: the bound config field changes when the operator
        # selects a different task, and a stale callback would write to the old one.
        try:
            overlay.alignment_area_changed.disconnect()
        except TypeError:
            pass  # nothing connected yet
        if spec.on_changed is not None:
            overlay.alignment_area_changed.connect(spec.on_changed)

        overlay.set_area(spec.area)
        overlay.set_visible(True)
        return overlay

    def set_rectangle_area(self, key: str, area: FibsemRectangle) -> None:
        """Reposition a rectangle from the config, without emitting a change.

        `set_area` is programmatic, so this cannot feed back into `on_changed`.
        """
        overlay = self._rectangles.get(key)
        if overlay is not None:
            overlay.set_area(area)

    def set_rectangle_visible(self, key: str, visible: bool) -> None:
        """Show or hide a rectangle, keeping its position.

        Editability follows visibility: `RectOverlay` decides whether to handle
        a press from `_interactive` alone, so a hidden-but-interactive rectangle
        would still swallow clicks meant for the one underneath it.
        """
        overlay = self._rectangles.get(key)
        if overlay is None:
            return
        overlay.set_visible(visible)
        overlay.set_editable(visible)
        label = self._legend_labels.get(key)
        if label is not None:
            label.setVisible(visible)

    def clear_rectangles(self) -> None:
        for overlay in self._rectangles.values():
            try:
                overlay.alignment_area_changed.disconnect()
            except TypeError:
                pass
            self.canvas.remove_overlay(overlay)
        self._rectangles.clear()
        self._clear_legend()

    def rectangle_keys(self) -> list:
        return list(self._rectangles)

    def _add_legend_entry(self, spec: RectangleSpec) -> None:
        label = QLabel(f"■ {spec.label}")
        label.setStyleSheet(f"color: {spec.colour}; font-size: 11px;")
        self.legend_row.addWidget(label)
        self._legend_labels[spec.key] = label
        if self.legend_row.count() == len(self._legend_labels):
            self.legend_row.addStretch()

    def _clear_legend(self) -> None:
        while self.legend_row.count():
            item = self.legend_row.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._legend_labels.clear()


class SalamiProtocolPage(QWidget):
    """Three-column protocol editor. See the module docstring."""

    def __init__(
        self,
        info_widget: QWidget,
        task_selector: QWidget,
        task_parameters: QWidget,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Column 1: what this experiment is ---
        col1 = QWidget()
        col1_layout = QVBoxLayout(col1)
        col1_layout.setContentsMargins(4, 4, 4, 4)
        col1_layout.addWidget(info_widget)
        col1_layout.addStretch()

        # --- Column 2: what its tasks do ---
        col2 = QWidget()
        col2_layout = QVBoxLayout(col2)
        col2_layout.setContentsMargins(4, 4, 4, 4)
        selector_row = QHBoxLayout()
        selector_row.setContentsMargins(0, 0, 0, 0)
        selector_row.addWidget(QLabel("Task"))
        selector_row.addWidget(task_selector, stretch=1)
        col2_layout.addLayout(selector_row)
        col2_layout.addWidget(task_parameters, stretch=1)

        # --- Column 3: what that looks like on the sample ---
        self.preview = SyncPreviewPanel(parent=self)

        splitter = QSplitter(Qt.Horizontal)  # type: ignore
        splitter.addWidget(_in_scroll_area(col1))
        splitter.addWidget(_in_scroll_area(col2))
        splitter.addWidget(self.preview)
        splitter.setSizes([300, 420, 700])
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter)

    # -- preview passthrough, so callers don't reach through .preview ----------

    def show_preview(self, image: FibsemImage, caption: str = "") -> None:
        self.preview.show_image(image, caption)

    def clear_preview(self) -> None:
        self.preview.clear_image()

    def draw_rectangle(self, spec: RectangleSpec):
        return self.preview.draw_rectangle(spec)

    def set_rectangle_area(self, key: str, area: FibsemRectangle) -> None:
        self.preview.set_rectangle_area(key, area)

    def set_rectangle_visible(self, key: str, visible: bool) -> None:
        self.preview.set_rectangle_visible(key, visible)

    def clear_rectangles(self) -> None:
        self.preview.clear_rectangles()

    @property
    def image_geometry(self):
        return self.preview.image_geometry

    def rectangle_area(self, key: str) -> Optional[FibsemRectangle]:
        """The region as it currently sits on screen, or None if not drawn."""
        overlay = self.preview._rectangles.get(key)
        return overlay.get_area() if overlay is not None else None


def preview_caption(result) -> str:
    """A one-line description of what a MicroscopeSyncResult acquired."""
    settings = result.image_settings
    beam = "FIB" if settings.beam_type is BeamType.ION else "SEM"
    verb = "Read from" if result.direction == "from_microscope" else "Applied to"
    return f"{beam} · {verb.lower()} microscope · {result.summary}"
