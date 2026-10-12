"""Editing a list of flat dataclasses in a form, one item at a time.

``build_control`` renders a ``List[SomeDataclass]`` field with this. Each item's
fields are built by ``build_control`` too, so an item field declares its label,
unit, bounds and choices with ``field_meta`` exactly as a task parameter does.
Fields marked ``advanced`` go behind a collapsed panel.

Nothing here is specific to any one list. An item class may optionally declare:

* ``ITEM_LABEL`` -- what one item is called in the editor ("HFW Step").
* ``RESET_ON_COPY`` -- fields that go back to their defaults when an item is
  added by copying the selected one, for values that must not be shared between
  two items (a key that names something both would then edit).

A field called ``name`` labels the item in the selector, and one called
``enabled`` that is False marks it as such.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import MISSING, fields
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, get_type_hints

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from fibsem.structures import get_fields_with_metadata
from fibsem.ui.widgets.custom_widgets import IconToolButton, TitledPanel, align_form

if TYPE_CHECKING:
    from fibsem.ui.widgets.form_builder import Control, FormDefaults


class DataclassListEditor(QWidget):
    """A selector over the items, with the selected item's fields below it.

    Emits :attr:`items_changed` after any edit, add, remove or reorder. The list
    returned by :meth:`get_items` holds the same item objects it was given, edited
    in place, so a caller holding them sees field edits immediately.
    """

    items_changed = pyqtSignal()

    def __init__(
        self,
        items: Optional[List[Any]],
        item_cls: Type[Any],
        defaults: Optional["FormDefaults"] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._items: List[Any] = list(items or [])
        self._item_cls = item_cls
        self._defaults = defaults
        self._item_label: str = getattr(item_cls, "ITEM_LABEL", "Item")
        field_names = {f.name for f in fields(item_cls)}
        self._name_field: Optional[str] = "name" if "name" in field_names else None
        self._controls: Dict[str, "Control"] = {}
        self._loading = False

        self._setup_ui()
        self._populate_selector()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        self.selector = QComboBox()
        self.btn_up = IconToolButton(
            "mdi:arrow-up", tooltip=f"Move this {self._item_label} earlier"
        )
        self.btn_down = IconToolButton(
            "mdi:arrow-down", tooltip=f"Move this {self._item_label} later"
        )
        self.btn_add = IconToolButton(
            "mdi:plus",
            tooltip=f"Add a {self._item_label}, copied from the selected one",
        )
        self.btn_remove = IconToolButton(
            "mdi:trash-can-outline", tooltip=f"Remove this {self._item_label}"
        )
        row.addWidget(self.selector, 1)
        for button in (self.btn_up, self.btn_down, self.btn_add, self.btn_remove):
            row.addWidget(button)
        layout.addLayout(row)

        self.fields_widget = QWidget()
        self.fields_layout = QGridLayout(self.fields_widget)
        self.fields_layout.setContentsMargins(0, 0, 0, 0)
        align_form(self.fields_layout)
        layout.addWidget(self.fields_widget)

        self.advanced_widget = QWidget()
        self.advanced_layout = QGridLayout(self.advanced_widget)
        self.advanced_layout.setContentsMargins(0, 0, 0, 0)
        align_form(self.advanced_layout)
        self.advanced_panel = TitledPanel(
            f"Advanced {self._item_label} Options", content=self.advanced_widget
        )
        self.advanced_panel.collapse()
        layout.addWidget(self.advanced_panel)

        self.selector.currentIndexChanged.connect(self._on_selection_changed)
        self.btn_up.clicked.connect(lambda: self._on_move(-1))
        self.btn_down.clicked.connect(lambda: self._on_move(1))
        self.btn_add.clicked.connect(self._on_add)
        self.btn_remove.clicked.connect(self._on_remove)

    # ------------------------------------------------------------------
    # The item list
    # ------------------------------------------------------------------

    @property
    def _current_index(self) -> int:
        return self.selector.currentIndex()

    @property
    def _current_item(self) -> Optional[Any]:
        index = self._current_index
        return self._items[index] if 0 <= index < len(self._items) else None

    def _item_text(self, index: int, item: Any) -> str:
        name = ""
        if self._name_field is not None:
            name = str(getattr(item, self._name_field, "") or "")
        text = f"{index + 1}. {name or f'{self._item_label} {index + 1}'}"
        if getattr(item, "enabled", True) is False:
            text += "  (disabled)"
        return text

    def _populate_selector(self, select_index: int = 0) -> None:
        self._loading = True
        self.selector.clear()
        for index, item in enumerate(self._items):
            self.selector.addItem(self._item_text(index, item))
        if self._items:
            self.selector.setCurrentIndex(
                max(0, min(select_index, len(self._items) - 1))
            )
        self._loading = False

        count = len(self._items)
        index = self._current_index
        # One item always remains: a list with none in it is not a state the
        # task can run, and an empty editor has nothing to copy a new item from.
        self.btn_remove.setEnabled(count > 1)
        self.btn_up.setEnabled(index > 0)
        self.btn_down.setEnabled(0 <= index < count - 1)
        self._rebuild_fields()

    def _refresh_item_text(self) -> None:
        index = self._current_index
        item = self._current_item
        if item is not None:
            self.selector.setItemText(index, self._item_text(index, item))

    # ------------------------------------------------------------------
    # Fields
    # ------------------------------------------------------------------

    @staticmethod
    def _clear(layout: QGridLayout) -> None:
        while layout.count():
            child = layout.takeAt(0)
            if child is not None and child.widget() is not None:
                child.widget().setParent(None)

    def _rebuild_fields(self) -> None:
        """Build a control per field of the selected item."""
        # Imported here: build_control builds this editor for a list field, so a
        # top-level import of form_builder would be a cycle.
        from fibsem.ui.widgets.form_builder import FormDefaults, build_control

        self._clear(self.fields_layout)
        self._clear(self.advanced_layout)
        self._controls = {}

        item = self._current_item
        self.fields_widget.setVisible(item is not None)
        self.advanced_panel.setVisible(item is not None)
        if item is None:
            return

        metadata = get_fields_with_metadata(self._item_cls)
        try:
            hints = get_type_hints(self._item_cls)
        except Exception as e:
            logging.warning(
                f"Could not resolve the types of {self._item_cls.__name__}: {e}"
            )
            hints = {}

        rows = {False: 0, True: 0}
        for f in fields(item):
            m = metadata.get(f.name, {})
            if m.get("hidden", False):
                continue
            control = build_control(
                m,
                getattr(item, f.name),
                annotation=hints.get(f.name),
                defaults=self._defaults or FormDefaults(),
            )
            if control is None:
                continue

            label = QLabel(m.get("label") or f.name.replace("_", " ").title())
            if m.get("tooltip"):
                label.setToolTip(m["tooltip"])
                control.widget.setToolTip(m["tooltip"])

            advanced = bool(m.get("advanced", False))
            grid = self.advanced_layout if advanced else self.fields_layout
            grid.addWidget(label, rows[advanced], 0)
            grid.addWidget(control.widget, rows[advanced], 1)
            rows[advanced] += 1

            if control.editable:
                control.connect(lambda name=f.name: self._on_field_changed(name))
            self._controls[f.name] = control

        self.advanced_panel.setVisible(rows[True] > 0)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        if self._loading:
            return
        index = self._current_index
        self.btn_up.setEnabled(index > 0)
        self.btn_down.setEnabled(0 <= index < len(self._items) - 1)
        self._rebuild_fields()

    def _on_field_changed(self, field_name: str) -> None:
        item = self._current_item
        control = self._controls.get(field_name)
        if self._loading or item is None or control is None:
            return
        setattr(item, field_name, control.read())
        self._refresh_item_text()
        self.items_changed.emit()

    def _on_add(self) -> None:
        current = self._current_item
        if current is not None:
            new_item = deepcopy(current)
            for name in getattr(self._item_cls, "RESET_ON_COPY", ()):
                setattr(new_item, name, self._field_default(name))
        else:
            new_item = self._item_cls()
        if self._name_field is not None:
            setattr(
                new_item,
                self._name_field,
                self._unique_name(str(getattr(new_item, self._name_field) or "")),
            )

        insert_at = self._current_index + 1 if self._items else 0
        self._items.insert(insert_at, new_item)
        self._populate_selector(select_index=insert_at)
        self.items_changed.emit()

    def _field_default(self, name: str) -> Any:
        for f in fields(self._item_cls):
            if f.name == name:
                if f.default is not MISSING:
                    return f.default
                if f.default_factory is not MISSING:  # type: ignore[misc]
                    return f.default_factory()  # type: ignore[misc]
        return None

    def _unique_name(self, base: str) -> str:
        base = base or self._item_label
        existing = {str(getattr(item, self._name_field, "")) for item in self._items}
        if base not in existing:
            return base
        suffix = 2
        while f"{base}-{suffix}" in existing:
            suffix += 1
        return f"{base}-{suffix}"

    def _on_remove(self) -> None:
        index = self._current_index
        if not (0 <= index < len(self._items)) or len(self._items) <= 1:
            return
        name = self._item_text(index, self._items[index])
        reply = QMessageBox.question(
            self,
            f"Remove {self._item_label}",
            f"Remove {name}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._items.pop(index)
        self._populate_selector(select_index=max(0, index - 1))
        self.items_changed.emit()

    def _on_move(self, delta: int) -> None:
        index = self._current_index
        target = index + delta
        if not (0 <= index < len(self._items) and 0 <= target < len(self._items)):
            return
        self._items[index], self._items[target] = (
            self._items[target],
            self._items[index],
        )
        self._populate_selector(select_index=target)
        self.items_changed.emit()

    # ------------------------------------------------------------------
    # Value access
    # ------------------------------------------------------------------

    def get_items(self) -> List[Any]:
        return self._items

    def set_items(self, items: Optional[List[Any]]) -> None:
        index = self._current_index
        self._items = list(items or [])
        self._populate_selector(select_index=max(0, index))
