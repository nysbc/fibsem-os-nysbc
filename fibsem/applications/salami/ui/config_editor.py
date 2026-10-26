"""Dataclass-driven config editors.

Renders an editor for any dataclass by walking `dataclasses.fields()` and
building a control per field — checkbox for `bool`, spinbox for numbers, combo
for enums and fixed choices — recursing into nested dataclasses as collapsible
sections. Add a field to a task config and its editor appears with no UI code to
change.

The control for one field comes from the shared
`fibsem.ui.widgets.form_builder.build_control`, the builder the AutoLamella task
form and the milling forms use. What stays here is the layout (the recursion,
the sections and the view spec) and three wrappers for what Salami needs on top:
an "unset" checkbox, a combo that keeps an unsupported value visible, and a
read-only display.

Nothing here touches the microscope. These widgets edit a *protocol*: values are
written into the config and applied only when the task runs. That rules out
`FibsemBeamSettingsWidget`, whose handlers push every keystroke straight to the
hardware — right for a live control panel, wrong for a protocol that may be days
from running, or running right now on a different slice.

Three details of this codebase shape the implementation:

- **Annotations may be strings.** `fibsem/structures.py` uses
  `from __future__ import annotations`, so `field.type` there is `'Optional[BeamSettings]'`
  rather than a type. Everything resolves through `typing.get_type_hints()`;
  comparing `field.type == bool` would silently fall through to a text box.
- **Core structures carry no form metadata.** Units therefore come from
  `FIELD_HINTS`, keyed by field name, rather than from the dataclasses — which
  keeps core `fibsem` structures untouched. An unhinted float still gets an
  editor, just in raw SI.
- **Optional is everywhere.** `BeamSettings` is ten `Optional` fields, and
  `None` means "leave the microscope as it is". `optional_control` makes that an
  explicit checkbox rather than overloading zero.
"""

import dataclasses
import functools
import inspect
import logging
import typing
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from fibsem.config import STANDARD_RESOLUTIONS_ZIP
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, field_meta
from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import TitledPanel
from fibsem.ui.widgets.form_builder import Control, FormDefaults, build_control
from fibsem.utils import format_value

# ---------------------------------------------------------------------------
# Display hints
# ---------------------------------------------------------------------------
# Keyed by field name, applied at any depth, in the shared form vocabulary
# (`fibsem.structures.DEFAULT_FIELD_METADATA`). `field_meta` checks the keys when
# this module is imported, so a misspelt one is an error rather than a field
# that silently renders without its unit. `scale` is the display multiplier:
# display = stored * scale, and the SI prefix is derived from it (1e6 + "m"
# renders as "μm").
#
# `microscope_parameter` names a key to query the microscope for
# (`FibsemMicroscope.get_available_values`) and renders a dropdown of exactly
# what the column supports, rather than a free number the hardware would reject
# when the task runs. It falls back to numeric entry when no microscope is
# connected, or when the backend reports nothing for that key.
#
# Two keys are Salami's own and sit beside the checked part. An `Optional`
# field is wrapped in a checkbox that means "unset"; `always_set` opts out of
# that, and `checkbox_label` / `checkbox_tooltip` reword it where unset means
# "feature off" rather than "leave the column alone".
#
# A field with no entry still gets an editor — it just shows raw SI. This is
# where to add a unit when a new field reads badly, and it is deliberately not
# metadata on the dataclasses, so core fibsem structures stay untouched.

RESOLUTION_ITEMS: List[Tuple[int, int]] = [
    tuple(value) for _, value in STANDARD_RESOLUTIONS_ZIP
]
_RESOLUTION_LABELS: Dict[Tuple[int, int], str] = {
    tuple(value): label for label, value in STANDARD_RESOLUTIONS_ZIP
}


def format_resolution(value: Any) -> str:
    """`(1536, 1024)` -> `"1536x1024"`, the label the image settings widget uses."""
    value = tuple(value)
    return _RESOLUTION_LABELS.get(value, "x".join(str(v) for v in value))


FIELD_HINTS: Dict[str, dict] = {
    "dwell_time": field_meta(scale=1e6, unit="s", decimals=2, minimum=0.0, maximum=1e6),
    "hfw": field_meta(scale=1e6, unit="m", decimals=2, minimum=0.0, maximum=1e5),
    # No unset checkbox: the spinbox is the value. `resolve_beam_defaults` seeds
    # it from the column's live working distance on connect, so it starts at a
    # real number rather than zero.
    "working_distance": {
        **field_meta(scale=1e3, unit="m", decimals=4, minimum=0.0, maximum=1e3),
        "always_set": True,
    },
    # Note the parameter names are the microscope's, not the field's: beam
    # current is "current" to get_available_values.
    "voltage": field_meta(
        scale=1e-3,
        unit="V",
        decimals=3,
        minimum=0.0,
        maximum=1e3,
        microscope_parameter="voltage",
    ),
    "beam_current": field_meta(
        scale=1e9,
        unit="A",
        decimals=4,
        minimum=0.0,
        maximum=1e6,
        microscope_parameter="current",
    ),
    "step_size": field_meta(scale=1e9, unit="m", decimals=2, minimum=0.0, maximum=1e6),
    "scan_rotation": field_meta(scale=1.0, unit="rad", decimals=4),
    # Fractions of the frame, 0-1.
    "left": field_meta(decimals=3, minimum=0.0, maximum=1.0, step=0.05),
    "top": field_meta(decimals=3, minimum=0.0, maximum=1.0, step=0.05),
    "width": field_meta(decimals=3, minimum=0.0, maximum=1.0, step=0.05),
    "height": field_meta(decimals=3, minimum=0.0, maximum=1.0, step=0.05),
    "resolution": field_meta(items=RESOLUTION_ITEMS, format_fn=format_resolution),
    # Optional integration features: unchecked is off, checked needs a real
    # count. Limits are the hardware ranges documented on the fields in
    # fibsem/structures.py, and they start at 2 because integrating one line or
    # one frame is the same as not integrating at all — so the minimum the
    # spinbox offers is the smallest value that actually does something.
    "line_integration": {
        **field_meta(
            minimum=2,
            maximum=255,
            tooltip="Average this many scans of each line. 2-255.",
        ),
        "checkbox_label": "Enable",
        "checkbox_tooltip": "Unchecked: no line integration.",
    },
    "scan_interlacing": {
        **field_meta(
            minimum=2,
            maximum=8,
            tooltip="Scan the frame in this many interlaced passes. 2-8.",
        ),
        "checkbox_label": "Enable",
        "checkbox_tooltip": "Unchecked: no scan interlacing.",
    },
    "frame_integration": {
        **field_meta(
            minimum=2, maximum=512, tooltip="Average this many frames. 2-512."
        ),
        "checkbox_label": "Enable",
        "checkbox_tooltip": "Unchecked: no frame integration.",
    },
    "every_n_steps": field_meta(
        minimum=1, maximum=1_000_000, tooltip="Run once every N slices."
    ),
    "start_step": field_meta(
        minimum=0, maximum=1_000_000, tooltip="First slice this may run on."
    ),
    "end_step": field_meta(
        minimum=-1,
        maximum=1_000_000,
        tooltip="Last slice, exclusive. -1 means no bound.",
    ),
}

# What a field falls back to for keys its hint does not declare. These are the
# bounds and precision the form had before it moved onto the shared builder, so
# the move does not quietly put a floor of zero under an unhinted float.
SALAMI_FORM_DEFAULTS = FormDefaults(
    float_range=(-1e10, 1e10),
    int_range=(-2147483648, 2147483647),
    step=1.0,
    decimals=2,
)


def format_field_name(name: str, hints: Optional[dict] = None) -> str:
    if hints and hints.get("label"):
        return hints["label"]
    return name.replace("_", " ").title()


def make_group_divider(parent: Optional[QWidget] = None) -> QWidget:
    """A prominent horizontal rule marking a break between groups of sections.

    Deliberately heavier than the 1px `#3a3d42` hairlines used elsewhere in the
    UI: this separates two kinds of setting rather than two rows, so it has to
    read as a group boundary at a glance.
    """
    container = QWidget(parent)
    box = QVBoxLayout(container)
    box.setContentsMargins(0, 10, 0, 6)
    box.setSpacing(0)

    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Plain)
    line.setFixedHeight(2)
    line.setStyleSheet(
        f"background-color: {stylesheets.GRAY_HIGHLIGHT_COLOR}; border: none;"
    )
    box.addWidget(line)
    return container


# ---------------------------------------------------------------------------
# Type resolution
# ---------------------------------------------------------------------------


def resolve_hints(cls: type) -> Dict[str, Any]:
    """Resolve a dataclass's annotations to real types.

    `typing.get_type_hints` is what turns `'Optional[BeamSettings]'` back into a
    type. It needs the defining module's namespace, so it can fail on a forward
    reference to something not importable at runtime; fall back to the raw
    annotations rather than losing the whole form.
    """
    try:
        return typing.get_type_hints(cls)
    except Exception:
        logging.debug(f"Could not resolve type hints for {cls}", exc_info=True)
        return {f.name: f.type for f in dataclasses.fields(cls)}


def unwrap_optional(annotation: Any) -> Tuple[Any, bool]:
    """Return (inner type, was_optional) for `Optional[T]` / `Union[T, None]`.

    Unions of several real types (e.g. `Optional[Union[Path, str]]`) collapse to
    the first non-None member, which is enough to pick a control.
    """
    if typing.get_origin(annotation) is Union:
        args = typing.get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        optional = len(non_none) < len(args)
        if non_none:
            return non_none[0], optional
        return Any, optional
    return annotation, False


def is_dataclass_type(annotation: Any) -> bool:
    return inspect.isclass(annotation) and dataclasses.is_dataclass(annotation)


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
# `build_control` renders one plain field. The wrappers below add what the
# Salami configs also need, and tag the Control with `kind` ("plain", "choice",
# "optional", "read_only") and `inner` (the wrapped control, or None) so callers
# and tests can tell them apart without knowing a widget class.


def _tag(control: Control, kind: str, inner: Optional[Control] = None) -> Control:
    control.kind = kind
    control.inner = inner
    return control


def _as_tuple(value: Any) -> Any:
    # A resolution is declared `Tuple[int, int]` but arrives from YAML as a
    # list, and `[1536, 1024] == (1536, 1024)` is False.
    return tuple(value) if isinstance(value, list) else value


def placeholder_for(annotation: Any) -> Any:
    """A concrete value to build a control from when the stored one is None.

    The numeric controls multiply the value by their scale and would raise on
    None, which would drop the row.
    """
    return 0 if annotation in (int, float) else ""


def choice_control(items: Sequence[Any], value: Any, metadata: dict) -> Control:
    """A combo over `items`, which must not be empty.

    Values are compared as tuples where applicable (see `_as_tuple`).

    An unset value shows the first choice rather than a placeholder entry, so
    the control always names a real setting. The config is seeded to match
    beforehand — see `tasks.resolve_beam_defaults` — so what is displayed is
    what is stored.
    """
    items = [_as_tuple(item) for item in items]
    # Built from the first item: the combo's own `set_value` snaps a numeric
    # value to the nearest item and raises on None, and this control does
    # neither — `write` below replaces it.
    control = build_control(
        {**metadata, "items": items}, items[0], defaults=SALAMI_FORM_DEFAULTS
    )
    combo = control.widget
    unit = metadata.get("unit") or ""

    def write(new: Any) -> None:
        if new is None or new == "":
            combo.setCurrentIndex(0)
            return
        target = _as_tuple(new)
        for i in range(combo.count()):
            if _as_tuple(combo.itemData(i)) == target:
                combo.setCurrentIndex(i)
                return
        # Keep an out-of-list value visible rather than snapping to the nearest
        # supported one: a protocol written on another machine, or set offline,
        # should read back as itself so the operator can see and change it.
        if unit and isinstance(new, (int, float)):
            label = format_value(new, unit)
        else:
            label = str(new)
        combo.addItem(f"{label} (not offered by this column)", new)
        combo.setCurrentIndex(combo.count() - 1)

    control.write = write
    write(value)
    return _tag(control, "choice")


def optional_control(
    inner: Control,
    value: Any,
    label: str = "Set",
    tooltip: str = "Unchecked: leave this parameter unchanged.",
) -> Control:
    """Wrap `inner` with a checkbox; unchecked reads as None.

    `None` covers two different intents in this codebase, so the label is
    configurable rather than fixed:

    - On a `BeamSettings` field it means "leave the microscope as it is", which
      is what a fresh protocol says — the beam block in
      `configs/experiment_config.yaml` is all nulls. Overloading zero would make
      "don't touch the voltage" indistinguishable from "set 0 V". Reads as *Set*.
    - On an optional *feature* like line integration it means the feature is off.
      Reads as *Enable*.

    Either way the stored value is None when unchecked; only the wording differs.
    `inner` must already be built from a concrete value (`placeholder_for`).
    """
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)

    check = QCheckBox(label)
    check.setToolTip(tooltip)
    check.setChecked(value is not None)
    inner.widget.setEnabled(value is not None)
    check.toggled.connect(inner.widget.setEnabled)

    layout.addWidget(check)
    layout.addWidget(inner.widget, stretch=1)

    def read() -> Any:
        return inner.read() if check.isChecked() else None

    def write(new: Any) -> None:
        check.setChecked(new is not None)
        # Explicit, not left to the toggled connection: a refresh writes with
        # the inputs' signals blocked.
        inner.widget.setEnabled(new is not None)
        if new is not None:
            inner.write(new)

    control = Control(
        widget=container,
        read=read,
        write=write,
        signals=(check.toggled, *inner.signals),
        inputs=(check, *inner.inputs),
    )
    control.check = check
    return _tag(control, "optional", inner=inner)


def _format_read_only(value: Any, unit: str) -> str:
    if value is None:
        return "not set"
    if unit and isinstance(value, (int, float)):
        return format_value(value, unit)
    return str(value)


def read_only_control(value: Any, reason: str = "", unit: str = "") -> Control:
    """Displays a value that must not be edited here.

    Formats through the field's unit hint, so a read-only working distance reads
    "4.00 mm" rather than the stored "0.004" — the point of showing it is that
    the operator can check it at a glance.
    """
    box = QLineEdit()
    box.setEnabled(False)
    if reason:
        box.setToolTip(reason)
    shown = [value]

    def write(new: Any) -> None:
        shown[0] = new
        box.setText(_format_read_only(new, unit))

    write(value)
    control = Control(widget=box, read=lambda: shown[0], write=write, editable=False)
    return _tag(control, "read_only")


def fetch_available_values(
    microscope: FibsemMicroscope,
    key: str,
    beam_type: Optional[BeamType] = None,
) -> List[Any]:
    """Ask the microscope which values it supports for `key`.

    Goes through `get_available_values_cached` rather than `get_available_values`:
    the cached wrapper is defined on `FibsemMicroscope` itself (only the uncached
    one is abstract), so it works on every backend and on the simulator, and the
    form is rebuilt on every task switch — repeatedly querying the column for a
    list that cannot change would be wasteful.

    Backends legitimately report nothing for a key they do not expose — Tescan has
    no `voltage` branch, for instance — so an empty list is a normal answer, not
    an error, and the caller falls back to numeric entry.
    """
    try:
        return list(microscope.get_available_values_cached(key, beam_type) or [])
    except Exception:
        logging.debug(
            f"Could not read available {key!r} values from the microscope",
            exc_info=True,
        )
        return []


def create_field_control(
    name: str,
    value: Any,
    annotation: Any,
    hints: Optional[dict] = None,
    microscope: Optional[FibsemMicroscope] = None,
    beam_type: Optional[BeamType] = None,
) -> Control:
    """Build the control for one editable field."""
    hints = dict(hints or {})
    inner_type, optional = unwrap_optional(annotation)

    # A field the microscope can enumerate becomes a dropdown of exactly those
    # values, so the protocol cannot store a current the column will reject at
    # step 0. Handled before the Optional wrapper rather than inside it: these
    # fields always show a concrete value, never an "unset" state.
    parameter = hints.get("microscope_parameter")
    if parameter and microscope is not None:
        values = fetch_available_values(microscope, parameter, beam_type)
        if values:
            unit = hints.get("unit") or ""
            return choice_control(
                values, value, {**hints, "format_fn": lambda v: format_value(v, unit)}
            )
        logging.debug(
            f"Microscope reported no available {parameter!r} values; "
            f"falling back to numeric entry for {name!r}."
        )

    # An Optional[str] is already handled by the builder: an empty box reads as
    # None, which is clearer than a checkbox beside a text field.
    if optional and inner_type is str:
        return _tag(
            build_control(
                hints, value, annotation=annotation, defaults=SALAMI_FORM_DEFAULTS
            ),
            "plain",
        )

    def build(concrete: Any) -> Control:
        items = hints.get("items")
        if not items and inspect.isclass(inner_type) and issubclass(inner_type, Enum):
            items = list(inner_type)
        if items:
            return choice_control(items, concrete, hints)
        control = build_control(
            hints, concrete, annotation=inner_type, defaults=SALAMI_FORM_DEFAULTS
        )
        return _tag(control, "plain" if control.editable else "read_only")

    if not optional:
        return build(value)

    # `always_set` opts a field out of the unset checkbox: the control alone is
    # the value. For a field the operator always wants to state outright, the
    # checkbox is a click in the way rather than a meaningful choice.
    if hints.get("always_set"):
        control = build(placeholder_for(inner_type) if value is None else value)
        write = control.write

        def write_unless_unset(new: Any) -> None:
            # Shows the placeholder until a value arrives, and must not store it:
            # a 0 written back would drive the column to zero working distance
            # rather than leave it alone.
            if new is not None:
                write(new)

        control.write = write_unless_unset
        return control

    return optional_control(
        build(placeholder_for(inner_type) if value is None else value),
        value,
        label=hints.get("checkbox_label", "Set"),
        tooltip=hints.get(
            "checkbox_tooltip", "Unchecked: leave this parameter unchanged."
        ),
    )


# ---------------------------------------------------------------------------
# The recursive editor
# ---------------------------------------------------------------------------


class DataclassEditorWidget(QWidget):
    """Editor generated from a dataclass instance.

    Scalar fields become a label/control grid; nested dataclasses become
    collapsible `TitledPanel` sections, recursively. Edits are written straight
    back onto the object with `setattr`, so the config the caller passed in is
    mutated in place and `config_changed` is purely a "now persist it" nudge.

    There is no Apply button on purpose: a protocol can be edited mid-run, and a
    half-applied config is worse than an eagerly saved one.

    Args:
        obj: dataclass instance to edit.
        exclude: dotted field paths to omit, e.g. `"state.stage_position"`.
        read_only: dotted paths to show but not allow editing, mapped to the
            reason (used as the tooltip).
        expanded: dotted paths of nested sections to open initially. Everything
            else starts collapsed, so the form stays scannable.
        flatten: dotted paths of nested dataclasses to render inline instead of
            as their own section. Use it for pure containers — `state` holds only
            a beam block once the stage position is excluded, and
            `State > Electron Beam > Voltage` is a level deeper than the operator
            needs to think.
        dividers: dotted paths of nested sections to precede with a prominent
            horizontal rule, marking a break between groups of settings. Applies
            to sections only — scalar rows live in a grid, where a full-width
            rule would land in the wrong place.
        inject: dotted path -> widget to place immediately before that section
            (and after its divider, if it has one). The widgets stay owned by the
            caller: rebuilding the form reparents them out rather than destroying
            them, so a caller can hold references and keep them wired up.
        hints: per-field display hints; defaults to `FIELD_HINTS`.
        microscope: used only to *read* which values the column supports, for
            fields with a `microscope_parameter` hint. Never written to.
    """

    config_changed = pyqtSignal()

    def __init__(
        self,
        obj: Any = None,
        exclude: Optional[set] = None,
        read_only: Optional[Dict[str, str]] = None,
        expanded: Optional[set] = None,
        flatten: Optional[set] = None,
        dividers: Optional[set] = None,
        inject: Optional[Dict[str, QWidget]] = None,
        hints: Optional[Dict[str, dict]] = None,
        microscope: Optional[FibsemMicroscope] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.obj = obj
        self.microscope = microscope
        self.inject = dict(inject or {})
        self.exclude = set(exclude or ())
        self.read_only = dict(read_only or {})
        self.expanded = set(expanded or ())
        self.flatten = set(flatten or ())
        self.dividers = set(dividers or ())
        self.hints = hints if hints is not None else FIELD_HINTS
        # Dotted path -> (owning object, field name, control).
        self.field_widgets: Dict[str, Tuple[Any, str, Control]] = {}
        self._loading = False

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)

        if obj is not None:
            self.set_object(obj)

    def set_microscope(self, microscope: Optional[FibsemMicroscope]) -> None:
        """Rebuild so `microscope_parameter` fields gain (or lose) their dropdowns."""
        if microscope is self.microscope:
            return
        self.microscope = microscope
        if self.obj is not None:
            self.set_object(self.obj)

    def set_object(self, obj: Any) -> None:
        """Rebuild the form for `obj`."""
        self.obj = obj
        self._clear()
        if obj is None:
            return

        self._loading = True
        try:
            body = self._build_for(obj, prefix="")
            self._layout.addWidget(body)
            self._layout.addStretch()
        finally:
            self._loading = False

    def _clear(self) -> None:
        # Injected widgets belong to the caller, which may still hold references
        # and signal connections. Reparent them out so they survive the rebuild;
        # everything else the form built is ours to destroy.
        for widget in self.inject.values():
            widget.setParent(None)

        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.field_widgets.clear()

    def _build_for(self, obj: Any, prefix: str) -> QWidget:
        """Build the controls for one dataclass level.

        Scalars go into a grid at the top; nested dataclasses follow as panels,
        so a section's own parameters are never buried under its children.
        """
        container = QWidget()
        box = QVBoxLayout(container)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)

        scalars = QWidget()
        grid = QGridLayout(scalars)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(1, 1)
        row = 0

        annotations = resolve_hints(type(obj))

        for field in dataclasses.fields(obj):
            path = f"{prefix}{field.name}"
            if path in self.exclude:
                continue

            annotation = annotations.get(field.name, field.type)
            value = getattr(obj, field.name, None)
            inner_type, _ = unwrap_optional(annotation)
            field_hints = dict(self.hints.get(field.name, {}))
            if not field_hints.get("tooltip") and field.metadata.get("tooltip"):
                field_hints["tooltip"] = field.metadata["tooltip"]

            # Nested dataclass -> its own collapsible section, unless flattened.
            if is_dataclass_type(inner_type):
                if value is None:
                    logging.debug(f"{path} is None; no editor rendered for it.")
                    continue
                nested = self._build_for(value, prefix=f"{path}.")
                if path in self.dividers:
                    box.addWidget(make_group_divider())
                injected = self.inject.get(path)
                if injected is not None:
                    box.addWidget(injected)
                    injected.setVisible(True)
                if path in self.flatten:
                    box.addWidget(nested)
                    continue
                panel = TitledPanel(format_field_name(field.name, field_hints))
                panel.set_content(nested)
                if path not in self.expanded:
                    panel.collapse()
                box.addWidget(panel)
                continue

            # One malformed field must not blank the whole tab. This form is
            # generated over arbitrary dataclasses, so a value that doesn't match
            # its annotation is a real possibility from hand-edited YAML.
            try:
                if path in self.read_only:
                    control = read_only_control(
                        value,
                        self.read_only[path],
                        unit=field_hints.get("unit") or "",
                    )
                else:
                    # Available-value queries are per beam. The beam is whichever
                    # one owns this field — a BeamSettings carries its own
                    # beam_type — so read it off the object rather than assuming
                    # the electron column.
                    control = create_field_control(
                        field.name,
                        value,
                        annotation,
                        field_hints,
                        microscope=self.microscope,
                        beam_type=getattr(obj, "beam_type", None),
                    )
            except Exception:
                logging.warning(
                    f"Could not build an editor for {path!r} (value={value!r}); "
                    "skipping that field.",
                    exc_info=True,
                )
                continue
            self.field_widgets[path] = (obj, field.name, control)

            if control.editable:
                control.connect(functools.partial(self._on_field_changed, path))

            label = QLabel(format_field_name(field.name, field_hints))
            tooltip = field_hints.get("tooltip")
            if tooltip:
                label.setToolTip(tooltip)
                if control.editable:
                    control.widget.setToolTip(tooltip)
            grid.addWidget(label, row, 0)
            grid.addWidget(control.widget, row, 1)
            row += 1

        # Only add the grid if it actually holds something.
        if row > 0:
            box.insertWidget(0, scalars)
        else:
            scalars.deleteLater()

        return container

    def _on_field_changed(self, path: str) -> None:
        if self._loading:
            return
        entry = self.field_widgets.get(path)
        if entry is None:
            return
        owner, field_name, control = entry
        setattr(owner, field_name, control.read())
        self.config_changed.emit()

    def refresh(self) -> None:
        """Re-read every control from the underlying objects."""
        self._loading = True
        try:
            for owner, field_name, control in self.field_widgets.values():
                control.set_blocked(True)
                try:
                    control.write(getattr(owner, field_name, None))
                finally:
                    control.set_blocked(False)
        finally:
            self._loading = False

    def set_fields_enabled(self, enabled: bool) -> None:
        """Grey out every generated control, as the image widget does while acquiring.

        Used while a microscope sync is in flight. A sync rewrites config values
        behind the form's back and then calls `refresh()`, so an edit made in the
        meantime would be silently discarded — better to make the form visibly
        inert for the couple of seconds the acquisition takes.

        Only the generated controls are touched, not this widget as a whole:
        injected widgets belong to the caller (the sync row has to keep its
        spinner animating and its own busy state) and section headers stay live
        so the form can still be collapsed and scrolled.

        Read-only fields stay disabled regardless — re-enabling them would undo
        the view spec.
        """
        for _owner, _field_name, control in self.field_widgets.values():
            control.widget.setEnabled(enabled and control.editable)
