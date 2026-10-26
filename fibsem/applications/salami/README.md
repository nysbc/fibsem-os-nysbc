# Salami — slice and view

A FIB-SEM slice-and-view (volume EM) application built on the core `fibsem`
package. The FIB removes a thin slice from the block face, the SEM images the
freshly exposed surface, and the pair repeats a few thousand times to build a
3D volume.

> **Status:** under active development. The experiment layer runs headless; the
> UI can create, open and reuse experiments, configure tasks, and run. A five
> slice run on the simulator acquires and saves an image per slice. Nothing has
> been through a real sample yet, and the gaps that matter are listed under
> [Not yet implemented](#not-yet-implemented) — read that before trusting a run.

---

## The two-file experiment

An experiment is a **directory** containing exactly two files:

```
Salami-2026-07-25-14-30/
├── experiment_config.yaml     # the protocol   — what to do
├── Experiment_Status.yaml     # the run state  — where we got to
├── Slice_Task/                # per-task output
├── Acquisition_Task/
└── Setup/                     # trench / fiducial reference images
```

The split is the central design decision, and it exists to support two
scenarios:

| Scenario | What happens |
|---|---|
| **Resume** — a run stops or crashes | status restores `current_step` and `next_task_index`; the protocol is unchanged |
| **Reuse** — same experiment, new sample | the protocol is copied and re-homed; a fresh status is created |

So: **config is reusable, status is per-run.** Anything sample-specific (stage
position) or machine-specific (output paths) belongs in status or is derived at
run time — never persisted into the protocol. `as_template()` enforces this.

### Path handling

`path` is stored in the config for provenance but **never trusted on load** —
it is always re-derived from where the YAML actually is. A moved or copied
experiment folder therefore still loads correctly, and a protocol written on a
colleague's machine can't send writes to their disk.

---

## Lifecycle

Constructing a `SalamiExperiment` does **not** touch the disk. Writing happens
in `initialise()`, called only once the target directory is settled.

| Operation | Menu | Behaviour |
|---|---|---|
| `SalamiExperiment.create(parent_dir, name)` | File → New Experiment | new directory from the default protocol |
| `SalamiExperiment.create(..., template=cfg)` | File → New Experiment From Protocol | copies and re-homes `cfg`; **never writes to the source** |
| `SalamiExperiment.open(path)` | File → Open Experiment | read-only load; nothing is written until the run continues |
| `config.export(filename)` | File → Export Protocol | writes a reusable protocol, stripped of sample state |

`create()` refuses to overwrite an existing experiment.

### Run states

`status.run_state` tracks one run: `created` → `setup` → `running` →
`paused` / `completed` / `failed`. Distinct from `status.setup_complete`, which
records whether the one-off preparation (trench, fiducial) has been done.

---

## The protocol — `SalamiExperimentConfig`

```yaml
name: Salami-2026-07-25-14-30
description: ''
path: ...                  # provenance only; re-derived on load
n_steps: 5000              # number of slices
advance:
  step_size: 1.0e-08       # slice thickness (m) — 10 nm
  advance_milling_pattern: true
  advance_beam_shift: true
tasks: [...]
```

`n_steps × advance.step_size` is the total depth sliced through — exposed as
`config.volume_depth`. A 50 µm region at 10 nm slices is 5000 steps.

For near-isotropic voxels, the slice thickness should roughly match the in-plane
pixel size (`hfw ÷ horizontal resolution`).

### `VolumeAdvanceConfig`

The per-slice advance, at experiment level rather than per task — multiple tasks
in one experiment share a single advance.

- `step_size` — slice thickness, in metres
- `advance_milling_pattern` — walks the slice pattern into the block
- `advance_beam_shift` — shifts the SEM in y to track the receding face

---

## Tasks

Tasks are dispatched through a registry, so the experiment class never
type-checks individual configs. Two ship today:

| Task | Config | Does |
|---|---|---|
| `MillingTask` | `MillingTaskConfig` | cuts one slice via `run_milling_task` |
| `AcquisitionTask` | `AcquisitionTaskConfig` | auto-functions, then images the face |
| `SenseAIAcquisitionTask` | `SenseAIAcquisitionTaskConfig` | sub-sampled frame through the external scan generator, optionally reconstructed |

They run in list order, once per step. The default protocol is
`Slice_Task` → `Acquisition_Task`: mill, then image.

### Where the parameters live

A frequent source of confusion — imaging parameters are split across two
objects, because voltage and current are *not* part of `ImageSettings`:

| Parameter | Field |
|---|---|
| SEM voltage, current, working distance | `AcquisitionTaskConfig.state.electron_beam` (`BeamSettings`) |
| resolution, dwell time, HFW | `AcquisitionTaskConfig.image` (`ImageSettings`) |
| slice pattern and geometry | `MillingTaskConfig.milling.stages[i].pattern` |
| milling current | `MillingTaskConfig.milling.stages[i].milling.milling_current` |

**Milling current and milling depth** are editable on the task form, one row per
stage above the generated fields. Currents come from the ion column's supported
values when connected and fall back to numeric entry in nA offline; depth is in
µm. Pattern *geometry* stays on the Milling tab, so two widgets never draw on
one FIB canvas — these two are the exception because neither has any geometry to
draw, and both are what change between samples.

#### Depth is not the height of the volume box

Two different axes, and calling both "depth" has already caused confusion:

| | Means | Set by | Changes when the box is resized? |
|---|---|---|---|
| **box width** | slice width, across the face | dragging the box | yes |
| **box height** | distance sliced *through* the block — sets the slice count | dragging the box | yes |
| **milling depth** | how far each cut reaches *into* the face | per stage, on the task form | **no** |

So a 10 µm box height at 100 nm slices is 100 slices, whatever the milling
depth is. The UI says "through the block" rather than "deep" for the box height
for this reason, and the box-height variable in `tasks.py` is `extent_m`, not
`depth_m`.

### The SenseAI acquisition task

Images the face through the Quantum Detectors external scan generator: a
sub-sampled scan, and a sparse reconstruction that fills in the pixels the beam
never visited. Four things about it differ from the ordinary acquisition task.

**It has no `ImageSettings`.** While the column is in external scan the
generator decides how many pixels are visited and for how long, so resolution
and dwell time live in `scan` (`SenseAIScanConfig`) and the one remaining
viewing parameter, HFW, sits on the beam block with voltage, current and working
distance. That is the arrangement `TaskImagingView` already describes for the
milling task, and it leaves exactly one home for each value rather than an
`ImageSettings` whose resolution the run ignores.

**External scan is set per acquisition, never once.** `microscope.acquire_image()`
in core calls `set_full_frame_scanning_mode()`, so any ordinary imaging takes
the scan back. The task sets external immediately before the frame and returns
it in a `finally`, so a failed acquisition cannot leave the column unable to
image. Auto-functions run *before* the handover — they drive the column's own
scan engine and would otherwise fight the generator for the beam.

**One 4-page TIFF per slice.** Input and Mask are the measurement and are
written losslessly; the reconstruction is scaled into page 0. See
`docs/design/senseai-control.md` for the layer order and why the file has to be
written with `photometric="minisblack"`.

**Reconstruction has its own `RunSchedule`.** Because the measurement is saved
in full, any slice can be reconstructed again afterwards in the SenseAI desktop
app — during a run the reconstruction is really for watching progress. Setting
it to every tenth slice keeps an eye on the volume without paying for it on
every one. Tuning the reconstruction is not this application's job; SenseAI
ships an Optuna study for that.

### The scan generator is loaded once, and never by the task

One `SenseAI_ModuleControl` per process, and the task will not create one. That
is not a convenience: the DLL's source registry outlives any Python object, so a
second module inherits every source the first one added, collides on *"source
already exists"*, and ends up with a generator that was never initialised. Every
step of that failure is logged and swallowed, so it reports success and only
shows up later as an acquisition that waits for a frame which never arrives.

- The **SenseAI tab's *Initialise Scan Generator*** builds the module and
  publishes it with `set_senseai_module`. Pressing it again adopts the existing
  one rather than building a second, and a module that failed to initialise is
  never published.
- A **headless run** passes its own module in on `RunContext.senseai_module`.
- The task calls `require_senseai_module`, which raises if there is no module or
  if it reports a `readiness_problem`. Refusing to run is deliberate: it is far
  easier to diagnose than a generator that silently never returns a frame.

Everything the task is allowed to do is per-slice — update the generator's
config, take the scan external, and give it back.

DLL paths are deliberately **not** in the protocol: they are machine-specific,
and a protocol has to survive being copied to another microscope.

**The preview goes back on the workflow update, not `sem_acquisition_signal`.**
That signal is a psygnal, so its handlers run in the *emitting* thread — and the
handler is the image widget, which draws on the quad-view canvas. Emitting it
from the run's worker thread put Qt drawing on a non-GUI thread. The frame also
needs real metadata (`senseai_preview_image` supplies it): the widget reads
`beam_type` to choose a pane and the canvas reads `pixel_size` for its scale
bar, and a bare `FibsemImage(array)` is dropped with *"Image metadata is None"*.

### Adding a task to a protocol

**Experiment tab → Add Task.** Pick a type and name it; the type list is built
from the registry, so a newly registered task type appears with no UI change.
The task is appended to `config.tasks`, its output directory is created, the
protocol is saved and the Protocol tab's selector switches to it.

Order in the list is run order, which is what makes a protocol like
`Slice_Task → Charge_Settling → Real_Acquisition` work: a throwaway acquisition
to settle charging, then the one that is kept.

Names are validated because a task name is not cosmetic — `BaseTask.output_dir`
joins it onto the experiment path and it becomes part of every image filename
under it. Duplicates are rejected (two tasks would write into one directory, and
`find_task` could only ever return the first), as is anything that is not a word
character, space, dash or dot.

**Add and Remove are both disabled while a run is in progress.** They change the
length of `config.tasks`, which `status.next_task_index` indexes into while
`_run_step` is iterating. That race is the whole reason `enabled` and
`skip_once` exist — use those mid-run instead. Removing a task leaves its output
directory and any images already acquired on disk, and clears any pending
one-shot skip naming it.

### Adding a task type

1. Define a config dataclass with a `TASK_TYPE` classvar, `to_dict`/`from_dict`,
   `enabled`, `schedule`, and a `should_run(step)`.
2. Subclass `BaseTask`, set `CONFIG_CLS`, implement `run()`.
3. Decorate with `@register_task`.
4. Add it to the `TaskConfig` union.

Optional hooks: `configure()` (set output paths), `apply_advance()` (offset
geometry for the step), `_clear_runtime_paths()` (for protocol reuse).

Nothing else needs changing — dispatch, serialisation and the run loop pick it
up automatically.

---

## Controlling when a task runs

Three independent controls, checked in this order:

**1. `enabled` (protocol).** The operator's on/off switch. Persists across a
resume; reset by `as_template()` so a fault on one sample doesn't silently
disable the task on the next.

```python
experiment.set_task_enabled("Acquisition_Task", False)
```

**2. `schedule` (protocol).** The planned cadence — `RunSchedule(every_n_steps,
start_step, end_step)`, with `end_step` exclusive and `-1` meaning no bound.
Kept separate from `enabled` so turning a task off doesn't destroy its schedule.

**3. `skip_once` (status).** Skip the next occurrence only, then resume as
normal. Checked *after* the schedule, so "next" means the next time the task
would actually have run — not merely the next step.

```python
experiment.skip_task_once("Slice_Task")
experiment.cancel_skip_once("Slice_Task")
```

All three are safe to change mid-run: none of them alters the length of
`config.tasks`, so `status.next_task_index` stays valid. **Removing** a task
mid-run would not be safe, which is why these exist.

`experiment.task_states()` returns a per-task summary for the UI.

---

## The run loop

```
for step in current_step .. n_steps:
    for i in next_task_index .. len(tasks):
        skip unless task.should_run(step)      # enabled + schedule
        skip once if pending
        cfg = runtime_task_config(task, path, step, advance)
        status.in_flight_task = task.name;  save
        run task
        status.in_flight_task = None;  next_task_index = i+1;  save
    current_step += 1;  next_task_index = 0;  save
```

Two properties matter more than they might look.

### Advance is derived, not accumulated

The stored protocol is always the **step-0 baseline**. Each step builds a
throwaway copy carrying that step's geometry:

```
y = baseline_y + step × step_size
```

So resuming at step 3000 lands at the right depth, there's no float drift
accumulated over thousands of additions, and machine-specific output paths never
contaminate the saved protocol. It also means a mid-run protocol edit is picked
up cleanly at the next step boundary.

**Which field moves depends on the pattern**, and `tasks.advance_pattern_y` owns
that decision. `RectanglePattern` is positioned by `point` (`define()` sets
`centre_y=self.point.y`), but `LinePattern.define()` ignores `point` entirely and
builds its shape from `start_y`/`end_y`. Shifting `point` alone moves a rectangle
and does *nothing* to a line.

Two consequences worth remembering:

- The pattern classes live in core `fibsem.milling.patterning`, shared with other
  applications, so this is handled on the Salami side rather than by changing
  them.
- Anything asserting that the advance works must check `pattern.define()`, not
  `pattern.point`. There was a period where the config field moved correctly the
  whole time the geometry stood still, and a test on the field would not have
  caught it.

Note the sign convention: pattern y is positive *up* the frame
(`draw_rectangle_shape` computes `cy = icy - centre_y / pixelsize`).

### The slice pattern is a thin rectangle

`DEFAULT_SLICE` is a `RectanglePattern` — 40 µm wide, one slice thickness tall,
cutting 5 µm into the face. Chosen over a line after comparing both on the
microscope: a line has no height to be the slice thickness and cannot be
positioned, since `define()` ignores its `point`.

**The pattern's height is the slice thickness, and `advance.step_size` owns it.**
`MillingTask.apply_slice_thickness` sets the height from it on the per-step
runtime copy, so the slab cut and the distance advanced cannot drift apart —
short of the advance leaves unmilled material between slices, over it re-cuts the
last one, and both are invisible until the stack is reconstructed.

So editing a slice pattern's height on the Milling tab has no effect; change the
slice thickness instead. Patterns that express their extent some other way (a
trench's upper/lower heights) are left alone.

### The volume is a box on the preview

Selecting a milling task draws a yellow region over the preview image. Its width
is the slice width, its height is the distance to slice through the block, and
its position is where on the face. It says nothing about how deep each cut goes
— that is the stage's milling depth, set on the task form. Drag it and the
pattern and the slice count follow:

```
n_steps = box depth / slice thickness
```

The box is never stored — `tasks.milling_region_for` rebuilds it from the
pattern and `n_steps`, and `tasks.apply_milling_region` reads it back the other
way. `ImageGeometry` handles the conversion between frame fractions and pattern
metres, including the y flip.

The **slice count is always derived**. Drag the box and it changes; change the
thickness and it changes, holding the volume fixed. It is not typed anywhere.

A preview image is required, because fractions of a frame only become metres via
its pixel size. Without one the panel says so instead of drawing a box that
cannot be converted.

### Resume is sub-step, because slicing is destructive

Status is written after **every task**, not just every step. A crash between the
mill and the image would otherwise re-enter the step at task 0 and cut the same
slice twice — destroying material you cannot get back.

For the genuinely ambiguous case — killed *during* a mill — `in_flight_task` is
set before the task runs and cleared after. If it's still set on open, an
unknown amount of material was removed and there is no safe automatic choice, so
the operator is asked:

```python
experiment.has_interrupted_task          # True
experiment.resolve_interrupted_task("retry")   # or "skip"
```

`run()` refuses to start while it's unresolved.

---

## The UI, in pages

`SalamiMainWindow` is a shell holding one tab per page, following
`autolamella/ui/AutoLamellaMainUI.py`. Each page is self-contained; the shell
owns the tab bar and the File menu.

| Page | Contents |
|---|---|
| **Microscope** | `SalamiUI`: quad view, and a strip of Connection / Image / Movement / Milling / SenseAI / Experiment (status + run controls) |
| **Protocol** | three columns — experiment information, task parameters, and the last sync's image with the task's editable regions drawn over it |

The Protocol page is disabled until an experiment exists, with a tooltip saying
why. It does not build its first two columns: it is handed the experiment info
group box and the task editor, which the shell reparents out of the Microscope
page's tab strip. Every handler keeps working because they are the same objects.

`SalamiUI` also runs standalone — it keeps its own Task Config tab and menu bar
when no shell is present.

## Worker ↔ UI

The experiment runs on a background thread, so the UI stays responsive and
`ask_user` can block. Qt only allows the main thread to touch widgets, so
everything the worker wants displayed goes through one signal:

```
worker thread                     main thread
ctx.emit(update) ──[signal]──►    _on_workflow_update(update)
while WAITING_FOR_UI_UPDATE: ◄──  WAITING_FOR_UI_UPDATE = False
```

A single payload type, `SalamiWorkflowUpdate`, carries **both** progress
reporting (step counters, timings, results) and UI actions (show these buttons,
display this image, load this milling config). Live objects — images, milling
configs — are excluded from `to_dict()` so serialisation stays cheap.

The reverse direction has no channel and needs none: the UI and the experiment
share the same config object, so edits are simply visible. Decisions come back
via flags the worker polls (`USER_RESPONSE`, `_workflow_stop_event`).

---

## File map

| File | Contains |
|---|---|
| `experiment.py` | `SalamiExperimentConfig`, `SalamiExperimentStatus`, `SalamiExperiment`, run loop, setup |
| `tasks.py` | task configs, `BaseTask`, the registry, `AcquisitionTask`, `MillingTask` |
| `structures.py` | `RunSchedule`, `AutoFunctionConfig`, `RunContext`, `VolumeAdvanceConfig` |
| `workflow_update_schema.py` | `SalamiWorkflowUpdate` — the worker → UI payload |
| `workflow_ui.py` | `ask_user`, `set_images_ui`, abort checking |
| `_default_settings.py` | `DEFAULT_SLICE`, `DEFAULT_TRENCH` |
| `configs/experiment_config.yaml` | the default protocol new experiments start from |
| `ui/main_window.py` | the shell: one tab per page, the menu bar, `salami-ui` entry point |
| `ui/SalamiUI.py` | the Microscope page — quad view plus the control strip |
| `ui/protocol_page.py` | the Protocol page — 3 columns, including the sync preview |
| `ui/add_task_dialog.py` | Add Task dialog — type dropdown from the registry, name validation |
| `ui/task_editor.py` | per-task view specs; which fields are shown, locked or hidden |
| `ui/config_editor.py` | generates an editor from any dataclass |
| `ui/qtdesigner_files/` | generated from `SalamiUI.ui` — **do not hand-edit** |

Tests are in `tests/salami/`, and need no microscope.

---

## Not yet implemented

- **Beam settings never reach the column, and `advance_beam_shift` is inert.**
  `AcquisitionTask.run` applies only `cfg.image`, via
  `microscope.acquire_image()`. `set_microscope_state` is commented out
  (`tasks.py:881`) and nothing calls `apply_beam_settings`, so the protocol's
  voltage, current and working distance are ignored at run time.

  The consequence worth knowing: `AcquisitionTask.apply_advance` computes
  `beam.shift` correctly for every step, onto a runtime copy that is then
  discarded. `MillingTask.apply_advance` *does* take effect, because its runtime
  stages are handed to `run_milling_task`. So **the mill walks into the block
  and the SEM view does not follow**, while both advance flags default to True.

  `set_microscope_state` cannot simply be uncommented — it forwards `None`
  fields into setters and moves the stage. `apply_beam_settings` exists for
  exactly this and explains why in its docstring; the fix is to call it from
  `AcquisitionTask.run`. Any test of it has to assert on what reached the
  microscope, not on what the config holds.
- **No drift correction.** `_create_fiducial` / `_confirm_fiducial` are stubs,
  so there is no fiducial and no per-slice alignment — the single biggest gap
  for real multi-hour runs.
- **No GIS deposition** step in setup, and no relief cuts (trench only).
- **No focus/WD compensation** for the receding face beyond periodic autofocus.
- The trench config adjusted during setup is held on the instance only
  (`experiment.trench_config`) and is not persisted.
