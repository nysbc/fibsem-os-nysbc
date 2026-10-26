# Working on fibsem-os-NYSBC

Context for anyone — human or agent — picking this up. A universal API for
FIB-SEM microscope control (OpenFIBSEM), NYSBC fork, plus applications on top.

- `fibsem/` — core: microscope abstraction, acquisition, milling, alignment, UI widgets
- `fibsem/applications/autolamella/` — cryo-lamella preparation
- `fibsem/applications/salami/` — slice-and-view volume EM (**active development**)

Active branch: `SNV_refactor`.

---

## How to work here

**Make the smallest change that works.** Readability and maintainability come
first. When adding a feature, look for an existing mechanism to extend before
inventing anything: a field on a config the editor already renders, a hook
already on `BaseTask`, a signal already routed to the main thread, a
`RunSchedule` already doing cadence. Add a new class or concept only when
reusing what exists would genuinely be more tedious or more confusing. Before
writing a new helper, name the existing thing that nearly does the job and say
why it does not fit. Given a fix and a refactor, take the fix.

**Do not change core `fibsem/` to solve an application problem.** Core modules
(`structures.py`, `milling/`, `microscope.py`, `ui/widgets/`) are shared with
applications nobody is currently working on; changing them means re-verifying
contracts across all of those. Absorb the difference in the application layer
instead. Example: `LinePattern.define()` ignores `BasePattern.point`, so
Salami's own `tasks.advance_pattern_y` moves `start_y`/`end_y` rather than
patching the core pattern class.

**Test invariants, not tuned constants.** Assert that a value is physical and
in the right relationship to its siblings, not that it equals 0.4. Exact-value
tests have to be rewritten every time someone retunes a default, and they test
the wrong thing — nobody cares that the number is 0.4, they care that it is not
0%, 150% or −20%. Comparing against a freshly constructed default is fine.

**Qt threading.** Runs happen on a background thread so the UI stays responsive
and prompts can block. Only the main thread may touch widgets. Worker → UI goes
through `SalamiUI.workflow_update_signal`, handled on the main thread. Never
call a widget method, or emit a **psygnal**, from a worker — psygnal handlers
run in the *emitting* thread, and `microscope.sem_acquisition_signal` is wired
to the image widget, which draws on the canvas.

**Protocol vs run state.** Configuration is a *reusable protocol* and must stay
independent of sample and machine. Stage positions, output paths and progress
belong in the run status, not the config.

**Don't write to disk from a constructor.** Creating an experiment object must
not create or overwrite files — the caller decides the directory first.

**Generated UI files.** Anything under `**/qtdesigner_files/` comes from
`pyuic5`. Edit the `.ui` and regenerate; hand edits are lost.

---

## Environment and tests

Use the conda env with `fibsem` installed editable (`senseai` on the dev
machine, `C:\ProgramData\miniforge3\envs\senseai`). Do **not** use the Enthought
AutoScript env even if it is first on `PATH` — it lacks `psygnal`, and
installing deps there risks upgrading numpy under AutoScript.

```bash
python -c "import psygnal, fibsem; print(fibsem.__file__)"   # sanity check
python -m pytest tests/salami -q                             # no microscope needed
```

- `fibsem/microscopes/simulator.py` provides a `DemoMicroscope`; most of the app
  can be exercised without hardware.
- The Qt suites (`test_regions.py`, `test_main_window.py`) are slow (~2 min) —
  run them in the background and the rest in the foreground.
- `tests/test_structures.py` has a **pre-existing** collection error (imports
  `CompustagePosition`, which does not exist). Unrelated.
- Runs leave stray `Alignment/` and `None/` directories in the repo root. Don't
  commit them.
- Qt `isVisible()` is always `False` under the offscreen platform. Use
  `isVisibleTo(parent)` or check enabled state instead.

---

## Salami, in brief

Slice-and-view: the FIB removes a thin slice from the block face, the SEM images
the freshly exposed surface, a few thousand times. Start with
`fibsem/applications/salami/README.md`; design decisions and what is left are in
`docs/design/salami-roadmap.md`.

**An experiment is a directory** with exactly two files: `experiment_config.yaml`
(the protocol — reusable) and `Experiment_Status.yaml` (the run state). That
split is the central design decision.

**The UI is a shell with pages** (`ui/main_window.py`), following
`autolamella/ui/AutoLamellaMainUI.py`: a **Microscope** page (`SalamiUI` — quad
view plus a widget strip) and a **Protocol** page (`ui/protocol_page.py` — three
columns: experiment info, task parameters, and the last sync's image with the
task's editable regions drawn on it).

**Task editors are generated** from the config dataclasses
(`ui/config_editor.py`); `ui/task_editor.py` holds the per-task view spec saying
what to hide, lock or flatten. Adding a field to a task config makes an editor
appear with no UI code.

Two domain facts shape everything:

- **Milling is destructive and not idempotent.** Re-running a step removes
  material twice. Resume is sub-step, and ambiguous cases ask the operator.
- **Runs last days.** 50 µm at 10 nm is 5000 steps. Crash recovery and
  unattended robustness are not nice-to-haves.

---

## Gotchas that cost real time

- **The generated editor binds controls to the owner object found at build
  time.** Anything writing to a nested config behind the form's back must
  **mutate**, not replace — assigning a new `FibsemRectangle` silently orphans
  its four spinboxes.
- **Selecting a task seeds its beam defaults, which emits `config_changed`
  before the rectangles are redrawn.** Any handler on that signal runs once with
  the new task and the previous task's rectangles. Dispatch on `TASK_TYPE`, not
  on which rectangles happen to be drawn.
- **Assert on `pattern.define()`, not `pattern.point`.** The advance once wrote
  to a field the default pattern never read, so the position moved correctly
  while the milled geometry stood still for a whole run.
- **The shipped `configs/experiment_config.yaml` is the real default**, not
  `DEFAULT_SLICE` — that constant only applies to a task with no stages. Change
  one, regenerate the other.
- **SenseAI: the DLL registry outlives the Python object.** A second
  `SenseAI_ModuleControl` inherits the first one's sources, collides on "source
  already exists", and yields a generator that never returns a frame — and every
  step of that failure is logged and swallowed, so it reports success. The DLL
  and generator load **once**, from the SenseAI tab; the task calls
  `require_senseai_module` and refuses if there isn't one.
- **`utils.format_value` emits U+03BC**, which a cp1252 console logging handler
  cannot encode; the entry is silently lost. `MicroscopeSyncResult.log_summary`
  works around it locally.
