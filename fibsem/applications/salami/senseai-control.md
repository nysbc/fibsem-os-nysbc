# SenseAI control (external scan generator)

Working notes on driving the Quantum Detectors SenseAI scan generator from
OpenFIBSEM. SenseAI replaces the microscope's own scan engine: the SEM beam is
put into external scan mode, SenseAI scans a sub-sampled pattern, and a sparse
reconstruction fills in the unmeasured pixels. For slice-and-view that is the
point — visiting a fraction of the pixels cuts the dwell time per slice, which
over a few thousand slices is the difference between a run that is practical and
one that is not. (The default config samples 30%; the actual speed-up has not
been measured here.)

Code:

- `fibsem/SenseAI_SG.py` — config loading, module control, capture stack format,
  and a simulator for working without the hardware.
- `fibsem/ui/SenseAIControlWidget.py` — the SenseAI tab in the Salami UI.
- `fibsem/config.py:224` — `DEFAULT_SENSEAI_DLL`, `DEFAULT_SENSEAI_CONFIG`.

Most of what follows was recovered by probing the DLL and reading files it
wrote, because it is not in the vendor documentation. Where something is
inferred rather than measured it says so.

## Working on this

The vendor Python package is installed in the `senseai` conda environment
alongside `fibsem`:

```
C:\ProgramData\miniforge3\envs\senseai\Lib\site-packages\senseai\   # python wrapper
C:\Program Files\SenseAI\SenseAI 2026.1.1\SenseAI.dll               # the actual engine
```

You do **not** need the microscope to work on most of this. Loading the DLL,
adding image sources, saving and loading TIFFs, and running a reconstruction on
a synthetic image all work standalone — the scan generator config in
`SenseAI_SG.py` also supports a `QDMock` generator (any name containing `WC`),
which is how the mock captures in `C:\Code\SenseAI_Project\senseai_captures\`
were produced.

Note the version skew: the installed Python package is `senseai-2025.11.2` while
the DLL is `2026.1.1`. Nothing has misbehaved because of it so far, but the
bundled HTML docs in the install directory describe 2026.1.1 and mention modules
(`datasets`) the installed package does not have.

## The Python package is a thin shim; the state is in the DLL

Every method in `senseai/` marshals a JSON blob and calls a `ctypes` binding.
There is no state on the Python side worth speaking of. Two consequences that
have already cost time:

**Constructing a new `SenseAI()` does not reset anything.** `init_senseai_dll`
calls `ctypes.WinDLL(path)`, and Windows returns the already-loaded module for a
path it has seen. The DLL's source registry survives. So re-clicking *Initialize
Scan Generator*, or building a fresh `SenseAI_ModuleControl`, inherits every
source from the previous attempt. Only `delete_*`, `reset()`, or restarting the
Python process clears it.

**Errors arrive as `RuntimeError` with the DLL's message inside.**
`base.py:83-87` raises `RuntimeError(f"Call failed: {Status} with error
'{Response}'.")`. The useful part is the `Response`. The DLL also logs to stderr
with a file and line (`Export.cpp:246`), which is usually more informative than
the exception — worth capturing when something fails.

## Sources: one flat registry, unique names

`list_contents()` returns a single namespace with these buckets:

```
Algorithms, Buffers, Detectors, Dictionaries, Images, ScanGenerators, Slicers, Tasks
```

Scan generators and detectors live in the same registry as images and
dictionaries. Adding any of them goes through one native `AddSource`, which
fails with **`This source already exists`** on a duplicate name:

```
E Export.cpp:246] Failed to add source -> This source already exists.
  Input: {"Name":"t", ..., "SourceType":"Image", ...}
```

Because the message comes from the shared `AddSource`, it does not tell you
which kind of source collided. Check `list_contents()`.

This is why `senseai.reset()` is the wrong recovery tool here: its docstring
says it deletes everything, and everything includes the scan generator and
detector, so recovering from a name clash that way tears down the external scan
path and requires a full re-init.

## The four-layer image model

SenseAI holds every image as four layers, and its capture step writes them to a
four-page TIFF (`Detector.cpp`, filename pattern `capture_{}_{}.tiff`). Four is
a hard convention — the DLL contains the string `File does not have 4 layers,
skipping:`.

The page order used by the **loader** is:

| page | layer | on acquisition |
|------|----------------|----------------------------------|
| 0 | Reconstruction | zeros (ignored on load, see below) |
| 1 | Reference | copy of Input |
| 2 | Input | the acquired frame, 0 where unmeasured |
| 3 | Mask | 65535 where measured, 0 elsewhere |

Recovered by writing a probe file with a different pattern on each page, loading
it, and reading each layer back with `get_image_array` to see which page it
matched (r = 1.000 on each). This matters because it is **not** the order you
would guess from `senseai.py:132-134`, where `add_image` writes Input before
Reference. Captures written by SenseAI cannot settle the question: they always
have Reference identical to Input, so pages 1 and 2 are byte-for-byte the same.

It is therefore possible that SenseAI's *writer* emits Input at page 1 and its
*loader* reads page 1 as Reference, and nobody would ever notice. Since
`build_capture_stack` also sets Reference to a copy of Input, we are immune
either way. Only revisit this if a genuinely different reference image is ever
needed.

**Mask semantics: 1 means measured.** In a real capture, every non-zero input
pixel lies inside the mask and none outside. In a uint16 file the measured value
is 65535, not 1 — the dtype maximum, so the mask page is visible rather than
black in the GUI. On load SenseAI normalises it back to a clean 0/1.

**dtype tracks the pipeline, not the format.** A raw capture is uint16, holding
the detector counts directly. Once an image has been through SenseAI, every
layer is float32 normalised to 0..1 (`set_image_array` casts to float32 on the
way in, `base.py:220-222`), and an export is float32. So a uint16 file has never
been reconstructed and a float32 one has. Acquisition stacks are kept uint16 to
preserve the raw counts; a reconstruction cannot be stored in one without either
scaling it or promoting the whole stack to float32.

### Signed samples and the unmeasured sentinel

`hw.get_detector_image` returns the analogue detector's frame as **int16** — the
samples are signed and in practice mostly negative. SenseAI's own capture writer
stores them **offset-binary**, shifted by half the range (`value + 32768`), which
is why its files are uint16 and ordered correctly.

Casting the frame straight to uint16 instead wraps every negative sample to the
top of the range. The symptom is a histogram of the right shape sitting about
32768 too high, with the genuinely bright pixels folded down into a small tail
near zero. `build_capture_stack` handles this in `_to_offset_binary`; do not
"simplify" it to an `astype`.

The detector also writes `INT16_MIN` at every pixel the beam never visited, and
the shift maps that sentinel exactly onto 0 — which is what SenseAI writes for an
unmeasured pixel. So the two conventions agree for free, and a correctly built
Input page has zeros at precisely the positions where the mask is 0.

Measured against a capture taken from the SenseAI desktop app over the same
sample, after the fix:

```
                    mean      std      unmeasured   pct[1,25,50,75,99]
ours                24868.9   2673.6   [0]          [20518 23352 24451 25789 35154]
SenseAI desktop     24692.8   2619.2   [0]          [20402 23217 24293 25586 34833]
```

The remaining ~1% is the difference between two separate acquisitions, not a
convention mismatch.

## Writing a TIFF that SenseAI will load

Use `save_capture_stack` in `fibsem/SenseAI_SG.py`. The one thing that matters:

```python
tifffile.imwrite(path, stack, photometric="minisblack")
```

Without `photometric`, `tifffile` reads the leading axis of a `(4, rows, cols)`
array as *samples*, like RGBA, and writes a single page:

| | naive `imwrite(path, stack)` | with `photometric="minisblack"` |
|---------------------|------------------------------|----------------------------------|
| IFDs | 1 | 4 |
| SamplesPerPixel | 4 | 1 |
| PlanarConfiguration | SEPARATE | CONTIG |
| Photometric | RGB | MINISBLACK |
| SenseAI `Size` | `[rows, cols, 4, 1]` | `[rows, cols, 1, 1]` |
| layers assigned | none | all four |

The `Size` in the response to `load_image` is the quickest diagnostic: a 4 in
the third slot means it was read as a 4-deep volume and the layers were never
assigned.

Nothing else is required. SenseAI's own writer uses LZW with a horizontal
predictor, `RowsPerStrip=2`, `NewSubfileType=PAGE` and a `PageNumber` tag of
`(n, 4)` on each page, but an uncompressed file with none of those loads
correctly. Matching them is optional.

### What `load_image` actually does

`senseai.py:74-97` builds `{"Name", "Origin": path, "Size": [1,1,1,1],
"Views": [...]}` and calls `AddImage` — the *same* DLL entry point as
`add_image`. The difference is only which half of the contract is used:
`load_image` sets `Origin` and stops; `add_image` omits it and then pushes
pixels with three `set_image_array` calls. The `Size` in the request is a
placeholder that the DLL overwrites from the file.

There is no layer or page parameter. The page-to-layer mapping is fixed in the
DLL and cannot be overridden from Python.

Inside the DLL the loader is ArrayFire's `af::loadImageNative`
(`src\api\c\imageio2.cpp:153`), which delegates to FreeImage — visible in the
stack trace of any failed load. FreeImage dispatches on file type, which is why
a path with an unrecognised extension dies with `Unknown File or Filetype`.

### Two red herrings

**`.MultiLayer` is not a load convention.** The DLL contains
`(.*)\.\w+\.MultiLayer$` and `Multi-layer file must be a tiff`, which look like
a way to ask for a multi-layer load. Passing `file.tiff.MultiLayer` as the origin
fails outright in FreeImage. Those strings belong to the `MakeMultiLayerTIFF` /
`DatasetSave` steps, not to `AddSource`.

**The `tifffile` "shaped" description is harmless.** `imwrite` writes
`{"shape": [4, rows, cols]}` into page 0's `ImageDescription` by default.
Suppressing it with `metadata=None` changes nothing; the file still fails. The
sample-vs-page structure is the whole cause.

### Page 0 is ignored on load

After loading a saved stack, the Reconstruction layer comes back **NaN-filled**,
not the zeros that were written. SenseAI initialises that layer rather than
reading page 0. So a reconstruction saved into page 0 will not come back when
the file is reloaded — worth knowing before treating these files as a complete
round-trip of a reconstructed slice.

## Reconstruction

`quick_recon` (`senseai/recon.py:28-33`) creates **three** sources named after
the image, on top of the image itself:

```
img01, Dictionary img01, Train img01, Recon img01     (+ task "Recon img01")
```

Every name is derived from the image name and the image name is fixed, so
calling `quick_recon` a second time fails at the first `add_image` with `This
source already exists`, before the reconstruction is even reached.

`SenseAI_ModuleControl.reconstruct` therefore **does not call `quick_recon`**. It
creates the image, dictionary and two instances itself on the first call, keeps
the step list that `quick_recon` would have built (`_recon_task_steps`, mirroring
`recon.py:35-77`), and for every later slice replaces the pixel data with
`set_image_array` and re-queues the same task. Sources are rebuilt only when the
frame shape changes, since an image's `Size` is fixed at `add_image`.

Reuse was checked against building fresh sources every time, scoring both on
held-out measured pixels (a fifth of the measured pixels withheld from the input,
then compared against the reconstruction there — with no ground truth that is the
only honest score available):

| | held-out correlation |
|--------------------------------------|------|
| fresh sources | +0.7685 |
| sources reused for a different image | +0.7668 |
| fresh sources on that same image | +0.7675 |

Reuse costs nothing, and correlation with the *previous* image is +0.008, so
nothing leaks between slices and no reset step is needed. Held-out and fit
correlations agree to three decimals, so the reconstruction generalises to pixels
it never saw rather than overfitting the measured ones.

Two things that are easy to get wrong when feeding it:

- **Normalise the input to 0..1.** SenseAI does this itself when loading from a
  file, but `add_image` pushes arrays through untouched. Raw uint16 counts score
  +0.43 against held-out pixels where normalised input scores +0.77.
- **Reduce the mask to 0/1.** The stack stores a measured pixel as 65535 so the
  layer is visible in the GUI, and SenseAI does not rescale the mask.

`recon_inputs_from_stack` does both.

### Parameters

Not fibsem's problem for now. The defaults (`iterations=50`, patch `20x20`,
`dict_size=36`) produce something good enough to look at, and because the input
and mask are saved in the capture stack, any slice can be re-reconstructed and
tuned properly in the SenseAI desktop app. It ships an Optuna-based hyperparameter
search — see `senseai/optimisation/study.html` in the bundled documentation.

Two measurements worth keeping so nobody re-derives them:

- **50 iterations is converged.** 400-500 iterations gains +0.001 at every size
  tested. If a reconstruction looks poor, iterations are not the reason.
- **Quality scales with frame size, not effort.** On a 50% sampled capture,
  128x128 scored +0.265, 256x256 +0.769, 512x512 +0.865. A small crop simply does
  not contain enough patches to learn a dictionary, so do not judge the pipeline
  by a cropped test.

### The buffer URL cache trap

`SenseAI._buffer_urls` (`senseai.py:40`) is a **class** attribute, shared by
every instance, mapping a params JSON to a registered buffer URL.
`delete_dictionary` purges its entries; `delete_image` and `delete_algorithm` do
not (there is a literal `# TODO: This everywhere` at `senseai.py:257`). So after
deleting and re-adding an image, `get_image_buffer` can reuse a URL registered
against the deleted source — stale pixels or a crash, and constructing a new
`SenseAI` object does not clear it.

`reconstruct` sidesteps this entirely by reading results with `get_image_array`,
which does not use the cache. That is also the right call on the merits:
`get_image_buffer` returns normalised uint8, `get_image_array` returns float32.
For anything but display, use the array.

## Threading in the widget

Acquisition runs on a `@thread_worker` so the UI stays responsive. Per the
project convention, the worker touches no widgets:

- `acquire_image()` — main thread. Reads the save checkbox and path, validates
  the directory *before* acquiring, and applies the UI values to the config with
  `update_scanGen_config()`.
- `_acquire_image()` — worker. Updates the scan generator, acquires, returns the
  raw `(image, mask)`.
- `_acquisition_returned()` — main thread, on `worker.returned`. Builds the
  capture stack, updates the viewer, saves the TIFF.

`notification_service.show_toast` is safe from a worker: it emits a Qt signal
rather than touching a widget directly.

## Known gaps

- **The reconstruction is not written back into the stack.** `reconstruct`
  returns the array and the widget displays it, but page 0 of the saved stack
  stays zero — a uint16 stack cannot hold a float32 0..1 reconstruction without
  scaling or promoting the whole stack, and SenseAI ignores page 0 on load
  anyway. Since the input and mask are saved, any slice can be re-reconstructed
  later, so this is a choice rather than a gap.
- **The viewer scales the raw signed frame.** `_update_eb_image_viewer` is given
  `senseAI_img` as the detector sent it, so the `INT16_MIN` sentinel at every
  unmeasured pixel drags the display range down and flattens the contrast.
  `build_capture_stack` already produces corrected data — `stack[INPUT_PAGE]`
  would display better.
- **Re-initialising the scan generator fails.** A second *Initialize Scan
  Generator* hits `This source already exists` inside `add_ScanGenerator`, which
  logs and swallows it, leaves `scanGen_added` False, and then
  `init_scan_generator` bails with "has not been added yet". The same
  delete-then-add treatment `reconstruct` uses would fix it.
- **The external-scan switch is a no-op.**
  `microscope._set(key="ElectronBeam_scan", value="external")` in
  `initialize_scan_generator` matches no key in any backend. ThermoMicroscope's
  `_set` ends in `logging.warning(f"Unknown key: {key}")` and returns, and the
  microscope log shows exactly that:

  ```
  WARNING — _set:3991 — Unknown key: ElectronBeam_scan (None)
  ```

  The key ThermoMicroscope handles is `"scantype_external"`, and it needs a
  `beam_type` (the call passes none). That acquisition works anyway suggests the
  column was already in external scan — from the SenseAI desktop app or an
  earlier session — rather than that the call did anything.

  Worse for unattended runs: `microscope.acquire_image()` calls
  `set_full_frame_scanning_mode()` internally, so **any ordinary acquisition puts
  the column back to full frame**. External scan therefore cannot be set once at
  initialisation; it has to be set immediately before each sub-sampled
  acquisition.
- **No `acquire_image` in the Salami task layer yet.** The SenseAI path is
  driven only from its tab; nothing in the run loop calls it.

## Probing recipes

Check what is registered, and clear a name clash without restarting:

```python
s.list_contents()
s.delete_instance("Recon img01"); s.delete_instance("Train img01")
s.delete_dictionary("Dictionary img01"); s.delete_image("img01")
```

Verify a saved capture is well formed, without the DLL:

```python
with tifffile.TiffFile(path) as tf:
    assert len(tf.pages) == 4
    assert tf.pages[0].tags["SamplesPerPixel"].value == 1
```

Verify SenseAI assigns the layers (needs the DLL, no microscope). Write a
different pattern to each page, then confirm each layer matches the page it
should:

```python
meta = s.load_image("probe", path)
meta["Size"]                          # want [rows, cols, 1, 1], not [..., 4, 1]
s.get_image_array("probe", "Input")   # compare against page 2
```

Assert invariants rather than exact values — that the mask is binary, that every
non-zero input pixel is inside the mask, that Reference equals Input — since the
absolute numbers depend on detector gain and on whether SenseAI has normalised.
