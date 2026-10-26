import json
import logging
import os
import time
from typing import Optional

import numpy as np
import tifffile

# from senseai import SenseAI


### scan generator and external detector imaging code here
### also code for senseAI processing and reconstruction here


# SenseAI holds every image as four layers and its capture step writes them to a
# 4-page TIFF (Detector.cpp -> "capture_{}_{}.tiff"). This page order was recovered
# by loading a probe file with a different pattern on each page and reading back
# which page each layer matched, so it is the order SenseAI's *loader* uses:
RECONSTRUCTION_PAGE = 0
REFERENCE_PAGE = 1
INPUT_PAGE = 2
MASK_PAGE = 3

SENSEAI_LAYERS = ("Reconstruction", "Reference", "Input", "Mask")
N_SENSEAI_LAYERS = len(SENSEAI_LAYERS)

# Value marking a measured pixel in a uint16 mask layer. SenseAI writes the dtype
# maximum rather than 1, so the mask page is visible in the SenseAI GUI.
MASK_MEASURED = 65535


def _to_offset_binary(image: np.ndarray) -> np.ndarray:
    """
    Convert a signed detector frame to the unsigned convention SenseAI stores.

    The analogue detector delivers *signed* samples, and SenseAI's own capture
    writer shifts them by half the range rather than casting. Casting straight to
    uint16 instead wraps every negative sample to the top of the range, which
    shifts the histogram by 32768 and folds the bright tail down to zero.

    The detector also writes the most negative value at pixels the beam never
    visited, and the shift maps that sentinel onto 0 — which is exactly what
    SenseAI writes for an unmeasured pixel.
    """
    if not np.issubdtype(image.dtype, np.signedinteger):
        return image

    offset = -int(np.iinfo(image.dtype).min)
    return np.clip(image.astype(np.int64) + offset, 0, 65535)


def build_capture_stack(
    image: np.ndarray,
    mask: np.ndarray = None,
    reconstruction: np.ndarray = None,
) -> np.ndarray:
    """
    Bundle a detector frame into SenseAI's four-layer image stack.

    The stack is uint16 to keep the raw detector counts. SenseAI only converts to
    float32 once the data enters its own pipeline, and a reconstructed image comes
    back normalised to 0..1, which would throw away the acquired dynamic range.
    A signed frame is shifted rather than cast — see _to_offset_binary.

    Args:
        image: The acquired (sub-sampled) frame, 2D.
        mask: Which pixels were measured; any non-zero value counts as measured.
            If None, every pixel is marked as measured.
        reconstruction: An existing reconstruction, if there is one. Zeros otherwise.

    Returns:
        (4, rows, cols) uint16 array ordered as SENSEAI_LAYERS.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}")

    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != image.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {image.shape}"
            )

    input_layer = _to_offset_binary(image)
    if mask is not None:
        # SenseAI writes 0 wherever the beam never went. The detector's own
        # sentinel already lands on 0 after the shift; this keeps that true for
        # any detector that reports unvisited pixels some other way.
        input_layer = np.where(mask != 0, input_layer, 0)

    stack = np.zeros((N_SENSEAI_LAYERS, *image.shape), dtype=np.uint16)
    stack[INPUT_PAGE] = input_layer
    # SenseAI defaults the reference to a copy of the input when none is supplied
    # (senseai.add_image) and its own captures are written that way, so a stack we
    # build is indistinguishable from one it wrote.
    stack[REFERENCE_PAGE] = input_layer

    if mask is None:
        stack[MASK_PAGE] = MASK_MEASURED
    else:
        stack[MASK_PAGE] = np.where(mask != 0, MASK_MEASURED, 0)

    if reconstruction is not None:
        reconstruction = np.asarray(reconstruction)
        if reconstruction.shape != image.shape:
            raise ValueError(
                f"Reconstruction shape {reconstruction.shape} does not match "
                f"image shape {image.shape}"
            )
        stack[RECONSTRUCTION_PAGE] = reconstruction

    return stack


def recon_inputs_from_stack(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pull the two layers a reconstruction needs out of a capture stack.

    Both conversions matter:

    * The image is normalised to 0..1, which is what SenseAI does itself when it
      loads an image from a file. Pushing raw uint16 counts through add_image
      skips that step, and measured on real data it roughly halves the quality of
      the result (correlation against held-out pixels 0.43 vs 0.77).
    * The mask is reduced to 0/1. The stack stores a measured pixel as
      MASK_MEASURED so the layer is visible in the GUI, but SenseAI treats the
      mask as an indicator and does not rescale it, so feeding the stored value
      back would scale the mask by 65535.

    Returns:
        (image, mask), both float32 and C-contiguous as the DLL requires.
    """
    image = stack[INPUT_PAGE].astype(np.float32) / np.iinfo(np.uint16).max
    mask = (stack[MASK_PAGE] != 0).astype(np.float32)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def _recon_task_steps(image: str, dictionary: str, train: str, recon: str) -> list:
    """
    The steps a sparse reconstruction runs, per iteration.

    This mirrors what senseai.recon.quick_recon builds internally (recon.py:35-77).
    We build it here rather than calling quick_recon because that function also
    *creates* the dictionary and instances, which can only be done once per name —
    a second call fails with "This source already exists". Keeping the steps means
    the sources can be created once and the task re-queued for every later slice.
    """
    return [
        {
            "Configuration": {"Epochs": 0, "Batches": 1, "Type": "TrainDictionary"},
            "Image": image,
            "Dictionary": dictionary,
            "Instance": train,
            "Type": "TrainDictionary",
            "Title": "Train Dictionary",
        },
        {
            "Configuration": {"Epochs": 0, "Batches": 1, "Type": "SparseCode"},
            "Image": image,
            "Dictionary": dictionary,
            "Instance": recon,
            "Detector": "",
            "ScanGenerator": "",
            "Slicer": "",
            "Type": "SparseCode",
            "Title": "Sparse-Code",
        },
        {
            "Configuration": {
                "Array": "FullXhat",
                "Destination": "Reconstruction",
                "Rescale": True,
                "Normalise": False,
                "Type": "Reconstruct",
            },
            "Image": image,
            "Dictionary": dictionary,
            "Instance": recon,
            "Type": "Reconstruct",
            "Title": "Reconstruct",
        },
    ]


def save_capture_stack(path: str, stack: np.ndarray) -> None:
    """
    Write a capture stack as a 4-page TIFF that SenseAI will load as four layers.

    photometric="minisblack" is not optional. Without it tifffile reads the leading
    axis as samples and writes a *single* page with SamplesPerPixel=4, which SenseAI
    loads as one 4-deep volume with no layers assigned.
    """
    if stack.ndim != 3 or stack.shape[0] != N_SENSEAI_LAYERS:
        raise ValueError(
            f"Expected a ({N_SENSEAI_LAYERS}, rows, cols) stack, got shape {stack.shape}"
        )

    tifffile.imwrite(path, stack, photometric="minisblack")
    logging.info(f"Saved capture stack to {path}")


def reconstruction_to_uint16(reconstruction: np.ndarray) -> np.ndarray:
    """
    Scale a reconstruction so it can be stored in a uint16 capture stack.

    A reconstruction comes back as float, roughly 0..1, and assigning that into
    a uint16 page would truncate every pixel to 0. Non-finite values are mapped
    to 0 rather than wrapping: SenseAI fills the Reconstruction layer with NaN
    until a task has written to it.
    """
    recon = np.nan_to_num(
        np.asarray(reconstruction, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0
    )
    return np.clip(recon * MASK_MEASURED, 0, MASK_MEASURED).astype(np.uint16)


# ---------------------------------------------------------------------------
# Column scan mode
# ---------------------------------------------------------------------------
#
# The scan generator can only drive the beam while the column is in external
# scan. Two things make this fiddly, and both are easy to get wrong:
#
#   * The key is `scantype_external`, and it needs a beam type. An unrecognised
#     key does not raise — ThermoMicroscope's `_set` ends in
#     `logging.warning(f"Unknown key: {key}")` and returns — so a typo here is
#     silent, and was: `_set("ElectronBeam_scan", "external")` never did
#     anything.
#   * It cannot be set once and left. `microscope.acquire_image()` calls
#     `set_full_frame_scanning_mode()` internally, so any ordinary acquisition
#     anywhere puts the column back to full frame. External scan has to be set
#     immediately before each sub-sampled acquisition.


def set_external_scan(microscope, beam_type=None) -> None:
    """Hand the column's scan over to the external scan generator."""
    from fibsem.structures import BeamType

    beam_type = beam_type or BeamType.ELECTRON
    microscope.set("scantype_external", None, beam_type)
    logging.debug(f"{beam_type.name} scan set to external.")


def set_internal_scan(microscope, beam_type=None) -> None:
    """Give the scan back to the column, so ordinary imaging works again."""
    from fibsem.structures import BeamType

    beam_type = beam_type or BeamType.ELECTRON
    microscope.set_full_frame_scanning_mode(beam_type)
    logging.debug(f"{beam_type.name} scan returned to full frame.")


# ---------------------------------------------------------------------------
# The process's SenseAI module
# ---------------------------------------------------------------------------
#
# One instance per process, because the thing it wraps is already one per
# process: the DLL's source registry outlives any Python object, so a second
# SenseAI_ModuleControl does not get a clean slate — it inherits every scan
# generator, detector and image the first one added, and collides with them on
# "This source already exists".
#
# A module-level instance is therefore modelling the constraint rather than
# working around it, and it is what lets a headless task reach the same scan
# generator the UI initialised.

_MODULE: Optional["SenseAI_ModuleControl"] = None


def set_senseai_module(module: Optional["SenseAI_ModuleControl"]) -> None:
    """Adopt a module built elsewhere, so everything else in the process shares it."""
    global _MODULE
    _MODULE = module


def get_senseai_module() -> Optional["SenseAI_ModuleControl"]:
    """The process's SenseAI module, or None if nothing has published one.

    Deliberately does not create one. The DLL and the scan generator are loaded
    once per process, from the SenseAI tab; anything that wanted a module and
    found none would be building a second one against a registry that already
    holds the first one's sources, which fails at "source already exists" and
    leaves a module that can never be initialised.
    """
    return _MODULE


def require_senseai_module(
    module: Optional["SenseAI_ModuleControl"] = None,
) -> "SenseAI_ModuleControl":
    """The module to acquire through, or raise explaining what is missing.

    `module` is the caller's own — a headless run passes the one it built. When
    it is None the process-wide one is used, which the SenseAI tab publishes
    when the operator initialises the scan generator.

    Raises rather than falling back to building one: a run that quietly creates
    a second module produces a scan generator that never returns a frame, which
    is far harder to diagnose than being told the generator is not initialised.
    """
    module = module if module is not None else _MODULE

    if module is None:
        raise RuntimeError(
            "No SenseAI scan generator is available. Initialise it on the "
            "SenseAI tab before running, or pass a module in for a headless run."
        )

    problem = module.readiness_problem()
    if problem is not None:
        raise RuntimeError(
            f"The SenseAI scan generator is not usable: {problem}. Re-initialise "
            "it on the SenseAI tab; if that keeps failing, the DLL still holds "
            "the sources from an earlier one and the application has to be "
            "restarted."
        )
    return module


class SenseAI_Config:
    def __init__(self, senseAI_WorkspaceJSON_path: str = None):

        self.senseAI_WorkspaceJSON_path = senseAI_WorkspaceJSON_path
        self.scanGen_config: dict = None
        self.detector_config: dict = None
        self.scanGen_initialised: bool = False

        # Always load: load_config_information() falls back to defaults for a
        # missing path, so skipping it here would leave the configs as None and
        # anything built from this config would fail on the first lookup.
        self.load_config_information(self.senseAI_WorkspaceJSON_path)

    def load_config_information(self, json_path: str = None):

        ## load the config from the json into python dicts
        if not json_path or json_path == "":
            logging.info("No config path provided, using default values")

            self.scanGen_config = {
                "ScanSize": [640, 480],
                "DwellTime": 0.010,
                "Pattern": "Linehop",
                "Sampling": 0.3,
                "ResetBuffer": True,
                "Name": "WCScanGen",
            }

            self.detector_config = {
                "Input": 0,
                "QDType": "qd_analogue",
                "Name": "WCDetector",
            }

            return

        try:
            workspace_config = self._load_json(json_path)
        except Exception as e:
            logging.error(f"Error loading Workspace JSON file: {e}")
            return

        self.scanGen_config = workspace_config["ScanGenerators"][0]
        self.detector_config = workspace_config["Detectors"][0]

        return

    def _load_json(self, json_path: str):

        with open(json_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        return data

    def update_config(self):

        ## update necessary parameters here

        pass


class SenseAI_ModuleControl:
    """
    This class provides control over the SenseAI module,
    including loading the dll, adding and updating scan generators and detectors,
    initializing scan generators, and getting images from detectors.

    """

    def __init__(self, dll_path: str = None, config: SenseAI_Config = None):

        self.dll_path = dll_path
        self.config = config

        self.dll_loaded: bool = False
        self.scanGen_added: bool = False
        self.detector_added: bool = False
        # Set only by a successful init_scan_generator. Separate from
        # scanGen_added, which only says the source was registered: a generator
        # can be added and still never have been initialised, and acquiring
        # against one that has not been is what leaves the caller waiting on a
        # frame that never arrives.
        self.scanGen_initialised: bool = False

        self.scanGen_name = "Quantum Detectors"
        self.detector_name = "Quantum Detectors"

        # Reconstruction sources are created once and reused for every later
        # slice; see reconstruct(). _recon_shape is the frame shape they were
        # built for — a different shape means they have to be rebuilt.
        self.recon_image_name = "img01"
        self._recon_shape: tuple = None
        self._recon_steps: list = None

        self.load_dll()

        if self.config is None:
            logging.info("No config provided, using default values")
            self.config = SenseAI_Config()

        if self.config is not None:
            self.add_ScanGenerator(self.config.scanGen_config)
            self.add_Detector(self.config.detector_config)

    def load_dll(self):
        """Load the SenseAI dll and initialize the SenseAI module."""
        if self.dll_path is None or self.dll_path == "":
            logging.info(
                "No dll path provided, cannot load SenseAI module. Loading Sim"
            )
            self.senseAI = SIM_SenseAI_Module()
            self.dll_loaded = False
            return

        try:
            # senseai.quick_recon is deliberately not used: it creates sources as a
            # side effect and so cannot be called twice. reconstruct() does the same
            # work with the sources kept alive between slices.
            from senseai import SenseAI

            self.senseAI = SenseAI(self.dll_path)
            self.dll_loaded = True
            logging.info(f"SenseAI module loaded successfully from {self.dll_path}")

        except Exception as e:
            logging.error(f"Error loading SenseAI module: {e}")

    @property
    def is_ready(self) -> bool:
        """True when this module can actually acquire.

        `dll_loaded` is deliberately not required: with no DLL path the module
        falls back to the simulator, which is a legitimate way to work offline.
        What matters is that the generator and detector were registered and the
        generator was initialised.
        """
        return bool(
            self.scanGen_added and self.detector_added and self.scanGen_initialised
        )

    def readiness_problem(self) -> Optional[str]:
        """Why this module cannot acquire, or None when it can.

        Worth stating precisely: every step of the setup logs its own failure
        and then carries on, so by the time acquisition fails the cause is well
        upstream of it.
        """
        if not self.scanGen_added:
            return (
                f"scan generator {self.scanGen_name!r} was not added "
                "(the DLL registry may already hold one from an earlier module)"
            )
        if not self.detector_added:
            return f"detector {self.detector_name!r} was not added"
        if not self.scanGen_initialised:
            return f"scan generator {self.scanGen_name!r} was never initialised"
        return None

    def add_ScanGenerator(self, scanGen_config: dict) -> None:
        """Add a scan generator with the given name and configuration."""

        name = scanGen_config["Name"]
        scangen_type = "QD"

        if name is not None:
            if "WC" in name:
                scangen_type = "QDMock"
            self.scanGen_name = name

        try:
            self.senseAI.hw.add_scan_generator(
                self.scanGen_name, scangen_type, scanGen_config
            )
            self.scanGen_added = True
            logging.info(
                f"Scan generator {self.scanGen_name} of type {scangen_type} added successfully"
            )
        except Exception as e:
            logging.error(f"Error adding scan generator: {e}")

    def add_Detector(self, detector_config: dict) -> None:
        """Add a detector with the given name and configuration."""

        name = detector_config["Name"]

        if name is not None:
            self.detector_name = name

        try:
            self.senseAI.hw.add_detector(
                self.detector_name, "ScanGenerator", self.scanGen_name, detector_config
            )
            self.detector_added = True
            logging.info(
                f"Detector {self.detector_name} added to Scan Gen {self.scanGen_name} successfully"
            )
        except Exception as e:
            logging.error(f"Error adding detector: {e}")

    def update_ScanGenerator(self, scanGen_config: dict) -> None:
        """Update the scan generator with the given name and configuration."""
        try:
            current_dir = os.getcwd()
            os.chdir(r"C:\Program Files\SenseAI\SenseAI 2026.1.1")

            self.senseAI.hw.update_scan_generator(self.scanGen_name, scanGen_config)
            logging.info(f"Scan generator {self.scanGen_name} updated successfully")

            os.chdir(current_dir)

        except Exception as e:
            logging.error(f"Error updating scan generator: {e}")

    def init_scan_generator(self) -> None:
        """Initialize the scan generator with the given name."""

        if not self.scanGen_added:
            logging.error(
                f"Scan generator {self.scanGen_name} has not been added yet, cannot initialize"
            )
            return

        if not self.detector_added:
            logging.error(
                f"Detector {self.detector_name} has not been added yet, cannot initialize"
            )
            return

        try:
            ## try work-around
            current_dir = os.getcwd()
            os.chdir(r"C:\Program Files\SenseAI\SenseAI 2026.1.1")

            self.senseAI.hw.init_scan_generator(self.scanGen_name)
            logging.info(f"Scan generator {self.scanGen_name} initialized successfully")
            self.scanGen_initialised = True

            ### running this method starts the scan on the scan engine
            ### pause the scanning so it is not scanning unnecessarily? (either way the beam is not scanning)
            self.senseAI.hw.set_scan_generator_state(self.scanGen_name, paused=True)

            os.chdir(current_dir)
        except Exception as e:
            logging.error(f"Error initializing scan generator: {e}")

    def get_detector_image(self):
        """
        Get an image from the detector with the given name.
        Returns the image as a numpy array, or None if there was an error.

        Args:
            name: Name of the detector to get the image from.
        Returns:
            Image as a numpy array, or None if there was an error.

        """
        # TODO: Deprecate this method
        if not self.scanGen_added:
            logging.error("Scan Gen not initialised")
            return None

        try:
            image = self.senseAI.hw.get_detector_image(self.detector_name)
            img = image[0]

            logging.info(
                f"Image acquired from detector {self.detector_name} successfully"
            )
            return image
        except Exception as e:
            logging.error(f"Error getting image from detector: {e}")
            return None

    def acquire_image(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Acquire a full image with the current scan generator settings
        Different from get detector image.
        This method blocks the thread while waiting for the scan generator to complete scan
        Pauses the scanning once done (meaning get detector might return nothing?)


        Args:
            None

        Returns:
            Tuple with Image array and Corresponding Mask
            output[0] = Image (np.ndarray) (uint16)
            output[1] = Mask (np.ndarray)

        """
        if not self.scanGen_added:
            logging.error("Scan Generator not Initialised")
            return None

        try:
            t1 = time.time()

            # set scan gen to start scanning
            self.senseAI.hw.set_scan_generator_state(self.scanGen_name, paused=False)
            # Wait for the scan gen
            self.senseAI.hw.wait_for_scan_gen_frame(self.scanGen_name)

            logging.info(
                f"Scan Generator {self.scanGen_name} scanning: Waiting for frame..."
            )
            self.senseAI.hw.wait_for_scan_gen_frame(self.scanGen_name)
            output = self.senseAI.hw.get_detector_image(self.detector_name)
            logging.info(
                f"Frame Received from detector {self.detector_name}: Pausing Scan"
            )
            self.senseAI.hw.set_scan_generator_state(self.scanGen_name, paused=True)

            t2 = time.time()
            logging.info(f"Image acquisition took {t2 - t1:.2f} seconds")

            # The output Image is 16 bit, do not normalise the output to 8bit or something else (e.g. for UI)
            # It is the responsibility of the receiver to adjust the output for their purpose

            return output

        except Exception as e:
            logging.error(f"Error With Image Acquisition: {e}")
            return None

    def reconstruct(
        self,
        stack: np.ndarray,
        iterations: int = 50,
        patch_shape: tuple = (10, 10, 1, 1),
        dict_size: int = 36,
    ) -> np.ndarray:
        """
        Run a sparse reconstruction on a capture stack and return the result.

        The image, dictionary and BPFA instances are created on the first call and
        reused for every later slice: only the pixel data is replaced and the task
        re-queued. Measured on real data, reuse scores the same as building fresh
        sources each time (held-out correlation 0.767 vs 0.767) with no trace of
        the previous slice (0.008 correlation against the image it replaced), and
        it avoids both re-training the dictionary from scratch and the duplicate
        source name that made a second reconstruction fail.

        Blocks until the task finishes, so call it from a worker thread.

        Args:
            stack: A capture stack from build_capture_stack().
            iterations: Task iterations. 50 is converged for a 256x256 crop of a
                50% sampled frame; 500 iterations gained +0.001.
            patch_shape: Dictionary patch shape.
            dict_size: Number of dictionary elements.

        Returns:
            The reconstruction as float32, roughly 0..1, dense over the whole
            frame. None if the reconstruction could not be run.
        """
        image, mask = recon_inputs_from_stack(stack)

        try:
            if self._recon_shape != image.shape or self._recon_steps is None:
                self._create_recon_sources(image, mask, patch_shape, dict_size)
            else:
                self._update_recon_image(image, mask)

            t1 = time.time()
            self.senseAI.tasks.queue_task(
                f"Recon {self.recon_image_name}",
                iterations,
                steps=self._recon_steps,
                block=True,
            )
            logging.info(f"Reconstruction took {time.time() - t1:.2f} seconds")

            # get_image_array, not get_image_buffer: the buffer path returns
            # normalised uint8 and caches a URL per source in a dict shared by
            # every SenseAI instance, which goes stale if a source is replaced.
            recon = self.senseAI.get_image_array(
                self.recon_image_name, "Reconstruction"
            )
            return np.asarray(recon)

        except Exception as e:
            logging.error(f"Error performing reconstruction: {e}")
            return None

    def _create_recon_sources(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        patch_shape: tuple,
        dict_size: int,
    ) -> None:
        """Create the image, dictionary and instances a reconstruction needs."""
        self._delete_recon_sources()

        name = self.recon_image_name
        self.senseAI.add_image(name, image=image, mask=mask)
        dictionary = self.senseAI.add_dictionary(
            f"Dictionary {name}",
            dict_size,
            "Random",
            list(patch_shape),
            OnesElement=True,
        )
        train = self.senseAI.add_instance(
            f"Train {name}", name, dictionary["Name"], batch_size=16000
        )
        recon = self.senseAI.add_instance(
            f"Recon {name}", name, dictionary["Name"], batch_size=32000
        )

        self._recon_steps = _recon_task_steps(
            name, dictionary["Name"], train["Name"], recon["Name"]
        )
        self._recon_shape = image.shape
        logging.info(f"Created reconstruction sources for {name} {image.shape}")

    def _update_recon_image(self, image: np.ndarray, mask: np.ndarray) -> None:
        """Replace the pixel data on the existing image source."""
        name = self.recon_image_name
        self.senseAI.set_image_array(name, "Input", image)
        self.senseAI.set_image_array(name, "Reference", image)
        self.senseAI.set_image_array(name, "Mask", mask)

    def _delete_recon_sources(self) -> None:
        """
        Remove any reconstruction sources left over from an earlier run.

        The names are fixed, so a leftover source from a previous attempt makes
        the next one fail with "This source already exists" — and the registry
        lives in the DLL, which outlives any SenseAI object. Instances go first:
        they reference the dictionary and the image.
        """
        name = self.recon_image_name
        for delete, source in (
            (self.senseAI.delete_instance, f"Recon {name}"),
            (self.senseAI.delete_instance, f"Train {name}"),
            (self.senseAI.delete_dictionary, f"Dictionary {name}"),
            (self.senseAI.delete_image, name),
        ):
            try:
                delete(source)
                logging.debug(f"Deleted stale reconstruction source {source}")
            except Exception:
                pass  # not being there is the normal case

        self._recon_shape = None
        self._recon_steps = None


class SIM_Tasks:
    """Stands in for senseai.tasks. A simulated task has nothing to run."""

    def queue_task(
        self, name: str, iterations: int = 10, steps: list = None, block: bool = False
    ) -> None:
        logging.info(
            f"Running task {name} for {iterations} iterations with "
            f"{len(steps) if steps else 0} steps SIMULATED"
        )


class SIM_SenseAI_Module:
    """
    Simulated SenseAI module, used when no dll path is given.

    Sources are kept in a dict keyed by name so the duplicate-name behaviour of
    the real registry is reproduced: adding a name twice raises, which is what
    makes the reconstruction source lifecycle worth testing here at all.
    """

    def __init__(self):
        logging.info("Initializing SIM SenseAI Module")

        self.hw = SIM_HW_Interface()
        self.tasks = SIM_Tasks()
        self.sources: dict = {}

    ### Implement other methods as needed / discovered

    def _add_source(self, name: str, value) -> dict:
        if name in self.sources:
            raise RuntimeError(
                "Call failed: Failed to add source with error "
                "'This source already exists'."
            )
        self.sources[name] = value
        return {"Name": name}

    def _delete_source(self, name: str) -> None:
        if name not in self.sources:
            raise RuntimeError(f"Call failed: Source not found: {name}")
        del self.sources[name]

    def add_image(
        self, name: str, image: np.ndarray, mask: np.ndarray = None, reference=None
    ) -> dict:
        logging.info(f"Adding image {name} with shape {image.shape} SIMULATED")
        if mask is None:
            mask = np.ones_like(image)
        return self._add_source(
            name, {"Input": image, "Reference": image, "Mask": mask}
        )

    def add_dictionary(
        self,
        name: str,
        size: int = 36,
        generator: str = "Random",
        patch_shape: list = None,
        **kwargs,
    ) -> dict:
        logging.info(f"Adding dictionary {name} SIMULATED")
        return self._add_source(name, {"Size": size, "PatchShape": patch_shape})

    def add_instance(self, name: str, image: str, dictionary: str, **kwargs) -> dict:
        logging.info(f"Adding instance {name} SIMULATED")
        return self._add_source(name, {"Image": image, "Dictionary": dictionary})

    def delete_image(self, name: str) -> None:
        self._delete_source(name)

    def delete_dictionary(self, name: str) -> None:
        self._delete_source(name)

    def delete_instance(self, name: str) -> None:
        self._delete_source(name)

    def list_contents(self) -> dict:
        return {"Sources": sorted(self.sources)}

    def set_image_array(self, name: str, layer: str, image: np.ndarray) -> None:
        if name not in self.sources:
            raise RuntimeError(f"Call failed: Source not found: {name}")
        self.sources[name][layer] = image

    def get_image_array(self, name: str, layer: str) -> np.ndarray:
        """
        Return a stand-in for a layer.

        The simulated Reconstruction fills the unmeasured pixels with the mean of
        the measured ones. That is a poor reconstruction but a truthful shape: it
        is dense, in the same range as the input, and it changes when the input
        changes, so a caller that mixes up slices or layers still shows up.
        """
        if name not in self.sources:
            raise RuntimeError(f"Call failed: Source not found: {name}")

        if layer != "Reconstruction":
            return self.sources[name][layer]

        image = np.asarray(self.sources[name]["Input"], dtype=np.float32)
        mask = np.asarray(self.sources[name]["Mask"]) != 0
        fill = image[mask].mean() if mask.any() else 0.0
        return np.where(mask, image, fill).astype(np.float32)

    def get_image_buffer(self, name: str, buffer_type: str) -> np.ndarray:
        logging.info(
            f"Getting image buffer {buffer_type} for image {name} from SIM SenseAI Module"
        )
        # In the simulated module, we will just return a random image as the reconstruction
        if buffer_type == "Reconstruction":
            recon_image = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
            logging.info(
                f"Returning simulated reconstruction image for {name} with shape {recon_image.shape}"
            )
            return recon_image
        else:
            logging.error(
                f"Buffer type {buffer_type} not recognized in SIM SenseAI Module"
            )
            return None


class SIM_HW_Interface:
    """
    Simulated hardware interface for SenseAI, used for testing purposes.
    This class simulates the hardware interface for SenseAI, allowing for testing without actual hardware.
    It provides methods to add scan generators and detectors, update scan generators, and initialize scan generators.
    It also provides a method to get an image from the detector.
    """

    def __init__(self):
        logging.info("Initializing SIM HW Interface")
        self.scanGenerator: dict = None
        self.detector: dict = None

    def add_scan_generator(self, name: str, type: str, config: dict) -> None:

        logging.info(
            f"Adding scan generator {name} of type {type} with config {config}"
        )
        self.scanGenerator = {"Name": name, "Type": type, "Config": config}

    def add_detector(self, name: str, type: str, source: str, config: dict) -> None:
        logging.info(
            f"Adding detector {name} of type {type} with source {source} and config {config}"
        )
        self.detector = {"Name": name, "Type": type, "Source": source, "Config": config}

    def update_scan_generator(self, name: str, config: dict) -> None:
        logging.info(f"Updating scan generator {name} with config {config}")
        if self.scanGenerator is not None and self.scanGenerator["Name"] == name:
            self.scanGenerator["Config"] = config
        else:
            logging.error(f"Scan generator {name} not found, cannot update")

    def init_scan_generator(self, name: str) -> None:
        """
        Initialize the scan generator with the given name.
        If the scan generator is not found, log an error.
        Args:
            name: Name of the scan generator to initialize.

        Returns:
            None

        """

        logging.info(f"Initializing scan generator {name}")
        if self.scanGenerator is not None and self.scanGenerator["Name"] == name:
            logging.info(
                f"Scan generator {name} initialized with config {self.scanGenerator['Config']}"
            )
        else:
            logging.error(f"Scan generator {name} not found, cannot initialize")

    def get_detector_image(self, name: str) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Get an image from the detector with the given name.
        If the detector is not found, return None.
        If the detector is found, return a simulated image based on the scan generator configuration.

        Args:
            name: Name of the detector to get the image from.
        Returns:
            List containing the simulated image and mask as 2D numpy arrays, or None if the detector is not found.

        """
        logging.info(f"Getting image from detector {name} SIMULATED ")
        if self.detector is not None and self.detector["Name"] == name:
            logging.info(
                f"Getting image from detector {name} with config {self.detector['Config']} SIMULATED"
            )
            # return a simulated image based on the config
            scan_size = self.scanGenerator["Config"]["ScanSize"]

            ## SCAN SIZE IS REVERSED in config for SenseAI, FLIP HERE

            image = np.random.randint(
                0, 256, (scan_size[1], scan_size[0]), dtype=np.uint8
            )

            sampling = self.scanGenerator["Config"]["Sampling"]

            image, mask = self._zero_random_pixels(image, sampling)

            return [image, mask]
        else:
            logging.error(f"Detector {name} not found, cannot get image")
            return None

    def wait_for_scan_gen_frame(self, name: str) -> None:
        """
        Simulate waiting for a scan generator frame to be ready.
        In this simulation, we will just sleep for a short time to simulate the wait.

        Args:
            name: Name of the scan generator to wait for.

        Returns:
            None

        """
        logging.info(f"Waiting for scan generator {name} frame SIMULATED")
        time.sleep(1)  # Simulate waiting for a frame to be ready
        logging.info(f"Scan generator {name} frame is ready SIMULATED")

    def set_scan_generator_state(self, name: str, paused: bool) -> None:
        """
        Simulate setting the state of the scan generator to paused or unpaused.
        In this simulation, we will just log the state change.

        Args:
            name: Name of the scan generator to set the state for.
            paused: Boolean indicating whether to pause or unpause the scan generator.

        Returns:

            None

        """
        state = "paused" if paused else "unpaused"
        logging.info(f"Setting scan generator {name} state to {state} SIMULATED")

    def _zero_random_pixels(
        self, image: np.ndarray, sample_pct: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Randomly set a percentage of pixels to 0 in a 2D numpy array.

        Args:
            image:      2D numpy array
            sample_pct: float between 0 and 1 (e.g. 0.25 for 25%)

        Returns:
            Tuple of the modified copy of the image and a mask marking the pixels
            that were *kept*. The hardware mask means "this pixel was measured",
            so the simulated one has to mean the same or anything checked against
            the simulator is checked against a backwards mask.
        """
        result = image.copy()

        total_pixels = image.size  # total number of pixels
        n_zero = int(total_pixels * (1 - sample_pct))  # how many to zero out

        # Pick random flat indices, then set them to 0
        indices = np.random.choice(total_pixels, size=n_zero, replace=False)
        result.flat[indices] = 0

        mask = np.ones(image.size, dtype=np.uint8)
        mask.flat[indices] = 0
        mask = mask.reshape(image.shape)

        return result, mask


def main():
    pass


if __name__ == "__main__":
    main()
