from fibsem.applications.salami.structures import DEFAULT_SLICE_THICKNESS
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning import RectanglePattern
from fibsem.milling.patterning.patterns2 import TrenchPattern
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import FibsemMillingSettings

# Width of the slab removed each slice: the lateral extent of the block face
# being collected.
DEFAULT_SLICE_WIDTH = 40e-6
# How far the cut reaches into the face. Not the slice thickness — that is the
# pattern's height.
DEFAULT_SLICE_DEPTH = 5e-6

# A slice is a wide, very thin slab: 40 um across and one slice thickness tall.
#
# A rectangle rather than a line, because a line has no height to be the slice
# thickness and `LinePattern.define()` ignores `point`, so the pattern could not
# be positioned or advanced. `RectanglePattern.define()` reads
# `centre_x/centre_y` straight off `point`.
DEFAULT_SLICE = FibsemMillingStage(
    pattern=RectanglePattern(
        width=DEFAULT_SLICE_WIDTH,
        height=DEFAULT_SLICE_THICKNESS,
        depth=DEFAULT_SLICE_DEPTH,
    ),
    name="Default Slice",
    milling=FibsemMillingSettings(milling_current=7.6e-9),
)

DEFAULT_TRENCH = FibsemMillingTaskConfig(
    name="Trench Milling",
    field_of_view=180e-6,
    stages=[
        FibsemMillingStage(
            name="Trench Milling 01",
            milling=FibsemMillingSettings(milling_current=7.6e-9),
            pattern=TrenchPattern(
                width=22e-6,
                upper_trench_height=32e-6,
                lower_trench_height=16e-6,
                spacing=25e-6,
                depth=1.25e-6,
            ),
        )
    ],
)
