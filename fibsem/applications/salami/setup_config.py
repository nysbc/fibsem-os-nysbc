"""The setup section of a Salami protocol: what Setup Experiment mills.

Two steps, each a full milling config (stages, patterns, currents) and a skip
switch:

    trench    Mill Trenches — as many stages as the sample needs
    fiducial  Mill Fiducial — the mark the slices align against

They are protocol, not run state: a protocol reused on another sample mills
the same trench and fiducial. Skipping is in the protocol too, so a re-run of
setup on a sample whose trench is already cut can leave it alone.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional

from fibsem.applications.salami._default_settings import (
    DEFAULT_FIDUCIAL,
    DEFAULT_TRENCH,
)
from fibsem.milling.tasks import FibsemMillingTaskConfig

TRENCH_KEY = "trench"
FIDUCIAL_KEY = "fiducial"

# Step key -> how the step is named in the editor and the setup prompts.
SETUP_STEP_LABELS: Dict[str, str] = {
    TRENCH_KEY: "Mill Trenches",
    FIDUCIAL_KEY: "Mill Fiducial",
}


@dataclass
class SetupStepConfig:
    milling: FibsemMillingTaskConfig
    # Leave this step out when setup runs, e.g. the trench is already milled.
    skip: bool = False

    def to_dict(self) -> dict:
        return {"skip": self.skip, "milling": self.milling.to_dict()}

    @classmethod
    def from_dict(
        cls, data: Optional[dict], default: FibsemMillingTaskConfig
    ) -> "SetupStepConfig":
        data = data or {}
        milling = data.get("milling")
        return cls(
            milling=FibsemMillingTaskConfig.from_dict(milling)
            if milling
            else deepcopy(default),
            skip=bool(data.get("skip", False)),
        )


@dataclass
class SalamiSetupConfig:
    trench: SetupStepConfig = field(
        default_factory=lambda: SetupStepConfig(milling=deepcopy(DEFAULT_TRENCH))
    )
    fiducial: SetupStepConfig = field(
        default_factory=lambda: SetupStepConfig(milling=deepcopy(DEFAULT_FIDUCIAL))
    )

    def step(self, key: str) -> SetupStepConfig:
        if key == TRENCH_KEY:
            return self.trench
        if key == FIDUCIAL_KEY:
            return self.fiducial
        raise KeyError(f"Unknown setup step {key!r}")

    def to_dict(self) -> dict:
        return {
            TRENCH_KEY: self.trench.to_dict(),
            FIDUCIAL_KEY: self.fiducial.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "SalamiSetupConfig":
        """Read the section; a protocol written before it existed gets defaults."""
        data = data or {}
        return cls(
            trench=SetupStepConfig.from_dict(data.get(TRENCH_KEY), DEFAULT_TRENCH),
            fiducial=SetupStepConfig.from_dict(
                data.get(FIDUCIAL_KEY), DEFAULT_FIDUCIAL
            ),
        )
