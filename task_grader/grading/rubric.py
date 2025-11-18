from dataclasses import dataclass
from typing import Literal, Mapping, get_args

ScoreScale = Literal["0-1", "0-5", "0-10", "percentage"]

# Numeric ranges for each score scale.
# Used only for validation + computing an overall numeric score.
SCORE_SCALE_NUMERIC_RANGES: dict[ScoreScale, tuple[int, int]] = {
    "0-1": (0, 1),
    "0-5": (0, 5),
    "0-10": (0, 10),
    "percentage": (0, 100),
}


def _build_score_scale_descriptions(
    ranges: Mapping[ScoreScale, tuple[int, int]],
) -> dict[ScoreScale, str]:
    """
    Generate human-readable descriptions for each score scale from its numeric range.

    This keeps SCORE_SCALE_NUMERIC_RANGES as the single source of truth, and
    avoids duplicating the same information in two separate dicts.
    """
    descriptions: dict[ScoreScale, str] = {}

    for scale, (lo, hi) in ranges.items():
        # Optional: special-case 0–1 to keep the slightly nicer "0 or 1" phrasing
        if lo == 0 and hi == 1:
            desc = "use an integer score of 0 or 1"
        else:
            desc = f"use an integer score from {lo} to {hi}"
        descriptions[scale] = desc

    return descriptions


# Map each score scale to its description (used as semantic guardrail in the grading prompt template)
SCORE_SCALE_DESCRIPTIONS: dict[ScoreScale, str] = _build_score_scale_descriptions(
    SCORE_SCALE_NUMERIC_RANGES
)


@dataclass
class Criterion:
    id: str
    name: str
    description: str
    weight: float
    scale: ScoreScale

    def __post_init__(self):
        if self.scale not in SCORE_SCALE_DESCRIPTIONS:
            raise ValueError(
                f"Invalid scale: {self.scale}. Must be one of {get_args(ScoreScale)}"
            )

        if self.weight <= 0:
            raise ValueError(f"Invalid weight: {self.weight}. Must be positive")


@dataclass
class Rubric:
    task_id: str
    title: str
    description: str
    overall_max_score: float
    min_passing_score: float
    criteria: list[Criterion]

    def __post_init__(self):
        if not self.criteria:
            raise ValueError("criteria must be non-empty")

        if self.min_passing_score <= 0:
            raise ValueError(
                f"Invalid min_passing_score: {self.min_passing_score}. Must be positive"
            )

        if self.min_passing_score > self.overall_max_score:
            raise ValueError(
                f"Invalid min_passing_score: {self.min_passing_score}. Must be less than or equal to overall_max_score"
            )
