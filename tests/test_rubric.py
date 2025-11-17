import pytest

from task_grader.grading.rubric import (
    ScoreScale,
    SCORE_SCALE_DESCRIPTIONS,
    Criterion,
    Rubric,
)
from typing import get_args


def test_score_scale_descriptions_cover_all_literals():
    """SCORE_SCALE_DESCRIPTIONS should have exactly one entry per ScoreScale literal."""
    literals = set(get_args(ScoreScale))
    mapping_keys = set(SCORE_SCALE_DESCRIPTIONS.keys())

    assert mapping_keys == literals, (
        "SCORE_SCALE_DESCRIPTIONS keys must match ScoreScale literals. "
        f"Missing: {literals - mapping_keys}, extra: {mapping_keys - literals}"
    )


def test_score_scale_description_strings_are_non_empty():
    """Each score scale description should be a non-empty string."""
    for scale, desc in SCORE_SCALE_DESCRIPTIONS.items():
        assert isinstance(desc, str)
        assert desc.strip(), f"Description for scale {scale!r} must not be empty."


def test_rubric_post_init_accepts_valid_scores():
    """Rubric should allow min_passing_score > 0 and <= overall_max_score."""
    rubric = Rubric(
        task_id="task-1",
        title="Sample Rubric",
        description="A test rubric.",
        overall_max_score=100,
        min_passing_score=60,
        criteria=[
            Criterion(
                id="clarity",
                name="Clarity",
                description="How clear the submission is.",
                weight=0.5,
                scale="0-10",
            )
        ],
    )
    assert rubric.overall_max_score == 100
    assert rubric.min_passing_score == 60


def test_rubric_post_init_rejects_non_positive_min_passing_score():
    """Rubric should raise if min_passing_score is not positive."""
    with pytest.raises(ValueError) as excinfo:
        Rubric(
            task_id="task-1",
            title="Sample Rubric",
            description="A test rubric.",
            overall_max_score=100,
            min_passing_score=0,
            criteria=[
                Criterion(
                    id="clarity",
                    name="Clarity of intent and scope",
                    description="How clearly the evaluation intent and scope are stated.",
                    weight=0.3,
                    scale="0-10",
                )
            ],
        )
    assert "Must be positive" in str(excinfo.value)


def test_rubric_post_init_rejects_min_passing_score_above_max():
    """Rubric should raise if min_passing_score > overall_max_score."""
    with pytest.raises(ValueError) as excinfo:
        Rubric(
            task_id="task-1",
            title="Sample Rubric",
            description="A test rubric.",
            overall_max_score=50,
            min_passing_score=60,
            criteria=[
                Criterion(
                    id="clarity",
                    name="Clarity of intent and scope",
                    description="How clearly the evaluation intent and scope are stated.",
                    weight=0.3,
                    scale="0-10",
                )
            ],
        )
    assert "less than or equal to overall_max_score" in str(excinfo.value)


def test_rubric_rejects_empty_criteria():
    """Rubric should raise an error if criteria list is empty."""
    with pytest.raises(ValueError) as excinfo:
        Rubric(
            task_id="task-empty",
            title="Empty Rubric",
            description="This rubric has no criteria.",
            overall_max_score=100,
            min_passing_score=50,
            criteria=[],  # empty list
        )
    msg = str(excinfo.value)
    assert "criteria" in msg.lower()
    assert "empty" in msg.lower()
