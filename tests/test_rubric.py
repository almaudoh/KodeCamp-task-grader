import json
import pytest
from pathlib import Path
from typing import get_args

from task_grader.grading.rubric import (
    ScoreScale,
    SCORE_SCALE_DESCRIPTIONS,
    Criterion,
    Rubric,
)


def test_score_scale_descriptions_cover_all_literals():
    """SCORE_SCALE_DESCRIPTIONS should have exactly one entry per ScoreScale literal."""
    literals = set(get_args(ScoreScale))
    mapping_keys = set(SCORE_SCALE_DESCRIPTIONS.keys())

    assert mapping_keys == literals, (
        "SCORE_SCALE_DESCRIPTIONS keys must match ScoreScale literals. "
        f"Missing: {literals - mapping_keys}, extra: {mapping_keys - literals}"  # type: ignore[arg-type]
    )


def test_score_scale_description_strings_are_non_empty():
    """Each score scale description should be a non-empty string."""
    for scale, desc in SCORE_SCALE_DESCRIPTIONS.items():
        assert isinstance(desc, str)
        assert desc.strip(), f"Description for scale {scale!r} must not be empty."


def test_score_scale_descriptions_correctness():
    """Verify that SCORE_SCALE_DESCRIPTIONS contains the expected human-readable strings."""
    expected_descriptions = {
        # Special case for 0-1
        "0-1": "use an integer score of 0 or 1",
        # General case (from {lo} to {hi})
        "0-5": "use an integer score from 0 to 5",
        "0-10": "use an integer score from 0 to 10",
        "percentage": "use an integer score from 0 to 100",
    }

    for scale, expected_desc in expected_descriptions.items():
        assert (
            scale in SCORE_SCALE_DESCRIPTIONS
        ), f"Missing description for scale {scale!r}"
        assert SCORE_SCALE_DESCRIPTIONS[scale] == expected_desc, (  # type: ignore[arg-type]
            f"Description mismatch for scale {scale!r}. "
            f"Expected: {expected_desc!r}, Got: {SCORE_SCALE_DESCRIPTIONS[scale]!r}"  # type: ignore[arg-type]
        )


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


def test_rubric_save_to_json(tmp_path: Path):
    """Test saving a Rubric object to a JSON file and verify content."""
    # Setup
    criterion = Criterion(
        id="clarity",
        name="Clarity",
        description="How clear the submission is.",
        weight=0.5,
        scale="0-10",
    )
    rubric = Rubric(
        task_id="task-101",
        title="I/O Test Rubric",
        description="Test description.",
        overall_max_score=100.0,
        min_passing_score=50.0,
        criteria=[criterion],
    )

    filename = "test_rubric_data"
    filepath = tmp_path / f"{filename}.json"

    # Action: Save the rubric
    rubric.save_to_json(dest_dir=tmp_path, filename=filename)

    # Assertion: Verify the file exists
    assert filepath.is_file()

    # Assertion: Verify the content
    with open(filepath, "r") as f:
        data = json.load(f)

    # Check top-level attributes
    assert data["task_id"] == "task-101"
    assert data["overall_max_score"] == 100.0
    assert data["min_passing_score"] == 50.0

    # Check nested criteria data
    assert len(data["criteria"]) == 1
    assert data["criteria"][0]["id"] == "clarity"
    assert data["criteria"][0]["scale"] == "0-10"


def test_rubric_load_from_json(tmp_path: Path):
    """Test loading a Rubric object from a valid JSON file."""
    # Setup: Create a file to load
    data_to_save = {
        "task_id": "task-202",
        "title": "Load Test",
        "description": "Loading a rubric from file.",
        "overall_max_score": 50.0,
        "min_passing_score": 25.0,
        "criteria": [
            {
                "id": "style",
                "name": "Code Style",
                "description": "Adherence to PEP8.",
                "weight": 0.2,
                "scale": "0-5",
            }
        ],
    }

    filename = "load_rubric_data"
    filepath = tmp_path / f"{filename}.json"

    with open(filepath, "w") as f:
        json.dump(data_to_save, f)

    # Action: Load the rubric
    loaded_rubric = Rubric.load_from_json(source_dir=tmp_path, filename=filename)

    # Assertion: Verify the loaded object's attributes
    assert isinstance(loaded_rubric, Rubric)
    assert loaded_rubric.task_id == "task-202"
    assert loaded_rubric.overall_max_score == 50.0
    assert loaded_rubric.min_passing_score == 25.0

    # Verify nested Criterion object
    assert len(loaded_rubric.criteria) == 1
    loaded_criterion = loaded_rubric.criteria[0]
    assert isinstance(loaded_criterion, Criterion)
    assert loaded_criterion.id == "style"
    assert loaded_criterion.scale == "0-5"


def test_rubric_load_from_json_file_not_found(tmp_path: Path):
    """Test loading raises FileNotFoundError when file is missing."""
    with pytest.raises(FileNotFoundError):
        # Action: Try to load a file that doesn't exist
        Rubric.load_from_json(source_dir=tmp_path, filename="non_existent_file")


def test_rubric_load_from_json_invalid_json(tmp_path: Path):
    """Test loading raises JSONDecodeError when file content is invalid JSON."""
    filename = "invalid_json_data"
    filepath = tmp_path / f"{filename}.json"

    # Setup: Write non-JSON content to the file
    with open(filepath, "w") as f:
        f.write("{'bad_key': 'unquoted_string'}")  # Invalid JSON syntax

    with pytest.raises(json.decoder.JSONDecodeError):
        # Action: Try to load the invalid file
        Rubric.load_from_json(source_dir=tmp_path, filename=filename)


def test_criterion_save_to_json(tmp_path: Path):
    """Test saving a Criterion object to a JSON file and verify content."""
    # 1. Setup
    criterion = Criterion(
        id="focus",
        name="Focus on Task",
        description="The extent to which the submission addresses the prompt.",
        weight=0.75,
        scale="percentage",
    )

    filename = "test_criterion_save"
    filepath = tmp_path / f"{filename}.json"

    # 2. Action: Save the criterion
    criterion.save_to_json(dest_dir=tmp_path, filename=filename)

    # 3. Assertion: Verify the file exists
    assert filepath.is_file()

    # 4. Assertion: Verify the content
    with open(filepath, "r") as f:
        data = json.load(f)

    assert data["id"] == "focus"
    assert data["weight"] == 0.75
    assert data["scale"] == "percentage"


def test_criterion_load_from_json(tmp_path: Path):
    """Test loading a Criterion object from a valid JSON file."""
    # 1. Setup: Create a file to load
    data_to_save = {
        "id": "correctness",
        "name": "Factual Correctness",
        "description": "Are all stated facts true?",
        "weight": 0.5,
        "scale": "0-1",
    }

    filename = "test_criterion_load"
    filepath = tmp_path / f"{filename}.json"

    with open(filepath, "w") as f:
        json.dump(data_to_save, f)

    # 2. Action: Load the criterion
    loaded_criterion = Criterion.load_from_json(source_dir=tmp_path, filename=filename)

    # 3. Assertion: Verify the loaded object's attributes
    assert isinstance(loaded_criterion, Criterion)
    assert loaded_criterion.id == "correctness"
    assert loaded_criterion.name == "Factual Correctness"
    assert loaded_criterion.weight == 0.5
    assert loaded_criterion.scale == "0-1"


def test_criterion_post_init_rejects_invalid_scale():
    """Criterion should raise ValueError if the scale is not a valid ScoreScale literal."""
    with pytest.raises(ValueError) as excinfo:
        Criterion(
            id="test-id",
            name="Test Name",
            description="Test Desc",
            weight=1.0,
            # Invalid scale
            scale="0-100",  # type: ignore[arg-type]
        )
    msg = str(excinfo.value)
    assert "Invalid scale: 0-100" in msg
    assert "Must be one of" in msg


@pytest.mark.parametrize("invalid_weight", [0.0, -0.1, -5.0])
def test_criterion_post_init_rejects_non_positive_weight(invalid_weight: float):
    """Criterion should raise ValueError if weight is non-positive."""
    with pytest.raises(ValueError) as excinfo:
        Criterion(
            id="test-id",
            name="Test Name",
            description="Test Desc",
            weight=invalid_weight,  # Non-positive weight
            scale="0-10",
        )
    msg = str(excinfo.value)
    assert f"Invalid weight: {invalid_weight}" in msg
    assert "Must be positive" in msg
