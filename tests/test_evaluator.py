# tests/test_evaluator.py

import pytest

from task_grader.grading.evaluator import (
    LLMTaskEvaluator,
    CriterionEvaluation,
)
from task_grader.grading.rubric import (
    Rubric,
    Criterion,
)


class FakeMessage:
    """Simple stand-in for a LangChain BaseMessage, holding only .content."""

    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """
    Minimal fake LLM that returns a preconfigured YAML response,
    ignoring the prompt content.
    """

    def __init__(self, response_text: str):
        self._response_text = response_text
        self.last_prompt: str | None = None

    def invoke(self, prompt: str) -> FakeMessage:
        # Capture prompt for potential assertions
        self.last_prompt = prompt
        return FakeMessage(self._response_text)


def make_sample_rubric() -> Rubric:
    """Helper to construct a small rubric for testing."""
    return Rubric(
        task_id="task-1",
        title="Sample Rubric",
        description="Evaluate how well the trainee designs a prompt template.",
        overall_max_score=100,
        min_passing_score=60,
        criteria=[
            Criterion(
                id="clarity",
                name="Clarity of intent and scope",
                description="How clearly the evaluation intent and scope are stated.",
                weight=0.5,
                scale="0-10",
            ),
            Criterion(
                id="structure",
                name="Prompt structure",
                description="How well-structured and modular the prompt is.",
                weight=0.5,
                scale="0-10",
            ),
        ],
    )


def make_base_template() -> str:
    """
    Minimal base template containing all placeholders used in evaluate()
    and by PromptBuilder.from_rubric.
    """
    return """
##Context##
You're an expert in {knowledge_area}.

Rubric:
{rubric}

Assignment:
{assignment}

Submission:
{submission}

Trainee: {trainee_name}
Cohort: {cohort_specifics}
Track: {track_name}

{other_enumerated_notes}
""".strip()


def test_evaluate_happy_path_case_insensitive_ids():
    rubric = make_sample_rubric()

    # YAML uses "Clarity" (capital C) while rubric id is "clarity"
    yaml_response = """```yaml
intro: "Short intro."
overall_evaluation: "Sentence one. Sentence two."
overall_verdict: "good"
criteria_specific_evaluations:
  - id: "Clarity"
    name: "Clarity of intent and scope"
    score_scale: "0-10"
    score: 8
    justification: "Generally clear."
  - id: "structure"
    name: "Prompt structure"
    score_scale: "0-10"
    score: 7
    justification: "Reasonably structured."
```"""

    fake_llm = FakeLLM(response_text=yaml_response)
    base_template = make_base_template()

    evaluator = LLMTaskEvaluator(llm=fake_llm, base_prompt_template=base_template)  # type: ignore[arg-type]

    result = evaluator.evaluate(
        rubric=rubric,
        assignment="Assignment text here",
        submission="Submission text here",
        trainee_name="Firstname Lastname",
        knowledge_area="prompt engineering",
        cohort_specifics="Agentic AI Track, Nov 2025",
        track_name="Agentic AI",
        other_notes="",
    )

    # Top-level fields
    assert result.intro == "Short intro."
    assert result.overall_evaluation == "Sentence one. Sentence two."
    assert result.overall_verdict == "good"

    # Criteria evaluations
    assert len(result.criteria) == 2
    ids = {c.id for c in result.criteria}
    # We expect canonical rubric ids to be used ("clarity", "structure")
    assert ids == {"clarity", "structure"}

    clarity_eval = next(c for c in result.criteria if c.id == "clarity")
    assert clarity_eval.score == 8
    assert clarity_eval.score_scale == "0-10"
    assert "clear" in clarity_eval.justification.lower()

    structure_eval = next(c for c in result.criteria if c.id == "structure")
    assert structure_eval.score == 7
    assert structure_eval.score_scale == "0-10"

    # Total score: weights 0.5 each, scores 8/10 and 7/10
    # normalized = (0.5 * 0.8 + 0.5 * 0.7) = 0.75
    # scaled to overall_max_score=100 -> 75.0
    assert result.total_score == pytest.approx(75.0)


def test_parse_yaml_missing_required_keys_raises():
    # Missing overall_verdict and criteria_specific_evaluations
    bad_yaml = """
intro: "Hi"
overall_evaluation: "Only two fields here."
"""
    with pytest.raises(ValueError) as excinfo:
        LLMTaskEvaluator._parse_yaml(bad_yaml)

    msg = str(excinfo.value)
    assert "Missing required keys" in msg
    assert "overall_verdict" in msg
    assert "criteria_specific_evaluations" in msg


def test_parse_yaml_non_mapping_top_level_raises():
    bad_yaml = """
- intro: "This should be a mapping, not a list."
"""
    with pytest.raises(ValueError) as excinfo:
        LLMTaskEvaluator._parse_yaml(bad_yaml)

    msg = str(excinfo.value)
    assert "Expected top-level YAML mapping" in msg


# ----------------------------------------------------------------------
# Tests for _build_criterion_evaluations
# ----------------------------------------------------------------------


def test_build_criterion_evaluations_unknown_id_raises():
    rubric = make_sample_rubric()
    data = {
        "intro": "x",
        "overall_evaluation": "y",
        "overall_verdict": "good",
        "criteria_specific_evaluations": [
            {
                "id": "nonexistent",
                "name": "Clarity of intent and scope",
                "score_scale": "0-10",
                "score": 8,
                "justification": "Text",
            }
        ],
    }

    with pytest.raises(ValueError) as excinfo:
        LLMTaskEvaluator._build_criterion_evaluations(data, rubric)

    msg = str(excinfo.value)
    assert "not found in rubric" in msg


def test_build_criterion_evaluations_score_out_of_range_raises():
    rubric = make_sample_rubric()
    data = {
        "intro": "x",
        "overall_evaluation": "y",
        "overall_verdict": "good",
        "criteria_specific_evaluations": [
            {
                "id": "clarity",
                "name": "Clarity of intent and scope",
                "score_scale": "0-10",
                "score": 999,  # clearly out of [0, 10]
                "justification": "Way too big.",
            }
        ],
    }

    with pytest.raises(ValueError) as excinfo:
        LLMTaskEvaluator._build_criterion_evaluations(data, rubric)

    msg = str(excinfo.value)
    assert "out of range" in msg
    assert "0-10" in msg


def test_build_criterion_evaluations_case_insensitive_id_mapping():
    """
    Ensure that criterion ids from YAML are matched case-insensitively
    against rubric ids, but the canonical rubric id is preserved in the result.
    """
    rubric = make_sample_rubric()
    data = {
        "intro": "x",
        "overall_evaluation": "y",
        "overall_verdict": "good",
        "criteria_specific_evaluations": [
            {
                "id": "ClArItY",  # weird casing
                "name": "Clarity of intent and scope",
                "score_scale": "0-10",
                "score": 9,
                "justification": "Looks good.",
            },
            {
                "id": "Structure",  # initial capital
                "name": "Prompt structure",
                "score_scale": "0-10",
                "score": 8,
                "justification": "Solid structure.",
            },
        ],
    }

    evals = LLMTaskEvaluator._build_criterion_evaluations(data, rubric)

    assert len(evals) == 2
    ids = {e.id for e in evals}
    # We expect canonical ids "clarity" and "structure", as defined in rubric
    assert ids == {"clarity", "structure"}


# ----------------------------------------------------------------------
# Tests for _compute_total_score
# ----------------------------------------------------------------------


def test_compute_total_score_respects_weights_and_scales():
    rubric = Rubric(
        task_id="scales-1",
        title="Mixed Scales Rubric",
        description="",
        overall_max_score=100,
        min_passing_score=50,
        criteria=[
            Criterion(
                id="crit_a",
                name="Criterion A",
                description="",
                weight=1.0,
                scale="0-10",
            ),
            Criterion(
                id="crit_b",
                name="Criterion B",
                description="",
                weight=1.0,
                scale="0-5",
            ),
        ],
    )

    # A: 8/10, B: 4/5 -> both 0.8 normalized
    evals = [
        CriterionEvaluation(
            id="crit_a",
            name="Criterion A",
            score_scale="0-10",
            score=8,
            justification="",
        ),
        CriterionEvaluation(
            id="crit_b",
            name="Criterion B",
            score_scale="0-5",
            score=4,
            justification="",
        ),
    ]

    total = LLMTaskEvaluator._compute_total_score(evals, rubric)

    # Both criteria have equal weight, both are at 0.8 of their max.
    # Normalized total = 0.8, scaled to overall_max_score=100 => 80
    assert total == pytest.approx(80.0)
