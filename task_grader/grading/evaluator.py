from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel

from .rubric import (
    SCORE_SCALE_DESCRIPTIONS,
    SCORE_SCALE_NUMERIC_RANGES,
    Rubric,
    Criterion,
    ScoreScale,
)
from .prompt_builder import PromptBuilder


@dataclass
class CriterionEvaluation:
    """Structured representation of the LLM's evaluation for a single criterion."""

    id: str
    name: str
    score_scale: ScoreScale
    score: int
    justification: str


@dataclass
class EvaluationResult:
    """
    Final structured result for a trainee's submission.

    - intro: short one-sentence summary
    - overall_evaluation: paragraph-level summary (3–5 sentences)
    - overall_verdict: qualitative verdict, e.g. "good" / "fail"
    - criteria: list of per-criterion evaluations
    - total_score: numeric score computed from rubric weights and score scales
    - raw_yaml: the raw YAML block returned by the LLM (for debugging/audit)
    """

    intro: str
    overall_evaluation: str
    overall_verdict: str
    criteria: list[CriterionEvaluation]
    total_score: float
    raw_yaml: str


class LLMTaskEvaluator:
    """
    Evaluate a trainee submission against a rubric using an LLM via LangChain.

    Typical usage:

        evaluator = LLMTaskEvaluator.from_ollama(
            model_name="llama3.2:3b",
            prompt_template_path="task_grader/grading/grading_prompt_template.txt"
        )

        result = evaluator.evaluate(
            rubric=my_rubric,
            assignment=assignment_text,
            submission=submission_text,
            trainee_name="Firstname Lastname",
            knowledge_area="prompt engineering",
            cohort_specifics="Agentic AI Track, Nov 2025",
            track_name="Agentic AI",
            other_notes="",  # optional extra constraints for the prompt
        )
    """

    def __init__(self, llm: BaseChatModel, base_prompt_template: str) -> None:
        self._llm = llm
        self._base_prompt_template = base_prompt_template

    # ---------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------

    @classmethod
    def from_ollama(
        cls,
        model_name: str,
        prompt_template_path: str | Path,
        temperature: float = 0.0,
        **ollama_kwargs: Any,
    ) -> "LLMTaskEvaluator":
        """
        Convenience constructor that builds a ChatOllama LLM and loads the base
        grading prompt template from disk.
        """
        llm = ChatOllama(model=model_name, temperature=temperature, **ollama_kwargs)

        template_path = Path(prompt_template_path)
        base_template = template_path.read_text(encoding="utf-8")

        return cls(llm=llm, base_prompt_template=base_template)

    @classmethod
    def from_groq(
        cls,
        model_name: str,
        prompt_template_path: str | Path,
        api_key: str = None,
        temperature: float = 0.0,
        **groq_kwargs: Any,
    ) -> "LLMTaskEvaluator":
        """
        Convenience constructor that builds a Groq LLM and loads the base
        grading prompt template from disk.
        """
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            **groq_kwargs,
        )

        template_path = Path(prompt_template_path)
        base_template = template_path.read_text(encoding="utf-8")

        return cls(llm=llm, base_prompt_template=base_template)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def evaluate(
        self,
        rubric: Rubric,
        assignment: str,
        submission: str,
        trainee_name: str,
        knowledge_area: str,
        cohort_specifics: str,
        track_name: str,
        other_notes: str = "",
    ) -> EvaluationResult:
        """
        Evaluate a trainee submission against the given rubric using the LLM.

        This method:
        - Builds a grading prompt from the base template and rubric
        - Calls the LLM to generate a YAML grading response
        - Parses and validates the YAML against the rubric and score scales
        - Computes an overall numeric score based on weights and scales
        """
        # Build the prompt using PromptBuilder.from_rubric
        builder = PromptBuilder.from_rubric(
            base_template=self._base_prompt_template,
            rubric=rubric,
            score_scale_literal=ScoreScale,
            scale_descriptions=SCORE_SCALE_DESCRIPTIONS,
            additional_notes=other_notes,
        )

        # Validate that no placeholders were missed
        missing = builder.validate_placeholders(
            {
                "knowledge_area",
                "cohort_specifics",
                "track_name",
                "assignment",
                "trainee_name",
                "submission",
                "other_enumerated_notes",
            }
        )

        if missing:
            raise ValueError(f"Unresolved placeholders in grading prompt: {missing}")

        prompt = builder.build(
            knowledge_area=knowledge_area,
            cohort_specifics=cohort_specifics,
            track_name=track_name,
            assignment=assignment,
            trainee_name=trainee_name,
            submission=submission,
            other_enumerated_notes="",  # can be used to append more notes via .format
        )

        # Call the LLM
        llm_output = self._llm.invoke(prompt)
        # For ChatOllama / LC chat models, invoke() returns a BaseMessage with .content
        raw_text = getattr(llm_output, "content", str(llm_output))

        # Extract YAML block and parse
        yaml_text = self._extract_yaml_block(raw_text)
        data = self._parse_yaml(yaml_text)

        # Validate structure and map onto rubric
        criterion_evals = self._build_criterion_evaluations(data, rubric)

        # Compute overall numeric score using rubric weights and score scales
        total_score = self._compute_total_score(criterion_evals, rubric)

        return EvaluationResult(
            intro=data["intro"],
            overall_evaluation=data["overall_evaluation"],
            overall_verdict=data["overall_verdict"],
            criteria=criterion_evals,
            total_score=total_score,
            raw_yaml=yaml_text,
        )

    # ---------------------------------------------------------------------
    # Helpers: YAML extraction / parsing / validation
    # ---------------------------------------------------------------------

    @staticmethod
    def _extract_yaml_block(text: str) -> str:
        """
        Extract the YAML block from the LLM output.

        If a ```yaml ... ``` fenced block exists, it is used.
        Otherwise, the entire output is treated as YAML.
        """
        fence_yaml = "```yaml"
        fence_plain = "```"

        if fence_yaml in text:
            start = text.index(fence_yaml) + len(fence_yaml)
            end = text.find(fence_plain, start)
            if end == -1:
                end = len(text)
            return text[start:end].strip()

        if fence_plain in text:
            start = text.index(fence_plain) + len(fence_plain)
            end = text.find(fence_plain, start)
            if end == -1:
                end = len(text)
            return text[start:end].strip()

        return text.strip()

    @staticmethod
    def _parse_yaml(yaml_text: str) -> dict[str, Any]:
        """Parse the YAML grading response into a Python dict, with basic shape validation."""
        try:
            data = yaml.safe_load(yaml_text)
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse YAML from LLM output: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"Expected top-level YAML mapping, got: {type(data)!r}")

        required_keys = {
            "intro",
            "overall_evaluation",
            "overall_verdict",
            "criteria_specific_evaluations",
        }
        missing = required_keys - data.keys()
        if missing:
            raise ValueError(f"Missing required keys in grading YAML: {missing}")

        if not isinstance(data["criteria_specific_evaluations"], list):
            raise ValueError(
                "criteria_specific_evaluations must be a list of criterion entries"
            )

        return data

    @staticmethod
    def _build_criterion_evaluations(
        data: dict[str, Any],
        rubric: Rubric,
    ) -> list[CriterionEvaluation]:
        """Validate and convert YAML criterion entries into CriterionEvaluation objects."""
        # Use lowercase keys for case-insensitive lookup, but keep canonical ids in the values
        rubric_by_lower_id: dict[str, Criterion] = {
            c.id.lower(): c for c in rubric.criteria
        }

        seen_ids: set[str] = set()
        evaluations: list[CriterionEvaluation] = []

        for item in data["criteria_specific_evaluations"]:
            if not isinstance(item, dict):
                raise ValueError(
                    "Each entry in criteria_specific_evaluations must be a mapping"
                )

            try:
                raw_id = item["id"]
                cid_lower = raw_id.lower()
                name = item["name"]
                score_scale = item["score_scale"]
                score = item["score"]
                justification = item["justification"]
            except KeyError as exc:
                raise ValueError(
                    f"Missing key in criterion evaluation entry: {exc}"
                ) from exc

            if cid_lower not in rubric_by_lower_id:
                raise ValueError(f"Criterion id {raw_id!r} not found in rubric")

            rubric_criterion = rubric_by_lower_id[cid_lower]
            canonical_id = rubric_criterion.id  # preserve the rubric's original casing

            # Ensure name and scale match the rubric
            if name != rubric_criterion.name:
                raise ValueError(
                    f"Criterion name mismatch for id {canonical_id!r}: "
                    f"rubric has {rubric_criterion.name!r}, YAML has {name!r}"
                )

            if score_scale != rubric_criterion.scale:
                raise ValueError(
                    f"Criterion scale mismatch for id {canonical_id!r}: "
                    f"rubric has {rubric_criterion.scale!r}, YAML has {score_scale!r}"
                )

            if score_scale not in SCORE_SCALE_NUMERIC_RANGES:
                raise ValueError(
                    f"Unsupported score_scale {score_scale!r} for id {canonical_id!r}"
                )

            min_score, max_score = SCORE_SCALE_NUMERIC_RANGES[score_scale]  # type: ignore[index]

            if not isinstance(score, int):
                raise ValueError(
                    f"Score for criterion {canonical_id!r} must be an integer, got {type(score)!r}"
                )

            if not (min_score <= score <= max_score):
                raise ValueError(
                    f"Score {score} for criterion {canonical_id!r} out of range "
                    f"for scale {score_scale!r} (expected {min_score}–{max_score})"
                )

            if not isinstance(justification, str):
                raise ValueError(
                    f"Justification for criterion {canonical_id!r} must be a string"
                )

            if canonical_id in seen_ids:
                raise ValueError(
                    f"Duplicate criterion id {canonical_id!r} in criteria_specific_evaluations"
                )

            seen_ids.add(canonical_id)

            evaluations.append(
                CriterionEvaluation(
                    id=canonical_id,  # <- store canonical id, not lowercased id
                    name=name,
                    score_scale=score_scale,  # type: ignore[arg-type]
                    score=score,
                    justification=justification,
                )
            )

        rubric_ids = {c.id for c in rubric.criteria}
        missing_ids = rubric_ids - seen_ids

        if missing_ids:
            raise ValueError(f"Missing evaluations for rubric criteria: {missing_ids}")

        extra_ids = seen_ids - rubric_ids

        if extra_ids:
            raise ValueError(
                f"Evaluations found for unknown rubric criteria: {extra_ids}"
            )

        return evaluations

    # ---------------------------------------------------------------------
    # Scoring
    # ---------------------------------------------------------------------

    @staticmethod
    def _compute_total_score(
        criterion_evals: list[CriterionEvaluation],
        rubric: Rubric,
    ) -> float:
        """
        Compute an overall numeric score using rubric weights and score scales.

        Assumptions:
        - Each Criterion.weight is a positive float.
        - We treat weights as relative; they do not have to sum exactly to 1.0.
        - For each criterion:
            normalized_score = score / max_score_for_scale
        - Weighted sum of normalized scores is then scaled to rubric.overall_max_score.
        """
        rubric_by_id: dict[str, Criterion] = {c.id: c for c in rubric.criteria}

        total_weight = sum(c.weight for c in rubric.criteria)
        if total_weight <= 0:
            raise ValueError("Sum of rubric criterion weights must be positive")

        weighted_sum = 0.0
        for ev in criterion_evals:
            rub_crit = rubric_by_id[ev.id]
            _, max_score = SCORE_SCALE_NUMERIC_RANGES[ev.score_scale]
            normalized = ev.score / max_score
            weighted_sum += normalized * rub_crit.weight

        normalized_total = weighted_sum / total_weight
        return normalized_total * rubric.overall_max_score
