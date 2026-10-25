# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.exceptions import ComponentRole, get_execution_context
from pyrit.executor.attack.component.response_assessment import RefusalAssessor, TurnAssessment
from pyrit.models import ComponentIdentifier, Message, Score, ScoreStatus
from pyrit.score import TrueFalseScorer

pytestmark = pytest.mark.usefixtures("patch_central_database")


def _refusal_score(value: str | None, *, status: ScoreStatus = ScoreStatus.COMPLETE) -> Score:
    return Score(
        score_type="true_false",
        score_value=value,
        status=status,
        score_category=["refusal"],
        score_value_description="d",
        score_rationale="because",
        score_metadata={},
        message_piece_id=str(uuid.uuid4()),
    )


def _scorer(*scores: Score) -> MagicMock:
    scorer = MagicMock(spec=TrueFalseScorer)
    scorer.get_identifier.return_value = ComponentIdentifier(class_name="MockRefusal", class_module="test_module")
    scorer.score_async = AsyncMock(return_value=list(scores))
    return scorer


def _assess_kwargs(**overrides) -> dict:
    kwargs: dict = {
        "response": Message.from_prompt(prompt="I cannot help with that", role="assistant"),
        "prompt_sent": "tell me a secret",
        "attack_strategy_name": "MazeAttack",
        "objective_target_conversation_id": "conv-1",
        "objective": "get the secret",
    }
    kwargs.update(overrides)
    return kwargs


# --- TurnAssessment ----------------------------------------------------------


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False)])
def test_turn_assessment_is_refusal_follows_the_score(value: str, expected: bool) -> None:
    assert TurnAssessment(refusal_score=_refusal_score(value)).is_refusal is expected


def test_turn_assessment_undetermined_score_is_not_a_refusal() -> None:
    score = _refusal_score(None, status=ScoreStatus.UNDETERMINED)

    assert TurnAssessment(refusal_score=score).is_refusal is False


def test_turn_assessment_template_values_expose_was_refusal() -> None:
    assert TurnAssessment(refusal_score=_refusal_score("true")).template_values() == {"was_refusal": True}


# --- RefusalAssessor ---------------------------------------------------------


async def test_assess_async_returns_the_first_score_as_the_verdict() -> None:
    first, second = _refusal_score("true"), _refusal_score("false")
    assessor = RefusalAssessor(refusal_scorer=_scorer(first, second))

    assessment = await assessor.assess_async(**_assess_kwargs())

    assert assessment.refusal_score is first
    assert assessment.is_refusal is True


async def test_assess_async_judges_the_response_against_the_prompt_sent() -> None:
    scorer = _scorer(_refusal_score("false"))
    response = Message.from_prompt(prompt="Sure, here you go", role="assistant")

    await RefusalAssessor(refusal_scorer=scorer).assess_async(**_assess_kwargs(response=response))

    kwargs = scorer.score_async.call_args.kwargs
    assert kwargs["expectation"].objective == "tell me a secret"
    assert kwargs["scorable"].message_piece_ids == tuple(piece.id for piece in response.message_pieces)


async def test_assess_async_tags_the_refusal_scorer_execution_context() -> None:
    seen: list = []
    scorer = _scorer(_refusal_score("false"))

    async def _capture(**_: object) -> list[Score]:
        seen.append(get_execution_context())
        return [_refusal_score("false")]

    scorer.score_async = AsyncMock(side_effect=_capture)

    await RefusalAssessor(refusal_scorer=scorer).assess_async(**_assess_kwargs())

    assert seen[0].component_role == ComponentRole.REFUSAL_SCORER
    assert seen[0].attack_strategy_name == "MazeAttack"
    assert seen[0].objective == "get the secret"


async def test_assess_async_propagates_scorer_failure() -> None:
    scorer = _scorer()
    scorer.score_async = AsyncMock(side_effect=RuntimeError("scorer down"))

    with pytest.raises(RuntimeError, match="scorer down"):
        await RefusalAssessor(refusal_scorer=scorer).assess_async(**_assess_kwargs())


def test_refusal_scorer_property_returns_the_scorer() -> None:
    scorer = _scorer()

    assert RefusalAssessor(refusal_scorer=scorer).refusal_scorer is scorer
