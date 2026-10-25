# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import uuid
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import get_mock_prompt_normalizer

from pyrit.exceptions import InvalidJsonException
from pyrit.executor.attack import AttackAdversarialConfig, AttackParameters, AttackScoringConfig, ConversationSession
from pyrit.executor.attack.component import ExperienceBank, StrategyDescription
from pyrit.executor.attack.multi_turn.emjo import EmjoAttack
from pyrit.executor.attack.multi_turn.role_driven_attack import RoleDrivenAttackContext
from pyrit.models import AttackOutcome, ComponentIdentifier, ConversationType, Message, MessagePiece, Score
from pyrit.prompt_target import CapabilityName, PromptTarget
from pyrit.score import TrueFalseScorer
from pyrit.score.true_false.float_scale_threshold_scorer import ORIGINAL_FLOAT_VALUE_KEY

pytestmark = pytest.mark.usefixtures("patch_central_database")

OBJECTIVE = "Test objective"

# Neutral placeholders: PyRIT ships no strategies, so the tests supply their own.
STRATEGY_A = StrategyDescription(name="strategy_a", definition="Placeholder definition A.")
STRATEGY_B = StrategyDescription(name="strategy_b", definition="Placeholder definition B.")


def _identifier(name: str) -> ComponentIdentifier:
    return ComponentIdentifier(class_name=name, class_module="test_module")


def _written(text: str) -> str:
    return json.dumps({"next_message": text, "rationale": "r"})


def _analysis(diagnosis: str = "too direct", *edits: str) -> str:
    return json.dumps({"diagnosis": diagnosis, "edits": list(edits)})


def _score(success: bool, graded: float, rationale: str = "judge says so") -> Score:
    return Score(
        score_type="true_false",
        score_value="true" if success else "false",
        score_category=["test"],
        score_value_description="d",
        score_rationale=rationale,
        score_metadata={ORIGINAL_FLOAT_VALUE_KEY: graded},
        message_piece_id=str(uuid.uuid4()),
        scorer_class_identifier=_identifier("MockScorer"),
    )


def _blocked_reply() -> Message:
    payload = json.dumps({"status_code": 400, "message": "content filter"})
    piece = MessagePiece(
        role="assistant", original_value=payload, original_value_data_type="error", response_error="blocked"
    )
    return Message(message_pieces=[piece])


def _chat_target() -> MagicMock:
    chat = MagicMock(spec=PromptTarget)
    chat.send_prompt_async = AsyncMock()
    chat.set_system_prompt_async = AsyncMock()
    chat.get_identifier.return_value = _identifier("MockChat")
    chat.configuration.capabilities.input_modalities = frozenset({frozenset({"text"})})
    chat.configuration.capabilities.output_modalities = frozenset({frozenset({"text"})})
    return chat


def _objective_target() -> MagicMock:
    target = MagicMock(spec=PromptTarget)
    target.send_prompt_async = AsyncMock()
    target.get_identifier.return_value = _identifier("MockTarget")
    target.configuration.capabilities.input_modalities = frozenset({frozenset({"text"})})
    target.configuration.capabilities.output_modalities = frozenset({frozenset({"text"})})
    return target


def _scorer() -> MagicMock:
    scorer = MagicMock(spec=TrueFalseScorer)
    scorer.score_async = AsyncMock()
    scorer.get_identifier.return_value = _identifier("MockScorer")
    return scorer


class _Harness:
    """Scripts what the agents and the objective target say, and records what they were sent."""

    def __init__(self, *, role_replies: list[str | Message], target_replies: list[str]) -> None:
        self.adversarial = _chat_target()
        self.objective_target = _objective_target()
        self.role_replies = deque(role_replies)
        self.target_replies = deque(target_replies)
        self.role_inputs: list[str] = []
        self.target_sends: list[tuple[str, str]] = []
        self.normalizer = get_mock_prompt_normalizer()
        self.normalizer.send_prompt_async = AsyncMock(side_effect=self._send)

    async def _send(self, **kwargs) -> Message:
        text = kwargs["message"].message_pieces[0].converted_value
        if kwargs["target"] is self.adversarial:
            self.role_inputs.append(text)
            reply = self.role_replies.popleft()
        else:
            self.target_sends.append((kwargs["conversation_id"], text))
            reply = self.target_replies.popleft()
        return reply if isinstance(reply, Message) else Message.from_prompt(prompt=reply, role="assistant")

    @property
    def system_prompts(self) -> list[str]:
        return [call.kwargs["system_prompt"] for call in self.adversarial.set_system_prompt_async.call_args_list]

    def attack(self, **overrides) -> EmjoAttack:
        kwargs: dict = {
            "objective_target": self.objective_target,
            "attack_adversarial_config": AttackAdversarialConfig(target=self.adversarial),
            "attack_scoring_config": AttackScoringConfig(objective_scorer=_scorer()),
            "prompt_normalizer": self.normalizer,
            "max_turns": 3,
        }
        kwargs.update(overrides)
        return EmjoAttack(**kwargs)


def _context(*, objective: str = OBJECTIVE, next_message: Message | None = None) -> RoleDrivenAttackContext:
    return RoleDrivenAttackContext(
        params=AttackParameters(objective=objective, next_message=next_message),
        session=ConversationSession(),
    )


async def _run(attack: EmjoAttack, context: RoleDrivenAttackContext, scores: list[Score]):
    with patch.object(attack, "_score_response_async", new_callable=AsyncMock, side_effect=scores):
        return await attack._perform_async(context=context)


# --- construction ------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"retrieval_k": -1}, "retrieval_k"),
        ({"high_score_threshold": 0}, "high_score_threshold"),
        ({"high_score_threshold": 1.5}, "high_score_threshold"),
        ({"strategies": [STRATEGY_A, STRATEGY_A]}, "unique"),
        ({"max_turns": 0}, "positive integer"),
        ({"attack_scoring_config": AttackScoringConfig()}, "Objective scorer must be provided"),
    ],
    ids=["negative-k", "zero-threshold", "high-threshold", "repeated-strategy", "zero-turns", "no-scorer"],
)
def test_init_rejects_invalid_settings(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _Harness(role_replies=[], target_replies=[]).attack(**overrides)


def test_init_requires_native_system_prompt_support_on_the_adversarial_target() -> None:
    harness = _Harness(role_replies=[], target_replies=[])
    harness.adversarial.configuration.includes.side_effect = lambda *, capability: (
        capability != CapabilityName.SYSTEM_PROMPT
    )

    with pytest.raises(ValueError, match=r"(?s)EmjoAttack .*supports_system_prompt"):
        harness.attack()


def test_init_strategies_are_optional() -> None:
    attack = _Harness(role_replies=[], target_replies=[]).attack()

    assert attack._render_strategies() == "(none supplied)"


def test_strategy_description_rejects_blank_values() -> None:
    with pytest.raises(ValueError):
        StrategyDescription(name=" ", definition="d")


def test_each_attack_gets_its_own_bank_unless_one_is_shared() -> None:
    harness = _Harness(role_replies=[], target_replies=[])
    shared = ExperienceBank()

    assert harness.attack().experience_bank is not harness.attack().experience_bank
    assert harness.attack(experience_bank=shared).experience_bank is shared


def test_identifier_reflects_the_settings_that_change_behavior() -> None:
    harness = _Harness(role_replies=[], target_replies=[])

    base = harness.attack().get_identifier()
    with_strategies = harness.attack(strategies=[STRATEGY_A]).get_identifier()
    other_k = harness.attack(retrieval_k=5).get_identifier()
    other_threshold = harness.attack(high_score_threshold=0.9).get_identifier()

    assert len({base.hash, with_strategies.hash, other_k.hash, other_threshold.hash}) == 4


# --- reply parsing -----------------------------------------------------------


def test_parse_analysis_reads_the_diagnosis_and_edits() -> None:
    analysis = EmjoAttack._parse_analysis(_analysis("why", "one", "two"))

    assert analysis.diagnosis == "why"
    assert analysis.edits == ("one", "two")


@pytest.mark.parametrize(
    "reply",
    [json.dumps({"diagnosis": "d", "edits": "x"}), '{"diagnosis": "d"}', "nope"],
    ids=["edits-not-a-list", "missing-key", "not-json"],
)
def test_parse_analysis_rejects_malformed_replies(reply: str) -> None:
    with pytest.raises(InvalidJsonException):
        EmjoAttack._parse_analysis(reply)


@pytest.mark.parametrize("reply", [_written(" "), '{"next_message": "x"}', "nope"], ids=["blank", "missing", "bad"])
def test_parse_message_rejects_malformed_replies(reply: str) -> None:
    with pytest.raises(InvalidJsonException):
        EmjoAttack._parse_message(reply)


# --- the loop ----------------------------------------------------------------


async def test_every_iteration_goes_to_the_target_on_a_fresh_conversation() -> None:
    harness = _Harness(
        role_replies=[_written("a"), _analysis(), _written("b"), _analysis(), _written("c")],
        target_replies=["r1", "r2", "r3"],
    )
    context = _context()

    result = await _run(harness.attack(), context, [_score(False, 0.1)] * 3)

    conversation_ids = [conversation_id for conversation_id, _ in harness.target_sends]
    assert len(set(conversation_ids)) == 3
    assert [text for _, text in harness.target_sends] == ["a", "b", "c"]
    assert result.outcome == AttackOutcome.FAILURE
    pruned = [ref for ref in context.related_conversations if ref.conversation_type == ConversationType.PRUNED]
    assert len(pruned) == 2


async def test_the_run_stops_at_the_first_success_and_remembers_it() -> None:
    harness = _Harness(role_replies=[_written("a"), _analysis(), _written("b")], target_replies=["r1", "r2"])
    attack = harness.attack()

    result = await _run(attack, _context(), [_score(False, 0.2), _score(True, 1.0)])

    assert result.outcome == AttackOutcome.SUCCESS
    assert result.executed_turns == 2
    found = await attack.experience_bank.similar_successes_async(objective=OBJECTIVE)
    assert [(experience.prompt, experience.score) for experience in found] == [("b", 1.0)]
    assert result.metadata["emjo_trace"][1]["remembered"] == "success"


async def test_a_later_run_is_shown_precedents_from_an_earlier_one() -> None:
    bank = ExperienceBank()
    first = _Harness(role_replies=[_written("the winning prompt")], target_replies=["r"])
    await _run(first.attack(experience_bank=bank, max_turns=1), _context(), [_score(True, 1.0)])
    second = _Harness(role_replies=[_written("new prompt")], target_replies=["r"])

    again = _context(objective=OBJECTIVE + " again")
    await _run(second.attack(experience_bank=bank, max_turns=1), again, [_score(True, 1.0)])

    attacker_input = second.role_inputs[0]
    assert "1. Task: Test objective\n   Prompt: the winning prompt" in attacker_input


async def test_the_first_iteration_has_no_precedents_no_best_attempt_and_no_feedback() -> None:
    harness = _Harness(role_replies=[_written("a")], target_replies=["r"])

    await _run(harness.attack(max_turns=1, strategies=[STRATEGY_A, STRATEGY_B]), _context(), [_score(True, 1.0)])

    attacker_input = harness.role_inputs[0]
    assert "- strategy_a: Placeholder definition A." in attacker_input
    assert "- strategy_b: Placeholder definition B." in attacker_input
    assert "Precedents (prompts that fulfilled similar tasks):\n(none)" in attacker_input
    assert "Best attempt so far:\n(none yet)" in attacker_input
    assert "(none yet; this is the first attempt)" in attacker_input


async def test_the_analyzer_reads_the_failed_attempt_and_its_diagnosis_reaches_the_next_prompt() -> None:
    harness = _Harness(
        role_replies=[_written("first try"), _analysis("too direct", "soften it", "add context"), _written("second")],
        target_replies=["a refusal", "ok"],
    )
    context = _context()

    await _run(harness.attack(max_turns=2), context, [_score(False, 0.3, "it refused"), _score(True, 1.0)])

    analyzer_input = harness.role_inputs[1]
    assert "The attempt's prompt:\nfirst try" in analyzer_input
    assert "The Target's response:\na refusal" in analyzer_input
    assert "Score: 0.30 on a scale from 0 to 1." in analyzer_input
    assert "Judge's explanation: it refused" in analyzer_input
    assert "Analyzer" in harness.system_prompts[1]
    next_attacker_input = harness.role_inputs[2]
    assert "too direct\nEdits to make:\n- soften it\n- add context" in next_attacker_input
    assert context.trace[0].details["diagnosis"] == "too direct"
    assert context.trace[0].details["edits"] == ["soften it", "add context"]


async def test_the_analyzer_is_not_asked_after_a_success_or_on_the_last_iteration() -> None:
    harness = _Harness(role_replies=[_written("a")], target_replies=["r"])

    await _run(harness.attack(max_turns=1), _context(), [_score(False, 0.2)])

    assert len(harness.role_inputs) == 1


async def test_the_best_scoring_attempt_so_far_is_shown_as_an_anchor() -> None:
    harness = _Harness(
        role_replies=[_written("good try"), _analysis(), _written("worse try"), _analysis(), _written("third")],
        target_replies=["r1", "r2", "r3"],
    )
    context = _context()

    await _run(harness.attack(), context, [_score(False, 0.8), _score(False, 0.75), _score(False, 0.1)])

    assert "Best attempt so far:\nScore 0.80: good try" in harness.role_inputs[-1]
    assert context.trace[0].details["remembered"] == "best_attempt"
    assert "remembered" not in context.trace[1].details


async def test_a_score_below_the_threshold_is_not_kept_as_the_best_attempt() -> None:
    harness = _Harness(role_replies=[_written("a"), _analysis(), _written("b")], target_replies=["r1", "r2"])

    await _run(harness.attack(max_turns=2), _context(), [_score(False, 0.5), _score(False, 0.4)])

    assert "Best attempt so far:\n(none yet)" in harness.role_inputs[2]


async def test_a_blocked_analyzer_means_the_next_iteration_goes_without_a_diagnosis() -> None:
    harness = _Harness(
        role_replies=[_written("a"), _blocked_reply(), _written("b")],
        target_replies=["r1", "r2"],
    )
    context = _context()

    result = await _run(harness.attack(max_turns=2), context, [_score(False, 0.2), _score(False, 0.2)])

    assert result.executed_turns == 2
    assert "diagnosis" not in context.trace[0].details
    assert "(none yet; this is the first attempt)" in harness.role_inputs[2]


async def test_a_caller_supplied_message_seeds_the_first_iteration_without_the_attacker() -> None:
    harness = _Harness(role_replies=[_analysis(), _written("second")], target_replies=["r1", "r2"])
    seed = Message.from_prompt(prompt="my own opener", role="user")
    context = _context(next_message=seed)

    await _run(harness.attack(max_turns=2), context, [_score(False, 0.2), _score(True, 1.0)])

    assert harness.target_sends[0][1] == "my own opener"
    assert context.trace[0].prompt_source == "seed"
    assert context.trace[1].prompt_source == "attacker"


async def test_the_trace_records_the_graded_score_of_every_iteration() -> None:
    harness = _Harness(role_replies=[_written("a"), _analysis(), _written("b")], target_replies=["r1", "r2"])

    result = await _run(harness.attack(max_turns=2), _context(), [_score(False, 0.4), _score(True, 1.0)])

    trace = result.metadata["emjo_trace"]
    assert [entry["graded_score"] for entry in trace] == [0.4, 1.0]
    assert [entry["score_value"] for entry in trace] == [False, True]


async def test_a_blocked_attacker_before_any_iteration_is_a_preparation_failure() -> None:
    harness = _Harness(role_replies=[_blocked_reply()], target_replies=[])

    result = await _run(harness.attack(), _context(), [])

    assert result.outcome == AttackOutcome.UNDETERMINED
    assert result.executed_turns == 0
    assert harness.target_sends == []
