# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import uuid
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import get_mock_prompt_normalizer

from pyrit.executor.attack import AttackAdversarialConfig, AttackParameters, AttackScoringConfig, ConversationSession
from pyrit.executor.attack.multi_turn.ramp import RampAttack
from pyrit.executor.attack.multi_turn.ramp_domain import (
    EmpiricalTransitioner,
    RampDomain,
    RampStrategy,
    SafetyPattern,
)
from pyrit.executor.attack.multi_turn.role_driven_attack import RoleDrivenAttackContext
from pyrit.models import AttackOutcome, ComponentIdentifier, Message, MessagePiece, Score, ScoreStatus
from pyrit.prompt_target import CapabilityName, PromptTarget
from pyrit.score import TrueFalseScorer

pytestmark = pytest.mark.usefixtures("patch_central_database")

OBJECTIVE = "Test objective"

# Neutral placeholders: PyRIT ships no strategies, so the tests supply their own.
CALM = RampStrategy(name="calm", definition="Placeholder definition for the calm strategy.", risk=1)
LOUD = RampStrategy(name="loud", definition="Placeholder definition for the loud strategy.", risk=2)


def _identifier(name: str) -> ComponentIdentifier:
    return ComponentIdentifier(class_name=name, class_module="test_module")


def _judgment(pattern: str = "borderline", clues: tuple[str, ...] = (), rationale: str = "because") -> str:
    return json.dumps({"safety_pattern": pattern, "clues": list(clues), "rationale": rationale})


def _written(text: str) -> str:
    return json.dumps({"next_message": text, "rationale": "r"})


def _score(value: str, *, status: ScoreStatus = ScoreStatus.COMPLETE) -> Score:
    return Score(
        score_type="true_false",
        score_value=None if status is ScoreStatus.UNDETERMINED else value,
        status=status,
        score_category=["test"],
        score_value_description="d",
        score_rationale="rationale",
        score_metadata={},
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
        self.target_sends: list[str] = []
        self.normalizer = get_mock_prompt_normalizer()
        self.normalizer.send_prompt_async = AsyncMock(side_effect=self._send)

    async def _send(self, **kwargs) -> Message:
        text = kwargs["message"].message_pieces[0].converted_value
        if kwargs["target"] is self.adversarial:
            self.role_inputs.append(text)
            reply = self.role_replies.popleft()
        else:
            self.target_sends.append(text)
            reply = self.target_replies.popleft()
        return reply if isinstance(reply, Message) else Message.from_prompt(prompt=reply, role="assistant")

    @property
    def system_prompts(self) -> list[str]:
        return [call.kwargs["system_prompt"] for call in self.adversarial.set_system_prompt_async.call_args_list]

    def attack(self, **overrides) -> RampAttack:
        kwargs: dict = {
            "objective_target": self.objective_target,
            "attack_adversarial_config": AttackAdversarialConfig(target=self.adversarial),
            "attack_scoring_config": AttackScoringConfig(objective_scorer=_scorer()),
            "prompt_normalizer": self.normalizer,
            "strategies": [CALM, LOUD],
            "max_turns": 3,
        }
        kwargs.update(overrides)
        return RampAttack(**kwargs)


def _context(*, next_message: Message | None = None) -> RoleDrivenAttackContext:
    return RoleDrivenAttackContext(
        params=AttackParameters(objective=OBJECTIVE, next_message=next_message),
        session=ConversationSession(),
    )


async def _run(attack: RampAttack, context: RoleDrivenAttackContext, scores: list[Score]):
    with patch.object(attack, "_score_response_async", new_callable=AsyncMock, side_effect=scores):
        return await attack._perform_async(context=context)


# --- construction ------------------------------------------------------------


def test_init_requires_at_least_one_strategy() -> None:
    with pytest.raises(ValueError, match="supply them"):
        _Harness(role_replies=[], target_replies=[]).attack(strategies=[])


def test_init_rejects_repeated_strategy_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        _Harness(role_replies=[], target_replies=[]).attack(strategies=[CALM, CALM])


@pytest.mark.parametrize(
    ("setting", "value"),
    [("planning_horizon", 0), ("risk_budget", 0), ("max_turns", 0)],
    ids=["horizon", "risk", "turns"],
)
def test_init_rejects_non_positive_settings(setting: str, value: float) -> None:
    with pytest.raises(ValueError):
        _Harness(role_replies=[], target_replies=[]).attack(**{setting: value})


def test_init_requires_an_objective_scorer() -> None:
    with pytest.raises(ValueError, match="Objective scorer must be provided"):
        _Harness(role_replies=[], target_replies=[]).attack(attack_scoring_config=AttackScoringConfig())


def test_init_requires_native_system_prompt_support_on_the_adversarial_target() -> None:
    harness = _Harness(role_replies=[], target_replies=[])
    harness.adversarial.configuration.includes.side_effect = lambda *, capability: (
        capability != CapabilityName.SYSTEM_PROMPT
    )

    with pytest.raises(ValueError, match=r"(?s)RampAttack .*supports_system_prompt"):
        harness.attack()


def test_init_loads_the_bundled_prompts() -> None:
    attack = _Harness(role_replies=[], target_replies=[]).attack()

    assert "Judge" in attack._judge_prompt.value
    assert "write the next user message" in attack._attacker_prompt.value


def test_identifier_reflects_the_strategies_and_planning_settings() -> None:
    harness = _Harness(role_replies=[], target_replies=[])

    base = harness.attack().get_identifier()
    other_strategies = harness.attack(strategies=[CALM]).get_identifier()
    other_horizon = harness.attack(planning_horizon=2).get_identifier()
    empirical = harness.attack(transitioner=EmpiricalTransitioner(records=[])).get_identifier()

    assert len({base.hash, other_strategies.hash, other_horizon.hash, empirical.hash}) == 4


# --- judge reply parsing -----------------------------------------------------


def test_parse_judgment_reads_every_field() -> None:
    judgment = RampAttack._parse_judgment(_judgment("LEAKY", ("a", "b"), "why"))

    assert judgment.pattern is SafetyPattern.LEAKY
    assert judgment.clues == ("a", "b")
    assert judgment.rationale == "why"


@pytest.mark.parametrize(
    "reply",
    [
        _judgment("explosive"),
        json.dumps({"safety_pattern": "leaky", "clues": "x", "rationale": "r"}),
        '{"safety_pattern": "leaky"}',
        "not json",
    ],
    ids=["unknown-pattern", "clues-not-a-list", "missing-keys", "not-json"],
)
def test_parse_judgment_rejects_malformed_replies(reply: str) -> None:
    from pyrit.exceptions import InvalidJsonException

    with pytest.raises(InvalidJsonException):
        RampAttack._parse_judgment(reply)


# --- the loop ----------------------------------------------------------------


async def test_loop_stops_at_the_first_success_and_traces_every_turn() -> None:
    harness = _Harness(
        role_replies=[
            _written("opening"),
            _judgment("leaky", ("first clue",)),
            _written("follow-up"),
            _judgment("leaky", ("second clue",)),
        ],
        target_replies=["partial answer", "full answer"],
    )
    context = _context()

    result = await _run(harness.attack(), context, [_score("false"), _score("true")])

    assert result.outcome == AttackOutcome.SUCCESS
    assert result.executed_turns == 2
    assert harness.target_sends == ["opening", "follow-up"]
    trace = result.metadata["ramp_trace"]
    assert [entry["turn"] for entry in trace] == [1, 2]
    assert trace[0]["strategy"] == "calm"
    assert trace[0]["judged_pattern"] == "leaky"
    assert trace[0]["new_clues"] == ["first clue"]
    assert trace[0]["state_before"]["budget"] == 3
    assert trace[0]["state_after"] == {"pattern": "leaky", "clues": 1, "refusal": 0, "risk": 1, "budget": 2}
    assert trace[1]["state_before"] == trace[0]["state_after"]
    assert trace[1]["score_value"] is True


async def test_the_planner_picks_the_cheapest_strategy_when_nothing_distinguishes_them() -> None:
    harness = _Harness(role_replies=[_written("a"), _judgment()], target_replies=["r"])
    context = _context()

    await _run(harness.attack(max_turns=1), context, [_score("false")])

    assert context.trace[0].details["strategy"] == "calm"
    assert context.trace[0].details["planned"][0] == "calm"


async def test_the_attack_model_is_told_the_strategy_and_the_clues_gathered() -> None:
    harness = _Harness(
        role_replies=[_written("a"), _judgment("borderline", ("a useful clue",)), _written("b"), _judgment()],
        target_replies=["r1", "r2"],
    )

    await _run(harness.attack(max_turns=2), _context(), [_score("false"), _score("false")])

    first_writer, _, second_writer, _ = harness.role_inputs
    assert "Strategy to apply this turn: calm" in first_writer
    assert "Definition: Placeholder definition for the calm strategy." in first_writer
    assert "Clues gathered so far:\n(none)" in first_writer
    assert "- a useful clue" in second_writer
    assert "User: a\nTarget: r1" in second_writer


async def test_the_judge_reads_the_latest_exchange_and_the_clues_already_known() -> None:
    harness = _Harness(
        role_replies=[_written("opening"), _judgment("borderline", ("known",))],
        target_replies=["the answer"],
    )

    await _run(harness.attack(max_turns=1), _context(), [_score("false")])

    judge_input = harness.role_inputs[1]
    assert "The message sent to the Target:\nopening" in judge_input
    assert "The Target's response:\nthe answer" in judge_input
    assert "Judge" in harness.system_prompts[1]


async def test_repeated_clues_are_not_counted_twice() -> None:
    harness = _Harness(
        role_replies=[
            _written("a"),
            _judgment("leaky", ("same clue",)),
            _written("b"),
            _judgment("leaky", ("Same Clue", "another")),
        ],
        target_replies=["r1", "r2"],
    )
    context = _context()

    await _run(harness.attack(max_turns=2), context, [_score("false"), _score("false")])

    assert context.trace[1].details["new_clues"] == ["another"]
    assert context.trace[1].details["state_after"]["clues"] == 2


async def test_a_judged_full_answer_is_not_trusted_without_the_objective_scorer() -> None:
    harness = _Harness(
        role_replies=[_written("a"), _judgment("success"), _written("b"), _judgment("borderline")],
        target_replies=["r1", "r2"],
    )
    context = _context()

    result = await _run(harness.attack(max_turns=2), context, [_score("false"), _score("false")])

    assert result.outcome == AttackOutcome.FAILURE
    assert context.trace[0].details["judged_pattern"] == "success"
    assert context.trace[0].details["state_after"]["pattern"] == "leaky"


async def test_refusals_raise_the_refusal_firmness_in_the_state() -> None:
    harness = _Harness(
        role_replies=[_written("a"), _judgment("strict_refusal"), _written("b"), _judgment("weak_refusal")],
        target_replies=["no", "no again"],
    )
    context = _context()

    await _run(harness.attack(max_turns=2), context, [_score("false"), _score("false")])

    assert context.trace[0].details["state_after"]["refusal"] == 2
    assert context.trace[1].details["state_after"]["refusal"] == 3


async def test_a_learned_transitioner_steers_the_choice_of_strategy() -> None:
    before = RampDomain.initial_state(budget=3).to_dict()
    records = [
        {"strategy": "calm", "state_before": before, "judged_pattern": "strict_refusal"},
        {"strategy": "loud", "state_before": before, "judged_pattern": "leaky"},
    ] * 3
    rounds = [_written("a"), _judgment(), _written("b"), _judgment(), _written("c"), _judgment()]
    harness = _Harness(role_replies=rounds, target_replies=["r1", "r2", "r3"])
    context = _context()
    transitioner = EmpiricalTransitioner(records=records, min_support=3)

    await _run(harness.attack(max_turns=3, transitioner=transitioner), context, [_score("false")] * 3)

    assert context.trace[0].details["strategy"] == "loud"


async def test_the_attack_stops_when_the_risk_budget_is_spent() -> None:
    harness = _Harness(role_replies=[_written("a"), _judgment()], target_replies=["r"])

    result = await _run(harness.attack(max_turns=5, risk_budget=1), _context(), [_score("false")])

    assert result.outcome == AttackOutcome.FAILURE
    assert result.executed_turns == 1
    assert "Risk budget of 1 spent" in (result.outcome_reason or "")


async def test_when_no_plan_fits_the_risk_budget_the_least_risky_strategy_is_used() -> None:
    harness = _Harness(role_replies=[_written("a"), _judgment()], target_replies=["r"])
    context = _context()

    await _run(harness.attack(strategies=[LOUD, CALM], max_turns=3, risk_budget=0.5), context, [_score("false")])

    assert context.trace[0].details["strategy"] == "calm"
    assert context.trace[0].details["planned"] == []


async def test_a_caller_supplied_message_seeds_the_first_turn_without_planning() -> None:
    harness = _Harness(
        role_replies=[_judgment("borderline"), _written("second"), _judgment()], target_replies=["r1", "r2"]
    )
    context = _context(next_message=Message.from_prompt(prompt="my own opener", role="user"))

    await _run(harness.attack(max_turns=2), context, [_score("false"), _score("false")])

    assert harness.target_sends[0] == "my own opener"
    assert context.trace[0].prompt_source == "seed"
    assert context.trace[0].details["strategy"] is None
    assert context.trace[0].details["state_after"]["risk"] == 0
    assert context.trace[1].prompt_source == "attacker"


async def test_a_blocked_judge_is_recorded_as_a_neutral_turn_and_the_attack_continues() -> None:
    harness = _Harness(
        role_replies=[_written("a"), _blocked_reply(), _written("b"), _judgment()], target_replies=["r1", "r2"]
    )
    context = _context()

    result = await _run(harness.attack(max_turns=2), context, [_score("false"), _score("false")])

    assert result.executed_turns == 2
    assert context.trace[0].details["judge_error"] == "blocked"
    assert context.trace[0].details["judged_pattern"] == "borderline"
    assert context.trace[0].details["new_clues"] == []


async def test_a_blocked_attack_model_before_any_turn_is_a_preparation_failure() -> None:
    harness = _Harness(role_replies=[_blocked_reply()], target_replies=[])

    result = await _run(harness.attack(), _context(), [])

    assert result.outcome == AttackOutcome.UNDETERMINED
    assert result.executed_turns == 0
    assert harness.target_sends == []


async def test_a_malformed_judge_reply_is_retried() -> None:
    harness = _Harness(
        role_replies=[_written("a"), _judgment("explosive"), _judgment("borderline")], target_replies=["r"]
    )
    context = _context()

    await _run(harness.attack(max_turns=1), context, [_score("false")])

    assert len(harness.role_inputs) == 3
    assert context.trace[0].details["judged_pattern"] == "borderline"


async def test_an_undetermined_final_score_is_reported_as_undetermined() -> None:
    harness = _Harness(role_replies=[_written("a"), _judgment()], target_replies=["r"])

    result = await _run(harness.attack(max_turns=1), _context(), [_score("", status=ScoreStatus.UNDETERMINED)])

    assert result.outcome == AttackOutcome.UNDETERMINED
