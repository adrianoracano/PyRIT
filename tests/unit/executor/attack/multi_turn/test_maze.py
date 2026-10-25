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
from pyrit.executor.attack.multi_turn.maze import (
    MazeAttack,
    MazeAttackContext,
    MazeMode,
    RecoveryMode,
    StrategistHistory,
    TargetHistory,
)
from pyrit.models import (
    AttackOutcome,
    ComponentIdentifier,
    ConversationType,
    Message,
    MessagePiece,
    Score,
    ScoreStatus,
)
from pyrit.prompt_target import CapabilityName, PromptTarget
from pyrit.score import TrueFalseScorer

pytestmark = pytest.mark.usefixtures("patch_central_database")

OBJECTIVE = "Test objective"


def _identifier(name: str) -> ComponentIdentifier:
    return ComponentIdentifier(class_name=name, class_module="test_module")


def _decision(
    mode: str = "ATTACK",
    framing: str = "academic",
    state: str = "cooperative",
    anchors: tuple[str, ...] = (),
    rationale: str = "because",
) -> str:
    return json.dumps(
        {
            "mode": mode,
            "framing": framing,
            "response_state": state,
            "anchors": list(anchors),
            "rationale": rationale,
        }
    )


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


def _scorer(name: str = "MockScorer") -> MagicMock:
    scorer = MagicMock(spec=TrueFalseScorer)
    scorer.score_async = AsyncMock()
    scorer.get_identifier.return_value = _identifier(name)
    return scorer


class _Harness:
    """Scripts what the role agents and the objective target say, and records what they were sent."""

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
        message = kwargs["message"]
        text = message.message_pieces[0].converted_value
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

    def attack(self, **overrides) -> MazeAttack:
        kwargs: dict = {
            "objective_target": self.objective_target,
            "attack_adversarial_config": AttackAdversarialConfig(target=self.adversarial),
            "attack_scoring_config": AttackScoringConfig(objective_scorer=_scorer()),
            "prompt_normalizer": self.normalizer,
            "max_turns": 4,
        }
        kwargs.update(overrides)
        return MazeAttack(**kwargs)


def _context(*, next_message: Message | None = None) -> MazeAttackContext:
    return MazeAttackContext(
        params=AttackParameters(objective=OBJECTIVE, next_message=next_message),
        session=ConversationSession(),
    )


async def _run(attack: MazeAttack, context: MazeAttackContext, scores: list[Score]):
    with patch.object(attack, "_score_response_async", new_callable=AsyncMock, side_effect=scores):
        return await attack._perform_async(context=context)


# --- construction ------------------------------------------------------------


def test_init_requires_an_objective_scorer() -> None:
    harness = _Harness(role_replies=[], target_replies=[])

    with pytest.raises(ValueError, match="Objective scorer must be provided"):
        harness.attack(attack_scoring_config=AttackScoringConfig())


@pytest.mark.parametrize("max_turns", [0, -1])
def test_init_rejects_non_positive_max_turns(max_turns: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _Harness(role_replies=[], target_replies=[]).attack(max_turns=max_turns)


def test_init_rejects_an_empty_framing_schedule() -> None:
    with pytest.raises(ValueError, match="at least one framing"):
        _Harness(role_replies=[], target_replies=[]).attack(framing_schedule=[])


def test_init_requires_native_system_prompt_support_on_the_adversarial_target() -> None:
    harness = _Harness(role_replies=[], target_replies=[])
    harness.adversarial.configuration.includes.side_effect = lambda *, capability: (
        capability != CapabilityName.SYSTEM_PROMPT
    )

    with pytest.raises(ValueError, match=r"(?s)MazeAttack .*supports_system_prompt"):
        harness.attack()


def test_init_loads_the_bundled_role_prompts() -> None:
    attack = _Harness(role_replies=[], target_replies=[]).attack()

    assert "Strategist" in attack._strategist_prompt.value
    assert "write the next user message" in attack._attacker_prompt.value
    assert "refused or keeps deflecting" in attack._recovery_prompt.value


def test_init_accepts_a_custom_role_prompt() -> None:
    attack = _Harness(role_replies=[], target_replies=[]).attack(strategist_system_prompt="custom {{ objective }}")

    assert attack._strategist_prompt.value == "custom {{ objective }}"


def test_identifier_differs_between_ablation_variants() -> None:
    harness = _Harness(role_replies=[], target_replies=[])

    full = harness.attack().get_identifier()
    no_history = harness.attack(target_history=TargetHistory.NONE).get_identifier()
    static_recovery = harness.attack(recovery=RecoveryMode.STATIC).get_identifier()

    assert len({full.hash, no_history.hash, static_recovery.hash}) == 3


# --- role reply parsing ------------------------------------------------------


def test_parse_decision_reads_every_field() -> None:
    decision = MazeAttack._parse_decision(_decision("recover", "defensive", "REFUSING", ("a", "b"), "why"))

    assert decision.mode is MazeMode.RECOVER
    assert decision.framing == "defensive"
    assert decision.response_state.value == "refusing"
    assert decision.anchors == ("a", "b")
    assert decision.rationale == "why"


@pytest.mark.parametrize(
    "reply",
    [
        _decision(mode="EXPLODE"),
        _decision(state="furious"),
        json.dumps(
            {"mode": "ATTACK", "framing": "f", "response_state": "cooperative", "anchors": "x", "rationale": "r"}
        ),
        '{"mode": "ATTACK"}',
        "not json",
    ],
    ids=["unknown-mode", "unknown-state", "anchors-not-a-list", "missing-keys", "not-json"],
)
def test_parse_decision_rejects_malformed_replies(reply: str) -> None:
    with pytest.raises(InvalidJsonException):
        MazeAttack._parse_decision(reply)


def test_parse_message_returns_the_next_message() -> None:
    assert MazeAttack._parse_message(_written("hello")) == "hello"


@pytest.mark.parametrize("reply", [_written("  "), '{"next_message": "x"}', "nope"], ids=["blank", "missing", "bad"])
def test_parse_message_rejects_malformed_replies(reply: str) -> None:
    with pytest.raises(InvalidJsonException):
        MazeAttack._parse_message(reply)


# --- the loop ----------------------------------------------------------------


async def test_loop_stops_at_the_first_success_and_sends_what_the_attacker_wrote() -> None:
    harness = _Harness(
        role_replies=[_decision(), _written("opening"), _decision(framing="technical"), _written("follow-up")],
        target_replies=["first answer", "second answer"],
    )
    attack = harness.attack()
    context = _context()

    result = await _run(attack, context, [_score("false"), _score("true")])

    assert result.outcome == AttackOutcome.SUCCESS
    assert result.executed_turns == 2
    assert [text for _, text in harness.target_sends] == ["opening", "follow-up"]
    assert [turn.response for turn in context.history] == ["first answer", "second answer"]


async def test_loop_uses_one_continuing_conversation_with_the_target_by_default() -> None:
    harness = _Harness(
        role_replies=[_decision(), _written("a"), _decision(), _written("b")], target_replies=["r1", "r2"]
    )
    context = _context()

    await _run(harness.attack(max_turns=2), context, [_score("false"), _score("false")])

    conversation_ids = {conversation_id for conversation_id, _ in harness.target_sends}
    assert conversation_ids == {context.session.conversation_id}


async def test_loop_fails_when_the_budget_is_spent_without_success() -> None:
    harness = _Harness(
        role_replies=[_decision(), _written("a"), _decision(), _written("b")], target_replies=["r1", "r2"]
    )

    result = await _run(harness.attack(max_turns=2), _context(), [_score("false"), _score("false")])

    assert result.outcome == AttackOutcome.FAILURE
    assert result.executed_turns == 2


async def test_each_agent_runs_on_its_own_tracked_conversation_with_its_own_system_prompt() -> None:
    harness = _Harness(role_replies=[_decision(), _written("opening")], target_replies=["answer"])
    context = _context()

    await _run(harness.attack(max_turns=1), context, [_score("true")])

    strategist_prompt, attacker_prompt = harness.system_prompts
    assert "Strategist" in strategist_prompt and OBJECTIVE in strategist_prompt
    assert "write the next user message" in attacker_prompt
    adversarial = [
        ref for ref in context.related_conversations if ref.conversation_type == ConversationType.ADVERSARIAL
    ]
    assert {ref.description for ref in adversarial} == {"MazeAttack strategist turn 1", "MazeAttack attacker turn 1"}


async def test_strategist_reads_the_dialogue_and_the_writer_gets_the_decision_record() -> None:
    harness = _Harness(
        role_replies=[
            _decision(),
            _written("opening"),
            _decision(framing="technical", anchors=("the process",)),
            _written("follow-up"),
        ],
        target_replies=["a helpful answer", "done"],
    )

    await _run(harness.attack(max_turns=2), _context(), [_score("false"), _score("true")])

    first_strategist, _, second_strategist, second_writer = harness.role_inputs
    assert "turn 1 of 2" in first_strategist
    assert "(no messages yet)" in first_strategist
    assert "User: opening\nTarget: a helpful answer" in second_strategist
    assert "- framing: technical" in second_writer
    assert "- anchors: the process" in second_writer


async def test_recover_decision_hands_the_turn_to_the_recovery_agent() -> None:
    harness = _Harness(
        role_replies=[
            _decision(),
            _written("opening"),
            _decision(mode="RECOVER", framing="defensive", state="refusing"),
            _written("repositioned"),
        ],
        target_replies=["I can't help with that", "ok"],
    )
    context = _context()

    await _run(harness.attack(max_turns=2), context, [_score("false"), _score("true")])

    assert "refused or keeps deflecting" in harness.system_prompts[3]
    assert [record.prompt_source for record in context.trace] == ["attacker", "recovery"]
    assert harness.target_sends[1][1] == "repositioned"


async def test_recover_on_the_first_turn_is_treated_as_attack() -> None:
    harness = _Harness(role_replies=[_decision(mode="RECOVER"), _written("opening")], target_replies=["ok"])
    context = _context()

    await _run(harness.attack(max_turns=1), context, [_score("true")])

    assert context.trace[0].details["mode"] == "ATTACK"
    assert context.trace[0].prompt_source == "attacker"


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        (RecoveryMode.STATIC, MazeAttack.STATIC_RECOVERY_MESSAGE),
        (RecoveryMode.WEAK, MazeAttack.WEAK_RECOVERY_MESSAGE),
    ],
)
async def test_simplified_recovery_sends_a_fixed_message_without_calling_the_recovery_agent(
    mode: RecoveryMode, message: str
) -> None:
    harness = _Harness(
        role_replies=[_decision(), _written("opening"), _decision(mode="RECOVER", state="refusing")],
        target_replies=["no", "fine"],
    )
    context = _context()

    await _run(harness.attack(max_turns=2, recovery=mode), context, [_score("false"), _score("true")])

    assert harness.target_sends[1][1] == message
    assert context.trace[1].prompt_source == f"{mode.value}_recovery"
    assert len(harness.role_inputs) == 3


async def test_framing_schedule_replaces_the_strategist() -> None:
    harness = _Harness(role_replies=[_written("a"), _written("b"), _written("c")], target_replies=["1", "2", "3"])
    context = _context()

    await _run(
        harness.attack(max_turns=3, framing_schedule=["academic", "technical"]),
        context,
        [_score("false"), _score("false"), _score("false")],
    )

    assert [record.details["framing"] for record in context.trace] == ["academic", "technical", "technical"]
    assert all("You never write messages" not in prompt for prompt in harness.system_prompts)


async def test_strategist_can_be_limited_to_the_last_exchange() -> None:
    harness = _Harness(
        role_replies=[_decision(), _written("p1"), _decision(), _written("p2"), _decision(), _written("p3")],
        target_replies=["r1", "r2", "r3"],
    )

    await _run(
        harness.attack(max_turns=3, strategist_history=StrategistHistory.LAST_TURN),
        _context(),
        [_score("false"), _score("false"), _score("false")],
    )

    third_strategist = harness.role_inputs[4]
    assert "User: p2\nTarget: r2" in third_strategist
    assert "p1" not in third_strategist


async def test_disabling_target_history_sends_each_message_on_a_fresh_conversation() -> None:
    harness = _Harness(
        role_replies=[_decision(), _written("a"), _decision(), _written("b")], target_replies=["r1", "r2"]
    )
    context = _context()

    await _run(harness.attack(max_turns=2, target_history=TargetHistory.NONE), context, [_score("false")] * 2)

    first, second = (conversation_id for conversation_id, _ in harness.target_sends)
    assert first != second
    pruned = [ref for ref in context.related_conversations if ref.conversation_type == ConversationType.PRUNED]
    assert [ref.conversation_id for ref in pruned] == [first]
    assert "User: a\nTarget: r1" in harness.role_inputs[2]


async def test_a_caller_supplied_message_seeds_the_first_turn_without_calling_the_agents() -> None:
    harness = _Harness(role_replies=[_decision(), _written("second")], target_replies=["r1", "r2"])
    seed = Message.from_prompt(prompt="my own opener", role="user")
    context = _context(next_message=seed)

    await _run(harness.attack(max_turns=2), context, [_score("false"), _score("true")])

    assert harness.target_sends[0][1] == "my own opener"
    assert context.trace[0].prompt_source == "seed"
    assert context.trace[1].prompt_source == "attacker"


async def test_the_trace_records_each_turn_in_the_result_metadata() -> None:
    harness = _Harness(
        role_replies=[_decision(framing="technical", anchors=("x",)), _written("opening")], target_replies=["answer"]
    )

    result = await _run(harness.attack(max_turns=1), _context(), [_score("true")])

    trace = result.metadata["maze_trace"]
    assert trace == [
        {
            "turn": 1,
            "mode": "ATTACK",
            "framing": "technical",
            "response_state": "cooperative",
            "anchors": ["x"],
            "rationale": "because",
            "prompt": "opening",
            "prompt_source": "attacker",
            "was_refusal": None,
            "score_value": True,
        }
    ]


# --- refusal scorer ----------------------------------------------------------


async def test_a_refusal_scorer_verdict_is_shown_to_the_strategist_and_traced() -> None:
    refusal_scorer = _scorer("RefusalScorer")
    refusal_scorer.score_async.return_value = [
        Score(
            score_type="true_false",
            score_value="true",
            score_category=["refusal"],
            score_value_description="d",
            score_rationale="r",
            score_metadata={},
            message_piece_id=str(uuid.uuid4()),
        )
    ]
    harness = _Harness(
        role_replies=[_decision(), _written("opening"), _decision(), _written("again")],
        target_replies=["no", "still no"],
    )
    context = _context()
    attack = harness.attack(
        max_turns=2,
        attack_scoring_config=AttackScoringConfig(objective_scorer=_scorer(), refusal_scorer=refusal_scorer),
    )

    await _run(attack, context, [_score("false"), _score("false")])

    assert "judged the Target's latest response to be a refusal" in harness.role_inputs[2]
    assert [record.was_refusal for record in context.trace] == [True, True]
    assert attack.get_attack_scoring_config().refusal_scorer is refusal_scorer


# --- failure handling --------------------------------------------------------


async def test_a_block_before_any_turn_is_a_preparation_failure() -> None:
    harness = _Harness(role_replies=[_blocked_reply()], target_replies=[])

    result = await _run(harness.attack(), _context(), [])

    assert result.outcome == AttackOutcome.UNDETERMINED
    assert result.executed_turns == 0
    assert harness.target_sends == []
    assert result.metadata["maze_trace"] == []


async def test_a_block_mid_run_truncates_the_attack_as_undetermined() -> None:
    harness = _Harness(role_replies=[_decision(), _written("opening"), _blocked_reply()], target_replies=["answer"])

    result = await _run(harness.attack(), _context(), [_score("false")])

    assert result.outcome == AttackOutcome.UNDETERMINED
    assert result.executed_turns == 1
    assert "blocked after 1 completed turn" in (result.outcome_reason or "")
    assert len(result.metadata["maze_trace"]) == 1


async def test_a_malformed_strategist_reply_is_retried() -> None:
    harness = _Harness(
        role_replies=[_decision(mode="EXPLODE"), _decision(), _written("opening")], target_replies=["answer"]
    )

    result = await _run(harness.attack(max_turns=1), _context(), [_score("true")])

    assert result.outcome == AttackOutcome.SUCCESS
    assert len(harness.role_inputs) == 3


async def test_an_undetermined_final_score_is_reported_as_undetermined() -> None:
    harness = _Harness(role_replies=[_decision(), _written("opening")], target_replies=["answer"])

    result = await _run(harness.attack(max_turns=1), _context(), [_score("", status=ScoreStatus.UNDETERMINED)])

    assert result.outcome == AttackOutcome.UNDETERMINED


# --- helpers -----------------------------------------------------------------


def test_describe_response_truncates_long_text() -> None:
    long_text = "x" * (MazeAttack.HISTORY_RESPONSE_CHAR_LIMIT + 50)

    described = MazeAttack._describe_response(Message.from_prompt(prompt=long_text, role="assistant"))

    assert described.endswith("[...]")
    assert len(described) < len(long_text)


def test_describe_response_notes_blocked_and_errored_responses() -> None:
    blocked = MessagePiece(role="assistant", original_value="x", original_value_data_type="error")
    blocked.response_error = "blocked"
    errored = MessagePiece(role="assistant", original_value="x", original_value_data_type="error")
    errored.response_error = "unknown"

    assert "blocked" in MazeAttack._describe_response(Message(message_pieces=[blocked]))
    assert "error" in MazeAttack._describe_response(Message(message_pieces=[errored]))
