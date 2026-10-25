# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pyrit.common.apply_defaults import REQUIRED_VALUE, apply_defaults
from pyrit.common.path import EXECUTOR_RAMP_PATH
from pyrit.exceptions import AdversarialChatResponseBlockedException, InvalidJsonException
from pyrit.executor.attack.component import StructuredRole, parse_json_object
from pyrit.executor.attack.component.risk_constrained_planner import RiskConstrainedPlanner
from pyrit.executor.attack.core.attack_config import AttackAdversarialConfig
from pyrit.executor.attack.multi_turn.ramp_domain import (
    RampDomain,
    RampState,
    RampStrategy,
    RampTransitioner,
    RuleTransitioner,
    SafetyPattern,
)
from pyrit.executor.attack.multi_turn.role_driven_attack import (
    RoleDrivenAttackContext,
    RoleDrivenConversationAttack,
    TurnTrace,
)
from pyrit.models import JsonSchemaDefinition, Message, SeedPrompt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyrit.executor.attack.component import PrependedConversationConfig
    from pyrit.executor.attack.core.attack_config import AttackConverterConfig, AttackScoringConfig
    from pyrit.prompt_normalizer import PromptNormalizer
    from pyrit.prompt_target.common.prompt_target import PromptTarget

logger = logging.getLogger(__name__)


class RampSystemPromptPaths(enum.Enum):
    """Enum for the predefined RAMP role system prompt paths."""

    JUDGE = Path(EXECUTOR_RAMP_PATH, "judge.yaml").resolve()
    ATTACKER = Path(EXECUTOR_RAMP_PATH, "attacker.yaml").resolve()


@dataclass(frozen=True)
class RampJudgment:
    """The Judge's reading of one objective-target response."""

    pattern: SafetyPattern
    clues: tuple[str, ...]
    rationale: str


@dataclass
class _RampRun:
    """Mutable state of one run: its agents, the symbolic state and the clues gathered."""

    judge: StructuredRole[RampJudgment]
    attacker: StructuredRole[str]
    state: RampState
    clues: list[str] = field(default_factory=list)
    strategy: RampStrategy | None = None
    state_before: RampState | None = None


class RampAttack(RoleDrivenConversationAttack):
    """
    Implementation of the RAMP (Risk-Aware Multi-Turn Planning) attack strategy [@liu2026ramp].

    RAMP treats multi-turn jailbreaking as planning. The dialogue is summarised as a small symbolic state:
    how the target last responded, how many task-relevant clues have been gathered, how firmly it has been
    refusing, how much risk has been spent, and how many turns remain. Each turn a Judge classifies the
    target's response and extracts clues, a transition model predicts how each strategy would change the
    state, and a planner searches a few turns ahead for the cheapest sequence that stays inside a risk
    budget and ends in a full answer. The first strategy of that plan is applied: the attack model turns it,
    the objective and the clues into the next message.

    PyRIT ships no strategy catalog. Pass the strategies to test as ``strategies``; each carries a name, a
    definition told to the attack model, and a risk. With the default ``RuleTransitioner`` strategies are
    indistinguishable except for their risk; an ``EmpiricalTransitioner`` fitted on earlier results'
    ``ramp_trace`` metadata teaches the planner which strategies work.

    The paper does not publish its prompts or how states are discretised, so the defaults here are written
    for PyRIT from the behavior it describes. The objective scorer, not the Judge, decides whether the
    objective was achieved. RAMP is text-only and does not use ``use_score_as_feedback``.
    """

    TRACE_METADATA_KEY: ClassVar[str] = "ramp_trace"

    JUDGE_SCHEMA: ClassVar[JsonSchemaDefinition] = {
        "type": "object",
        "properties": {
            "safety_pattern": {"type": "string", "enum": [pattern.name.lower() for pattern in SafetyPattern]},
            "clues": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": ["safety_pattern", "clues", "rationale"],
        "additionalProperties": False,
    }

    MESSAGE_SCHEMA: ClassVar[JsonSchemaDefinition] = {
        "type": "object",
        "properties": {
            "next_message": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": ["next_message", "rationale"],
        "additionalProperties": False,
    }

    @apply_defaults
    def __init__(
        self,
        *,
        objective_target: PromptTarget = REQUIRED_VALUE,  # type: ignore[ty:invalid-parameter-default]
        attack_adversarial_config: AttackAdversarialConfig,
        strategies: Sequence[RampStrategy],
        attack_converter_config: AttackConverterConfig | None = None,
        attack_scoring_config: AttackScoringConfig | None = None,
        prompt_normalizer: PromptNormalizer | None = None,
        prepended_conversation_config: PrependedConversationConfig | None = None,
        max_turns: int = 5,
        judge_system_prompt: str | SeedPrompt | None = None,
        attacker_system_prompt: str | SeedPrompt | None = None,
        transitioner: RampTransitioner | None = None,
        planning_horizon: int = 3,
        risk_budget: float = 10.0,
    ) -> None:
        """
        Initialize the RAMP attack strategy.

        Args:
            objective_target (PromptTarget): The target system to attack.
            attack_adversarial_config (AttackAdversarialConfig): Supplies the chat target that runs the Judge
                and the attack model. Only its ``target`` is used.
            strategies (Sequence[RampStrategy]): The strategies the planner may choose between, in
                preference order. There is no default; supply the ones to test.
            attack_converter_config (AttackConverterConfig | None): Converters applied to the messages sent
                to, and the responses received from, the objective target. Defaults to None.
            attack_scoring_config (AttackScoringConfig | None): Scoring configuration. The objective scorer
                is required. An optional refusal scorer adds a refusal verdict to the trace. Defaults to None.
            prompt_normalizer (PromptNormalizer | None): The prompt normalizer to use. Defaults to None.
            prepended_conversation_config (PrependedConversationConfig | None): Configuration for
                prepended-conversation handling. Defaults to None.
            max_turns (int): Maximum number of turns. Defaults to 5, the budget the paper evaluates.
            judge_system_prompt (str | SeedPrompt | None): System prompt for the Judge, rendered with
                ``objective`` and ``max_turns``. Defaults to the bundled prompt.
            attacker_system_prompt (str | SeedPrompt | None): System prompt for the attack model. Defaults to
                the bundled prompt.
            transitioner (RampTransitioner | None): Predicts how a strategy changes the state. Defaults to a
                ``RuleTransitioner``.
            planning_horizon (int): How many turns the planner looks ahead. Defaults to 3.
            risk_budget (float): Total risk the attack may spend. The attack stops when it is spent.
                Defaults to 10.0.

        Raises:
            ValueError: If no strategies are given or their names repeat, ``planning_horizon`` is not
                positive, ``risk_budget`` is not positive, no objective scorer is provided, or the
                adversarial target lacks native system-prompt support.
        """
        RampAttack.validate_constructor_parameters(
            {"strategies": strategies, "planning_horizon": planning_horizon, "risk_budget": risk_budget}
        )
        super().__init__(
            objective_target=objective_target,
            attack_adversarial_config=attack_adversarial_config,
            context_type=RoleDrivenAttackContext,
            attack_converter_config=attack_converter_config,
            attack_scoring_config=attack_scoring_config,
            prompt_normalizer=prompt_normalizer,
            prepended_conversation_config=prepended_conversation_config,
            max_turns=max_turns,
            attack_name="RampAttack",
        )
        self._strategies = tuple(strategies)
        self._transitioner = transitioner or RuleTransitioner()
        self._planning_horizon = planning_horizon
        self._risk_budget = risk_budget
        self._planner: RiskConstrainedPlanner[RampState, RampStrategy] = RiskConstrainedPlanner(
            actions=self._strategies,
            transition=self._transitioner.predict,
            step_cost=RampDomain.step_cost,
            step_risk=RampDomain.step_risk,
            is_goal=RampDomain.is_goal,
            heuristic=RampDomain.heuristic,
        )

        required = ["objective", "max_turns"]
        self._judge_prompt = self._resolve_role_prompt(
            value=judge_system_prompt,
            default=RampSystemPromptPaths.JUDGE.value,
            component_name="RAMP judge",
            required_parameters=required,
        )
        self._attacker_prompt = self._resolve_role_prompt(
            value=attacker_system_prompt,
            default=RampSystemPromptPaths.ATTACKER.value,
            component_name="RAMP attacker",
            required_parameters=required,
        )

    @classmethod
    def validate_constructor_parameters(cls, params: dict[str, Any]) -> None:
        """
        Validate the strategy catalog and planning settings without constructing an attack.

        Raises:
            ValueError: If the catalog is empty or repeats a name, or a planning setting is not positive.
        """
        super().validate_constructor_parameters(params)
        if "strategies" in params:
            names = [strategy.name for strategy in params["strategies"]]
            if not names:
                raise ValueError("RampAttack needs at least one strategy; PyRIT ships none, so supply them.")
            if len(set(names)) != len(names):
                raise ValueError("RampAttack strategy names must be unique.")
        if "planning_horizon" in params and params["planning_horizon"] <= 0:
            raise ValueError("planning_horizon must be a positive integer.")
        if "risk_budget" in params and params["risk_budget"] <= 0:
            raise ValueError("risk_budget must be positive.")

    def get_attack_adversarial_config(self) -> AttackAdversarialConfig | None:
        """
        Get the effective adversarial configuration used by this strategy.

        Returns:
            AttackAdversarialConfig | None: The adversarial target with the attack model's system prompt.
                The Judge prompt is part of the attack identity as a parameter.
        """
        adversarial_chat = getattr(self, "_adversarial_chat", None)
        if adversarial_chat is None:
            return None
        return AttackAdversarialConfig(
            target=adversarial_chat,
            system_prompt=self._attacker_prompt,
            first_message=None,
            adversarial_prompt_template=None,
        )

    def _identifier_params(self) -> dict[str, Any]:
        return {
            "planning_horizon": self._planning_horizon,
            "risk_budget": self._risk_budget,
            "transitioner": type(self._transitioner).__name__,
            "strategies": [
                {"name": strategy.name, "definition": strategy.definition, "risk": strategy.risk}
                for strategy in self._strategies
            ],
            "judge_system_prompt": self._judge_prompt.value,
        }

    def _begin_run(self, *, context: RoleDrivenAttackContext) -> _RampRun:
        return _RampRun(
            judge=self._build_role(
                context=context, parse=self._parse_judgment, schema=self.JUDGE_SCHEMA, label="RAMP judge"
            ),
            attacker=self._build_role(
                context=context, parse=self._parse_message, schema=self.MESSAGE_SCHEMA, label="RAMP attacker"
            ),
            state=RampDomain.initial_state(budget=self._max_turns - context.executed_turns),
        )

    async def _plan_turn_async(self, *, context: RoleDrivenAttackContext, run: _RampRun) -> tuple[TurnTrace, Message]:
        """
        Choose a strategy by planning, and write the message that applies it.

        A caller-supplied ``next_message`` seeds the first turn and bypasses planning and the attack model.

        Args:
            context (RoleDrivenAttackContext): The attack context.
            run (_RampRun): The run state.

        Returns:
            tuple[TurnTrace, Message]: The turn's trace (without outcome) and the message to send.
        """
        turn_number = context.executed_turns + 1
        run.state_before = run.state
        seed_message, context.next_message = context.next_message, None
        if seed_message is not None:
            run.strategy = None
            trace = TurnTrace(
                turn=turn_number,
                prompt=seed_message.get_value(),
                prompt_source="seed",
                details={"strategy": None, "planned": [], "plan_reached_goal": False},
            )
            return trace, seed_message.duplicate()

        strategy, planned, reached_goal = self._choose_strategy(run=run)
        run.strategy = strategy
        prompt = await self._ask_role_async(
            context=context,
            role=run.attacker,
            role_name="attacker",
            system_prompt=self._attacker_prompt,
            user_text=self._build_attacker_input(context=context, run=run, strategy=strategy),
        )
        trace = TurnTrace(
            turn=turn_number,
            prompt=prompt,
            prompt_source="attacker",
            details={"strategy": strategy.name, "planned": planned, "plan_reached_goal": reached_goal},
        )
        return trace, Message.from_prompt(prompt=prompt, role="user")

    def _choose_strategy(self, *, run: _RampRun) -> tuple[RampStrategy, list[str], bool]:
        """
        Pick the strategy for the coming turn by planning from the current state.

        Falls back to the least risky strategy when no plan fits inside the risk budget.

        Args:
            run (_RampRun): The run state.

        Returns:
            tuple[RampStrategy, list[str], bool]: The strategy, the names of the planned sequence, and
                whether that sequence is expected to reach a full answer.
        """
        plan = self._planner.plan(
            initial=run.state,
            horizon=min(self._planning_horizon, run.state.budget),
            max_risk=self._risk_budget - run.state.risk,
        )
        if plan is None or plan.first_action is None:
            return min(self._strategies, key=lambda strategy: strategy.risk), [], False
        return plan.first_action, [strategy.name for strategy in plan.actions], plan.reached_goal

    async def _observe_response_async(
        self, *, context: RoleDrivenAttackContext, run: _RampRun, trace: TurnTrace, response: Message
    ) -> None:
        """
        Have the Judge read the response and move the symbolic state on.

        If the Judge is blocked the turn is recorded as a neutral response with no clues rather than
        losing a response that has already been obtained.

        Args:
            context (RoleDrivenAttackContext): The attack context.
            run (_RampRun): The run state.
            trace (TurnTrace): The trace of the turn that was just played.
            response (Message): The objective target's response.
        """
        try:
            judgment = await self._ask_role_async(
                context=context,
                role=run.judge,
                role_name="judge",
                system_prompt=self._judge_prompt,
                user_text=self._build_judge_input(context=context, run=run),
            )
        except AdversarialChatResponseBlockedException as blocked:
            logger.warning("The RAMP judge was blocked; recording a neutral response: %s", blocked)
            judgment = RampJudgment(pattern=SafetyPattern.BORDERLINE, clues=(), rationale="Judge blocked.")
            trace.details["judge_error"] = "blocked"

        added = self._add_clues(run=run, clues=judgment.clues)
        before = run.state_before if run.state_before is not None else run.state
        # A full answer is only confirmed by the objective scorer, so the state never rests on it.
        observed = SafetyPattern(min(judgment.pattern, SafetyPattern.LEAKY))
        run.state = RampDomain.advance(state=before, strategy=run.strategy, pattern=observed, new_clues=len(added))
        trace.details.update(
            {
                "state_before": before.to_dict(),
                "judged_pattern": judgment.pattern.name.lower(),
                "new_clues": added,
                "judge_rationale": judgment.rationale,
                "state_after": run.state.to_dict(),
            }
        )

    @staticmethod
    def _add_clues(*, run: _RampRun, clues: Sequence[str]) -> list[str]:
        known = {clue.lower() for clue in run.clues}
        added: list[str] = []
        for clue in clues:
            text = clue.strip()
            if text and text.lower() not in known:
                known.add(text.lower())
                run.clues.append(text)
                added.append(text)
        return added

    def _stop_reason(self, *, context: RoleDrivenAttackContext, run: _RampRun) -> str | None:
        if run.state.risk >= self._risk_budget:
            return f"Risk budget of {self._risk_budget} spent after {context.executed_turns} turn(s)."
        return None

    @classmethod
    def _parse_judgment(cls, response_text: str) -> RampJudgment:
        """
        Parse a Judge reply.

        Args:
            response_text (str): The raw reply.

        Returns:
            RampJudgment: The classification and clues.

        Raises:
            InvalidJsonException: If the reply is malformed or names an unknown safety pattern.
        """
        data = parse_json_object(response_text, schema=cls.JUDGE_SCHEMA)
        try:
            pattern = SafetyPattern[str(data["safety_pattern"]).upper()]
        except KeyError as exc:
            raise InvalidJsonException(message=f"Unknown safety_pattern in reply: {response_text}") from exc
        clues = data["clues"]
        if not isinstance(clues, list):
            raise InvalidJsonException(message=f"'clues' must be a list in reply: {response_text}")
        return RampJudgment(pattern=pattern, clues=tuple(str(clue) for clue in clues), rationale=str(data["rationale"]))

    @classmethod
    def _parse_message(cls, response_text: str) -> str:
        """
        Parse an attack-model reply into the message to send.

        Args:
            response_text (str): The raw reply.

        Returns:
            str: The message text.

        Raises:
            InvalidJsonException: If the reply is malformed or the message is blank.
        """
        data = parse_json_object(response_text, schema=cls.MESSAGE_SCHEMA)
        next_message = str(data["next_message"])
        if not next_message.strip():
            raise InvalidJsonException(message=f"'next_message' must contain non-whitespace text: {response_text}")
        return next_message

    def _build_attacker_input(self, *, context: RoleDrivenAttackContext, run: _RampRun, strategy: RampStrategy) -> str:
        """
        Build what the attack model reads this turn.

        Args:
            context (RoleDrivenAttackContext): The attack context.
            run (_RampRun): The run state holding the clues gathered.
            strategy (RampStrategy): The strategy the planner chose.

        Returns:
            str: The attack model's input.
        """
        parts = [self._budget_line(context=context)]
        if context.prior_context:
            parts.append(f"Context that came before this dialogue:\n{context.prior_context}")
        parts.append(f"Strategy to apply this turn: {strategy.name}\nDefinition: {strategy.definition}")
        parts.append(f"Clues gathered so far:\n{self._render_clues(run.clues)}")
        parts.append(f"Conversation so far:\n{self._render_history(context.history)}")
        parts.append("Write the next message.")
        return "\n\n".join(parts)

    def _build_judge_input(self, *, context: RoleDrivenAttackContext, run: _RampRun) -> str:
        """
        Build what the Judge reads after a turn.

        Args:
            context (RoleDrivenAttackContext): The attack context, whose latest exchange is the turn played.
            run (_RampRun): The run state holding the clues already known.

        Returns:
            str: The Judge's input.
        """
        latest = context.history[-1]
        return "\n\n".join(
            [
                self._budget_line(context=context),
                f"Clues already known:\n{self._render_clues(run.clues)}",
                f"The message sent to the Target:\n{latest.prompt}",
                f"The Target's response:\n{latest.response}",
                "Classify the response.",
            ]
        )

    @staticmethod
    def _render_clues(clues: Sequence[str]) -> str:
        return "\n".join(f"- {clue}" for clue in clues) if clues else "(none)"
