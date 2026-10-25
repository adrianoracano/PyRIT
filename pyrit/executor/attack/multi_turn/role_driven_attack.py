# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from pyrit.exceptions import AdversarialChatResponseBlockedException, ComponentRole, execution_context
from pyrit.executor.attack.component import (
    ConversationManager,
    PrependedConversationConfig,
    RefusalAssessor,
    StructuredRole,
    TurnAssessment,
)
from pyrit.executor.attack.component.adversarial_conversation_manager import _raise_for_adversarial_error
from pyrit.executor.attack.component.conversation_manager import build_conversation_context_string_async
from pyrit.executor.attack.core.attack_config import AttackAdversarialConfig, AttackConverterConfig, AttackScoringConfig
from pyrit.executor.attack.core.attack_preparation import (
    AttackPreparationFailure,
    AttackPreparationFailureKind,
)
from pyrit.executor.attack.core.attack_scoring import score_attack_response_async
from pyrit.executor.attack.core.attack_strategy import attack_outcome_from_score
from pyrit.executor.attack.multi_turn.multi_turn_attack_strategy import (
    ConversationSession,
    MultiTurnAttackContext,
    MultiTurnAttackStrategy,
)
from pyrit.memory import CentralMemory
from pyrit.models import (
    AtomicAttackIdentifier,
    AttackOutcome,
    AttackResult,
    ConversationReference,
    ConversationType,
    JsonSchemaDefinition,
    Message,
    Score,
    SeedPrompt,
)
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import CapabilityName, TargetRequirements
from pyrit.score.score_utils import score_is_true

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pyrit.models import ComponentIdentifier
    from pyrit.prompt_target.common.prompt_target import PromptTarget

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class Exchange:
    """One exchange with the objective target, as shown to the agents."""

    prompt: str
    response: str


@dataclass
class TurnTrace:
    """
    What happened in one turn, kept in the attack result for later analysis.

    ``details`` holds what is specific to the attack (for example a decision record) and is flattened
    into the serialized trace.
    """

    turn: int
    prompt: str
    prompt_source: str
    was_refusal: bool | None = None
    score_value: bool | float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the trace to a JSON-serializable dict.

        Returns:
            dict[str, Any]: The attack-specific details followed by the common fields.
        """
        return {
            "turn": self.turn,
            **self.details,
            "prompt": self.prompt,
            "prompt_source": self.prompt_source,
            "was_refusal": self.was_refusal,
            "score_value": self.score_value,
        }


@dataclass
class RoleDrivenAttackContext(MultiTurnAttackContext[Any]):
    """Context for attacks whose messages are written by stateless LLM agents."""

    # Exchanges with the objective target so far, as the agents see them
    history: list[Exchange] = field(default_factory=list)

    # Per-turn records
    trace: list[TurnTrace] = field(default_factory=list)

    # The prepended conversation rendered as text, shown to the agents as prior context
    prior_context: str = ""


class RoleDrivenConversationAttack(MultiTurnAttackStrategy[RoleDrivenAttackContext, AttackResult]):
    """
    Base for multi-turn attacks in which stateless LLM agents plan and write each message.

    The agents run on one adversarial chat target, each with its own system prompt, and read the
    conversation as text in their input rather than as chat history. The conversation with the
    objective target is a single continuing dialogue unless the subclass says otherwise.

    The base owns what every such attack shares: configuration and validation, running a
    ``StructuredRole`` on a tracked conversation, sending to the objective target, refusal assessment,
    objective scoring, the turn loop with its stop conditions, and building the result. A subclass
    supplies the planning of a turn and, optionally, what it learns from the response.
    """

    # The role agents each run on a fresh conversation that starts with a system prompt.
    ROLE_TARGET_REQUIREMENTS: ClassVar[TargetRequirements] = TargetRequirements(
        native_required=frozenset({CapabilityName.SYSTEM_PROMPT}),
    )

    # Longest target response shown to the agents; longer responses are cut so a long dialogue stays
    # within a small model's context window.
    HISTORY_RESPONSE_CHAR_LIMIT: ClassVar[int] = 2000

    # Key under which the serialized per-turn trace is stored in the attack result metadata.
    TRACE_METADATA_KEY: ClassVar[str] = "trace"

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        attack_adversarial_config: AttackAdversarialConfig,
        context_type: type[RoleDrivenAttackContext],
        attack_converter_config: AttackConverterConfig | None = None,
        attack_scoring_config: AttackScoringConfig | None = None,
        prompt_normalizer: PromptNormalizer | None = None,
        prepended_conversation_config: PrependedConversationConfig | None = None,
        max_turns: int = 10,
        attack_name: str = "Attack",
    ) -> None:
        """
        Initialize the shared configuration.

        Args:
            objective_target (PromptTarget): The target system to attack.
            attack_adversarial_config (AttackAdversarialConfig): Supplies the chat target that runs the
                agents. Only its ``target`` is used.
            context_type (type[RoleDrivenAttackContext]): The context class the subclass uses.
            attack_converter_config (AttackConverterConfig | None): Converters applied to the messages
                sent to, and the responses received from, the objective target. Defaults to None.
            attack_scoring_config (AttackScoringConfig | None): Scoring configuration. The objective
                scorer is required. An optional refusal scorer adds a refusal verdict each turn.
                Defaults to None.
            prompt_normalizer (PromptNormalizer | None): The prompt normalizer to use. Defaults to None.
            prepended_conversation_config (PrependedConversationConfig | None): Configuration for
                prepended-conversation handling. Defaults to None.
            max_turns (int): Maximum number of turns. Defaults to 10.
            attack_name (str): Name used in error messages. Defaults to "Attack".

        Raises:
            ValueError: If no objective scorer is provided, ``max_turns`` is not positive, or the
                adversarial target lacks native system-prompt support.
        """
        super().__init__(
            objective_target=objective_target,
            logger=logger,
            context_type=context_type,
            prepended_conversation_config=prepended_conversation_config,
        )
        self._memory = CentralMemory.get_memory_instance()

        attack_converter_config = attack_converter_config or AttackConverterConfig()
        self._request_converters = attack_converter_config.request_converters
        self._response_converters = attack_converter_config.response_converters

        attack_scoring_config = attack_scoring_config or AttackScoringConfig()
        if attack_scoring_config.objective_scorer is None:
            raise ValueError("Objective scorer must be provided in the attack scoring configuration.")
        self._objective_scorer = attack_scoring_config.objective_scorer
        self._auxiliary_scorers = attack_scoring_config.auxiliary_scorers
        self._use_score_as_feedback = attack_scoring_config.use_score_as_feedback
        self._refusal_assessor = (
            RefusalAssessor(refusal_scorer=attack_scoring_config.refusal_scorer)
            if attack_scoring_config.refusal_scorer is not None
            else None
        )

        self._adversarial_chat = attack_adversarial_config.target
        try:
            self.ROLE_TARGET_REQUIREMENTS.validate(target=self._adversarial_chat)
        except ValueError as exc:
            raise ValueError(f"{attack_name} {exc}") from exc

        type(self).validate_constructor_parameters({"max_turns": max_turns})
        self._max_turns = max_turns

        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()
        self._conversation_manager = ConversationManager(prompt_normalizer=self._prompt_normalizer)

    @classmethod
    def validate_constructor_parameters(cls, params: dict[str, Any]) -> None:
        """
        Validate the turn bound without constructing an attack.

        Raises:
            ValueError: If turns are not positive.
        """
        super().validate_constructor_parameters(params)
        if "max_turns" in params and params["max_turns"] <= 0:
            raise ValueError("Maximum turns must be a positive integer.")

    @staticmethod
    def _resolve_role_prompt(
        *, value: str | SeedPrompt | None, default: Path, component_name: str, required_parameters: list[str]
    ) -> SeedPrompt:
        error_message = f"{component_name} prompt must support {', '.join(required_parameters)}"
        if value is None:
            return SeedPrompt.from_yaml_with_required_parameters(
                template_path=default,
                required_parameters=required_parameters,
                error_message=error_message,
            )
        return SeedPrompt.from_value_with_required_parameters(
            value,
            required_parameters=required_parameters,
            error_message=error_message,
            component_name=f"{component_name} system prompt",
        )

    def get_attack_scoring_config(self) -> AttackScoringConfig | None:
        """
        Get the attack scoring configuration used by this strategy.

        Returns:
            AttackScoringConfig | None: The objective, refusal and auxiliary scorers in use.
        """
        return AttackScoringConfig(
            objective_scorer=self._objective_scorer,
            refusal_scorer=self._refusal_assessor.refusal_scorer if self._refusal_assessor is not None else None,
            auxiliary_scorers=self._auxiliary_scorers,
            use_score_as_feedback=self._use_score_as_feedback,
        )

    def _identifier_params(self) -> dict[str, Any]:
        """
        Behavior-defining settings of the subclass, added to the attack identifier.

        Returns:
            dict[str, Any]: The settings. Empty by default.
        """
        return {}

    def _build_identifier(self) -> ComponentIdentifier:
        """
        Build the identifier, including the settings that change the attack's behavior.

        Returns:
            ComponentIdentifier: The attack identifier.
        """
        return self._create_identifier(params={"max_turns": self._max_turns, **self._identifier_params()})

    def _validate_context(self, *, context: RoleDrivenAttackContext) -> None:
        """
        Validate the context before executing the attack.

        Args:
            context (RoleDrivenAttackContext): The context to validate.

        Raises:
            ValueError: If the objective is missing or the turn budget is already spent.
        """
        if not context.objective:
            raise ValueError("Attack objective must be provided")
        if context.executed_turns >= self._max_turns:
            raise ValueError("Already exceeded max turns")

    async def _setup_async(self, *, context: RoleDrivenAttackContext) -> None:
        """
        Prepare the strategy for execution.

        Args:
            context (RoleDrivenAttackContext): The attack context.
        """
        context.session = ConversationSession()
        await self._conversation_manager.initialize_context_async(
            context=context,
            target=self._objective_target,
            conversation_id=context.session.conversation_id,
            request_converters=self._request_converters,
            prepended_conversation_config=self._prepended_conversation_config,
            max_turns=self._max_turns,
            memory_labels=self._memory_labels,
        )
        context.prior_context = await build_conversation_context_string_async(context.prepended_conversation or [])

    @abstractmethod
    def _begin_run(self, *, context: RoleDrivenAttackContext) -> Any:
        """
        Build whatever one run needs, such as its agents.

        Args:
            context (RoleDrivenAttackContext): The attack context supplying the run's identifiers.

        Returns:
            Any: The run state, handed back to ``_plan_turn_async`` and ``_observe_response_async``.
        """

    @abstractmethod
    async def _plan_turn_async(self, *, context: RoleDrivenAttackContext, run: Any) -> tuple[TurnTrace, Message]:
        """
        Decide how to shape the next message and write it.

        Args:
            context (RoleDrivenAttackContext): The attack context.
            run (Any): The run state from ``_begin_run``.

        Returns:
            tuple[TurnTrace, Message]: The turn's trace (without outcome) and the message to send.
        """

    async def _observe_response_async(
        self, *, context: RoleDrivenAttackContext, run: Any, trace: TurnTrace, response: Message
    ) -> None:
        """
        Learn from the objective target's response before it is scored.

        Args:
            context (RoleDrivenAttackContext): The attack context.
            run (Any): The run state from ``_begin_run``.
            trace (TurnTrace): The trace of the turn that was just played.
            response (Message): The objective target's response.
        """

    def _stop_reason(self, *, context: RoleDrivenAttackContext, run: Any) -> str | None:
        """
        Say why the attack should stop before the turn budget is spent.

        Args:
            context (RoleDrivenAttackContext): The attack context.
            run (Any): The run state from ``_begin_run``.

        Returns:
            str | None: The reason, or None to keep going.
        """
        return None

    async def _prepare_target_conversation_async(self, *, context: RoleDrivenAttackContext) -> None:
        """
        Get the objective-target conversation ready for the next send.

        By default the dialogue continues, rotating the conversation only for targets that cannot hold
        a multi-turn one.

        Args:
            context (RoleDrivenAttackContext): The attack context.
        """
        await self._rotate_conversation_for_single_turn_target_async(context=context)

    async def _perform_async(self, *, context: RoleDrivenAttackContext) -> AttackResult:
        """
        Run the turn loop until the objective is achieved, the attack stops, or the turn budget is spent.

        Args:
            context (RoleDrivenAttackContext): The attack context.

        Returns:
            AttackResult: The result of the attack execution.
        """
        logger.info(f"Starting {self.__class__.__name__} with objective: {context.objective}")
        run = self._begin_run(context=context)
        achieved_objective = False
        stop_reason: str | None = None

        while context.executed_turns < self._max_turns and not achieved_objective and stop_reason is None:
            try:
                trace, message = await self._plan_turn_async(context=context, run=run)
            except AdversarialChatResponseBlockedException as blocked:
                return self._create_blocked_result(context=context, blocked=blocked)

            response = await self._send_prompt_to_objective_target_async(context=context, message=message)
            context.last_response = response
            context.history.append(Exchange(prompt=trace.prompt, response=self._describe_response(response)))

            context.last_assessment = await self._assess_response_async(context=context, message=message)
            trace.was_refusal = context.last_assessment.is_refusal if context.last_assessment is not None else None
            await self._observe_response_async(context=context, run=run, trace=trace, response=response)

            context.last_score = await self._score_response_async(context=context)
            achieved_objective = score_is_true(context.last_score)
            trace.score_value = self._score_value(context.last_score)
            context.trace.append(trace)
            context.executed_turns += 1
            stop_reason = None if achieved_objective else self._stop_reason(context=context, run=run)

        return self._create_result_from_context(context=context, stop_reason=stop_reason)

    def _build_role(
        self,
        *,
        context: RoleDrivenAttackContext,
        parse: Callable[[str], T],
        schema: JsonSchemaDefinition,
        label: str,
    ) -> StructuredRole[T]:
        """
        Build one agent for a run.

        Args:
            context (RoleDrivenAttackContext): The attack context supplying the run's identifiers.
            parse (Callable[[str], T]): Turns the agent's reply text into its result.
            schema (JsonSchemaDefinition): The JSON schema of the agent's reply.
            label (str): Name used in error messages.

        Returns:
            StructuredRole[T]: The agent.
        """
        return StructuredRole(
            target=self._adversarial_chat,
            parse=parse,
            response_json_schema=schema,
            component_role=ComponentRole.ADVERSARIAL_CHAT,
            role_label=label,
            raise_for_error=_raise_for_adversarial_error,
            prompt_normalizer=self._prompt_normalizer,
            attack_strategy_name=self.__class__.__name__,
            objective=context.objective,
            objective_target_conversation_id=context.session.conversation_id,
        )

    async def _ask_role_async(
        self,
        *,
        context: RoleDrivenAttackContext,
        role: StructuredRole[T],
        role_name: str,
        system_prompt: SeedPrompt,
        user_text: str,
    ) -> T:
        """
        Ask one agent a self-contained question on a conversation that is tracked on the context.

        Args:
            context (RoleDrivenAttackContext): The attack context.
            role (StructuredRole[T]): The agent to ask.
            role_name (str): Name of the agent, recorded on the conversation reference.
            system_prompt (SeedPrompt): The agent's system prompt template, rendered with ``objective``
                and ``max_turns``.
            user_text (str): The input for this turn.

        Returns:
            T: The agent's parsed reply.
        """
        conversation_id = str(uuid.uuid4())
        context.related_conversations.add(
            ConversationReference(
                conversation_id=conversation_id,
                conversation_type=ConversationType.ADVERSARIAL,
                description=f"{self.__class__.__name__} {role_name} turn {context.executed_turns + 1}",
            )
        )
        return await role.ask_async(
            prompt_text=user_text,
            system_prompt=system_prompt.render_template_value(objective=context.objective, max_turns=self._max_turns),
            conversation_id=conversation_id,
        )

    def _budget_line(self, *, context: RoleDrivenAttackContext) -> str:
        turn_number = context.executed_turns + 1
        return f"This is turn {turn_number} of {self._max_turns} ({self._max_turns - turn_number} remaining after it)."

    @staticmethod
    def _render_history(history: list[Exchange]) -> str:
        if not history:
            return "(no messages yet)"
        return "\n\n".join(f"User: {exchange.prompt}\nTarget: {exchange.response}" for exchange in history)

    @classmethod
    def _describe_response(cls, response: Message) -> str:
        """
        Describe an objective-target response in text for the agents.

        Args:
            response (Message): The response.

        Returns:
            str: The response text, or a short notice when it was blocked, errored or empty.
        """
        if any(piece.is_blocked() for piece in response.message_pieces):
            return "[The Target's response was blocked.]"
        if response.is_error():
            return "[The Target returned an error.]"
        pieces = response.get_pieces_by_type(data_type="text")
        text = "\n".join(piece.converted_value for piece in pieces if piece.converted_value)
        if not text:
            return "[The Target returned an empty response.]"
        if len(text) > cls.HISTORY_RESPONSE_CHAR_LIMIT:
            return text[: cls.HISTORY_RESPONSE_CHAR_LIMIT] + " [...]"
        return text

    @staticmethod
    def _score_value(score: Score | None) -> bool | float | None:
        if score is None or score.is_undetermined:
            return None
        return score.get_value()

    async def _send_prompt_to_objective_target_async(
        self, *, context: RoleDrivenAttackContext, message: Message
    ) -> Message:
        """
        Send a message to the objective target.

        Args:
            context (RoleDrivenAttackContext): The attack context.
            message (Message): The message to send.

        Returns:
            Message: The target's response.

        Raises:
            ValueError: If the target returns no response.
        """
        await self._prepare_target_conversation_async(context=context)

        with execution_context(
            component_role=ComponentRole.OBJECTIVE_TARGET,
            attack_strategy_name=self.__class__.__name__,
            component_identifier=self._objective_target.get_identifier(),
            objective_target_conversation_id=context.session.conversation_id,
            objective=context.objective,
        ):
            context._record_objective_target_invocation(conversation_id=context.session.conversation_id)
            response = await self._prompt_normalizer.send_prompt_async(
                message=message,
                conversation_id=context.session.conversation_id,
                request_converter_configurations=self._request_converters,
                response_converter_configurations=self._response_converters,
                target=self._objective_target,
                normalizer_overrides=self._get_prepended_normalizer_overrides(
                    prepended_history_send_context=context.prepended_history_send_context,
                ),
                send_context=context.prepended_history_send_context,
            )

        if response is None:
            raise ValueError(
                "Received no response from the target system. "
                "Please check the target configuration and ensure it is reachable."
            )
        return response

    async def _assess_response_async(
        self, *, context: RoleDrivenAttackContext, message: Message
    ) -> TurnAssessment | None:
        """
        Assess whether the objective target refused the message it was just sent.

        Args:
            context (RoleDrivenAttackContext): The attack context holding the latest response.
            message (Message): The message that was sent.

        Returns:
            TurnAssessment | None: The assessment, or None when no refusal scorer is configured.
        """
        if self._refusal_assessor is None or context.last_response is None:
            return None
        return await self._refusal_assessor.assess_async(
            response=context.last_response,
            prompt_sent=message.get_value(),
            attack_strategy_name=self.__class__.__name__,
            objective_target_conversation_id=context.session.conversation_id,
            objective=context.objective,
        )

    async def _score_response_async(self, *, context: RoleDrivenAttackContext) -> Score | None:
        """
        Evaluate the objective target's latest response with the objective scorer.

        Args:
            context (RoleDrivenAttackContext): The attack context holding the latest response.

        Returns:
            Score | None: The objective score, or None when there is nothing to score.
        """
        if not context.last_response:
            logger.warning("No response available in context to score")
            return None

        with execution_context(
            component_role=ComponentRole.UNKNOWN,
            attack_strategy_name=self.__class__.__name__,
            objective_target_conversation_id=context.session.conversation_id,
            objective=context.objective,
        ):
            scoring_results = await score_attack_response_async(
                response=context.last_response,
                objective_scorer=self._objective_scorer,
                auxiliary_scorers=self._auxiliary_scorers,
                expectation=context.expectation,
            )
        objective_scores = scoring_results["objective_scores"]
        return objective_scores[0] if objective_scores else None

    def _create_blocked_result(
        self, *, context: RoleDrivenAttackContext, blocked: AdversarialChatResponseBlockedException
    ) -> AttackResult:
        """
        Build the result for a run whose adversarial target was blocked.

        A block produced no new data, so the outcome is UNDETERMINED either way. Before any turn has run
        the objective target was never reached, which is recorded as a preparation failure.

        Args:
            context (RoleDrivenAttackContext): The attack context.
            blocked (AdversarialChatResponseBlockedException): The block.

        Returns:
            AttackResult: The UNDETERMINED result.
        """
        if context.executed_turns:
            return self._create_attack_result(
                context=context,
                outcome=AttackOutcome.UNDETERMINED,
                outcome_reason=(
                    f"Adversarial chat was blocked after {context.executed_turns} completed turn(s), "
                    f"truncating the attack before it reached its turn limit. Details: {blocked}"
                ),
            )
        kind = AttackPreparationFailureKind.from_exception(blocked)
        failure = AttackPreparationFailure(kind=kind, reason=f"{kind.default_reason} Details: {blocked}")
        return self._create_attack_result(
            context=context,
            outcome=AttackOutcome.UNDETERMINED,
            outcome_reason=failure.reason,
            metadata=failure.to_metadata(),
        )

    def _create_result_from_context(
        self, *, context: RoleDrivenAttackContext, stop_reason: str | None = None
    ) -> AttackResult:
        """
        Build the result for a run that reached its end.

        Args:
            context (RoleDrivenAttackContext): The attack context.
            stop_reason (str | None): Why the attack stopped early, if it did. Defaults to None.

        Returns:
            AttackResult: The result.
        """
        if context.last_score is not None:
            return self._create_attack_result(
                context=context,
                outcome=attack_outcome_from_score(context.last_score),
                outcome_reason=stop_reason,
            )
        if context.last_response is not None:
            return self._create_attack_result(
                context=context,
                outcome=AttackOutcome.UNDETERMINED,
                outcome_reason="Objective scorer returned no score",
            )
        return self._create_attack_result(context=context, outcome=AttackOutcome.FAILURE)

    def _create_attack_result(
        self,
        *,
        context: RoleDrivenAttackContext,
        outcome: AttackOutcome,
        outcome_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AttackResult:
        """
        Create a result from the current attack context.

        Args:
            context (RoleDrivenAttackContext): The attack context.
            outcome (AttackOutcome): The outcome.
            outcome_reason (str | None): Why the outcome was reached. Defaults to None.
            metadata (dict[str, Any] | None): Extra metadata. Defaults to None.

        Returns:
            AttackResult: The result, carrying the per-turn trace in its metadata.
        """
        return AttackResult(
            atomic_attack_identifier=AtomicAttackIdentifier.build(attack_identifier=self.get_identifier()),
            conversation_id=context.session.conversation_id,
            objective=context.objective,
            outcome=outcome,
            outcome_reason=outcome_reason,
            executed_turns=context.executed_turns,
            last_response=context.last_response.get_piece() if context.last_response else None,
            automated_score=context.last_score,
            related_conversations=context.related_conversations,
            metadata={**(metadata or {}), self.TRACE_METADATA_KEY: [record.to_dict() for record in context.trace]},
            labels=context.memory_labels,
        )

    async def _teardown_async(self, *, context: RoleDrivenAttackContext) -> None:
        """Clean up after attack execution."""
