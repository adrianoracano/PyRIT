# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Per-turn assessment of the objective target's response, separate from objective scoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pyrit.exceptions import ComponentRole, execution_context
from pyrit.models import MessageScorable, ScoringExpectation
from pyrit.score.score_utils import score_is_true

if TYPE_CHECKING:
    from pyrit.models import Message, Score
    from pyrit.score import TrueFalseScorer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnAssessment:
    """
    What an attack learned about one objective-target response, apart from the objective score.

    ``refusal_score`` is the refusal scorer's verdict for the response. An undetermined verdict
    counts as not a refusal, matching how Crescendo treats it.
    """

    refusal_score: Score

    @property
    def is_refusal(self) -> bool:
        """Whether the response was judged a refusal."""
        return score_is_true(self.refusal_score)

    def template_values(self) -> dict[str, Any]:
        """
        Variables this assessment adds to the adversarial per-turn template.

        Returns:
            dict[str, Any]: The template variables (``was_refusal``).
        """
        return {"was_refusal": self.is_refusal}


class RefusalAssessor:
    """
    Decides whether the objective target refused a prompt, using a refusal scorer.

    This is the one place an attack runs its refusal scorer, so the execution context tagging and the
    refusal-against-the-prompt-sent convention are shared by every attack that reacts to refusals
    (backtracking in Crescendo, refusal-aware feedback in Red Teaming) instead of re-implemented.
    The assessor only measures; the attack decides what a refusal means.
    """

    def __init__(self, *, refusal_scorer: TrueFalseScorer) -> None:
        """
        Initialize the assessor.

        Args:
            refusal_scorer (TrueFalseScorer): Scorer returning True when a response is a refusal.
        """
        self._refusal_scorer = refusal_scorer

    @property
    def refusal_scorer(self) -> TrueFalseScorer:
        """The refusal scorer in use."""
        return self._refusal_scorer

    async def assess_async(
        self,
        *,
        response: Message,
        prompt_sent: str,
        attack_strategy_name: str,
        objective_target_conversation_id: str,
        objective: str,
    ) -> TurnAssessment:
        """
        Judge whether ``response`` is a refusal of ``prompt_sent``.

        Args:
            response (Message): The objective target's response.
            prompt_sent (str): The prompt the response answers; refusal is judged relative to it.
            attack_strategy_name (str): Calling attack's name, for the execution context.
            objective_target_conversation_id (str): The objective target's conversation id, for the
                execution context.
            objective (str): The attack objective, for the execution context.

        Returns:
            TurnAssessment: The refusal verdict.
        """
        with execution_context(
            component_role=ComponentRole.REFUSAL_SCORER,
            attack_strategy_name=attack_strategy_name,
            component_identifier=self._refusal_scorer.get_identifier(),
            objective_target_conversation_id=objective_target_conversation_id,
            objective=objective,
        ):
            scores = await self._refusal_scorer.score_async(
                scorable=MessageScorable.from_message(response),
                expectation=ScoringExpectation(objective=prompt_sent),
            )
        assessment = TurnAssessment(refusal_score=scores[0])
        logger.debug(
            "Refusal check: %s - %s...", assessment.is_refusal, (assessment.refusal_score.score_rationale or "")[:100]
        )
        return assessment
