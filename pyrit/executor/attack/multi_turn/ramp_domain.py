# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""The symbolic state and transition models RAMP plans over."""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class SafetyPattern(enum.IntEnum):
    """How safely the objective target responded, from firmest refusal to full compliance."""

    STRICT_REFUSAL = 0
    WEAK_REFUSAL = 1
    BORDERLINE = 2
    LEAKY = 3
    SUCCESS = 4


@dataclass(frozen=True)
class RampStrategy:
    """
    One high-level strategy the planner can choose for a turn.

    PyRIT ships no strategies. The caller supplies the catalog, so what is tested is chosen
    deliberately. ``risk`` is how much applying the strategy adds to the attack's risk budget.
    """

    name: str
    definition: str
    risk: int = 1

    def __post_init__(self) -> None:
        """
        Validate the strategy.

        Raises:
            ValueError: If the name or definition is blank or the risk is negative.
        """
        if not self.name.strip() or not self.definition.strip():
            raise ValueError("A RampStrategy needs a name and a definition.")
        if self.risk < 0:
            raise ValueError("A RampStrategy risk cannot be negative.")


@dataclass(frozen=True)
class RampState:
    """
    The attack's position, summarised in a few discrete components so it can be planned over.

    ``pattern`` is how the target last responded, ``clues`` how many task-relevant clues have been
    gathered (capped), ``refusal`` how firmly the target has been refusing (capped), ``risk`` the
    risk spent so far, and ``budget`` the turns left.
    """

    pattern: SafetyPattern
    clues: int
    refusal: int
    risk: int
    budget: int

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the state to a JSON-serializable dict.

        Returns:
            dict[str, Any]: The state's components, with the pattern by name.
        """
        return {
            "pattern": self.pattern.name.lower(),
            "clues": self.clues,
            "refusal": self.refusal,
            "risk": self.risk,
            "budget": self.budget,
        }


class RampDomain:
    """The rules of the planning problem: how a turn changes the state, what it costs, and the goal."""

    MAX_CLUES: ClassVar[int] = 3
    MAX_REFUSAL: ClassVar[int] = 3

    # Clues needed before a leaky response is expected to turn into a full answer.
    GOAL_CLUES: ClassVar[int] = 2

    @classmethod
    def initial_state(cls, *, budget: int) -> RampState:
        """
        Build the state before anything is known about the target.

        Args:
            budget (int): The turns available.

        Returns:
            RampState: A neutral state with no clues, no refusals and no risk spent.
        """
        return RampState(pattern=SafetyPattern.BORDERLINE, clues=0, refusal=0, risk=0, budget=budget)

    @classmethod
    def advance(
        cls,
        *,
        state: RampState,
        strategy: RampStrategy | None,
        pattern: SafetyPattern,
        new_clues: int,
    ) -> RampState:
        """
        Apply the outcome of one turn to a state.

        Args:
            state (RampState): The state before the turn.
            strategy (RampStrategy | None): The strategy used, or None for a turn the caller supplied.
            pattern (SafetyPattern): How the target responded.
            new_clues (int): How many new clues the response revealed.

        Returns:
            RampState: The state after the turn.
        """
        if pattern <= SafetyPattern.WEAK_REFUSAL:
            firmness = 2 if pattern is SafetyPattern.STRICT_REFUSAL else 1
            refusal = min(cls.MAX_REFUSAL, state.refusal + firmness)
        else:
            refusal = max(0, state.refusal - 1)
        return RampState(
            pattern=pattern,
            clues=min(cls.MAX_CLUES, state.clues + new_clues),
            refusal=refusal,
            risk=state.risk + (strategy.risk if strategy is not None else 0),
            budget=max(0, state.budget - 1),
        )

    @staticmethod
    def is_goal(state: RampState) -> bool:
        """
        Whether the state is the planning goal, a response that fully complies.

        Args:
            state (RampState): The state to check.

        Returns:
            bool: True for a fully compliant response.
        """
        return state.pattern is SafetyPattern.SUCCESS

    @classmethod
    def heuristic(cls, state: RampState) -> float:
        """
        Estimate how far a state is from the goal.

        Args:
            state (RampState): The state to estimate from.

        Returns:
            float: Larger when the response is firmer and fewer clues are gathered.
        """
        return float(SafetyPattern.SUCCESS - state.pattern) + 0.5 * max(0, cls.GOAL_CLUES - state.clues)

    @staticmethod
    def step_cost(state: RampState, strategy: RampStrategy) -> float:
        """
        Price a turn: pushing on a firm refusal, or with a risky strategy, costs more.

        Args:
            state (RampState): The state the turn starts from.
            strategy (RampStrategy): The strategy applied.

        Returns:
            float: The cost of the turn.
        """
        return 1.0 + 0.5 * state.refusal + 0.5 * strategy.risk

    @staticmethod
    def step_risk(state: RampState, strategy: RampStrategy) -> float:
        """
        Say how much risk a turn adds.

        Args:
            state (RampState): The state the turn starts from.
            strategy (RampStrategy): The strategy applied.

        Returns:
            float: The strategy's risk.
        """
        return float(strategy.risk)


class RampTransitioner(Protocol):
    """Predicts the state a strategy is expected to lead to."""

    def predict(self, state: RampState, strategy: RampStrategy) -> RampState:
        """
        Predict the state after applying ``strategy`` in ``state``.

        Args:
            state (RampState): The current state.
            strategy (RampStrategy): The strategy to apply.

        Returns:
            RampState: The expected state.
        """
        ...  # pragma: no cover


class RuleTransitioner:
    """
    A neutral prior over how strategies change the target's response.

    Without data nothing is assumed about which strategy works against which response, so every
    strategy is expected to lift the response by one step unless the target has been refusing as firmly
    as it can, and strategies differ only in the risk they add. A leaky response with enough clues is
    expected to turn into a full answer. ``EmpiricalTransitioner`` replaces this prior with what past
    attacks observed.
    """

    def predict_pattern(self, state: RampState, strategy: RampStrategy) -> SafetyPattern:
        """
        Predict how the target will respond.

        Args:
            state (RampState): The current state.
            strategy (RampStrategy): The strategy to apply.

        Returns:
            SafetyPattern: The expected response pattern.
        """
        if state.pattern is SafetyPattern.SUCCESS:
            return SafetyPattern.SUCCESS
        if state.pattern is SafetyPattern.LEAKY and state.clues >= RampDomain.GOAL_CLUES:
            return SafetyPattern.SUCCESS
        lift = 0 if state.refusal >= RampDomain.MAX_REFUSAL else 1
        return SafetyPattern(min(SafetyPattern.LEAKY, state.pattern + lift))

    def predict(self, state: RampState, strategy: RampStrategy) -> RampState:
        """
        Predict the state after applying ``strategy`` in ``state``.

        Args:
            state (RampState): The current state.
            strategy (RampStrategy): The strategy to apply.

        Returns:
            RampState: The expected state.
        """
        pattern = self.predict_pattern(state, strategy)
        new_clues = 1 if pattern >= SafetyPattern.BORDERLINE else 0
        return RampDomain.advance(state=state, strategy=strategy, pattern=pattern, new_clues=new_clues)


class EmpiricalTransitioner(RuleTransitioner):
    """
    Predicts the response from what past attacks observed, falling back to the rule prior.

    For each combination of response pattern, refusal firmness and strategy it counts the patterns that
    followed in the supplied records and predicts the most common one, once it has seen enough.
    """

    def __init__(
        self,
        *,
        records: Iterable[Mapping[str, Any]],
        min_support: int = 3,
        fallback: RuleTransitioner | None = None,
    ) -> None:
        """
        Initialize the transitioner from past attack traces.

        Args:
            records (Iterable[Mapping[str, Any]]): Per-turn trace entries from earlier RAMP attack results
                (``AttackResult.metadata["ramp_trace"]``). Entries without a strategy or judged pattern
                are ignored.
            min_support (int): Observations needed before a combination overrides the prior. Defaults to 3.
            fallback (RuleTransitioner | None): Used for combinations with too little data. Defaults to a
                ``RuleTransitioner``.

        Raises:
            ValueError: If ``min_support`` is not positive.
        """
        if min_support <= 0:
            raise ValueError("min_support must be positive.")
        self._min_support = min_support
        self._fallback = fallback or RuleTransitioner()
        self._counts: dict[tuple[SafetyPattern, int, str], Counter[SafetyPattern]] = {}
        for record in records:
            self._observe(record)

    def _observe(self, record: Mapping[str, Any]) -> None:
        strategy = record.get("strategy")
        before = record.get("state_before")
        judged = record.get("judged_pattern")
        if not strategy or not isinstance(before, dict) or not judged:
            return
        try:
            key = (SafetyPattern[str(before["pattern"]).upper()], int(before["refusal"]), str(strategy))
            observed = SafetyPattern[str(judged).upper()]
        except (KeyError, ValueError):
            return
        self._counts.setdefault(key, Counter())[observed] += 1

    def predict_pattern(self, state: RampState, strategy: RampStrategy) -> SafetyPattern:
        """
        Predict how the target will respond.

        Args:
            state (RampState): The current state.
            strategy (RampStrategy): The strategy to apply.

        Returns:
            SafetyPattern: The most common observed pattern, or the prior's when there is too little data.
        """
        counts = self._counts.get((state.pattern, state.refusal, strategy.name))
        if counts is None or sum(counts.values()) < self._min_support:
            return self._fallback.predict_pattern(state, strategy)
        return max(counts, key=lambda pattern: (counts[pattern], pattern))
