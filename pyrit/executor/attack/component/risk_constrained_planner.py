# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Finite-horizon best-first planning over a symbolic state space with a risk budget."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence

StateT = TypeVar("StateT", bound="Hashable")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True)
class Plan(Generic[StateT, ActionT]):
    """
    A sequence of actions and where it is expected to lead.

    ``reached_goal`` is False when no plan within the horizon and risk budget reached the goal. The plan is
    then the one that gets closest to it.
    """

    actions: tuple[ActionT, ...]
    final_state: StateT
    cost: float
    risk: float
    reached_goal: bool

    @property
    def first_action(self) -> ActionT | None:
        """The action to take now, or None when the plan is empty."""
        return self.actions[0] if self.actions else None


@dataclass(frozen=True)
class _Label(Generic[StateT, ActionT]):
    state: StateT
    actions: tuple[ActionT, ...]
    cost: float
    risk: float


class RiskConstrainedPlanner(Generic[StateT, ActionT]):
    """
    Searches action sequences that reach a goal state within a horizon, cheapest first, under a risk budget.

    This is a label-setting best-first search. A label is a partial plan: the state it reaches and its
    accumulated cost and risk. Labels are expanded in order of cost plus a heuristic estimate of the
    remaining distance to the goal. A label is dropped when it exceeds the risk budget, or when another
    label reaches the same state with no more cost and no more risk. The search ends at the first goal
    label, or when it runs out of labels, or after ``max_expansions`` expansions.

    The planner is deterministic and calls no model, so it can be tested without one. The domain is
    supplied as functions: ``transition`` predicts the state an action leads to, ``step_cost`` and
    ``step_risk`` price an action in a state, and ``is_goal`` recognises the goal.
    """

    def __init__(
        self,
        *,
        actions: Sequence[ActionT],
        transition: Callable[[StateT, ActionT], StateT],
        step_cost: Callable[[StateT, ActionT], float],
        step_risk: Callable[[StateT, ActionT], float],
        is_goal: Callable[[StateT], bool],
        heuristic: Callable[[StateT], float] | None = None,
        max_expansions: int = 2000,
    ) -> None:
        """
        Initialize the planner.

        Args:
            actions (Sequence[ActionT]): The actions available in every state, in preference order. Ties
                in the search are broken in favor of earlier actions.
            transition (Callable[[StateT, ActionT], StateT]): Predicts the state an action leads to.
            step_cost (Callable[[StateT, ActionT], float]): The cost of taking an action in a state.
            step_risk (Callable[[StateT, ActionT], float]): The risk of taking an action in a state.
            is_goal (Callable[[StateT], bool]): Whether a state is the goal.
            heuristic (Callable[[StateT], float] | None): Estimated remaining cost to the goal, used to
                order the search and to rank plans that do not reach it. Defaults to zero everywhere.
            max_expansions (int): Upper bound on label expansions, which bounds the search time.
                Defaults to 2000.

        Raises:
            ValueError: If there are no actions or ``max_expansions`` is not positive.
        """
        if not actions:
            raise ValueError("The planner needs at least one action.")
        if max_expansions <= 0:
            raise ValueError("max_expansions must be positive.")
        self._actions = tuple(actions)
        self._transition = transition
        self._step_cost = step_cost
        self._step_risk = step_risk
        self._is_goal = is_goal
        self._heuristic = heuristic or (lambda _state: 0.0)
        self._max_expansions = max_expansions

    def plan(self, *, initial: StateT, horizon: int, max_risk: float) -> Plan[StateT, ActionT] | None:
        """
        Find the best plan from ``initial``.

        Args:
            initial (StateT): The current state.
            horizon (int): The most actions a plan may contain.
            max_risk (float): The most risk a plan may accumulate.

        Returns:
            Plan[StateT, ActionT] | None: The cheapest plan that reaches the goal; failing that, the plan
                that ends closest to it; or None when every first action exceeds the risk budget.
        """
        root = _Label[StateT, ActionT](state=initial, actions=(), cost=0.0, risk=0.0)
        if self._is_goal(initial):
            return self._to_plan(root, reached_goal=True)

        counter = itertools.count()
        frontier: list[tuple[float, int, _Label[StateT, ActionT]]] = [(self._priority(root), next(counter), root)]
        frontier_labels: dict[StateT, list[_Label[StateT, ActionT]]] = {initial: [root]}
        best_partial: _Label[StateT, ActionT] | None = None

        for _ in range(self._max_expansions):
            if not frontier:
                break
            _, _, label = heapq.heappop(frontier)
            if self._is_goal(label.state):
                return self._to_plan(label, reached_goal=True)
            if len(label.actions) >= horizon:
                best_partial = self._closer(best_partial, label)
                continue
            for action in self._actions:
                child = self._expand(label, action)
                if child.risk > max_risk or self._is_dominated(child, frontier_labels):
                    continue
                frontier_labels.setdefault(child.state, []).append(child)
                heapq.heappush(frontier, (self._priority(child), next(counter), child))
                if child.actions:
                    best_partial = self._closer(best_partial, child)

        return self._to_plan(best_partial, reached_goal=False) if best_partial is not None else None

    def _expand(self, label: _Label[StateT, ActionT], action: ActionT) -> _Label[StateT, ActionT]:
        return _Label(
            state=self._transition(label.state, action),
            actions=(*label.actions, action),
            cost=label.cost + self._step_cost(label.state, action),
            risk=label.risk + self._step_risk(label.state, action),
        )

    def _priority(self, label: _Label[StateT, ActionT]) -> float:
        return label.cost + self._heuristic(label.state)

    @staticmethod
    def _is_dominated(candidate: _Label[StateT, ActionT], known: dict[StateT, list[_Label[StateT, ActionT]]]) -> bool:
        return any(
            other.cost <= candidate.cost and other.risk <= candidate.risk for other in known.get(candidate.state, [])
        )

    def _closer(
        self, current: _Label[StateT, ActionT] | None, candidate: _Label[StateT, ActionT]
    ) -> _Label[StateT, ActionT]:
        if current is None:
            return candidate
        current_key = (self._heuristic(current.state), current.cost, current.risk)
        candidate_key = (self._heuristic(candidate.state), candidate.cost, candidate.risk)
        return candidate if candidate_key < current_key else current

    @staticmethod
    def _to_plan(label: _Label[StateT, ActionT], *, reached_goal: bool) -> Plan[StateT, ActionT]:
        return Plan(
            actions=label.actions,
            final_state=label.state,
            cost=label.cost,
            risk=label.risk,
            reached_goal=reached_goal,
        )
