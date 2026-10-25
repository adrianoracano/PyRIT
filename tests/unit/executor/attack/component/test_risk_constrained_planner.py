# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass

import pytest

from pyrit.executor.attack.component.risk_constrained_planner import Plan, RiskConstrainedPlanner


@dataclass(frozen=True)
class _Step:
    name: str
    gain: int
    cost: float
    risk: float


SMALL = _Step("small", gain=1, cost=1.0, risk=1.0)
LEAP = _Step("leap", gain=2, cost=3.0, risk=0.0)
GOAL = 3


def _planner(*actions: _Step, max_expansions: int = 2000) -> RiskConstrainedPlanner[int, _Step]:
    return RiskConstrainedPlanner(
        actions=actions or (SMALL, LEAP),
        transition=lambda state, action: min(GOAL, state + action.gain),
        step_cost=lambda _state, action: action.cost,
        step_risk=lambda _state, action: action.risk,
        is_goal=lambda state: state == GOAL,
        heuristic=lambda state: float(GOAL - state),
        max_expansions=max_expansions,
    )


def _names(plan: Plan[int, _Step] | None) -> list[str]:
    assert plan is not None
    return [action.name for action in plan.actions]


def test_plan_returns_the_cheapest_plan_that_reaches_the_goal() -> None:
    plan = _planner().plan(initial=0, horizon=3, max_risk=10)

    assert _names(plan) == ["small", "small", "small"]
    assert plan is not None and plan.reached_goal and plan.cost == 3.0 and plan.risk == 3.0


def test_plan_stays_inside_the_risk_budget() -> None:
    plan = _planner().plan(initial=0, horizon=3, max_risk=2)

    assert plan is not None and plan.reached_goal
    assert sorted(_names(plan)) == ["leap", "small"]
    assert plan.risk == 1.0 and plan.cost == 4.0


def test_plan_respects_the_horizon() -> None:
    plan = _planner().plan(initial=0, horizon=2, max_risk=10)

    assert plan is not None and plan.reached_goal
    assert len(plan.actions) == 2


def test_plan_that_cannot_reach_the_goal_is_the_one_that_ends_closest() -> None:
    plan = _planner().plan(initial=0, horizon=1, max_risk=10)

    assert plan is not None
    assert not plan.reached_goal
    assert _names(plan) == ["leap"]
    assert plan.final_state == 2


def test_plan_is_none_when_every_first_action_exceeds_the_risk_budget() -> None:
    assert _planner(SMALL).plan(initial=0, horizon=3, max_risk=0.5) is None


def test_plan_from_the_goal_is_empty_and_reached() -> None:
    plan = _planner().plan(initial=GOAL, horizon=3, max_risk=10)

    assert plan is not None
    assert plan.actions == ()
    assert plan.first_action is None
    assert plan.reached_goal


def test_first_action_is_the_head_of_the_plan() -> None:
    plan = _planner().plan(initial=0, horizon=3, max_risk=10)

    assert plan is not None
    assert plan.first_action is SMALL


def test_ties_are_broken_in_favor_of_earlier_actions() -> None:
    twin = _Step("twin", gain=1, cost=1.0, risk=1.0)

    plan = _planner(SMALL, twin).plan(initial=0, horizon=3, max_risk=10)

    assert _names(plan) == ["small", "small", "small"]


def test_dominated_labels_are_not_expanded() -> None:
    idle = _Step("idle", gain=0, cost=1.0, risk=0.0)
    calls: list[int] = []

    def transition(state: int, action: _Step) -> int:
        calls.append(state)
        return min(GOAL, state + action.gain)

    planner = RiskConstrainedPlanner(
        actions=(idle, SMALL),
        transition=transition,
        step_cost=lambda _state, action: action.cost,
        step_risk=lambda _state, action: action.risk,
        is_goal=lambda state: state == GOAL,
        heuristic=lambda state: float(GOAL - state),
    )

    plan = planner.plan(initial=0, horizon=12, max_risk=100)

    # Idling returns to a state already reached more cheaply, so it is dominated. Without that pruning
    # the search would branch over every sequence of twelve actions.
    assert plan is not None and plan.reached_goal
    assert len(calls) < 100


def test_search_is_bounded_by_max_expansions() -> None:
    plan = _planner(max_expansions=1).plan(initial=0, horizon=3, max_risk=10)

    assert plan is not None
    assert not plan.reached_goal


def test_planner_rejects_an_empty_action_set() -> None:
    with pytest.raises(ValueError, match="at least one action"):
        RiskConstrainedPlanner(
            actions=(),
            transition=lambda state, _action: state,
            step_cost=lambda _state, _action: 1.0,
            step_risk=lambda _state, _action: 0.0,
            is_goal=lambda _state: False,
        )


def test_planner_rejects_a_non_positive_expansion_bound() -> None:
    with pytest.raises(ValueError, match="max_expansions"):
        _planner(max_expansions=0)
