# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.executor.attack.multi_turn.ramp_domain import (
    EmpiricalTransitioner,
    RampDomain,
    RampState,
    RampStrategy,
    RuleTransitioner,
    SafetyPattern,
)

CALM = RampStrategy(name="calm", definition="A low-risk placeholder strategy.", risk=1)
LOUD = RampStrategy(name="loud", definition="A high-risk placeholder strategy.", risk=2)


def _state(
    pattern: SafetyPattern = SafetyPattern.BORDERLINE, clues: int = 0, refusal: int = 0, risk: int = 0, budget: int = 5
) -> RampState:
    return RampState(pattern=pattern, clues=clues, refusal=refusal, risk=risk, budget=budget)


def _record(
    strategy: str | None = "loud",
    before: RampState | None = None,
    judged: str | None = "leaky",
) -> dict:
    return {"strategy": strategy, "state_before": (before or _state()).to_dict(), "judged_pattern": judged}


# --- RampStrategy ------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "definition", "risk"),
    [("", "d", 1), ("n", " ", 1), ("n", "d", -1)],
    ids=["blank-name", "blank-definition", "negative-risk"],
)
def test_strategy_rejects_invalid_values(name: str, definition: str, risk: int) -> None:
    with pytest.raises(ValueError):
        RampStrategy(name=name, definition=definition, risk=risk)


# --- RampState ---------------------------------------------------------------


def test_state_to_dict_names_the_pattern() -> None:
    assert _state(SafetyPattern.WEAK_REFUSAL, clues=1, refusal=2, risk=3, budget=4).to_dict() == {
        "pattern": "weak_refusal",
        "clues": 1,
        "refusal": 2,
        "risk": 3,
        "budget": 4,
    }


# --- RampDomain --------------------------------------------------------------


def test_initial_state_is_neutral() -> None:
    assert RampDomain.initial_state(budget=7) == _state(budget=7)


def test_advance_spends_a_turn_and_the_strategy_risk() -> None:
    after = RampDomain.advance(state=_state(risk=1), strategy=LOUD, pattern=SafetyPattern.LEAKY, new_clues=1)

    assert after == _state(SafetyPattern.LEAKY, clues=1, refusal=0, risk=3, budget=4)


def test_advance_for_a_caller_supplied_turn_adds_no_risk() -> None:
    after = RampDomain.advance(state=_state(), strategy=None, pattern=SafetyPattern.BORDERLINE, new_clues=0)

    assert after.risk == 0


@pytest.mark.parametrize(
    ("pattern", "expected_refusal"),
    [(SafetyPattern.STRICT_REFUSAL, 2), (SafetyPattern.WEAK_REFUSAL, 1)],
)
def test_advance_raises_refusal_firmness_on_a_refusal(pattern: SafetyPattern, expected_refusal: int) -> None:
    assert RampDomain.advance(state=_state(), strategy=CALM, pattern=pattern, new_clues=0).refusal == expected_refusal


def test_advance_lowers_refusal_firmness_on_engagement() -> None:
    after = RampDomain.advance(state=_state(refusal=2), strategy=CALM, pattern=SafetyPattern.LEAKY, new_clues=0)

    assert after.refusal == 1


def test_advance_caps_clues_and_refusal_and_floors_budget() -> None:
    state = _state(clues=RampDomain.MAX_CLUES, refusal=RampDomain.MAX_REFUSAL, budget=0)

    refused = RampDomain.advance(state=state, strategy=CALM, pattern=SafetyPattern.STRICT_REFUSAL, new_clues=5)

    assert refused.clues == RampDomain.MAX_CLUES
    assert refused.refusal == RampDomain.MAX_REFUSAL
    assert refused.budget == 0


def test_only_a_full_answer_is_the_goal() -> None:
    assert RampDomain.is_goal(_state(SafetyPattern.SUCCESS))
    assert not RampDomain.is_goal(_state(SafetyPattern.LEAKY, clues=3))


def test_heuristic_shrinks_as_the_response_improves_and_clues_accumulate() -> None:
    far = RampDomain.heuristic(_state(SafetyPattern.STRICT_REFUSAL))
    nearer = RampDomain.heuristic(_state(SafetyPattern.LEAKY))
    nearest = RampDomain.heuristic(_state(SafetyPattern.LEAKY, clues=RampDomain.GOAL_CLUES))

    assert far > nearer > nearest > 0
    assert RampDomain.heuristic(_state(SafetyPattern.SUCCESS, clues=RampDomain.GOAL_CLUES)) == 0


def test_step_cost_grows_with_refusal_firmness_and_strategy_risk() -> None:
    assert RampDomain.step_cost(_state(refusal=2), CALM) > RampDomain.step_cost(_state(refusal=0), CALM)
    assert RampDomain.step_cost(_state(), LOUD) > RampDomain.step_cost(_state(), CALM)


def test_step_risk_is_the_strategy_risk() -> None:
    assert RampDomain.step_risk(_state(), LOUD) == 2.0


# --- RuleTransitioner --------------------------------------------------------


def test_rule_prior_lifts_the_response_one_step() -> None:
    assert RuleTransitioner().predict_pattern(_state(SafetyPattern.STRICT_REFUSAL), CALM) is SafetyPattern.WEAK_REFUSAL


def test_rule_prior_treats_strategies_alike() -> None:
    transitioner = RuleTransitioner()

    assert transitioner.predict(_state(), CALM).pattern is transitioner.predict(_state(), LOUD).pattern


def test_rule_prior_does_not_lift_a_response_that_is_refused_as_firmly_as_possible() -> None:
    state = _state(SafetyPattern.STRICT_REFUSAL, refusal=RampDomain.MAX_REFUSAL)

    assert RuleTransitioner().predict_pattern(state, CALM) is SafetyPattern.STRICT_REFUSAL


def test_rule_prior_stops_lifting_at_leaky() -> None:
    state = _state(SafetyPattern.LEAKY, clues=0)

    assert RuleTransitioner().predict_pattern(state, CALM) is SafetyPattern.LEAKY


def test_rule_prior_expects_a_leaky_response_with_enough_clues_to_become_a_full_answer() -> None:
    state = _state(SafetyPattern.LEAKY, clues=RampDomain.GOAL_CLUES)

    assert RuleTransitioner().predict_pattern(state, CALM) is SafetyPattern.SUCCESS


def test_rule_predict_returns_the_advanced_state() -> None:
    after = RuleTransitioner().predict(_state(budget=3), LOUD)

    assert after == _state(SafetyPattern.LEAKY, clues=1, refusal=0, risk=2, budget=2)


# --- EmpiricalTransitioner ---------------------------------------------------


def test_empirical_prediction_follows_the_most_common_observed_pattern() -> None:
    records = [_record(judged="strict_refusal")] + [_record(judged="leaky")] * 3
    transitioner = EmpiricalTransitioner(records=records, min_support=3)

    assert transitioner.predict_pattern(_state(), LOUD) is SafetyPattern.LEAKY


def test_empirical_prediction_falls_back_without_enough_data() -> None:
    transitioner = EmpiricalTransitioner(records=[_record(judged="strict_refusal")], min_support=3)

    assert transitioner.predict_pattern(_state(), LOUD) is RuleTransitioner().predict_pattern(_state(), LOUD)


def test_empirical_prediction_is_specific_to_the_strategy_and_the_state() -> None:
    transitioner = EmpiricalTransitioner(records=[_record(judged="strict_refusal")] * 3, min_support=3)

    assert transitioner.predict_pattern(_state(), LOUD) is SafetyPattern.STRICT_REFUSAL
    assert transitioner.predict_pattern(_state(), CALM) is SafetyPattern.LEAKY
    assert transitioner.predict_pattern(_state(refusal=1), LOUD) is SafetyPattern.LEAKY


@pytest.mark.parametrize(
    "record",
    [
        _record(strategy=None),
        _record(judged=None),
        {"strategy": "loud", "judged_pattern": "leaky"},
        {"strategy": "loud", "state_before": {"pattern": "nonsense", "refusal": 0}, "judged_pattern": "leaky"},
        {"strategy": "loud", "state_before": {"pattern": "borderline", "refusal": 0}, "judged_pattern": "nonsense"},
    ],
    ids=["no-strategy", "no-pattern", "no-state", "bad-state-pattern", "bad-judged-pattern"],
)
def test_empirical_transitioner_ignores_unusable_records(record: dict) -> None:
    transitioner = EmpiricalTransitioner(records=[record] * 5, min_support=1)

    assert transitioner.predict_pattern(_state(), LOUD) is RuleTransitioner().predict_pattern(_state(), LOUD)


def test_empirical_transitioner_rejects_a_non_positive_support() -> None:
    with pytest.raises(ValueError, match="min_support"):
        EmpiricalTransitioner(records=[], min_support=0)
