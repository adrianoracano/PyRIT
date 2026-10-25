# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ruff: noqa: F401

"""Attack components module."""

from typing import TYPE_CHECKING

from pyrit.common.lazy_imports import get_lazy_dir, resolve_lazy_export

if TYPE_CHECKING:
    from pyrit.executor.attack.component.adversarial_conversation_manager import (
        AdversarialReply,
        AdversarialTurn,
        _AdversarialConversationManager,
    )
    from pyrit.executor.attack.component.conversation_manager import (
        ConversationManager,
        ConversationState,
        build_conversation_context_string_async,
        get_adversarial_chat_messages,
        get_prepended_turn_count,
        mark_messages_as_simulated,
    )
    from pyrit.executor.attack.component.experience_bank import (
        EmbeddingTextEmbedder,
        Experience,
        ExperienceBank,
        HashingTextEmbedder,
        TextEmbedder,
    )
    from pyrit.executor.attack.component.prepended_conversation_config import PrependedConversationConfig
    from pyrit.executor.attack.component.response_assessment import RefusalAssessor, TurnAssessment
    from pyrit.executor.attack.component.risk_constrained_planner import Plan, RiskConstrainedPlanner
    from pyrit.executor.attack.component.strategy_description import StrategyDescription
    from pyrit.executor.attack.component.structured_role import StructuredRole, parse_json_object

_LAZY_EXPORTS: dict[str, str | tuple[str, str | None]] = {
    "_AdversarialConversationManager": "pyrit.executor.attack.component.adversarial_conversation_manager",
    "AdversarialReply": "pyrit.executor.attack.component.adversarial_conversation_manager",
    "AdversarialTurn": "pyrit.executor.attack.component.adversarial_conversation_manager",
    "build_conversation_context_string_async": "pyrit.executor.attack.component.conversation_manager",
    "ConversationManager": "pyrit.executor.attack.component.conversation_manager",
    "ConversationState": "pyrit.executor.attack.component.conversation_manager",
    "EmbeddingTextEmbedder": "pyrit.executor.attack.component.experience_bank",
    "Experience": "pyrit.executor.attack.component.experience_bank",
    "ExperienceBank": "pyrit.executor.attack.component.experience_bank",
    "get_adversarial_chat_messages": "pyrit.executor.attack.component.conversation_manager",
    "HashingTextEmbedder": "pyrit.executor.attack.component.experience_bank",
    "get_prepended_turn_count": "pyrit.executor.attack.component.conversation_manager",
    "mark_messages_as_simulated": "pyrit.executor.attack.component.conversation_manager",
    "parse_json_object": "pyrit.executor.attack.component.structured_role",
    "PrependedConversationConfig": "pyrit.executor.attack.component.prepended_conversation_config",
    "Plan": "pyrit.executor.attack.component.risk_constrained_planner",
    "RefusalAssessor": "pyrit.executor.attack.component.response_assessment",
    "RiskConstrainedPlanner": "pyrit.executor.attack.component.risk_constrained_planner",
    "StrategyDescription": "pyrit.executor.attack.component.strategy_description",
    "StructuredRole": "pyrit.executor.attack.component.structured_role",
    "TextEmbedder": "pyrit.executor.attack.component.experience_bank",
    "TurnAssessment": "pyrit.executor.attack.component.response_assessment",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> object:
    return resolve_lazy_export(
        name=name,
        module_name=__name__,
        module_globals=globals(),
        exports=_LAZY_EXPORTS,
    )


def __dir__() -> list[str]:
    return get_lazy_dir(module_globals=globals(), exports=_LAZY_EXPORTS)
