# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Schema-validated JSON exchange with a chat target playing a named role in an attack."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Generic, TypeVar
from uuid import uuid4

from pyrit.exceptions import (
    ComponentRole,
    InvalidJsonException,
    PyritException,
    execution_context,
    remove_markdown_json,
)
from pyrit.models import JsonResponseConfig, JsonSchemaDefinition, Message
from pyrit.prompt_normalizer import PromptNormalizer, send_json_with_retry_async

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyrit.prompt_target import PromptTarget

logger = logging.getLogger(__name__)

T = TypeVar("T")


def camel_to_snake(name: str) -> str:
    """
    Convert a ``camelCase`` or ``PascalCase`` identifier to ``snake_case``.

    Args:
        name (str): The identifier to convert.

    Returns:
        str: The snake_case form of the identifier.
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def build_json_prompt_metadata(*, response_json_schema: JsonSchemaDefinition | None) -> dict[str, Any]:
    """
    Build the request metadata that asks a target for a JSON reply of an optional shape.

    When a schema is declared, returns ``response_format`` plus the schema under
    ``JSON_SCHEMA_METADATA_KEY`` so schema-aware targets can natively constrain the reply.
    When no schema is declared, returns an empty dict so the raw-text behavior is unchanged.

    Args:
        response_json_schema (JsonSchemaDefinition | None): The schema to forward, or None.

    Returns:
        dict[str, Any]: The prompt metadata (empty when no schema).
    """
    if response_json_schema is None:
        return {}
    return JsonResponseConfig(enabled=True, json_schema=response_json_schema).to_metadata()


def parse_json_object(response_text: str, *, schema: JsonSchemaDefinition | None = None) -> dict[str, Any]:
    """
    Decode a reply into a JSON object and validate its keys against ``schema``.

    Markdown code fences are stripped and keys are normalized from camelCase to snake_case before
    validation, so a backend that drifts to ``nextMessage`` still parses without burning a retry.
    Required and permitted keys are read from ``schema`` itself (its ``required`` list and
    ``properties`` map, honoring ``additionalProperties``) so the schema stays the single source of
    truth. When ``schema`` is None only the "decodes to an object" invariant is enforced.

    Args:
        response_text (str): The raw reply text.
        schema (JsonSchemaDefinition | None): The schema to validate against, or None.

    Returns:
        dict[str, Any]: The decoded object with snake_case keys.

    Raises:
        InvalidJsonException: If the reply is not valid JSON, does not decode to an object, is
            missing a required key, or carries a key the schema forbids.
    """
    cleaned = remove_markdown_json(response_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise InvalidJsonException(message=f"Invalid JSON encountered: {cleaned}") from e

    if not isinstance(parsed, dict):
        raise InvalidJsonException(
            message=f"Response must be a JSON object, but decoded to {type(parsed).__name__}: {cleaned}"
        )

    normalized = {camel_to_snake(key): value for key, value in parsed.items()}
    if schema is None:
        return normalized

    required_keys = {camel_to_snake(key) for key in schema.get("required", [])}
    missing_keys = required_keys - normalized.keys()
    if missing_keys:
        raise InvalidJsonException(message=f"Missing required keys {missing_keys} in JSON response: {cleaned}")

    if schema.get("additionalProperties", True) is False:
        allowed_keys = {camel_to_snake(key) for key in schema.get("properties", {})}
        extra_keys = normalized.keys() - allowed_keys
        if extra_keys:
            raise InvalidJsonException(message=f"Unexpected keys {extra_keys} found in JSON response: {cleaned}")

    return normalized


def raise_for_role_error(response: Message, *, role_label: str) -> None:
    """
    Raise a ``PyritException`` when a role's reply carries an error.

    This is the default error translation for ``StructuredRole``. Roles whose callers branch on
    specific failure categories (such as the adversarial chat) inject their own translation instead.

    Args:
        response (Message): The reply to inspect.
        role_label (str): Human-readable role name used in the error message.

    Raises:
        PyritException: If any piece of the reply represents an error.
    """
    if not response.is_error():
        return
    detail = next(
        (piece.converted_value for piece in response.message_pieces if piece.has_error()),
        "unknown error",
    )
    raise PyritException(message=f"{role_label} returned an error: {detail}")


class StructuredRole(Generic[T]):
    """
    A chat target playing one named role in an attack, exchanging schema-validated JSON.

    The role owns the mechanics every LLM-backed attack role repeats: asking the target for a JSON
    reply (forwarding the schema as request metadata), tagging the call with an execution context,
    rolling back and retrying malformed replies on a clean conversation, and parsing the reply into
    a typed value. It holds no conversation state of its own: callers decide whether a call joins an
    existing conversation (``send_async``) or stands alone (``ask_async``).
    """

    def __init__(
        self,
        *,
        target: PromptTarget,
        parse: Callable[[str], T],
        response_json_schema: JsonSchemaDefinition | None = None,
        component_role: ComponentRole = ComponentRole.UNKNOWN,
        role_label: str = "Structured role",
        raise_for_error: Callable[[Message], None] | None = None,
        prompt_normalizer: PromptNormalizer | None = None,
        attack_strategy_name: str | None = None,
        objective: str | None = None,
        objective_target_conversation_id: str | None = None,
    ) -> None:
        """
        Initialize the role.

        Args:
            target (PromptTarget): The chat target playing the role.
            parse (Callable[[str], T]): Turns the reply text into the typed result. Must raise
                ``InvalidJsonException`` on a bad reply to trigger a retry; build it on
                ``parse_json_object`` for schema-validated objects.
            response_json_schema (JsonSchemaDefinition | None): Schema forwarded to the target as
                request metadata so schema-aware targets can constrain their reply. Defaults to None.
            component_role (ComponentRole): Role tag for the execution context. Defaults to UNKNOWN.
            role_label (str): Name used in default error messages. Defaults to "Structured role".
            raise_for_error (Callable[[Message], None] | None): Raises a terminal exception when a
                reply carries an error. Defaults to a generic ``PyritException``.
            prompt_normalizer (PromptNormalizer | None): Normalizer to send through. Defaults to a
                new one.
            attack_strategy_name (str | None): Calling attack's name, for the execution context.
            objective (str | None): The attack objective, for the execution context.
            objective_target_conversation_id (str | None): The objective target's conversation id,
                for execution-context correlation.
        """
        self._target = target
        self._parse = parse
        self._response_json_schema = response_json_schema
        self._component_role = component_role
        self._role_label = role_label
        self._raise_for_error = raise_for_error or self._raise_for_default_error
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()
        self._attack_strategy_name = attack_strategy_name
        self._objective = objective
        self._objective_target_conversation_id = objective_target_conversation_id

    @property
    def target(self) -> PromptTarget:
        """The chat target playing the role."""
        return self._target

    @property
    def response_json_schema(self) -> JsonSchemaDefinition | None:
        """The schema forwarded to the target, if any."""
        return self._response_json_schema

    def build_message(self, *, prompt_text: str) -> Message:
        """
        Build a user message that asks for this role's JSON reply.

        Args:
            prompt_text (str): The text to send.

        Returns:
            Message: A user message carrying the schema request metadata.
        """
        prompt_metadata = build_json_prompt_metadata(response_json_schema=self._response_json_schema)
        return Message.from_prompt(prompt=prompt_text, role="user", prompt_metadata=prompt_metadata or None)

    async def send_async(self, *, message: Message, conversation_id: str) -> T:
        """
        Send a message on an existing conversation and parse the reply.

        Each retry runs on a clean conversation history: the failed turn is rolled back out of
        memory before the message is resent, instead of replaying the model's own malformed reply.

        Args:
            message (Message): The message to send, already carrying any schema request metadata.
            conversation_id (str): The conversation to send on.

        Returns:
            T: The parsed reply.

        Raises:
            PyritException: If the reply carries an error (as translated by ``raise_for_error``).
            ValueError: If the target returns no response.
            InvalidJsonException: If the reply still fails to parse after the retry budget.
        """

        def _parse_response(response: Message) -> T:
            self._raise_for_error(response)
            return self._parse(response.get_value())

        with execution_context(
            component_role=self._component_role,
            attack_strategy_name=self._attack_strategy_name,
            component_identifier=self._target.get_identifier(),
            objective_target_conversation_id=self._objective_target_conversation_id,
            objective=self._objective,
        ):
            return await send_json_with_retry_async(
                normalizer=self._prompt_normalizer,
                target=self._target,
                message=message,
                conversation_id=conversation_id,
                parse=_parse_response,
            )

    async def ask_async(
        self,
        *,
        prompt_text: str,
        system_prompt: str | None = None,
        conversation_id: str | None = None,
    ) -> T:
        """
        Ask the role one self-contained question and parse the reply.

        With no ``conversation_id`` the call runs on a fresh conversation, so the role sees only
        ``system_prompt`` and ``prompt_text`` and nothing it said before. A caller that wants to track
        the conversation supplies its own new id. ``system_prompt`` can only be set on a conversation
        with no messages, so pair it with a new id, never with one already in use.

        Args:
            prompt_text (str): The text to send.
            system_prompt (str | None): Rendered system prompt to set before sending. Defaults to None.
            conversation_id (str | None): Conversation to use. Defaults to a new one.

        Returns:
            T: The parsed reply.
        """
        resolved_conversation_id = conversation_id or str(uuid4())
        if system_prompt is not None:
            await self._target.set_system_prompt_async(
                system_prompt=system_prompt, conversation_id=resolved_conversation_id
            )
        return await self.send_async(
            message=self.build_message(prompt_text=prompt_text), conversation_id=resolved_conversation_id
        )

    def _raise_for_default_error(self, response: Message) -> None:
        raise_for_role_error(response, role_label=self._role_label)
