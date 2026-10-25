# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock

import pytest
from unit.mocks import get_mock_prompt_normalizer

from pyrit.exceptions import ComponentRole, InvalidJsonException, PyritException
from pyrit.executor.attack.component.structured_role import (
    StructuredRole,
    build_json_prompt_metadata,
    camel_to_snake,
    parse_json_object,
    raise_for_role_error,
)
from pyrit.models import JSON_SCHEMA_METADATA_KEY, ComponentIdentifier, Message, MessagePiece
from pyrit.prompt_target import PromptTarget

pytestmark = pytest.mark.usefixtures("patch_central_database")

SCHEMA: dict = {
    "type": "object",
    "properties": {"mode": {"type": "string"}, "framing": {"type": "string"}},
    "required": ["mode", "framing"],
    "additionalProperties": False,
}

OPEN_SCHEMA: dict = {"type": "object", "properties": {"mode": {"type": "string"}}}

VALID_JSON = '{"mode": "ATTACK", "framing": "academic"}'


def _target() -> MagicMock:
    target = MagicMock(spec=PromptTarget)
    target.get_identifier.return_value = ComponentIdentifier(class_name="MockChat", class_module="test_module")
    target.set_system_prompt_async = AsyncMock()
    return target


def _normalizer(*replies: str) -> MagicMock:
    normalizer = get_mock_prompt_normalizer()
    normalizer.send_prompt_async = AsyncMock(
        side_effect=[Message.from_prompt(prompt=reply, role="assistant") for reply in replies]
    )
    return normalizer


def _error_response(detail: str = "boom") -> Message:
    piece = MessagePiece(
        role="assistant", original_value=detail, original_value_data_type="error", response_error="unknown"
    )
    return Message(message_pieces=[piece])


def _role(*, normalizer: MagicMock, target: MagicMock | None = None, **overrides) -> StructuredRole[dict]:
    kwargs: dict = {
        "target": target or _target(),
        "parse": lambda text: parse_json_object(text, schema=SCHEMA),
        "response_json_schema": SCHEMA,
        "prompt_normalizer": normalizer,
    }
    kwargs.update(overrides)
    return StructuredRole(**kwargs)


# --- camel_to_snake ----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [("nextMessage", "next_message"), ("NextMessage", "next_message"), ("next_message", "next_message")],
)
def test_camel_to_snake(name: str, expected: str) -> None:
    assert camel_to_snake(name) == expected


# --- build_json_prompt_metadata ----------------------------------------------


def test_build_json_prompt_metadata_without_schema_is_empty() -> None:
    assert build_json_prompt_metadata(response_json_schema=None) == {}


def test_build_json_prompt_metadata_with_schema_forwards_it() -> None:
    metadata = build_json_prompt_metadata(response_json_schema=SCHEMA)

    assert metadata["response_format"] == "json"
    assert metadata[JSON_SCHEMA_METADATA_KEY] == SCHEMA


# --- parse_json_object -------------------------------------------------------


def test_parse_json_object_returns_decoded_object() -> None:
    assert parse_json_object(VALID_JSON, schema=SCHEMA) == {"mode": "ATTACK", "framing": "academic"}


def test_parse_json_object_normalizes_camel_case_keys() -> None:
    schema = {"type": "object", "properties": {"response_state": {"type": "string"}}, "required": ["responseState"]}

    assert parse_json_object('{"responseState": "refusing"}', schema=schema) == {"response_state": "refusing"}


def test_parse_json_object_strips_markdown_fences() -> None:
    assert parse_json_object(f"```json\n{VALID_JSON}\n```", schema=SCHEMA)["mode"] == "ATTACK"


def test_parse_json_object_without_schema_accepts_any_object() -> None:
    assert parse_json_object('{"anything": 1}') == {"anything": 1}


def test_parse_json_object_invalid_json_raises() -> None:
    with pytest.raises(InvalidJsonException, match="Invalid JSON"):
        parse_json_object("not json", schema=SCHEMA)


@pytest.mark.parametrize("response_text", ["[]", "null", '"text"', "42"], ids=["array", "null", "string", "number"])
def test_parse_json_object_non_object_raises(response_text: str) -> None:
    with pytest.raises(InvalidJsonException, match="must be a JSON object"):
        parse_json_object(response_text, schema=SCHEMA)


def test_parse_json_object_missing_required_key_raises() -> None:
    with pytest.raises(InvalidJsonException, match="Missing required keys"):
        parse_json_object('{"mode": "ATTACK"}', schema=SCHEMA)


def test_parse_json_object_extra_key_raises_when_additional_properties_forbidden() -> None:
    with pytest.raises(InvalidJsonException, match="Unexpected keys"):
        parse_json_object('{"mode": "ATTACK", "framing": "x", "surprise": 1}', schema=SCHEMA)


def test_parse_json_object_extra_key_allowed_when_additional_properties_open() -> None:
    assert parse_json_object('{"mode": "a", "surprise": 1}', schema=OPEN_SCHEMA)["surprise"] == 1


# --- raise_for_role_error ----------------------------------------------------


def test_raise_for_role_error_passes_clean_response() -> None:
    raise_for_role_error(Message.from_prompt(prompt=VALID_JSON, role="assistant"), role_label="Strategist")


def test_raise_for_role_error_raises_labelled_exception() -> None:
    with pytest.raises(PyritException, match="Strategist returned an error: boom"):
        raise_for_role_error(_error_response("boom"), role_label="Strategist")


# --- StructuredRole.build_message --------------------------------------------


def test_build_message_carries_schema_metadata() -> None:
    role = _role(normalizer=_normalizer())

    piece = role.build_message(prompt_text="hello").message_pieces[0]

    assert piece.original_value == "hello"
    assert JSON_SCHEMA_METADATA_KEY in piece.prompt_metadata


def test_build_message_without_schema_has_no_metadata() -> None:
    role = _role(normalizer=_normalizer(), response_json_schema=None)

    assert not role.build_message(prompt_text="hello").message_pieces[0].prompt_metadata


# --- StructuredRole.send_async -----------------------------------------------


async def test_send_async_parses_reply_on_the_given_conversation() -> None:
    normalizer = _normalizer(VALID_JSON)
    role = _role(normalizer=normalizer)

    result = await role.send_async(message=role.build_message(prompt_text="hi"), conversation_id="conv-1")

    assert result == {"mode": "ATTACK", "framing": "academic"}
    assert normalizer.send_prompt_async.call_args.kwargs["conversation_id"] == "conv-1"


async def test_send_async_retries_malformed_reply_then_succeeds() -> None:
    normalizer = _normalizer("totally not json", VALID_JSON)
    role = _role(normalizer=normalizer)

    result = await role.send_async(message=role.build_message(prompt_text="hi"), conversation_id="conv-1")

    assert result["mode"] == "ATTACK"
    assert normalizer.send_prompt_async.call_count == 2


async def test_send_async_error_reply_uses_default_translation_without_retrying() -> None:
    normalizer = get_mock_prompt_normalizer()
    normalizer.send_prompt_async = AsyncMock(return_value=_error_response("boom"))
    role = _role(normalizer=normalizer, role_label="Strategist")

    with pytest.raises(PyritException, match="Strategist returned an error"):
        await role.send_async(message=role.build_message(prompt_text="hi"), conversation_id="conv-1")

    normalizer.send_prompt_async.assert_awaited_once()


async def test_send_async_uses_injected_error_translation() -> None:
    normalizer = get_mock_prompt_normalizer()
    normalizer.send_prompt_async = AsyncMock(return_value=_error_response())
    translate = MagicMock(side_effect=RuntimeError("translated"))
    role = _role(normalizer=normalizer, raise_for_error=translate)

    with pytest.raises(RuntimeError, match="translated"):
        await role.send_async(message=role.build_message(prompt_text="hi"), conversation_id="conv-1")

    translate.assert_called_once()


async def test_send_async_tags_execution_context() -> None:
    from pyrit.exceptions import get_execution_context

    seen: list = []
    normalizer = get_mock_prompt_normalizer()

    async def _capture(**_: object) -> Message:
        seen.append(get_execution_context())
        return Message.from_prompt(prompt=VALID_JSON, role="assistant")

    normalizer.send_prompt_async = AsyncMock(side_effect=_capture)
    role = _role(
        normalizer=normalizer,
        component_role=ComponentRole.ADVERSARIAL_CHAT,
        attack_strategy_name="MazeAttack",
        objective="obj",
    )

    await role.send_async(message=role.build_message(prompt_text="hi"), conversation_id="conv-1")

    assert seen[0].component_role == ComponentRole.ADVERSARIAL_CHAT
    assert seen[0].attack_strategy_name == "MazeAttack"
    assert seen[0].objective == "obj"


# --- StructuredRole.ask_async ------------------------------------------------


async def test_ask_async_sets_system_prompt_on_a_fresh_conversation() -> None:
    target = _target()
    normalizer = _normalizer(VALID_JSON)
    role = _role(normalizer=normalizer, target=target)

    await role.ask_async(prompt_text="hi", system_prompt="be a strategist")

    system_kwargs = target.set_system_prompt_async.call_args.kwargs
    assert system_kwargs["system_prompt"] == "be a strategist"
    assert system_kwargs["conversation_id"] == normalizer.send_prompt_async.call_args.kwargs["conversation_id"]


async def test_ask_async_uses_a_new_conversation_each_call() -> None:
    normalizer = _normalizer(VALID_JSON, VALID_JSON)
    role = _role(normalizer=normalizer)

    await role.ask_async(prompt_text="one")
    await role.ask_async(prompt_text="two")

    first, second = (call.kwargs["conversation_id"] for call in normalizer.send_prompt_async.call_args_list)
    assert first != second


async def test_ask_async_joins_an_existing_conversation() -> None:
    target = _target()
    normalizer = _normalizer(VALID_JSON)
    role = _role(normalizer=normalizer, target=target)

    await role.ask_async(prompt_text="hi", conversation_id="conv-9")

    assert normalizer.send_prompt_async.call_args.kwargs["conversation_id"] == "conv-9"
    target.set_system_prompt_async.assert_not_called()


async def test_ask_async_sets_system_prompt_on_the_conversation_id_it_is_given() -> None:
    target = _target()
    normalizer = _normalizer(VALID_JSON)
    role = _role(normalizer=normalizer, target=target)

    await role.ask_async(prompt_text="hi", system_prompt="be a strategist", conversation_id="conv-new")

    assert target.set_system_prompt_async.call_args.kwargs["conversation_id"] == "conv-new"
    assert normalizer.send_prompt_async.call_args.kwargs["conversation_id"] == "conv-new"
