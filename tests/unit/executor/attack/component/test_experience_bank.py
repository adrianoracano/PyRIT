# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.executor.attack.component.experience_bank import (
    EmbeddingTextEmbedder,
    ExperienceBank,
    HashingTextEmbedder,
)
from pyrit.models import EmbeddingData, EmbeddingResponse, EmbeddingSupport, EmbeddingUsageInformation

# --- HashingTextEmbedder -----------------------------------------------------


async def test_hashing_embedding_is_unit_length_and_deterministic() -> None:
    embedder = HashingTextEmbedder()

    first = await embedder.embed_async("the quick brown fox")
    second = await embedder.embed_async("the quick brown fox")

    assert first == second
    assert math.isclose(sum(value * value for value in first), 1.0, rel_tol=1e-9)


async def test_hashing_embedding_of_text_without_words_is_all_zeros() -> None:
    assert set(await HashingTextEmbedder(dimensions=8).embed_async("... !!!")) == {0.0}


async def test_hashing_embedding_ranks_shared_wording_higher() -> None:
    embedder = HashingTextEmbedder()
    query, near, far = (
        await embedder.embed_async("describe how to pick a lock"),
        await embedder.embed_async("explain how to pick a lock quickly"),
        await embedder.embed_async("write a poem about autumn leaves"),
    )

    def similarity(first: list[float], second: list[float]) -> float:
        return sum(a * b for a, b in zip(first, second, strict=True))

    assert similarity(query, near) > similarity(query, far)


def test_hashing_embedder_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        HashingTextEmbedder(dimensions=0)


# --- EmbeddingTextEmbedder ---------------------------------------------------


async def test_embedding_adapter_returns_the_normalized_first_vector() -> None:
    generator = MagicMock(spec=EmbeddingSupport)
    generator.generate_text_embedding_async = AsyncMock(
        return_value=EmbeddingResponse(
            model="m",
            object="list",
            usage=EmbeddingUsageInformation(prompt_tokens=1, total_tokens=1),
            data=[EmbeddingData(embedding=[3.0, 4.0], index=0, object="embedding")],
        )
    )

    vector = await EmbeddingTextEmbedder(embedding_generator=generator).embed_async("text")

    assert vector == pytest.approx([0.6, 0.8])
    generator.generate_text_embedding_async.assert_awaited_once()


# --- ExperienceBank ----------------------------------------------------------


async def test_bank_returns_the_most_similar_successes_first() -> None:
    bank = ExperienceBank()
    await bank.add_success_async(objective="explain how to pick a lock", prompt="lock prompt")
    await bank.add_success_async(objective="write a poem about autumn leaves", prompt="poem prompt")

    found = await bank.similar_successes_async(objective="describe how to pick a lock", limit=1)

    assert [experience.prompt for experience in found] == ["lock prompt"]


async def test_bank_with_nothing_remembered_finds_nothing() -> None:
    assert await ExperienceBank().similar_successes_async(objective="anything") == []


async def test_bank_limit_zero_finds_nothing() -> None:
    bank = ExperienceBank()
    await bank.add_success_async(objective="a", prompt="p")

    assert await bank.similar_successes_async(objective="a", limit=0) == []


async def test_bank_includes_an_identical_objective() -> None:
    bank = ExperienceBank()
    await bank.add_success_async(objective="same objective", prompt="earlier success", score=0.9)

    found = await bank.similar_successes_async(objective="same objective")

    assert found[0].prompt == "earlier success"
    assert found[0].score == 0.9


async def test_bank_len_counts_successes() -> None:
    bank = ExperienceBank()
    await bank.add_success_async(objective="a", prompt="p")
    await bank.add_success_async(objective="b", prompt="q")

    assert len(bank) == 2


async def test_bank_exports_and_imports_successes() -> None:
    source = ExperienceBank()
    await source.add_success_async(objective="explain how to pick a lock", prompt="lock prompt", score=0.8)
    records = source.export_successes()

    restored = ExperienceBank()
    imported = await restored.import_successes_async([*records, {"objective": "missing a prompt"}, {"prompt": "x"}])

    assert imported == 1
    assert records == [{"objective": "explain how to pick a lock", "prompt": "lock prompt", "score": 0.8}]
    found = await restored.similar_successes_async(objective="pick a lock")
    assert found[0].prompt == "lock prompt"


async def test_bank_reports_its_embedder() -> None:
    assert ExperienceBank().embedder_name == "HashingTextEmbedder"
