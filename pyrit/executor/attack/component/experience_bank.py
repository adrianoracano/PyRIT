# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Embedding-based memory of what worked against earlier objectives, reused to warm-start new ones."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from pyrit.models import EmbeddingSupport


class TextEmbedder(Protocol):
    """Turns text into a vector for similarity search."""

    async def embed_async(self, text: str) -> list[float]:
        """
        Embed ``text``.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: The embedding.
        """
        ...  # pragma: no cover


class HashingTextEmbedder:
    """
    A dependency-free embedder that hashes word unigrams and bigrams into a fixed-size vector.

    Similarity reflects shared wording, not meaning. It needs no model or network call, which makes it a
    sensible default for local runs and tests; pass an ``EmbeddingTextEmbedder`` for semantic similarity.
    """

    def __init__(self, *, dimensions: int = 512) -> None:
        """
        Initialize the embedder.

        Args:
            dimensions (int): Size of the vector. Defaults to 512.

        Raises:
            ValueError: If ``dimensions`` is not positive.
        """
        if dimensions <= 0:
            raise ValueError("dimensions must be positive.")
        self._dimensions = dimensions

    async def embed_async(self, text: str) -> list[float]:
        """
        Embed ``text``.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: A unit-length vector, or all zeros for text without words.
        """
        words = re.findall(r"\w+", text.lower())
        features = [*words, *(f"{first} {second}" for first, second in zip(words, words[1:], strict=False))]
        vector = [0.0] * self._dimensions
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self._dimensions] += 1.0 if digest[4] % 2 else -1.0
        return _normalized(vector)


class EmbeddingTextEmbedder:
    """Adapts a PyRIT embedding generator to the ``TextEmbedder`` protocol."""

    def __init__(self, *, embedding_generator: EmbeddingSupport) -> None:
        """
        Initialize the adapter.

        Args:
            embedding_generator (EmbeddingSupport): The embedding generator to call.
        """
        self._embedding_generator = embedding_generator

    async def embed_async(self, text: str) -> list[float]:
        """
        Embed ``text`` with the wrapped generator.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: The embedding of the first returned item.
        """
        response = await self._embedding_generator.generate_text_embedding_async(text)
        return _normalized(list(response.data[0].embedding))


def _normalized(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def _cosine(first: Sequence[float], second: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(first, second, strict=False))


@dataclass(frozen=True)
class Experience:
    """A prompt that achieved an objective."""

    objective: str
    prompt: str
    score: float


class ExperienceBank:
    """
    Remembers prompts that achieved their objective, so a new, similar objective can start from patterns
    that worked.

    The bank is shared by every run of the attack it is given to, which is how experience carries over from
    one objective to the next. Writes are serialized so concurrent runs can share it.
    """

    def __init__(self, *, embedder: TextEmbedder | None = None) -> None:
        """
        Initialize an empty bank.

        Args:
            embedder (TextEmbedder | None): Embeds objectives for similarity search. Defaults to a
                ``HashingTextEmbedder``.
        """
        self._embedder = embedder or HashingTextEmbedder()
        self._successes: list[tuple[list[float], Experience]] = []
        self._lock = asyncio.Lock()

    @property
    def embedder_name(self) -> str:
        """The class name of the embedder in use."""
        return type(self._embedder).__name__

    def __len__(self) -> int:
        """
        Count the successful prompts remembered.

        Returns:
            int: The number of successes in the bank.
        """
        return len(self._successes)

    async def add_success_async(self, *, objective: str, prompt: str, score: float = 1.0) -> None:
        """
        Remember a prompt that achieved its objective.

        Args:
            objective (str): The objective it achieved.
            prompt (str): The prompt.
            score (float): The score it earned. Defaults to 1.0.
        """
        vector = await self._embedder.embed_async(objective)
        async with self._lock:
            self._successes.append((vector, Experience(objective=objective, prompt=prompt, score=score)))

    async def similar_successes_async(self, *, objective: str, limit: int = 2) -> list[Experience]:
        """
        Find the remembered successes whose objectives are most similar to ``objective``.

        Args:
            objective (str): The objective to find precedents for.
            limit (int): The most precedents to return. Defaults to 2.

        Returns:
            list[Experience]: Up to ``limit`` precedents, most similar first. An identical objective is
                included, so a repeated objective starts from its own earlier success.
        """
        if not self._successes or limit <= 0:
            return []
        query = await self._embedder.embed_async(objective)
        async with self._lock:
            ranked = sorted(self._successes, key=lambda entry: _cosine(query, entry[0]), reverse=True)
        return [experience for _, experience in ranked[:limit]]

    def export_successes(self) -> list[dict[str, Any]]:
        """
        Export the remembered successes for storage.

        Returns:
            list[dict[str, Any]]: One JSON-serializable dict per success.
        """
        return [
            {"objective": experience.objective, "prompt": experience.prompt, "score": experience.score}
            for _, experience in self._successes
        ]

    async def import_successes_async(self, records: Iterable[Mapping[str, Any]]) -> int:
        """
        Remember successes exported earlier.

        Args:
            records (Iterable[Mapping[str, Any]]): Dicts as returned by ``export_successes``. Entries
                missing an objective or prompt are skipped.

        Returns:
            int: How many were imported.
        """
        imported = 0
        for record in records:
            objective, prompt = record.get("objective"), record.get("prompt")
            if not isinstance(objective, str) or not isinstance(prompt, str):
                continue
            await self.add_success_async(objective=objective, prompt=prompt, score=float(record.get("score", 1.0)))
            imported += 1
        return imported
