# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A named strategy, described in words, that an attack tells its attack model to draw on."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyDescription:
    """
    A strategy as the attack model is told about it.

    PyRIT ships no strategy catalogs. The caller supplies the strategies to test, so what is tested is
    chosen deliberately.
    """

    name: str
    definition: str

    def __post_init__(self) -> None:
        """
        Validate the description.

        Raises:
            ValueError: If the name or definition is blank.
        """
        if not self.name.strip() or not self.definition.strip():
            raise ValueError("A StrategyDescription needs a name and a definition.")
