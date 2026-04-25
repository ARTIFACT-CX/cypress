"""
AREA: worker · MODELS · BASE

Abstract interface every voice model implementation must satisfy. The goal
is to keep model-specific code (weights paths, tokenizers, streaming
generation loops) behind a uniform surface so the command handler doesn't
care which model is loaded.

SWAP: this is the single interface boundary for models. Keep it small —
anything model-specific belongs in the concrete subclass, not here.
"""

from abc import ABC, abstractmethod
from typing import Type


class Model(ABC):
    # Subclasses set this at class-body level. `load_model` commands from
    # the host look up concrete classes in REGISTRY using this name.
    name: str = ""

    @abstractmethod
    async def load(self) -> None:
        """Acquire any weights / GPU memory. Blocking allowed."""

    @abstractmethod
    async def unload(self) -> None:
        """Release weights / GPU memory. Safe to call multiple times."""

    # TODO: streaming generation interface — arrives when Moshi lands. Will
    # likely expose an async-iterable of output frames given an async-
    # iterable of input frames, plus a side-channel for text tokens (the
    # Moshi "inner monologue" we use for tool-call interception).


# Populated by @register from each concrete module. The command handler
# iterates this to answer `list_models` and looks up entries by name for
# `load_model`.
REGISTRY: dict[str, Type[Model]] = {}


def register(cls: Type[Model]) -> Type[Model]:
    # Decorator used at the bottom of each concrete model file:
    #
    #   @register
    #   class Moshi(Model):
    #       name = "moshi"
    #       ...
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a non-empty `name`")
    REGISTRY[cls.name] = cls
    return cls
