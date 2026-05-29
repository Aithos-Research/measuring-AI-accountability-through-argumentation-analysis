from __future__ import annotations

from .moralchoice import MoralChoice
from .perspectives import MoralChoicePerspectives
from .perturbations import MoralChoicePerturbations

__all__ = [
    "MoralChoice",
    "MoralChoicePerspectives",
    "MoralChoicePerturbations",
]


def load_base(**kwargs):
    """
    Convenience loader for merged MoralChoice dilemmas.

    Keyword args are split: known __init__ args go to MoralChoice, the rest go to build_merged().
    """
    init_kwargs = {}
    for key in ("ambiguity", "template", "include_system_instruction", "include_both_orderings", "split"):
        if key in kwargs:
            init_kwargs[key] = kwargs.pop(key)

    return MoralChoice(**init_kwargs).build_merged(**kwargs)


def load_perspectives(**kwargs):
    """
    Load MoralChoice perspectives (flattened).
    """
    init_kwargs = {}
    for key in ("ambiguity", "data_file"):
        if key in kwargs:
            init_kwargs[key] = kwargs.pop(key)

    return MoralChoicePerspectives(**init_kwargs).load(**kwargs)


def load_perturbations(**kwargs):
    """
    Load MoralChoice perturbations (flattened).
    """
    init_kwargs = {}
    for key in ("ambiguity", "data_file"):
        if key in kwargs:
            init_kwargs[key] = kwargs.pop(key)

    return MoralChoicePerturbations(**init_kwargs).load(**kwargs)

