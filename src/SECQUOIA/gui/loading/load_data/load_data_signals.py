"""Signal helpers shared by the Load_data tree builders."""

from __future__ import annotations

import contextlib

__all__ = ["reconnect"]

_SLOT_REGISTRY_ATTR = "_secquoia_slots"


def reconnect(owner, signal_name: str, slot) -> None:
    """Connect slot to owner.<signal_name>, replacing previous slot."""
    registry = getattr(owner, _SLOT_REGISTRY_ATTR, None)
    if registry is None:
        registry = {}
        setattr(owner, _SLOT_REGISTRY_ATTR, registry)

    signal = getattr(owner, signal_name)
    previous = registry.pop(signal_name, None)
    if previous is not None:
        with contextlib.suppress(TypeError, RuntimeError):
            signal.disconnect(previous)

    signal.connect(slot)
    registry[signal_name] = slot
