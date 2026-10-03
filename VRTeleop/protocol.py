from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ControllerState:
    """One normalized WebXR controller sample (quaternion order: x, y, z, w)."""

    hand: str
    position: np.ndarray
    quaternion: np.ndarray
    squeeze: bool
    trigger: float
    buttons: frozenset[str] = field(default_factory=frozenset)
    source_timestamp_ms: float | None = None
    received_monotonic: float = field(default_factory=time.monotonic)


def _vector(mapping: Mapping[str, Any], keys: tuple[str, ...], label: str) -> np.ndarray:
    try:
        value = np.asarray([float(mapping[key]) for key in keys], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: expected numeric keys {keys}") from exc
    if not np.all(np.isfinite(value)):
        raise ValueError(f"invalid {label}: values must be finite")
    return value


def parse_controller_packet(packet: Mapping[str, Any], hand: str = "right") -> ControllerState:
    """Parse one browser packet and reject incomplete controller poses."""

    raw = packet.get(f"{hand}Controller")
    if not isinstance(raw, Mapping):
        raise ValueError(f"packet has no {hand}Controller")
    position = raw.get("position")
    quaternion = raw.get("quaternion")
    if not isinstance(position, Mapping) or not isinstance(quaternion, Mapping):
        raise ValueError(f"{hand} controller pose is incomplete")
    quat = _vector(quaternion, ("x", "y", "z", "w"), "quaternion")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        raise ValueError("invalid quaternion: zero norm")
    buttons_raw = raw.get("buttons", {})
    buttons = frozenset(
        str(name) for name, pressed in buttons_raw.items() if pressed
    ) if isinstance(buttons_raw, Mapping) else frozenset()
    trigger = float(raw.get("trigger", 0.0) or 0.0)
    timestamp = packet.get("timestamp")
    return ControllerState(
        hand=hand,
        position=_vector(position, ("x", "y", "z"), "position"),
        quaternion=quat / norm,
        squeeze=bool(raw.get("gripActive", False) or "squeeze" in buttons),
        trigger=float(np.clip(trigger, 0.0, 1.0)),
        buttons=buttons,
        source_timestamp_ms=float(timestamp) if timestamp is not None else None,
    )


class LatestControllerState:
    """Thread-safe latest-value buffer shared by WebSocket and control loops."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: ControllerState | None = None

    def set(self, state: ControllerState) -> None:
        with self._lock:
            self._state = state

    def get(self) -> ControllerState | None:
        with self._lock:
            return self._state
