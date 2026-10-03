import numpy as np

from VRTeleop.controller import (
    ClutchedTarget,
    damped_least_squares,
    orientation_error,
    quaternion_to_matrix,
)
from VRTeleop.protocol import ControllerState, LatestControllerState, parse_controller_packet


def test_parse_right_controller_packet() -> None:
    state = parse_controller_packet({
        "timestamp": 1234,
        "rightController": {
            "position": {"x": 1, "y": 2, "z": 3},
            "quaternion": {"x": 0, "y": 0, "z": 0, "w": 2},
            "gripActive": True,
            "trigger": 0.75,
            "buttons": {"a": True, "b": False},
        },
    })
    np.testing.assert_allclose(state.position, [1, 2, 3])
    np.testing.assert_allclose(state.quaternion, [0, 0, 0, 1])
    assert state.squeeze
    assert state.trigger == 0.75
    assert state.buttons == frozenset({"a"})


def test_latest_controller_buffer() -> None:
    latest = LatestControllerState()
    assert latest.get() is None
    state = ControllerState("right", np.zeros(3), np.array([0, 0, 0, 1]), False, 0.0)
    latest.set(state)
    assert latest.get() is state


def test_quaternion_and_orientation_error_identity() -> None:
    rotation = quaternion_to_matrix(np.array([0, 0, 0, 1]))
    np.testing.assert_allclose(rotation, np.eye(3))
    np.testing.assert_allclose(orientation_error(rotation, rotation), np.zeros(3))


def test_damped_least_squares_identity() -> None:
    jacobian = np.eye(6)
    twist = np.arange(6, dtype=float)
    result = damped_least_squares(jacobian, twist, damping=0.0)
    np.testing.assert_allclose(result, twist)


def test_clutch_has_no_jump_and_release_clears_reference() -> None:
    target = ClutchedTarget()
    state = ControllerState("right", np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0, 1]), True, 0.0)
    pose = target.update(state, np.array([0.1, 0.2, 0.3]), np.eye(3), 0.5, 0.5)
    assert pose is not None
    np.testing.assert_allclose(pose[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(pose[1], np.eye(3))
    released = ControllerState("right", state.position, state.quaternion, False, 0.0)
    assert target.update(released, pose[0], pose[1], 0.5, 0.5) is None
    assert not target.active
