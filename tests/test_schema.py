import numpy as np
import pytest

from franka_kitchen_vla.schema import ROBOT_STATE_NAMES, robot_state_from_observation


def test_robot_state_contains_qpos_then_qvel_only() -> None:
    observation = np.arange(59, dtype=np.float64)
    state = robot_state_from_observation(observation)
    np.testing.assert_array_equal(state, np.arange(18, dtype=np.float32))
    assert state.shape == (18,)
    assert state.dtype == np.float32
    assert len(ROBOT_STATE_NAMES) == 18


def test_robot_state_rejects_wrong_observation_shape() -> None:
    with pytest.raises(ValueError, match="59-D"):
        robot_state_from_observation(np.zeros(18, dtype=np.float32))
