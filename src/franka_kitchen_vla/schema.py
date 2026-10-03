from __future__ import annotations

import numpy as np


ROBOT_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "right_finger",
    "left_finger",
]

ROBOT_QPOS_DIM = len(ROBOT_JOINT_NAMES)
ROBOT_QVEL_NAMES = [f"{name}_velocity" for name in ROBOT_JOINT_NAMES]
ROBOT_STATE_NAMES = [*ROBOT_JOINT_NAMES, *ROBOT_QVEL_NAMES]
ROBOT_STATE_DIM = len(ROBOT_STATE_NAMES)


def robot_state_from_observation(observation: np.ndarray) -> np.ndarray:
    """Return robot qpos + qvel from a FrankaKitchen 59-D observation."""
    flat = np.asarray(observation)
    if flat.shape[-1] != 59:
        raise ValueError(f"Expected a 59-D FrankaKitchen observation, got {flat.shape}.")
    return np.asarray(flat[..., :ROBOT_STATE_DIM], dtype=np.float32)
