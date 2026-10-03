from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .tasks import completion_steps, instruction_for, unpack_observation

DEFAULT_DATASET_ROOT = Path("data/minari")
DEFAULT_DATASET_ID = "D4RL/kitchen/partial-v2"


def load_dataset(dataset_id: str = DEFAULT_DATASET_ID, root: str | Path = DEFAULT_DATASET_ROOT):
    os.environ["MINARI_DATASETS_PATH"] = str(Path(root).resolve())
    import minari

    return minari.load_dataset(dataset_id)


def make_replay_environment(dataset: Any):
    env = dataset.recover_environment(
        render_mode=None,
        robot_noise_ratio=0.0,
        object_noise_ratio=0.0,
    )
    # The TCP site would otherwise be rendered as a small dark ball inside the gripper in
    # every frame; it is a modelling aid rather than real geometry. See runtime.py.
    from .runtime import hide_end_effector_site

    hide_end_effector_site(env.unwrapped.model)
    return env


def set_state_from_observation(env: Any, flat_observation: np.ndarray) -> None:
    qpos, qvel = unpack_observation(flat_observation)
    unwrapped = env.unwrapped
    unwrapped.data.qpos[:] = qpos
    unwrapped.data.qvel[:] = qvel
    mujoco.mj_forward(unwrapped.model, unwrapped.data)


def episode_task_data(episode: Any) -> tuple[list[tuple[int, str]], str]:
    events = completion_steps(episode.observations["observation"])
    return events, instruction_for(task for _, task in events)
