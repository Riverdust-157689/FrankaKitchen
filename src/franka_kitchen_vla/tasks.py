from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

# These indices and goals are part of the FrankaKitchen-v1 task definition.
# They mirror OBS_ELEMENT_INDICES / OBS_ELEMENT_GOALS in
# gymnasium_robotics.envs.franka_kitchen.kitchen_env, which is the authority for
# how the environment decides that a task has been completed.
#
# IMPORTANT: the indices are positions in MuJoCo's 30-D qpos vector
# (9 robot joints + 21 object joints), exactly as KitchenEnv uses them via
# `self.data.qpos[OBS_ELEMENT_INDICES[task]]`. They are NOT positions in the 59-D
# observation. Use `unpack_observation` first and index the returned qpos.
TASK_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}

TASK_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}

TASK_PHRASES = {
    "bottom burner": "turn on the bottom burner",
    "top burner": "turn on the top burner",
    "light switch": "turn on the light switch",
    "slide cabinet": "open the sliding cabinet",
    "hinge cabinet": "open the hinged cabinet",
    "microwave": "open the microwave door",
    "kettle": "move the kettle to the top-left burner",
}

# Compact phrasing is meant for the future ``selected object(s) -> instruction``
# front end, where the upstream selector emits object names rather than sentences.
TASK_SHORT_PHRASES = {
    "bottom burner": "bottom burner",
    "top burner": "top burner",
    "light switch": "light switch",
    "slide cabinet": "slide cabinet",
    "hinge cabinet": "hinge cabinet",
    "microwave": "microwave",
    "kettle": "kettle",
}

PROMPT_STYLES = ("natural", "compact")

# The environment considers a task complete below this norm threshold.
BONUS_THRESHOLD = 0.3

# ``observation`` layout of FrankaKitchen-v1: robot qpos (9), robot qvel (9),
# object qpos (21), object qvel (20). Object joints in TASK_INDICES are already
# expressed in the object-qpos coordinate frame, so they index the 59-D vector
# directly when it is flattened as a whole.
_OBJECT_QPOS_OFFSET = 18


def unpack_observation(flat_observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover MuJoCo qpos/qvel from FrankaKitchen's 59-D observation.

    FrankaKitchen concatenates robot qpos (9), robot qvel (9), object qpos
    (21), and object qvel (20). The dataset values contain small observation
    noise, which is negligible for visual reconstruction.
    """
    flat = np.asarray(flat_observation)
    if flat.shape[-1] != 59:
        raise ValueError(f"Expected a 59-D FrankaKitchen observation, got {flat.shape}.")
    qpos = np.concatenate((flat[..., :9], flat[..., 18:39]), axis=-1)
    qvel = np.concatenate((flat[..., 9:18], flat[..., 39:59]), axis=-1)
    return qpos, qvel


def _goal_distance_frames(
    values: np.ndarray, task: str, threshold: float
) -> np.ndarray:
    goal = TASK_GOALS[task]
    values = np.asarray(values)
    return np.linalg.norm(values - goal, axis=-1) < threshold


def completion_events(
    observations: np.ndarray,
    achieved_goal: Mapping[str, np.ndarray] | None = None,
    desired_goal: Mapping[str, np.ndarray] | None = None,
    threshold: float = BONUS_THRESHOLD,
) -> list[tuple[int, str]]:
    """Return each reached task once, ordered by its first completion frame.

    When the episode carries ``achieved_goal`` / ``desired_goal`` (Minari stores
    them for every task listed in the recording environment's
    ``tasks_to_complete``), those values are used directly: they are the very
    numbers ``KitchenEnv.compute_reward`` compares, so the labels then match the
    environment bit for bit. Tasks that are absent from the goal dictionaries
    (for example ``bottom burner`` in ``partial-v2``) fall back to a geometric
    test on the 59-D observation.

    The fallback is accurate: over all 621 ``partial-v2`` episodes it agrees
    with the goal-dictionary answer for 620 episodes, the single difference
    being a one-frame boundary jitter on the kettle.
    """
    flat = np.asarray(observations)
    if flat.ndim != 2 or flat.shape[-1] != 59:
        raise ValueError(f"Expected observations shaped (T, 59), got {flat.shape}.")
    qpos, _ = unpack_observation(flat)

    events: list[tuple[int, str]] = []
    for task, indices in TASK_INDICES.items():
        if (
            achieved_goal is not None
            and desired_goal is not None
            and task in achieved_goal
            and task in desired_goal
        ):
            achieved = np.asarray(achieved_goal[task])
            desired = np.asarray(desired_goal[task])
            distances = np.linalg.norm(achieved - desired, axis=-1)
        else:
            distances = np.linalg.norm(qpos[:, indices] - TASK_GOALS[task], axis=-1)
        hits = np.flatnonzero(distances < threshold)
        if hits.size:
            events.append((int(hits[0]), task))
    return sorted(events)


def completion_steps(
    flat_observations: np.ndarray, threshold: float = BONUS_THRESHOLD
) -> list[tuple[int, str]]:
    """Geometric-only completion detection (kept for callers that lack goals)."""
    return completion_events(flat_observations, threshold=threshold)


def goal_entries(
    episode: object,
) -> tuple[Mapping[str, np.ndarray] | None, Mapping[str, np.ndarray] | None]:
    """Pull ``achieved_goal`` / ``desired_goal`` out of a Minari episode if present."""
    observations = getattr(episode, "observations", None)
    if not isinstance(observations, Mapping):
        return None, None
    achieved = observations.get("achieved_goal")
    desired = observations.get("desired_goal")
    if not isinstance(achieved, Mapping) or not isinstance(desired, Mapping):
        return None, None
    return achieved, desired


def episode_completion_events(episode: object) -> list[tuple[int, str]]:
    """Completion events for a Minari episode, preferring the environment's goals."""
    achieved, desired = goal_entries(episode)
    flat = episode.observations["observation"]  # type: ignore[attr-defined]
    return completion_events(flat, achieved_goal=achieved, desired_goal=desired)


def instruction_for(tasks: Iterable[str], style: str = "natural") -> str:
    """Turn an ordered task selection into the language prompt used by SmolVLA."""
    if style not in PROMPT_STYLES:
        raise ValueError(f"Unknown prompt style {style!r}; expected one of {PROMPT_STYLES}.")
    phrases_table = TASK_PHRASES if style == "natural" else TASK_SHORT_PHRASES
    phrases = []
    for task in tasks:
        if task not in phrases_table:
            raise ValueError(f"Unknown task {task!r}; available: {sorted(phrases_table)}")
        phrases.append(phrases_table[task])
    if not phrases:
        raise ValueError("At least one task is required to build an instruction.")
    if style == "compact":
        return ", ".join(phrases)
    if len(phrases) == 1:
        return phrases[0].capitalize() + "."
    return (", then ".join(phrases)).capitalize() + "."


def sequences_in_dataset(dataset: object, max_episodes: int | None = None) -> dict[tuple[str, ...], int]:
    """Count how often each ordered task sequence occurs in a Minari dataset."""
    counts: dict[tuple[str, ...], int] = {}
    for index, episode in enumerate(dataset.iterate_episodes()):  # type: ignore[attr-defined]
        if max_episodes is not None and index >= max_episodes:
            break
        key = tuple(task for _, task in episode_completion_events(episode))
        counts[key] = counts.get(key, 0) + 1
    return counts


def format_sequence(tasks: Sequence[str]) -> str:
    return " -> ".join(tasks)
