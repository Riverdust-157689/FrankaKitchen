import numpy as np
import pytest

from franka_kitchen_vla.tasks import (
    TASK_GOALS,
    TASK_INDICES,
    completion_events,
    completion_steps,
    instruction_for,
    unpack_observation,
)


def flat_from_qpos(qpos: np.ndarray) -> np.ndarray:
    """Pack a (T, 30) MuJoCo qpos into FrankaKitchen's 59-D observation layout."""
    qpos = np.asarray(qpos, dtype=float)
    qvel = np.zeros((qpos.shape[0], 29))
    return np.concatenate((qpos[:, :9], qvel[:, :9], qpos[:, 9:], qvel[:, 9:]), axis=1)


def test_unpack_observation_layout():
    flat = np.arange(59)
    qpos, qvel = unpack_observation(flat)
    np.testing.assert_array_equal(qpos, np.r_[np.arange(9), np.arange(18, 39)])
    np.testing.assert_array_equal(qvel, np.r_[np.arange(9, 18), np.arange(39, 59)])


def test_completion_steps_orders_first_hits():
    qpos = np.zeros((4, 30))
    qpos[:, 23:30] = np.array([-0.269, 0.35, 1.62, 1.0, 0.0, 0.0, 0.0])
    qpos[1:, TASK_INDICES["microwave"]] = TASK_GOALS["microwave"]
    qpos[2:, TASK_INDICES["light switch"]] = TASK_GOALS["light switch"]
    flat = flat_from_qpos(qpos)
    assert completion_steps(flat) == [(1, "microwave"), (2, "light switch")]


def test_completion_indices_are_qpos_indices_not_observation_indices():
    """A goal placed at the 59-D observation position must NOT be detected.

    OBS_ELEMENT_INDICES are MuJoCo qpos indices; the env reads
    ``data.qpos[OBS_ELEMENT_INDICES[task]]``. Indexing the 59-D observation directly
    is a real bug that silently produces wrong labels, so pin the behaviour down.
    """
    qpos = np.zeros((3, 30))
    # 'light switch' is qpos[17:19]; put the goal at the 59-D position 17/18 instead.
    qpos[1, 23:30] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    flat = flat_from_qpos(qpos)
    flat[1, 17:19] = TASK_GOALS["light switch"]
    assert completion_steps(flat) == []


def test_completion_events_prefers_the_goal_dictionaries():
    """When achieved_goal/desired_goal disagree with the observation, goals win."""
    qpos = np.zeros((3, 30))
    flat = flat_from_qpos(qpos)
    # Observation still shows the microwave closed...
    assert completion_steps(flat) == []
    # ...but the recorded goals say it is open from frame 1 onwards.
    achieved = {"microwave": np.array([[0.0], [0.0], [-0.75]])}
    desired = {"microwave": np.array([[-0.75]])}
    assert completion_events(flat, achieved_goal=achieved, desired_goal=desired) == [
        (2, "microwave")
    ]


def test_completion_events_falls_back_for_tasks_without_goals():
    qpos = np.zeros((3, 30))
    qpos[2, TASK_INDICES["bottom burner"]] = TASK_GOALS["bottom burner"]
    flat = flat_from_qpos(qpos)
    achieved = {"microwave": np.zeros((3, 1))}
    desired = {"microwave": np.array([[-0.75]])}
    events = completion_events(flat, achieved_goal=achieved, desired_goal=desired)
    assert events == [(2, "bottom burner")]


def test_instruction_preserves_sequence():
    assert instruction_for(["microwave", "kettle"]) == (
        "Open the microwave door, then move the kettle to the top-left burner."
    )


def test_instruction_single_task_and_compact_style():
    assert instruction_for(["light switch"]) == "Turn on the light switch."
    assert instruction_for(["microwave", "kettle"], style="compact") == "microwave, kettle"


def test_instruction_rejects_unknown_task_and_style():
    with pytest.raises(ValueError):
        instruction_for(["not a task"])
    with pytest.raises(ValueError):
        instruction_for(["microwave"], style="shouty")
    with pytest.raises(ValueError):
        instruction_for([])


def test_all_prompts_fit_smolvla_tokenizer_limit():
    """The four-task prompts must fit tokenizer_max_length=48 of the base model."""
    longest = instruction_for(list(TASK_INDICES))
    assert len(longest.split()) < 48  # cheap guard; exact tokenisation is checked in report
