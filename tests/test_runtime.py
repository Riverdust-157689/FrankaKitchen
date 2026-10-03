from franka_kitchen_vla.runtime import ordered_prefix_length
import pytest


def test_ordered_prefix_length_accepts_exact_prefix() -> None:
    requested = ["microwave", "kettle", "light switch", "slide cabinet"]
    assert ordered_prefix_length(requested, ["microwave", "kettle"]) == 2
    assert ordered_prefix_length(requested, requested) == 4


def test_ordered_prefix_length_stops_at_wrong_order() -> None:
    requested = ["microwave", "kettle", "light switch", "slide cabinet"]
    assert ordered_prefix_length(requested, ["kettle", "microwave"]) == 0
    assert ordered_prefix_length(requested, ["microwave", "light switch", "kettle"]) == 1


def test_ordered_prefix_length_ignores_unrequested_probe_events() -> None:
    requested = ["microwave", "kettle"]
    completed = ["bottom burner", "microwave", "kettle"]
    assert ordered_prefix_length(requested, completed) == 2


# --------------------------------------------------------------------------------------
# One seed assigns both noise streams (see LoadedPolicy.assign_seed)
# --------------------------------------------------------------------------------------
class _StubConfig:
    chunk_size = 50
    max_action_dim = 32


class _StubPolicy:
    config = _StubConfig()

    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


def _loaded_policy():
    torch = pytest.importorskip("torch")

    from franka_kitchen_vla.runtime import LoadedPolicy

    return LoadedPolicy(
        policy=_StubPolicy(),
        preprocess=None,
        postprocess=None,
        device=torch.device("cpu"),
        camera_keys=[],
        policy_path="stub",
    )


def _draws(loaded_policy, count: int = 4):
    return [loaded_policy._draw_noise().flatten()[0].item() for _ in range(count)]


def test_assign_seed_assigns_that_exact_value_to_the_generator():
    """The seed handed in IS the generator seed -- not a derived or mixed value."""
    torch = pytest.importorskip("torch")

    loaded = _loaded_policy()
    loaded.assign_seed(7)
    drawn = _draws(loaded, 3)

    reference = torch.Generator(device="cpu").manual_seed(7)
    expected = [
        torch.normal(0.0, 1.0, (1, 50, 32), generator=reference).flatten()[0].item()
        for _ in range(3)
    ]
    assert drawn == expected


def test_assign_seed_replays_identical_noise_for_the_same_seed():
    """Two rollouts sharing a seed share their noise bit for bit -- this is what makes
    the paired comparison in kitchen-instruction-probe meaningful."""
    loaded = _loaded_policy()
    loaded.assign_seed(0)
    first = _draws(loaded)
    loaded.assign_seed(0)
    second = _draws(loaded)
    assert first == second
    # ...but successive replannings inside one episode must not repeat one sample
    assert len(set(first)) == len(first)


def test_assign_seed_gives_different_episodes_different_noise():
    loaded = _loaded_policy()
    loaded.assign_seed(0)
    first = _draws(loaded)
    loaded.assign_seed(1)
    second = _draws(loaded)
    assert first != second


def test_assign_seed_starts_a_fresh_episode():
    loaded = _loaded_policy()
    loaded.assign_seed(3)
    _draws(loaded, 2)
    assert loaded.noise_draws == 2
    loaded.assign_seed(4)
    assert loaded.noise_draws == 0
    assert loaded.policy.resets == 2


def test_env_observation_noise_realisation_follows_the_episode_seed():
    """The other half of the contract: the SAME value also decides the environment's
    observation noise, so pairing holds for both streams at once."""
    import numpy as np

    from franka_kitchen_vla.runtime import make_kitchen_env

    tasks = ["microwave", "kettle"]
    noisy = make_kitchen_env(tasks, robot_noise_ratio=0.01, max_episode_steps=50)
    clean = make_kitchen_env(tasks, robot_noise_ratio=0.0, max_episode_steps=50)
    try:
        def observe(env, seed):
            observation, _ = env.reset(seed=seed)
            return np.asarray(observation["observation"], dtype=float)

        assert np.array_equal(observe(noisy, 0), observe(noisy, 0))
        assert not np.array_equal(observe(noisy, 0), observe(noisy, 1))
        # With noise off the seed changes nothing at all: the initial state is fixed, so
        # every episode starts from the canonical reset regardless of the seed.
        assert np.array_equal(observe(clean, 0), observe(clean, 7))
    finally:
        noisy.close()
        clean.close()
