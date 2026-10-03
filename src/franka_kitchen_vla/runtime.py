from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .schema import ROBOT_QPOS_DIM, robot_state_from_observation


@dataclass
class LoadedPolicy:
    policy: Any
    preprocess: Any
    postprocess: Any
    device: Any
    camera_keys: list[str]
    policy_path: str
    n_action_steps: int = 0
    noise_draws: int = 0
    _noise_generator: Any = None

    def assign_seed(self, episode_seed: int) -> None:
        """Assign ``episode_seed`` to this episode's two noise streams.

        One value is assigned to both sources of randomness in a rollout:

        * the **environment's observation noise**, where ``rollout`` passes the very same
          value to ``env.reset(seed=episode_seed)``; that streams from numpy's generator;
        * the **policy's flow-matching sampling noise**, assigned here to a
          ``torch.Generator``.

        SmolVLA integrates a flow-matching ODE from a standard-normal sample, so the
        injected noise decides the action chunk. Assigning the *same* seed makes the k-th
        replanning step of an episode use the k-th draw of that seed: two rollouts sharing
        a seed therefore share their noise bit for bit, which is what makes the paired
        comparison in ``kitchen-instruction-probe`` meaningful (otherwise sampling
        randomness cannot be told apart from a reaction to the language). Different
        episodes are given different seeds, so one run averages over both noises.

        Also clears the policy's action queue, so call this once at the start of every
        episode, not between replannings.
        """
        import torch

        self.policy.reset()
        self.noise_draws = 0
        self._noise_generator = torch.Generator(device=self.device).manual_seed(
            int(episode_seed) % (2**63)
        )

    def _draw_noise(self):
        import torch

        shape = (1, self.policy.config.chunk_size, self.policy.config.max_action_dim)
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=self.device,
            generator=self._noise_generator,
        )
        self.noise_draws += 1
        return noise

    def select_action(
        self, state: np.ndarray, frames: dict[str, np.ndarray], instruction: str, robot_type: str
    ) -> np.ndarray:
        import torch

        from lerobot.policies.utils import prepare_observation_for_inference

        raw: dict[str, np.ndarray] = {
            "observation.state": np.asarray(state, dtype=np.float32),
        }
        for key in self.camera_keys:
            name = key.split(".")[-1]
            raw[key] = np.asarray(frames[name], dtype=np.uint8)
        batch = prepare_observation_for_inference(
            raw, device=self.device, task=instruction, robot_type=robot_type
        )
        batch = self.preprocess(batch)
        noise = self._draw_noise()
        with torch.inference_mode():
            action = self.policy.select_action(batch, noise=noise)
        return self.postprocess(action).squeeze(0).detach().cpu().numpy()


def load_policy(
    policy_path: str | Path,
    device: str = "cuda",
    execution_horizon: int | None = None,
) -> LoadedPolicy:
    """Load a fine-tuned SmolVLA checkpoint together with its saved processors.

    The processors are the ones written during training, so the normalization
    statistics and the rename map are the exact ones the weights were trained
    with; do not rebuild them by hand.
    """
    import torch
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAPolicy

    policy_path = Path(policy_path).expanduser()
    if not policy_path.is_dir():
        raise FileNotFoundError(
            f"Local policy directory does not exist: {policy_path}. "
            "Pass a saved checkpoint directory ending in 'pretrained_model'."
        )
    missing = [
        name for name in ("config.json", "model.safetensors") if not (policy_path / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Incomplete policy checkpoint {policy_path}: missing {', '.join(missing)}"
        )
    policy_path = policy_path.resolve()

    resolved_device = torch.device(device)
    policy = SmolVLAPolicy.from_pretrained(str(policy_path))
    if execution_horizon is not None:
        # Inference-only knob: how many of the predicted chunk_size actions are
        # queued before the policy replans. The training loss always covers the
        # full chunk, so changing this never invalidates the checkpoint.
        if not 1 <= execution_horizon <= policy.config.chunk_size:
            raise ValueError(
                f"--execution-horizon must be in [1, {policy.config.chunk_size}], "
                f"got {execution_horizon}"
            )
        policy.config.n_action_steps = execution_horizon
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        str(policy_path),
        preprocessor_overrides={"device_processor": {"device": str(resolved_device)}},
    )
    policy.to(resolved_device)
    policy.eval()
    camera_keys = [key for key in policy.config.image_features]
    if not camera_keys:
        raise ValueError(
            f"{policy_path} declares no image features; it was not trained on image observations."
        )
    return LoadedPolicy(
        policy=policy,
        preprocess=preprocess,
        postprocess=postprocess,
        device=resolved_device,
        camera_keys=camera_keys,
        policy_path=str(policy_path),
        n_action_steps=int(policy.config.n_action_steps),
    )


END_EFFECTOR_SITE = "end_effector"


def hide_end_effector_site(model) -> bool:
    """Make the ``end_effector`` site invisible in renders. Returns whether it was found.

    MuJoCo draws every site as a sphere, so this model's TCP marker shows up as a 1 cm grey
    ball (``rgba=[0.5, 0.5, 0.5, 1]``) that is dark on its unlit side -- i.e. a small black
    dot sitting in the gripper in every rendered frame. It is a modelling aid, not geometry,
    and does not exist on a real robot, so it is hidden before rendering.

    Alpha 0 is used rather than a zero size because it leaves the site's geometry untouched;
    the two were measured to produce pixel-identical images. Only this one site is touched --
    the task sites (knobs, kettle, light switch, ...) are left alone.
    """
    import mujoco

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, END_EFFECTOR_SITE)
    if site_id < 0:
        return False
    model.site_rgba[site_id, 3] = 0.0
    return True


def make_kitchen_env(
    tasks: Sequence[str],
    terminate_on_tasks_completed: bool = True,
    robot_noise_ratio: float = 0.0,
    object_noise_ratio: float = 0.0,
    max_episode_steps: int | None = None,
):
    """Build FrankaKitchen-v1 directly (not via Minari) for closed-loop rollouts."""
    import gymnasium as gym
    import gymnasium_robotics

    from .tasks import TASK_GOALS

    unknown = set(tasks).difference(TASK_GOALS)
    if unknown:
        raise ValueError(f"Unknown task(s): {sorted(unknown)}; available: {sorted(TASK_GOALS)}")
    gym.register_envs(gymnasium_robotics)
    kwargs: dict[str, Any] = {
        "tasks_to_complete": list(tasks),
        "terminate_on_tasks_completed": terminate_on_tasks_completed,
        # Removing a finished task makes info["step_task_completions"] mean
        # "newly completed in this step", which is what we want for ordering.
        "remove_task_when_completed": True,
        "robot_noise_ratio": robot_noise_ratio,
        "object_noise_ratio": object_noise_ratio,
    }
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    env = gym.make("FrankaKitchen-v1", **kwargs)
    hide_end_effector_site(env.unwrapped.model)
    return env


def ordered_prefix_length(requested: Sequence[str], completed: Sequence[str]) -> int:
    """Number of requested tasks completed in the instructed order from the start.

    The Kitchen environment treats ``tasks_to_complete`` as a set, so its native
    termination signal only means that all requested goals were reached.  Sequence
    evaluation must additionally compare the order of the first-completion events.
    Completions unrelated to the requested sequence are ignored here; the closed-loop
    evaluator only asks the environment to track requested tasks, while the instruction
    probe deliberately tracks all seven tasks.
    """
    requested_list = list(requested)
    requested_set = set(requested_list)
    relevant = [task for task in completed if task in requested_set]
    length = 0
    for expected, actual in zip(requested_list, relevant, strict=False):
        if actual != expected:
            break
        length += 1
    return length


@dataclass
class Rollout:
    instruction: str
    seed: int
    tasks_requested: list[str]
    completed: list[str] = field(default_factory=list)
    ordered_prefix_length: int = 0
    all_tasks_completed: bool = False
    success: bool = False
    steps: int = 0
    actions: np.ndarray = field(default_factory=lambda: np.zeros((0, 9), dtype=np.float32))
    robot_qpos: np.ndarray = field(default_factory=lambda: np.zeros((0, 9), dtype=np.float32))
    videos: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "instruction": self.instruction,
            "seed": self.seed,
            "tasks_requested": list(self.tasks_requested),
            "completed": list(self.completed),
            "ordered_prefix_length": self.ordered_prefix_length,
            "all_tasks_completed": bool(self.all_tasks_completed),
            "success": bool(self.success),
            "steps": self.steps,
            "videos": dict(self.videos),
        }


def rollout(
    loaded: LoadedPolicy,
    env: Any,
    rig: Any,
    instruction: str,
    seed: int,
    max_steps: int,
    tasks_requested: Sequence[str] = (),
    action_clip: float = 1.0,
    record_dir: Path | None = None,
    fps: int = 12,
    robot_type: str = "franka_kitchen",
) -> Rollout:
    """Run one closed-loop episode and record what actually happened.

    Actions are clipped to the environment's Box(-1, 1) action space; a flow
    matching policy that has not converged will happily emit values outside it,
    and letting those through produces NaNs rather than a readable failure.
    """
    import imageio.v2 as imageio

    observation, _ = env.reset(seed=seed)
    # The same `seed` is assigned to both noise streams: env.reset above seeds the
    # environment's observation noise, assign_seed seeds the policy's sampling noise.
    loaded.assign_seed(seed)
    completed: list[str] = []
    action_log: list[np.ndarray] = []
    qpos_log: list[np.ndarray] = []
    writers: dict[str, Any] = {}
    videos: dict[str, str] = {}
    if record_dir is not None and rig is not None:
        record_dir.mkdir(parents=True, exist_ok=True)
        for name in rig.names:
            path = record_dir / f"seed{seed:05d}_{name}.mp4"
            writers[name] = imageio.get_writer(
                path, fps=fps, codec="libx264", macro_block_size=1
            )
            videos[name] = str(path)

    step = 0
    try:
        for step in range(max_steps):
            frames = rig.render(env.unwrapped.data) if rig is not None else {}
            for name, writer in writers.items():
                writer.append_data(frames[name])
            state = robot_state_from_observation(observation["observation"])
            action = loaded.select_action(state, frames, instruction, robot_type)
            action = np.clip(np.asarray(action, dtype=np.float32), -action_clip, action_clip)
            action_log.append(action)
            qpos_log.append(state[:ROBOT_QPOS_DIM])
            observation, _, terminated, truncated, info = env.step(action)
            for task in info["step_task_completions"]:
                if task not in completed:
                    completed.append(task)
            if terminated or truncated:
                break
    finally:
        for writer in writers.values():
            writer.close()

    requested = list(tasks_requested)
    prefix_length = ordered_prefix_length(requested, completed)
    all_tasks_completed = bool(requested) and all(task in completed for task in requested)
    return Rollout(
        instruction=instruction,
        seed=seed,
        tasks_requested=requested,
        completed=completed,
        ordered_prefix_length=prefix_length,
        all_tasks_completed=all_tasks_completed,
        success=bool(requested) and prefix_length == len(requested),
        steps=step + 1,
        actions=np.asarray(action_log, dtype=np.float32),
        robot_qpos=np.asarray(qpos_log, dtype=np.float32),
        videos=videos,
    )
