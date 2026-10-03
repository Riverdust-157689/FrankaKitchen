from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a Minari Franka Kitchen demonstration.")
    parser.add_argument("--dataset-id", default="D4RL/kitchen/complete-v2")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/minari"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--mode", choices=("actions", "states"), default="actions")
    parser.add_argument(
        "--init-state",
        choices=("episode", "env"),
        default=None,
        help=(
            "Where to start an actions replay. 'episode' (default for --mode actions) restores "
            "the episode's own first observation, which is required to reproduce the expert: "
            "the recorded episodes do not all start at the environment's reset state. "
            "'env' replays straight after env.reset() and measures that drift."
        ),
    )
    parser.add_argument(
        "--all-tasks-env",
        action="store_true",
        help=(
            "For --mode actions, build FrankaKitchen-v1 with all 7 tasks as goals so that "
            "bottom burner / top burner / hinge cabinet are reported too. The dataset's own "
            "env spec only covers the 4 tasks listed in tasks_to_complete."
        ),
    )
    parser.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    parser.add_argument("--cameras", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("previews/replay"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    from .cameras import CameraRig
    from .minari_utils import load_dataset, make_replay_environment, set_state_from_observation
    from .tasks import completion_steps, instruction_for

    dataset = load_dataset(args.dataset_id, args.dataset_root)
    episodes = list(dataset.iterate_episodes(episode_indices=[args.episode]))
    if not episodes:
        raise IndexError(f"Episode {args.episode} is not present in {args.dataset_id}.")
    episode = episodes[0]
    init_state = args.init_state or ("episode" if args.mode == "actions" else "env")
    if args.all_tasks_env and args.mode == "actions":
        from .tasks import TASK_INDICES
        from .runtime import make_kitchen_env

        env = make_kitchen_env(list(TASK_INDICES), terminate_on_tasks_completed=False)
    else:
        env = make_replay_environment(dataset)
    env.reset(seed=0)
    rig = None if args.no_video else CameraRig(env.unwrapped.model, args.camera_config)
    selected_cameras = args.cameras or (rig.names if rig else [])
    writers: dict[str, imageio.Writer] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fps = int(env.unwrapped.metadata.get("render_fps", 12))
    if rig:
        for name in selected_cameras:
            writers[name] = imageio.get_writer(
                args.output_dir / f"episode_{args.episode:04d}_{name}.mp4",
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )

    flat_observations = episode.observations["observation"]
    step_count = min(len(episode.actions), args.max_steps or len(episode.actions))
    rmse: list[float] = []
    unique_completions: list[str] = []
    reward_sum = 0.0
    try:
        observation, _ = env.reset(seed=0)
        if args.mode == "actions" and init_state == "episode":
            set_state_from_observation(env, flat_observations[0])
        for step in range(step_count):
            if args.mode == "states":
                set_state_from_observation(env, flat_observations[step])
            else:
                rmse.append(
                    float(
                        np.sqrt(
                            np.mean(
                                (observation["observation"] - flat_observations[step]) ** 2
                            )
                        )
                    )
                )
            if rig:
                for name, frame in rig.render(env.unwrapped.data, selected_cameras).items():
                    writers[name].append_data(frame)
            if args.mode == "actions":
                observation, reward, _, _, info = env.step(episode.actions[step])
                reward_sum += float(reward)
                for task in info["step_task_completions"]:
                    if task not in unique_completions:
                        unique_completions.append(task)

        if args.mode == "states":
            unique_completions = [
                task for frame, task in completion_steps(flat_observations[: step_count + 1])
                if frame <= step_count
            ]
    finally:
        for writer in writers.values():
            writer.close()
        if rig:
            rig.close()
        env.close()

    report = {
        "dataset_id": args.dataset_id,
        "episode": args.episode,
        "mode": args.mode,
        "init_state": init_state if args.mode == "actions" else None,
        "all_tasks_env": bool(args.all_tasks_env and args.mode == "actions"),
        "steps": step_count,
        "completed_tasks": unique_completions,
        "instruction": instruction_for(unique_completions) if unique_completions else None,
        "success": set(dataset.spec.env_spec.kwargs.get("tasks_to_complete", [])).issubset(
            unique_completions
        ),
        "replayed_reward_sum": reward_sum if args.mode == "actions" else None,
        "observation_rmse_mean": float(np.mean(rmse)) if rmse else None,
        "observation_rmse_max": float(np.max(rmse)) if rmse else None,
        "videos": [str(args.output_dir / f"episode_{args.episode:04d}_{name}.mp4") for name in writers],
    }
    report_path = args.output_dir / f"episode_{args.episode:04d}_{args.mode}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
