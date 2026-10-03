from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Closed-loop evaluation of a fine-tuned SmolVLA checkpoint in FrankaKitchen-v1. "
            "Success is decided by the environment's own goal thresholds, never by the model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  kitchen-eval --policy-path outputs/train/run/checkpoints/last/pretrained_model \\\n"
            "      --tasks microwave kettle --episodes 20 --record\n\n"
            "  # instruction control: same seeds, deliberately wrong order\n"
            "  kitchen-eval --policy-path ... --tasks kettle microwave \\\n"
            "      --seeds 0 1 2 3 4\n"
        ),
    )
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument(
        "--instruction",
        help="Override the instruction string instead of building it from --tasks.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=("natural", "compact"),
        default="natural",
        help="Must match the style used by kitchen-to-lerobot.",
    )
    parser.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help=(
            "Explicit per-episode seeds. Overrides --episodes/--seed. Each value is assigned "
            "to BOTH noise streams of that episode: the environment's observation noise "
            "(env.reset(seed=...)) and the policy's flow-matching sampling noise "
            "(LoadedPolicy.assign_seed)."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Base seed; episode i is assigned seed+i (see --seeds). This single value assigns "
            "BOTH noise streams: the environment's observation noise and the policy's "
            "flow-matching sampling noise. It does not change the initial state, which is "
            "fixed by the environment's reset_model()."
        ),
    )
    parser.add_argument(
        "--robot-noise-ratio",
        type=float,
        default=0.0,
        help=(
            "Amplitude of the environment's observation noise on the robot joints: "
            "noise = ratio * 0.1 * U(-1,1) per joint. This is OBSERVATION noise, not a reset "
            "perturbation -- the initial state never changes. The realisation is decided by "
            "--seed (changing this ratio rescales the same drawn uniforms rather than "
            "redrawing them), and since the noisy qpos is fed back through the controller it "
            "also perturbs the trajectory. Environment default is 0.01; 0.0 switches it off."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=10,
        help=(
            "How many actions of each predicted chunk are executed before replanning "
            "(default 10). The base config uses 50, which is fully open-loop for a step "
            "and reacts far too slowly in a 280-step episode."
        ),
    )
    parser.add_argument("--action-clip", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument(
        "--record", action="store_true", help="Write one mp4 per camera per episode."
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        help="Where --record writes; default <output-dir>/videos.",
    )
    parser.add_argument(
        "--terminate-on-success",
        action="store_true",
        help="Stop the episode once every requested task is complete (saves time).",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Append to --output-dir instead of refusing when it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    from .cameras import CameraRig
    from .runtime import load_policy, make_kitchen_env, rollout
    from .tasks import TASK_GOALS, instruction_for

    unknown = set(args.tasks).difference(TASK_GOALS)
    if unknown:
        raise ValueError(f"Unknown task(s): {sorted(unknown)}; available: {sorted(TASK_GOALS)}")
    instruction = args.instruction or instruction_for(args.tasks, style=args.prompt_style)
    seeds = args.seeds or [args.seed + index for index in range(args.episodes)]

    if args.output_dir.exists() and not args.allow_existing_output:
        raise FileExistsError(
            f"{args.output_dir} already exists. Pick a new path or pass --allow-existing-output."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_policy(
        args.policy_path,
        device=args.device,
        execution_horizon=args.execution_horizon,
    )
    print(
        f"policy={args.policy_path}\n"
        f"  cameras={loaded.camera_keys} chunk={loaded.policy.config.chunk_size} "
        f"execution_horizon={loaded.n_action_steps} device={args.device}"
    )
    print(f"  instruction: {instruction}")
    print(
        f"  noise: one seed per episode, assigned to both streams ({seeds[0]}..{seeds[-1]})\n"
        f"    env observation noise : robot_noise_ratio={args.robot_noise_ratio} "
        f"(amplitude only; realisation comes from the episode seed)\n"
        f"    policy sampling noise : flow-matching generator seeded from that same episode seed"
    )

    env = make_kitchen_env(
        args.tasks,
        terminate_on_tasks_completed=args.terminate_on_success,
        robot_noise_ratio=args.robot_noise_ratio,
        # Object noise only perturbs observation dims 18:59, which never reach the policy
        # (schema.robot_state_from_observation keeps the first 18) and never reach the
        # controller, so it cannot change a rollout. Pinned off rather than exposed.
        object_noise_ratio=0.0,
        max_episode_steps=args.max_steps,
    )
    rig = CameraRig(env.unwrapped.model, args.camera_config)
    missing = set(loaded.camera_keys).difference(f"observation.images.{n}" for n in rig.names)
    if missing:
        raise ValueError(
            f"The checkpoint expects cameras {sorted(missing)} but configs/cameras.yaml "
            f"provides {rig.names}. The camera key names must match the training dataset."
        )
    fps = int(env.unwrapped.metadata.get("render_fps", 12))
    record_root = None
    if args.record or args.record_dir is not None:
        record_root = args.record_dir or (args.output_dir / "videos")

    results = []
    try:
        for seed in seeds:
            episode_dir = (record_root / f"episode_seed{seed:05d}") if record_root else None
            result = rollout(
                loaded,
                env,
                rig,
                instruction,
                seed=seed,
                max_steps=args.max_steps,
                tasks_requested=args.tasks,
                action_clip=args.action_clip,
                record_dir=episode_dir,
                fps=fps,
            )
            results.append(result)
            flag = "OK  " if result.success else "FAIL"
            print(
                f"  {flag} seed={seed:<4d} steps={result.steps:<4d} "
                f"completed={result.completed}"
            )
    finally:
        rig.close()
        env.close()

    successes = [result.success for result in results]
    all_tasks_completed = [result.all_tasks_completed for result in results]
    task_completion_rate = {
        task: float(np.mean([task in result.completed for result in results]))
        for task in args.tasks
    }
    ordered_prefix_success_rate = {
        f"{index}:{task}": float(
            np.mean([result.ordered_prefix_length >= index for result in results])
        )
        for index, task in enumerate(args.tasks, start=1)
    }
    report = {
        "policy_path": str(args.policy_path),
        "tasks": args.tasks,
        "instruction": instruction,
        "prompt_style": args.prompt_style,
        "execution_horizon": args.execution_horizon,
        "noise": {
            "episode_seed_assigns": ["env_observation_noise", "policy_sampling_noise"],
            "robot_noise_ratio": args.robot_noise_ratio,
            "object_noise_ratio": 0.0,
        },
        "max_steps": args.max_steps,
        "camera_config": str(args.camera_config),
        "camera_keys_expected_by_policy": loaded.camera_keys,
        "seeds": seeds,
        "task_completion_rate": task_completion_rate,
        "ordered_prefix_success_rate": ordered_prefix_success_rate,
        "all_tasks_completion_rate": (
            float(np.mean(all_tasks_completed)) if all_tasks_completed else None
        ),
        "success_rate": float(np.mean(successes)) if successes else None,
        "episodes": [result.to_json() for result in results],
    }
    report_path = args.output_dir / f"report_{'_'.join(args.tasks)[:60]}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nper-task completion rate:")
    for task, rate in task_completion_rate.items():
        print(f"  {task:<16s} {rate:.3f}")
    print("ordered-prefix success rate:")
    for stage, rate in ordered_prefix_success_rate.items():
        print(f"  {stage:<18s} {rate:.3f}")
    print(
        f"all_tasks_completion_rate = {report['all_tasks_completion_rate']:.3f}\n"
        f"ordered_sequence_success_rate = {report['success_rate']:.3f}  "
        f"({sum(successes)}/{len(results)})"
    )
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
