from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Instruction-sensitivity probe. Runs two instructions from the SAME reset state "
            "on the same seeds and reports how far the two action trajectories diverge. "
            "A language-blind policy produces bit-identical rollouts; a policy that reads "
            "the instruction produces different ones."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Recommended pairs (all in-distribution if training used full sequences):\n"
            "  1. correct vs reversed order of the same task set (tests order sensitivity)\n"
            "  2. the instructed set vs an unrelated set (tests task sensitivity)\n\n"
            "Example:\n"
            "  kitchen-instruction-probe --policy-path ... \\\n"
            "      --tasks-a microwave kettle --reverse-of-a --seeds 0 1 2 3 4\n"
        ),
    )
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument(
        "--tasks-a",
        required=True,
        help="Comma-separated task list for instruction A, e.g. 'microwave,kettle'.",
    )
    parser.add_argument(
        "--tasks-b",
        help="Comma-separated task list for instruction B. Default: reversed --tasks-a.",
    )
    parser.add_argument(
        "--reverse-of-a",
        action="store_true",
        help="Use the reversed --tasks-a as instruction B (explicit form of the default).",
    )
    parser.add_argument("--instruction-a", help="Raw instruction string overriding --tasks-a.")
    parser.add_argument("--instruction-b", help="Raw instruction string overriding --tasks-b.")
    parser.add_argument(
        "--prompt-style", choices=("natural", "compact"), default="natural"
    )
    parser.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--steps",
        type=int,
        default=120,
        help="Rollout length per instruction (default 120, ~10 s at 12 Hz).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--action-clip", type=float, default=1.0)
    parser.add_argument(
        "--change-tolerance",
        type=float,
        default=1e-3,
        help="Per-step action L2 above this counts as 'the policy did something different'.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/probe"))
    parser.add_argument(
        "--allow-existing-output", action="store_true"
    )
    return parser.parse_args()


def parse_tasks(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def compare_rollouts(a, b, tolerance: float) -> dict:
    """Numeric divergence between two rollouts of the same length."""
    steps = min(len(a.actions), len(b.actions))
    if steps == 0:
        return {"comparable_steps": 0}
    delta = a.actions[:steps] - b.actions[:steps]
    per_step = np.linalg.norm(delta, axis=-1)
    qpos_delta = np.linalg.norm(
        a.robot_qpos[:steps] - b.robot_qpos[:steps], axis=-1
    )
    changed = per_step > tolerance
    return {
        "comparable_steps": int(steps),
        "identical_steps": int(np.count_nonzero(per_step <= tolerance)),
        "changed_step_fraction": float(changed.mean()),
        "action_l2_mean": float(per_step.mean()),
        "action_l2_p50": float(np.percentile(per_step, 50)),
        "action_l2_max": float(per_step.max()),
        "robot_qpos_l2_final": float(qpos_delta[-1]),
        "robot_qpos_l2_max": float(qpos_delta.max()),
        "same_trajectory": bool(changed.mean() == 0.0),
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    from .cameras import CameraRig
    from .runtime import load_policy, make_kitchen_env, rollout
    from .tasks import TASK_GOALS, TASK_INDICES, instruction_for

    if args.tasks_b and args.reverse_of_a:
        raise SystemExit("pass either --tasks-b or --reverse-of-a, not both")
    tasks_a = parse_tasks(args.tasks_a)
    if args.reverse_of_a or not args.tasks_b:
        tasks_b = list(reversed(tasks_a))
    else:
        tasks_b = parse_tasks(args.tasks_b)
    for name, tasks in (("--tasks-a", tasks_a), ("--tasks-b", tasks_b)):
        unknown = set(tasks).difference(TASK_GOALS)
        if unknown:
            raise ValueError(f"{name}: unknown task(s) {sorted(unknown)}")
        if not tasks:
            raise ValueError(f"{name} is empty")
    instruction_a = args.instruction_a or instruction_for(tasks_a, style=args.prompt_style)
    instruction_b = args.instruction_b or instruction_for(tasks_b, style=args.prompt_style)

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
    # Goal set = every task, so we can see which object the policy actually went for.
    env = make_kitchen_env(
        list(TASK_INDICES),
        terminate_on_tasks_completed=False,
        max_episode_steps=args.steps,
    )
    rig = CameraRig(env.unwrapped.model, args.camera_config)
    fps = int(env.unwrapped.metadata.get("render_fps", 12))

    print(f"policy      : {args.policy_path}")
    print(f"instruction A: {instruction_a}")
    print(f"instruction B: {instruction_b}")
    print(f"seeds        : {args.seeds}   steps: {args.steps}")
    print(
        f"noise        : "
        + "both arms are assigned the same episode seed, so both noise streams match"
        + "\n"
    )

    per_seed = []
    try:
        for seed in args.seeds:
            result_a = rollout(
                loaded, env, rig, instruction_a, seed=seed, max_steps=args.steps,
                tasks_requested=tasks_a, action_clip=args.action_clip, fps=fps,
            )
            result_b = rollout(
                loaded, env, rig, instruction_b, seed=seed, max_steps=args.steps,
                tasks_requested=tasks_b, action_clip=args.action_clip, fps=fps,
            )
            divergence = compare_rollouts(result_a, result_b, args.change_tolerance)
            entry = {
                "seed": seed,
                "a": result_a.to_json(),
                "b": result_b.to_json(),
                "divergence": divergence,
            }
            per_seed.append(entry)
            print(
                f"  seed={seed:<4d} A completed={result_a.completed} "
                f"| B completed={result_b.completed} "
                f"| changed_steps={divergence.get('changed_step_fraction', float('nan')):.2f} "
                f"actionL2_mean={divergence.get('action_l2_mean', float('nan')):.4f}"
            )
    finally:
        rig.close()
        env.close()

    def first_task_of(instruction_key: str) -> float:
        hits = 0
        for entry in per_seed:
            requested = entry[instruction_key]["tasks_requested"]
            completed = entry[instruction_key]["completed"]
            if requested and completed and completed[0] == requested[0]:
                hits += 1
        return hits / len(per_seed) if per_seed else float("nan")

    changed = [entry["divergence"].get("changed_step_fraction", 0.0) for entry in per_seed]
    l2 = [entry["divergence"].get("action_l2_mean", 0.0) for entry in per_seed]
    summary = {
        "policy_path": str(args.policy_path),
        "instruction_a": instruction_a,
        "instruction_b": instruction_b,
        "tasks_a": tasks_a,
        "tasks_b": tasks_b,
        "seeds": args.seeds,
        "steps": args.steps,
        "execution_horizon": args.execution_horizon,
        "noise": {
            "episode_seed_assigns": ["env_observation_noise", "policy_sampling_noise"],
            "arms_share_the_episode_seed": True,
        },
        "seeds_with_identical_rollouts": int(
            sum(1 for entry in per_seed if entry["divergence"].get("same_trajectory"))
        ),
        "mean_changed_step_fraction": float(np.mean(changed)) if changed else None,
        "mean_action_l2": float(np.mean(l2)) if l2 else None,
        "first_task_matches_instruction_a_rate": first_task_of("a"),
        "first_task_matches_instruction_b_rate": first_task_of("b"),
        "per_seed": per_seed,
    }
    report_path = args.output_dir / "probe_report.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n--- summary ---")
    print(f"  seeds whose rollouts were bit-identical : {summary['seeds_with_identical_rollouts']}/{len(per_seed)}")
    print(f"  mean fraction of steps that differ      : {summary['mean_changed_step_fraction']:.3f}")
    print(f"  mean per-step action L2                 : {summary['mean_action_l2']:.4f}")
    print(f"  first completed == first instructed (A) : {summary['first_task_matches_instruction_a_rate']:.2f}")
    print(f"  first completed == first instructed (B) : {summary['first_task_matches_instruction_b_rate']:.2f}")
    print(f"\nreport: {report_path}")
    if summary["seeds_with_identical_rollouts"] == len(per_seed) and per_seed:
        print(
            "\nWARNING: every pair of rollouts was identical under identical noise, so the "
            "policy is not reacting to the language input in this setup."
        )


if __name__ == "__main__":
    main()
