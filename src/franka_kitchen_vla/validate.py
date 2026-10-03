from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check the two things the training labels depend on: (1) that the geometric "
            "task-completion detection agrees with the goal values the environment used, "
            "and (2) that the recorded expert actions still reproduce the same task order "
            "when replayed in the current FrankaKitchen-v1 build."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-id", default="D4RL/kitchen/partial-v2")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/minari"))
    parser.add_argument(
        "--episodes",
        type=int,
        default=25,
        help="Number of episodes to replay (labels are always checked on every scanned episode).",
    )
    parser.add_argument("--episode-stride", type=int, default=None, help="Default: spread over the set.")
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument(
        "--max-label-episodes",
        type=int,
        default=None,
        help="Cap the (cheap) label comparison pass; default is every episode.",
    )
    parser.add_argument(
        "--replay-goal-set",
        choices=("all", "dataset"),
        default="all",
        help=(
            "all (default): build FrankaKitchen-v1 with all 7 tasks as goals, so bottom "
            "burner, top burner and hinge cabinet can be reported too. dataset: reuse the "
            "recording environment, whose goal set covers only the 4 tasks in the spec."
        ),
    )
    parser.add_argument("--no-replay", action="store_true", help="Only run the label check.")
    parser.add_argument(
        "--replay-start",
        choices=("episode-init", "env-reset"),
        default="episode-init",
        help=(
            "episode-init (default): restore the episode's own first observation before "
            "replaying, which is required because the recorded episodes do not all start at "
            "env.reset(). env-reset: replay straight after env.reset() to expose that drift."
        ),
    )
    parser.add_argument("--output", type=Path, help="Write the JSON report here.")
    return parser.parse_args()


def _env_authoritative_events(episode) -> list[tuple[int, str]]:
    """Completion events computed the way KitchenEnv.compute_reward does."""
    from .tasks import BONUS_THRESHOLD

    achieved = episode.observations["achieved_goal"]
    desired = episode.observations["desired_goal"]
    events = []
    for task in achieved:
        distance = np.linalg.norm(np.asarray(achieved[task]) - np.asarray(desired[task]), axis=-1)
        hits = np.flatnonzero(distance < BONUS_THRESHOLD)
        if hits.size:
            events.append((int(hits[0]), task))
    return sorted(events)


def check_labels(args: argparse.Namespace) -> dict:
    from .minari_utils import load_dataset
    from .tasks import episode_completion_events, goal_entries

    dataset = load_dataset(args.dataset_id, args.dataset_root)
    compared = 0
    exact = 0
    disagreements = []
    fallback_only = 0
    for index, episode in enumerate(dataset.iterate_episodes()):
        if args.max_label_episodes is not None and index >= args.max_label_episodes:
            break
        achieved, desired = goal_entries(episode)
        if not achieved or not desired:
            continue
        authoritative = _env_authoritative_events(episode)
        detected = episode_completion_events(episode)
        restricted = [(frame, task) for frame, task in detected if task in achieved]
        compared += 1
        if restricted == authoritative:
            exact += 1
        elif len(disagreements) < 10:
            disagreements.append(
                {
                    "episode": index,
                    "env_goal_events": authoritative,
                    "geometric_events": restricted,
                }
            )
        fallback_only += sum(1 for _, task in detected if task not in achieved)
    return {
        "episodes_compared": compared,
        "exact_matches": exact,
        "agreement_rate": (exact / compared) if compared else None,
        "disagreements": disagreements,
        "events_from_geometric_fallback": fallback_only,
    }


def replay_episode(env, episode, init_from_episode: bool) -> tuple[list[tuple[int, str]], float]:
    from .minari_utils import set_state_from_observation

    env.reset(seed=0)
    if init_from_episode:
        set_state_from_observation(env, np.asarray(episode.observations["observation"])[0])
    first_seen: dict[str, int] = {}
    reward_sum = 0.0
    for step, action in enumerate(episode.actions):
        _, reward, terminated, truncated, info = env.step(action)
        reward_sum += float(reward)
        for task in info["step_task_completions"]:
            first_seen.setdefault(task, step)
        if terminated or truncated:
            break
    return sorted((frame, task) for task, frame in first_seen.items()), reward_sum


def check_replay(args: argparse.Namespace) -> dict:
    import gymnasium as gym
    import gymnasium_robotics

    from .minari_utils import load_dataset, make_replay_environment
    from .tasks import TASK_INDICES, episode_completion_events

    dataset = load_dataset(args.dataset_id, args.dataset_root)
    total = int(dataset.total_episodes)
    if args.episode_stride is None:
        stride = max(1, total // max(1, args.episodes))
    else:
        stride = args.episode_stride
    indices = list(range(args.start_episode, total, stride))[: args.episodes]

    if args.replay_goal_set == "all":
        gym.register_envs(gymnasium_robotics)
        env = gym.make(
            "FrankaKitchen-v1",
            tasks_to_complete=list(TASK_INDICES),
            remove_task_when_completed=False,
            terminate_on_tasks_completed=False,
            robot_noise_ratio=0.0,
            object_noise_ratio=0.0,
        )
    else:
        env = make_replay_environment(dataset)

    per_task = {task: {"expected": 0, "reproduced": 0} for task in TASK_INDICES}
    exact = 0
    results = []
    try:
        for episode_index in indices:
            episode = list(dataset.iterate_episodes(episode_indices=[episode_index]))[0]
            expected = [task for _, task in episode_completion_events(episode)]
            reproduced, reward_sum = replay_episode(
                env, episode, args.replay_start == "episode-init"
            )
            reproduced_tasks = [task for _, task in reproduced]
            match = expected == reproduced_tasks
            exact += match
            for task in expected:
                per_task[task]["expected"] += 1
                if task in reproduced_tasks:
                    per_task[task]["reproduced"] += 1
            results.append(
                {
                    "episode": episode_index,
                    "actions": int(episode.actions.shape[0]),
                    "expected": expected,
                    "reproduced": reproduced_tasks,
                    "replay_reward_sum": reward_sum,
                    "exact_order_match": bool(match),
                }
            )
    finally:
        env.close()

    recall = {
        task: (values["reproduced"] / values["expected"]) if values["expected"] else None
        for task, values in per_task.items()
    }
    return {
        "episodes_replayed": len(indices),
        "exact_order_matches": exact,
        "exact_order_rate": (exact / len(indices)) if indices else None,
        "per_task_reproduction_recall": recall,
        "per_task_counts": per_task,
        "replay_start": args.replay_start,
        "replay_goal_set": args.replay_goal_set,
        "episodes": results,
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    report: dict = {"dataset_id": args.dataset_id}
    print(f"label check on {args.dataset_id} ...", flush=True)
    report["labels"] = check_labels(args)
    labels = report["labels"]
    rate = labels["agreement_rate"]
    if rate is None:
        print("  no episodes compared")
    else:
        print(
            "  goal-authoritative vs geometric detection: "
            f"{labels['exact_matches']}/{labels['episodes_compared']} exact ({rate * 100:.2f}%)"
        )
    print(f"  events that only the geometric fallback can see: {labels['events_from_geometric_fallback']}")
    for item in labels["disagreements"]:
        print(f"    ep{item['episode']}: env={item['env_goal_events']} geom={item['geometric_events']}")

    if not args.no_replay:
        print(f"replaying expert actions (episodes={args.episodes}) ...", flush=True)
        report["replay"] = check_replay(args)
        replay = report["replay"]
        exact_rate = replay["exact_order_rate"]
        if exact_rate is None:
            print("  no episodes replayed")
        else:
            print(
                "  exact task-order reproduction: "
                f"{replay['exact_order_matches']}/{replay['episodes_replayed']} "
                f"({exact_rate * 100:.1f}%)"
            )
        print("  per-task recall:")
        for task, value in replay["per_task_reproduction_recall"].items():
            counts = replay["per_task_counts"][task]
            shown = f"{value * 100:5.1f}%" if value is not None else "    n/a"
            print(f"    {task:16s} {shown}  ({counts['reproduced']}/{counts['expected']})")
        for entry in replay["episodes"]:
            flag = "OK " if entry["exact_order_match"] else "DIFF"
            print(f"    {flag} ep{entry['episode']:<4d} expected={entry['expected']}")
            if not entry["exact_order_match"]:
                print(f"         reproduced={entry['reproduced']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
