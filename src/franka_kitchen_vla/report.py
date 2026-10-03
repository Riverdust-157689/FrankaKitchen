from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Describe a Minari D4RL Kitchen dataset and/or a converted LeRobot dataset: "
            "which tasks exist, how they are ordered, and which prompt string each order "
            "produces."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=None,
        help="Minari id to inspect; repeatable. Default: complete-v2, partial-v2, mixed-v2.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/minari"))
    parser.add_argument("--max-episodes", type=int, help="Stop after this many source episodes.")
    parser.add_argument("--prompt-style", choices=("natural", "compact"), default="natural")
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        action="append",
        default=None,
        help="Also describe a converted LeRobot dataset directory; repeatable.",
    )
    parser.add_argument("--json-out", type=Path, help="Write the full report as JSON.")
    parser.add_argument(
        "--top",
        type=int,
        default=40,
        help="How many task sequences to list (default 40, 0 = all).",
    )
    return parser.parse_args()


def prompt_token_lengths(prompts: list[str]) -> dict[str, int]:
    """Return token counts per prompt, or {} when transformers is unavailable."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return {}
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", local_files_only=True
        )
    except Exception:
        try:
            from transformers import AutoTokenizer as _T

            tokenizer = _T.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        except Exception:
            return {}
    return {p: len(tokenizer(p, add_special_tokens=True)["input_ids"]) for p in prompts}


def describe_minari(args: argparse.Namespace, dataset_id: str) -> dict:
    from .minari_utils import load_dataset
    from .tasks import (
        PROMPT_STYLES,
        TASK_GOALS,
        TASK_INDICES,
        TASK_PHRASES,
        TASK_SHORT_PHRASES,
        episode_completion_events,
        instruction_for,
    )

    dataset = load_dataset(dataset_id, args.dataset_root)
    spec = dataset.spec
    spec_kwargs = dict(getattr(spec.env_spec, "kwargs", {}) or {})
    goal_set = list(spec_kwargs.get("tasks_to_complete", []))

    length_histogram: Counter[int] = Counter()
    sequence_counts: Counter[tuple[str, ...]] = Counter()
    sequence_first_frames: dict[tuple[str, ...], list[list[int]]] = {}
    first_task: Counter[str] = Counter()
    per_task_appearances: Counter[str] = Counter()
    per_task_first_frame: dict[str, list[int]] = {task: [] for task in TASK_INDICES}
    goal_keys: set[str] = set()
    episodes = 0
    steps = 0

    for index, episode in enumerate(dataset.iterate_episodes()):
        if args.max_episodes is not None and index >= args.max_episodes:
            break
        episodes += 1
        flat = np.asarray(episode.observations["observation"])
        steps += int(episode.actions.shape[0])
        length_histogram[int(episode.actions.shape[0])] += 1
        achieved = episode.observations.get("achieved_goal", {})
        goal_keys |= set(achieved.keys())
        events = episode_completion_events(episode)
        sequence = tuple(task for _, task in events)
        sequence_counts[sequence] += 1
        sequence_first_frames.setdefault(sequence, []).append([frame for frame, _ in events])
        if sequence:
            first_task[sequence[0]] += 1
        for frame, task in events:
            per_task_appearances[task] += 1
            per_task_first_frame[task].append(frame)

    prompt_episode_counts: Counter[str] = Counter()
    for sequence, count in sequence_counts.items():
        if sequence:
            prompt_episode_counts[instruction_for(sequence, style=args.prompt_style)] += count

    ordered = sorted(sequence_counts.items(), key=lambda item: (-item[1], item[0]))
    if args.top:
        ordered = ordered[: args.top]

    sequences = []
    for sequence, count in ordered:
        frame_matrix = np.asarray(sequence_first_frames.get(sequence, []), dtype=float)
        sequences.append(
            {
                "tasks": list(sequence),
                "episodes": count,
                "instruction": instruction_for(sequence, style=args.prompt_style) if sequence else None,
                "completion_frame_mean": (
                    frame_matrix.mean(axis=0).round(1).tolist() if frame_matrix.size else []
                ),
                "completion_frame_std": (
                    frame_matrix.std(axis=0).round(1).tolist() if frame_matrix.size else []
                ),
                "tokens": None,
            }
        )

    prompts = [entry["instruction"] for entry in sequences if entry["instruction"]]
    lengths = prompt_token_lengths(prompts)
    for entry in sequences:
        if entry["instruction"]:
            entry["tokens"] = lengths.get(entry["instruction"])

    task_table = []
    for task, indices in TASK_INDICES.items():
        frames = per_task_first_frame[task]
        task_table.append(
            {
                "task": task,
                "observation_indices": indices.tolist(),
                "goal": TASK_GOALS[task].tolist(),
                "phrase_natural": TASK_PHRASES[task],
                "phrase_compact": TASK_SHORT_PHRASES[task],
                "in_achieved_goal": task in goal_keys,
                "episodes_completing": per_task_appearances[task],
                "first_completion_frame_mean": float(np.mean(frames)) if frames else None,
                "first_completion_frame_p10": float(np.percentile(frames, 10)) if frames else None,
                "first_completion_frame_p90": float(np.percentile(frames, 90)) if frames else None,
            }
        )

    return {
        "dataset_id": dataset_id,
        "env_id": spec.env_spec.id if getattr(spec, "env_spec", None) else None,
        "tasks_to_complete_in_env_spec": goal_set,
        "goal_keys_present_in_data": sorted(goal_keys),
        "total_episodes_declared": int(dataset.total_episodes),
        "total_steps_declared": int(dataset.total_steps),
        "episodes_scanned": episodes,
        "steps_scanned": steps,
        "episode_length": {
            "min": min(length_histogram) if length_histogram else None,
            "max": max(length_histogram) if length_histogram else None,
            "mean": (steps / episodes) if episodes else None,
        },
        "episode_length_histogram": dict(sorted(length_histogram.items())),
        "distinct_sequences": len(sequence_counts),
        "sequence_length_histogram": dict(
            sorted(Counter(len(seq) for seq in sequence_counts).items())
        ),
        "first_task_counts": dict(first_task),
        "tasks": task_table,
        "sequences": sequences,
        "prompt_styles": list(PROMPT_STYLES),
    }


def describe_lerobot(root: Path, prompt_style: str) -> dict:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    root = root.resolve()
    repo_id = f"local/{root.name}"
    meta = LeRobotDatasetMetadata(repo_id, root=root)

    camera_keys = list(meta.camera_keys)
    features = {
        key: {"dtype": value["dtype"], "shape": list(value["shape"])}
        for key, value in meta.features.items()
    }

    tasks = []
    task_frame_counts: Counter[str] = Counter()
    task_episode_counts: Counter[str] = Counter()
    tasks_parquet = root / "meta" / "tasks.parquet"
    tasks_jsonl = root / "meta" / "tasks.jsonl"
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None and tasks_parquet.is_file():
        frame = pd.read_parquet(tasks_parquet)
        for task, row in frame.iterrows():
            tasks.append({"task_index": int(row["task_index"]), "task": str(task)})
    elif tasks_jsonl.is_file():
        for line in tasks_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                tasks.append(
                    {
                        "task_index": payload.get("task_index", payload.get("index")),
                        "task": payload.get("task"),
                    }
                )

    # Per-task frame/episode counts come from the episode metadata, which records the
    # episode length and its task list; reading it avoids touching the video files.
    episodes_dir = root / "meta" / "episodes"
    if pd is not None and episodes_dir.is_dir():
        try:
            for parquet in sorted(episodes_dir.rglob("*.parquet")):
                frame = pd.read_parquet(parquet, columns=["tasks", "length"])
                for task_list, length in zip(frame["tasks"], frame["length"], strict=False):
                    # pandas hands this column back as a numpy object array, not a list,
                    # so branching on list/tuple alone would stringify the whole array.
                    names = [task_list] if isinstance(task_list, str) else list(task_list)
                    for name in names:
                        task_frame_counts[str(name)] += int(length)
                        task_episode_counts[str(name)] += 1
        except Exception as exc:  # diagnostics only
            task_frame_counts[f"<unavailable: {exc}>"] = 0

    manifest = None
    manifest_path = root / "conversion_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return {
        "root": str(root),
        "repo_id": repo_id,
        "fps": int(meta.fps),
        "robot_type": getattr(meta, "robot_type", None),
        "total_episodes": int(meta.total_episodes),
        "total_frames": int(meta.total_frames),
        "camera_keys": camera_keys,
        "features": features,
        "tasks": tasks,
        "task_frame_counts": dict(task_frame_counts),
        "task_episode_counts": dict(task_episode_counts),
        "conversion_manifest": manifest,
        "prompt_style_hint": (
            manifest.get("prompt_style") if isinstance(manifest, dict) else None
        )
        or prompt_style,
    }


def print_minari(report: dict, top: int) -> None:
    print("=" * 78)
    print(f"Minari dataset: {report['dataset_id']}")
    print("=" * 78)
    print(f"  env                     : {report['env_id']}")
    print(f"  tasks_to_complete (spec): {report['tasks_to_complete_in_env_spec']}")
    print(f"  goal keys stored in data: {report['goal_keys_present_in_data']}")
    print(
        f"  episodes/steps declared : {report['total_episodes_declared']} / "
        f"{report['total_steps_declared']}"
    )
    print(f"  scanned                 : {report['episodes_scanned']} episodes, {report['steps_scanned']} steps")
    length = report["episode_length"]
    print(f"  episode length          : min {length['min']} max {length['max']} mean {length['mean']:.1f}")
    print(f"  distinct task sequences : {report['distinct_sequences']}")
    print(f"  sequence length counts  : {report['sequence_length_histogram']}")
    print()
    print("  Task vocabulary (7 tasks, from OBS_ELEMENT_INDICES/GOALS):")
    print(
        f"    {'task':16s} {'qpos idx':32s} {'goal':28s} {'eps':>5s} "
        f"{'first frame p10/p50/p90':>24s}  goal-dict"
    )
    for entry in report["tasks"]:
        frames = [
            entry["first_completion_frame_p10"],
            entry["first_completion_frame_mean"],
            entry["first_completion_frame_p90"],
        ]
        spread = (
            "/".join(f"{value:.0f}" for value in frames) if frames[0] is not None else "-"
        )
        print(
            f"    {entry['task']:16s} {str(entry['observation_indices']):12s} "
            f"{str(entry['goal']):28s} {entry['episodes_completing']:5d} {spread:>24s}"
            f"  {'yes' if entry['in_achieved_goal'] else 'no (geometric fallback)'}"
        )
    print()
    print(f"  Sequences and the exact prompt they produce (showing {len(report['sequences'])}):")
    for entry in report["sequences"]:
        token_text = f"{entry['tokens']:>3d} tok" if entry["tokens"] else "   ? tok"
        print(f"    [{entry['episodes']:3d} eps] {token_text}  {entry['instruction']}")
        if entry["completion_frame_mean"]:
            spread = ", ".join(
                f"{int(mean)}±{int(std)}"
                for mean, std in zip(
                    entry["completion_frame_mean"], entry["completion_frame_std"], strict=False
                )
            )
            print(f"                 completion frames (mean±std): {spread}")


def print_lerobot(report: dict) -> None:
    print("=" * 78)
    print(f"LeRobot dataset: {report['root']}")
    print("=" * 78)
    print(f"  repo_id        : {report['repo_id']}")
    print(f"  fps            : {report['fps']}   robot_type: {report['robot_type']}")
    print(f"  episodes/frames: {report['total_episodes']} / {report['total_frames']}")
    print(f"  camera keys    : {report['camera_keys']}")
    print(f"  prompt style   : {report['prompt_style_hint']}")
    print("  features:")
    for key, value in report["features"].items():
        print(f"    {key:34s} {value['dtype']:8s} {value['shape']}")
    print(f"  distinct task strings: {len(report['tasks'])}")
    for entry in report["tasks"]:
        frames = report["task_frame_counts"].get(str(entry["task"]))
        episodes = report["task_episode_counts"].get(str(entry["task"]))
        suffix = f"  ({episodes} episodes, {frames} frames)" if frames else ""
        print(f"    [{entry['task_index']}] {entry['task']}{suffix}")
    manifest = report.get("conversion_manifest")
    if manifest:
        print("  conversion manifest:")
        for key in (
            "source_dataset",
            "source_episodes_read",
            "lerobot_episodes_written",
            "emit_prefixes",
            "resolution",
            "camera_names",
            "excluded_sequences",
        ):
            if key in manifest:
                print(f"    {key:26s}: {manifest[key]}")


def main() -> None:
    args = parse_args()
    dataset_ids = args.dataset_id or [
        "D4RL/kitchen/complete-v2",
        "D4RL/kitchen/partial-v2",
        "D4RL/kitchen/mixed-v2",
    ]
    report: dict = {"minari": [], "lerobot": []}
    for dataset_id in dataset_ids:
        entry = describe_minari(args, dataset_id)
        report["minari"].append(entry)
        print_minari(entry, args.top)
        print()
    for root in args.lerobot_root or []:
        try:
            entry = describe_lerobot(root, args.prompt_style)
        except Exception as exc:  # keep the Minari half of the report useful
            print(f"!! could not read LeRobot dataset {root}: {type(exc).__name__}: {exc}\n")
            continue
        report["lerobot"].append(entry)
        print_lerobot(entry)
        print()
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
