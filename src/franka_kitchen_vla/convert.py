from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .schema import ROBOT_JOINT_NAMES, ROBOT_QVEL_NAMES, ROBOT_STATE_NAMES, robot_state_from_observation

MANIFEST_NAME = "conversion_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Minari D4RL Kitchen demonstrations to a LeRobot v3 dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # smoke test: one episode, one task sequence\n"
            "  kitchen-to-lerobot --repo-id local/fk_smoke "
            "--output-root data/lerobot/fk_smoke --max-episodes 1\n\n"
            "  # full prefixes (1,2,3,4-task instructions) with 256x256 cameras\n"
            "  kitchen-to-lerobot --repo-id local/fk_partial "
            "--output-root data/lerobot/fk_partial --emit-prefixes\n\n"
            "  # hold out one ordering entirely to test instruction following\n"
            "  kitchen-to-lerobot --repo-id local/fk_holdout "
            "--output-root data/lerobot/fk_holdout --emit-prefixes \\\n"
            '      --exclude-sequence "kettle,light switch,slide cabinet,hinge cabinet"\n'
        ),
    )
    parser.add_argument("--dataset-id", default="D4RL/kitchen/partial-v2")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/minari"))
    parser.add_argument("--repo-id", default="local/franka_kitchen_partial")
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/lerobot/franka_kitchen_partial")
    )
    parser.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="Subset of camera keys from the camera config; default is all of them.",
    )
    parser.add_argument("--width", type=int, help="Override the camera-config render width.")
    parser.add_argument("--height", type=int, help="Override the camera-config render height.")
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Read at most this many source episodes (applied after --episode-offset).",
    )
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument(
        "--episode-stride",
        type=int,
        default=1,
        help="Take every Nth episode; use with --episode-offset to shard manual workers.",
    )
    parser.add_argument(
        "--emit-prefixes",
        action="store_true",
        help=(
            "Also create task-sequence prefixes that begin at the original reset state. "
            "Each source episode becomes one episode per completed task, which is what "
            "teaches single-task and short-sequence instructions."
        ),
    )
    parser.add_argument(
        "--prefix-length",
        type=int,
        help=(
            "Write exactly one prefix of this many completed tasks per source episode. "
            "For example, --prefix-length 2 keeps reset -> completion of task 2 and "
            "drops the 1-, 3- and 4-task variants. Mutually exclusive with --emit-prefixes."
        ),
    )
    parser.add_argument(
        "--prompt-style",
        choices=("natural", "compact"),
        default="natural",
        help="natural: 'Open the microwave door, then ...'; compact: 'microwave, kettle'.",
    )
    parser.add_argument(
        "--exclude-sequence",
        action="append",
        default=[],
        metavar="A,B,C",
        help=(
            "Drop every source episode whose completed-task order equals this comma-separated "
            "list. Repeatable. Use it to keep an ordering completely unseen by training."
        ),
    )
    parser.add_argument(
        "--min-tasks",
        type=int,
        default=1,
        help="Skip source episodes that complete fewer than this many tasks.",
    )
    parser.add_argument(
        "--min-episode-steps",
        type=int,
        default=30,
        help=(
            "Skip source episodes shorter than this. partial-v2/mixed-v2 end with 18 "
            "one-step episodes that are the leftover tail of a split demonstration: their "
            "first frame is already at several goals, so they yield a 1-frame episode "
            "carrying a 4-task instruction, which is pure noise. Default 30. Pass 1 to keep "
            "everything."
        ),
    )
    parser.add_argument(
        "--min-first-completion-frame",
        type=int,
        default=1,
        help=(
            "Skip source episodes whose earliest completion happens before this frame, i.e. "
            "episodes that start with a task already satisfied. Those record nothing about "
            "how to do the task. Pass 0 to keep everything."
        ),
    )
    parser.add_argument("--images", action="store_true", help="Store PNG images instead of MP4 videos.")
    parser.add_argument(
        "--robot-type", default="franka_kitchen", help="LeRobot robot_type stored in metadata."
    )
    parser.add_argument(
        "--no-frame-cache",
        action="store_true",
        help="Re-render frames for every prefix instead of rendering each frame once.",
    )
    parser.add_argument(
        "--max-frame-cache-mb",
        type=int,
        default=1024,
        help=(
            "Upper bound on the per-episode frame cache used to render each frame only "
            "once for all prefixes. LeRobot buffers the same frames again while writing, "
            "so peak memory is roughly twice this. Episodes above the bound fall back to "
            "re-rendering. Default 1024."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete --output-root when it already exists instead of failing.",
    )
    parser.add_argument("--quiet", action="store_true", help="Do not print per-episode progress.")
    return parser.parse_args()


def _features(camera_names: Sequence[str], height: int, width: int, use_videos: bool) -> dict:
    image_dtype = "video" if use_videos else "image"
    features: dict[str, dict[str, Any]] = {
        f"observation.images.{name}": {
            "dtype": image_dtype,
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
        for name in camera_names
    }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (len(ROBOT_STATE_NAMES),),
        "names": ROBOT_STATE_NAMES,
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (9,),
        "names": ROBOT_QVEL_NAMES,
    }
    return features


def parse_exclude_sequence(text: str) -> tuple[str, ...]:
    tasks = tuple(part.strip() for part in text.split(",") if part.strip())
    if not tasks:
        raise ValueError(f"--exclude-sequence {text!r} does not name any task.")
    return tasks


def plan_variants(
    events: Sequence[tuple[int, str]],
    emit_prefixes: bool,
    prefix_length: int | None = None,
) -> list[tuple[int, tuple[str, ...]]]:
    """Map one source episode's completion events to (end_frame, tasks) output variants.

    ``end_frame`` is the last action index that executes the variant: the frame at
    which the variant's final task first satisfies its goal, following
    ``completion_steps``. With ``emit_prefixes`` a source episode that completes
    four tasks becomes four LeRobot episodes (1-, 2-, 3- and 4-task instructions),
    each starting at the original reset state.
    """
    if not events:
        return []
    if prefix_length is not None:
        if len(events) < prefix_length:
            return []
        frame, _ = events[prefix_length - 1]
        return [(frame, tuple(task for _, task in events[:prefix_length]))]
    if emit_prefixes:
        return [
            (frame, tuple(task for _, task in events[: index + 1]))
            for index, (frame, _) in enumerate(events)
        ]
    return [(events[-1][0], tuple(task for _, task in events))]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    if args.episode_stride < 1:
        raise SystemExit("--episode-stride must be >= 1")
    if args.episode_offset < 0:
        raise SystemExit("--episode-offset must be >= 0")
    if args.min_tasks < 1:
        raise SystemExit("--min-tasks must be >= 1")
    if args.prefix_length is not None and args.prefix_length < 1:
        raise SystemExit("--prefix-length must be >= 1")
    if args.prefix_length is not None and args.emit_prefixes:
        raise SystemExit("pass either --prefix-length or --emit-prefixes, not both")
    if args.min_episode_steps < 1:
        raise SystemExit("--min-episode-steps must be >= 1")
    if args.min_first_completion_frame < 0:
        raise SystemExit("--min-first-completion-frame must be >= 0")
    if args.overwrite:
        import shutil

        resolved = args.output_root.resolve()
        if resolved in (Path("/"), Path.home().resolve()):
            raise SystemExit(f"refusing to delete {resolved}")
        shutil.rmtree(args.output_root, ignore_errors=True)
    elif args.output_root.exists():
        raise FileExistsError(
            f"Output {args.output_root} already exists. Choose a new path, move it, or pass --overwrite."
        )

    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            'LeRobot dataset support is missing. Run: uv pip install -e ".[train]".'
        ) from exc

    from .cameras import CameraRig
    from .minari_utils import load_dataset, make_replay_environment, set_state_from_observation
    from .tasks import episode_completion_events, instruction_for

    excluded = {parse_exclude_sequence(text) for text in args.exclude_sequence}

    source = load_dataset(args.dataset_id, args.dataset_root)
    env = make_replay_environment(source)
    env.reset(seed=0)
    rig = CameraRig(
        env.unwrapped.model,
        args.camera_config,
        width=args.width,
        height=args.height,
        only=args.cameras,
    )
    use_videos = not args.images
    output = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output_root,
        fps=int(env.unwrapped.metadata.get("render_fps", 12)),
        robot_type=args.robot_type,
        features=_features(rig.names, rig.height, rig.width, use_videos),
        use_videos=use_videos,
        image_writer_processes=0,
        image_writer_threads=max(2, len(rig.names) * 2),
    )

    stats: Counter[str] = Counter()
    sequence_counts: Counter[tuple[str, ...]] = Counter()
    skipped_excluded: Counter[tuple[str, ...]] = Counter()
    prompt_counts: Counter[str] = Counter()
    task_frame_counts: Counter[str] = Counter()
    source_episodes_read = 0
    converted = 0
    cache_frames = 0

    def render_frames(flat: np.ndarray, last_frame: int) -> list[dict[str, np.ndarray]]:
        """Render frames 0..last_frame once; returns per-frame image dicts."""
        nonlocal cache_frames
        frames: list[dict[str, np.ndarray]] = []
        for step in range(last_frame + 1):
            set_state_from_observation(env, flat[step])
            frames.append(rig.render(env.unwrapped.data))
            cache_frames += 1
        return frames

    try:
        iterator = source.iterate_episodes()
        for index, episode in enumerate(iterator):
            if index < args.episode_offset:
                continue
            if (index - args.episode_offset) % args.episode_stride:
                continue
            if args.max_episodes is not None and source_episodes_read >= args.max_episodes:
                break
            source_episodes_read += 1

            flat = np.asarray(episode.observations["observation"])
            num_actions = int(episode.actions.shape[0])
            if num_actions < args.min_episode_steps:
                stats["skipped_too_short"] += 1
                continue
            events = episode_completion_events(episode)
            if len(events) < args.min_tasks:
                stats["skipped_too_few_tasks"] += 1
                continue
            if events[0][0] < args.min_first_completion_frame:
                stats["skipped_already_complete_at_start"] += 1
                continue
            sequence = tuple(task for _, task in events)
            sequence_counts[sequence] += 1
            if sequence in excluded:
                skipped_excluded[sequence] += 1
                stats["skipped_excluded_sequence"] += 1
                continue

            variants = plan_variants(events, args.emit_prefixes, args.prefix_length)
            if not variants:
                stats["skipped_no_variants"] += 1
                continue

            max_last = max(last for last, _ in variants)
            max_last = min(max_last, len(episode.actions) - 1)
            cache: list[dict[str, np.ndarray]] | None = None
            frames_to_cache = max_last + 1
            cache_bytes = frames_to_cache * len(rig.names) * rig.width * rig.height * 3
            if (
                not args.no_frame_cache
                and len(variants) > 1
                and cache_bytes <= args.max_frame_cache_mb * 1024 * 1024
            ):
                cache = render_frames(flat, max_last)
            elif not args.no_frame_cache and len(variants) > 1:
                stats["episodes_over_frame_cache_budget"] += 1

            for last_frame, tasks in variants:
                last_frame = min(last_frame, len(episode.actions) - 1)
                task = instruction_for(tasks, style=args.prompt_style)
                prompt_counts[task] += 1
                for step in range(last_frame + 1):
                    if cache is not None:
                        frames = cache[step]
                    else:
                        set_state_from_observation(env, flat[step])
                        frames = rig.render(env.unwrapped.data)
                        cache_frames += 1
                    frame = {f"observation.images.{name}": img for name, img in frames.items()}
                    frame["observation.state"] = robot_state_from_observation(flat[step])
                    frame["action"] = np.asarray(episode.actions[step], dtype=np.float32)
                    frame["task"] = task
                    output.add_frame(frame)
                output.save_episode()
                converted += 1
                task_frame_counts[task] += last_frame + 1
            stats["source_episodes_converted"] += 1
            if not args.quiet and stats["source_episodes_converted"] % 10 == 0:
                print(
                    f"  ... {stats['source_episodes_converted']} source episodes -> "
                    f"{converted} LeRobot episodes ({rig.names}, "
                    f"{rig.width}x{rig.height})",
                    flush=True,
                )
        output.finalize()
    finally:
        rig.close()
        env.close()

    manifest = {
        "source_dataset": args.dataset_id,
        "source_episodes_read": source_episodes_read,
        "lerobot_episodes_written": converted,
        "repo_id": args.repo_id,
        "output_root": str(args.output_root),
        "camera_names": rig.names,
        "camera_config": str(args.camera_config),
        "camera_config_sha256": _sha256(args.camera_config),
        "resolution": [rig.width, rig.height],
        "use_videos": use_videos,
        "prompt_style": args.prompt_style,
        "robot_type": args.robot_type,
        "state_layout": {
            "shape": [len(ROBOT_STATE_NAMES)],
            "names": ROBOT_STATE_NAMES,
            "description": "robot qpos (9) followed by robot qvel (9); no object state",
        },
        "emit_prefixes": args.emit_prefixes,
        "prefix_length": args.prefix_length,
        "min_episode_steps": args.min_episode_steps,
        "min_first_completion_frame": args.min_first_completion_frame,
        "episode_offset": args.episode_offset,
        "episode_stride": args.episode_stride,
        "max_episodes": args.max_episodes,
        "excluded_sequences": sorted(",".join(seq) for seq in excluded),
        "frames_rendered": cache_frames,
        "counters": dict(sorted(stats.items())),
        "skipped_excluded_sequences": {
            " -> ".join(seq): count for seq, count in sorted(skipped_excluded.items())
        },
        "source_sequences": {
            " -> ".join(seq): count for seq, count in sorted(sequence_counts.items())
        },
        "written_prompts": {
            task: {"episodes": prompt_counts[task], "frames": task_frame_counts[task]}
            for task in sorted(prompt_counts)
        },
        "smolvla_finetune_hint": (
            "lerobot-train --policy.path=<smolvla_base_dir> --policy.input_features=null "
            "--policy.device=cuda "
            f"--dataset.repo_id={args.repo_id} --dataset.root={args.output_root}"
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "source_sequences"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
