#!/usr/bin/env python
"""Render every camera in a config from the same simulation states and build a labelled sheet.

Use it to choose the global + end-effector cameras before converting a dataset:

    python scripts/preview_cameras.py \
        --camera-config configs/cameras_candidates.yaml \
        --dataset-id D4RL/kitchen/partial-v2 --episode 0 \
        --output previews/camera_candidates

The script writes one PNG grid (`sheet.png`), the individual frames, and a
`preview_manifest.json` recording, for every camera, its kind, parameters and the
hash of the config file -- so the sheet can be tied back to an exact configuration.

Frames are taken at evenly spaced *task completions* rather than evenly spaced
time, so each row shows the scene at the moments that actually matter:
before the first task, just after each completion, and at the end.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    parser.add_argument("--dataset-id", default="D4RL/kitchen/complete-v2")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/minari"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("previews/camera_candidates"))
    parser.add_argument("--columns", type=int, default=5, help="How many states to sample.")
    parser.add_argument("--scale", type=int, default=1, help="Integer upscale for the sheet.")
    parser.add_argument("--label-height", type=int, default=18)
    return parser.parse_args()


def pick_frames(flat: np.ndarray, columns: int) -> list[tuple[int, str]]:
    from franka_kitchen_vla.tasks import episode_completion_events

    class _Episode:
        def __init__(self, observations):
            self.observations = observations

    events = episode_completion_events(_Episode({"observation": flat}))
    total = flat.shape[0] - 1
    picks: list[tuple[int, str]] = [(0, "reset")]
    for frame, task in events:
        picks.append((min(frame, total), f"after {task}"))
    if not events:
        picks = [(int(round(i * total / max(1, columns - 1))), f"frame {int(round(i * total / max(1, columns - 1)))}")
                 for i in range(columns)]
    if len(picks) > columns:
        # Keep the reset frame plus an even sample of the completions.
        step = (len(picks) - 1) / (columns - 1)
        picks = [picks[int(round(i * step))] for i in range(columns)]
    return picks


def compose_sheet(
    rows: list[tuple[str, list[np.ndarray]]],
    column_labels: list[str],
    scale: int,
    label_height: int,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    if not rows:
        raise ValueError("nothing to render")
    tile_h, tile_w = rows[0][1][0].shape[:2]
    label_w = 150
    width = label_w + len(rows[0][1]) * tile_w
    height = label_height + len(rows) * (tile_h + label_height)
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    for column, label in enumerate(column_labels):
        draw.text((label_w + column * tile_w + 4, 4), label[:40], fill=(0, 0, 0))
    for row_index, (name, frames) in enumerate(rows):
        top = label_height + row_index * (tile_h + label_height)
        draw.text((4, top + 4), name[:40], fill=(0, 0, 0))
        for column, frame in enumerate(frames):
            image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
            sheet.paste(image, (label_w + column * tile_w, top + label_height))
    if scale > 1:
        sheet = sheet.resize((sheet.width * scale, sheet.height * scale), Image.NEAREST)
    return np.asarray(sheet)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    from PIL import Image

    from franka_kitchen_vla.cameras import CameraRig
    from franka_kitchen_vla.minari_utils import (
        load_dataset,
        make_replay_environment,
        set_state_from_observation,
    )

    dataset = load_dataset(args.dataset_id, args.dataset_root)
    episodes = list(dataset.iterate_episodes(episode_indices=[args.episode]))
    if not episodes:
        raise SystemExit(f"episode {args.episode} not in {args.dataset_id}")
    episode = episodes[0]
    flat = np.asarray(episode.observations["observation"])

    env = make_replay_environment(dataset)
    env.reset(seed=0)
    rig = CameraRig(env.unwrapped.model, args.camera_config)
    args.output.mkdir(parents=True, exist_ok=True)

    picks = pick_frames(flat, args.columns)
    column_labels = [f"{index}: {label}" for index, (_, label) in enumerate(picks)]
    rows: list[tuple[str, list[np.ndarray]]] = []
    # Effective world pose of every camera, per sampled frame. `eye`/`target` in the
    # YAML are offsets in the attached body's local frame, so this is the only place
    # that shows where the camera actually ends up in the scene.
    world_poses: dict[str, list[dict[str, list[float]]]] = {name: [] for name in rig.names}
    try:
        frames_by_camera: dict[str, list[np.ndarray]] = {name: [] for name in rig.names}
        for frame_index, _ in picks:
            set_state_from_observation(env, flat[frame_index])
            for name in rig.names:
                # One camera at a time, so scene.camera holds that camera's pose.
                image = rig.render(env.unwrapped.data, [name])[name]
                scene_camera = rig.renderer.scene.camera[0]
                world_poses[name].append(
                    {
                        "pos": [float(value) for value in scene_camera.pos],
                        "forward": [float(value) for value in scene_camera.forward],
                        "up": [float(value) for value in scene_camera.up],
                    }
                )
                frames_by_camera[name].append(image)
                Image.fromarray(image).save(
                    args.output / f"{name}_frame{frame_index:04d}.png"
                )
        for name in rig.names:
            rows.append((name, frames_by_camera[name]))
    finally:
        rig.close()
        env.close()

    sheet = compose_sheet(rows, column_labels, args.scale, args.label_height)
    sheet_path = args.output / "sheet.png"
    Image.fromarray(sheet).save(sheet_path)

    manifest = {
        "dataset_id": args.dataset_id,
        "episode": args.episode,
        "camera_config": str(args.camera_config),
        "frames_sampled": [
            {"frame": frame, "label": label} for frame, label in picks
        ],
        "cameras": [
            {
                "name": camera.name,
                "kind": camera.kind,
                "values": {k: v for k, v in camera.values.items()},
            }
            for camera in rig.configs
        ],
        "camera_world_poses": {
            "note": (
                "pos/forward/up are the effective MuJoCo camera pose in world coordinates; "
                "for kind=attached this equals body_world_pos + R_body @ eye/target."
            ),
            "frames_sampled": [frame for frame, _ in picks],
            "by_camera": world_poses,
        },
        "resolution": [rig.width, rig.height],
        "sheet": str(sheet_path),
    }
    (args.output / "preview_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nsheet: {sheet_path}")


if __name__ == "__main__":
    main()
