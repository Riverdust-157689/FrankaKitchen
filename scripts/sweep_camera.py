#!/usr/bin/env python
"""Sweep a camera's parameters and build a labelled contact sheet.

An ``attached`` camera has exactly three parameters::

    eye    [x, y, z]  position, in the frame named by `site:` (or `body:`)
    target [x, y, z]  look-at point, in the same frame
    roll   degrees    spin about the view axis (optional, default 0)

Prefer `site: end_effector` over `body: panda0_link7` on this robot: the two frames
differ by 21 cm along +z and a 45 degree rotation about z, so in the end_effector frame
`eye_y` really is "along the jaws" and `eye = [0, 0, z]` is the gripper centre line.

This tool renders the full cross product of whatever you pass, so position and
aim can be explored together -- they are not independent knobs on a look-at
camera, because moving ``eye`` while ``target`` stays put also rotates the view.

Notes that save time on ``panda0_link7``:

* local +z is the gripper approach axis; the hand mesh spans z = 0.03..0.24, so
  ``eye_z`` must be <= -0.04 or the camera sits inside the hand and renders its
  black inner surface dead-centre;
* only the direction ``target - eye`` matters, so the distance to the target is
  free -- move it along the same ray and nothing changes;
* tilting the aim away from +z brings the gripper into frame, and how much that
  costs depends strongly on the direction it tilts toward.

Reading the output
------------------
``motion``     mean frame-to-frame pixel change. Near zero means the camera is
               staring at something static and carries little information.

There is deliberately **no "quality score"**. An earlier version scored the
near-black fraction of the centre of the frame as "occlusion", which was wrong on
two counts: it mostly tracked how dark the surface being looked at was (the granite
countertop versus the bright splashback), and a wrist camera is *supposed* to have
the gripper in frame, so penalising that points the wrong way. Judge the frames by
eye; the pose sheet shows where the camera is and what it aims at.

Examples
--------
How far back, and how far ahead it looks:

    python scripts/sweep_camera.py --camera wrist --eye-z -0.30 -0.20 -0.12

Tilt the aim toward +y vs +x, without moving the camera:

    python scripts/sweep_camera.py --camera wrist --eye-z -0.20 \\
        --target-y 0.0 0.18 --target-x 0.0 0.18

Spin the image:

    python scripts/sweep_camera.py --camera wrist --roll -90 -45 0 45 90
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-config", type=Path, default=Path("configs/cameras_candidates.yaml"))
    parser.add_argument("--camera", default="wrist", help="Name of the camera entry to sweep.")
    parser.add_argument("--dataset-id", default="D4RL/kitchen/partial-v2")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/minari"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--states", type=int, default=4, help="How many simulation states to sample.")

    # position
    parser.add_argument("--eye-x", nargs="+", type=float, default=None, help="Default: keep the config value.")
    parser.add_argument("--eye-y", nargs="+", type=float, default=None)
    parser.add_argument("--eye-z", nargs="+", type=float, default=None)

    # aim point
    parser.add_argument("--target-x", nargs="+", type=float, default=None)
    parser.add_argument("--target-y", nargs="+", type=float, default=None)
    parser.add_argument("--target-z", nargs="+", type=float, default=None)

    # spin
    parser.add_argument(
        "--roll",
        nargs="+",
        type=float,
        default=None,
        help="Degrees about the view axis. Positive rotates the image content counter-clockwise.",
    )

    parser.add_argument(
        "--context-camera",
        default="",
        help=(
            "Camera used for the third-person POSE sheet, in which the swept camera's eye and "
            "target are drawn as red balls. Default: auto -- the first free/fixed camera in the "
            "config, preferring a name starting with 'global'. Pass 'none' to disable."
        ),
    )
    parser.add_argument(
        "--pose-zoom",
        type=float,
        default=2.5,
        help="Zoom factor for the pose sheet, cropped around the eye/target markers.",
    )
    parser.add_argument("--pose-margin", type=int, default=70, help="Crop margin in pixels.")
    parser.add_argument(
        "--pose-radius",
        type=float,
        default=0.025,
        help="Radius in metres of the MuJoCo-rendered red balls on the pose sheet.",
    )
    parser.add_argument(
        "--mark-target-in-frames",
        action="store_true",
        help=(
            "Also draw the target as a red ball inside each first-person sweep frame. It always "
            "lands at the image centre (the target is on the view axis by construction), so it "
            "mainly shows how far away the aim point is."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("previews/camera_sweep"))
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--max-combinations", type=int, default=80, help="Safety limit on sheet size.")
    parser.add_argument("--top", type=int, default=3, help="How many combinations to list by frame-to-frame change.")
    return parser.parse_args()


def build_pose_sheet(env, flat, state_indices, poses, args, rig, Image, ImageDraw, ImageFont):
    """第三方视角 + 两个红球 + 一根白杆：每组参数一块，标出相机在哪、瞄哪。

    球和杆都是**MuJoCo 自己渲染的 3D 图元**（`mark_pose_in_scene` 在 `update_scene` 之后注入
    场景），所以尺寸随距离变化、并被手臂正确遮挡。复用调用方的 rig，不新建 GL 上下文。
    """
    from franka_kitchen_vla.cameras import mark_pose_in_scene
    from franka_kitchen_vla.minari_utils import set_state_from_observation

    model = env.unwrapped.model
    data = env.unwrapped.data

    def third_person_camera():
        """挑一个不是 attached 的相机做第三方视角（free/fixed 都行）。"""
        requested = args.context_camera
        if requested and requested != "none":
            if requested in rig.names and requested != args.camera:
                return requested
            print(f"note: {requested!r} 不可用，自动挑一个第三方视角")
        if args.camera.startswith("global"):
            return None  # 被扫的本身就是全局视角，没有"外部"视角可言
        candidates = [c.name for c in rig.configs if c.kind in ("free", "fixed") and c.name != args.camera]
        for preferred in ("global", "global_free_default", "global_free_high", "global_free_low"):
            if preferred in candidates:
                return preferred
        return candidates[0] if candidates else None

    context_name = third_person_camera()
    if context_name is None:
        print("note: 找不到可用的第三方视角相机，跳过位姿图")
        return None
    print(f"位姿图使用第三方视角：{context_name!r}")

    state_index = state_indices[len(state_indices) // 2]
    tiles = []
    for label, eye, target in poses:
        set_state_from_observation(env, flat[state_index])
        # 红球和杆都由 MuJoCo 真渲染：在 update_scene 之后、render 之前注入场景图元
        image = Image.fromarray(
            rig.render(
                data,
                [context_name],
                annotate=lambda scene, _data, e=eye, t=target: mark_pose_in_scene(
                    scene, e, t, radius=args.pose_radius
                ),
            )[context_name].copy()
        )
        camera = rig.renderer.scene.camera[0]  # 只渲染了一个相机，所以这就是它自己的位姿
        height, width = np.asarray(image).shape[:2]
        fovy = float(model.vis.global_.fovy)
        scale = (height / 2) / np.tan(np.radians(fovy / 2))
        cam_pos = np.array(camera.pos)
        cam_forward = np.array(camera.forward)
        cam_up = np.array(camera.up)
        cam_right = np.cross(cam_forward, cam_up)

        def project(point):
            delta = np.asarray(point) - cam_pos
            depth = float(delta @ cam_forward)
            if depth <= 1e-6:
                return None
            return (
                width / 2 + (delta @ cam_right) / depth * scale,
                height / 2 - (delta @ cam_up) / depth * scale,
            )

        visible = [p for p in (project(eye), project(target)) if p is not None]
        if len(visible) == 2:
            (x0, y0), (x1, y1) = visible
            left = max(0, int(min(x0, x1)) - args.pose_margin)
            right = min(width, int(max(x0, x1)) + args.pose_margin)
            top = max(0, int(min(y0, y1)) - args.pose_margin)
            bottom = min(height, int(max(y0, y1)) + args.pose_margin)
            if right - left < 80:
                middle = (left + right) // 2
                left, right = max(0, middle - 60), min(width, middle + 60)
            if bottom - top < 80:
                middle = (top + bottom) // 2
                top, bottom = max(0, middle - 60), min(height, middle + 60)
            image = image.crop((left, top, right, bottom))
        zoom = max(1.0, float(args.pose_zoom))
        image = image.resize(
            (max(1, int(image.width * zoom)), max(1, int(image.height * zoom))), Image.LANCZOS
        )
        tiles.append(np.asarray(image))

    if not tiles:
        return None
    columns = min(3, len(tiles))
    rows_count = (len(tiles) + columns - 1) // columns
    cell_w = max(tile.shape[1] for tile in tiles)
    cell_h = max(tile.shape[0] for tile in tiles)
    # 标签画在拼好的整张图上（画在图块上会被裁剪截断），所以格子要按文字宽度撑开
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:  # 老版本 Pillow 的 load_default() 不接受 size
        font = ImageFont.load_default()
    scratch_draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    labels = [label[:70] for label, _, _ in poses]
    text_w = max(int(scratch_draw.textlength(text, font=font)) for text in labels) + 14
    cell_w = max(cell_w, text_w)
    cell_h = max(cell_h, 26)
    sheet = Image.new("RGB", (columns * cell_w, rows_count * cell_h), (25, 25, 25))
    for index, tile in enumerate(tiles):
        sheet.paste(Image.fromarray(tile), ((index % columns) * cell_w, (index // columns) * cell_h))
    draw = ImageDraw.Draw(sheet)
    for index, text in enumerate(labels):
        left = (index % columns) * cell_w
        top = (index // columns) * cell_h
        draw.rectangle([left, top, left + cell_w, top + 22], fill=(20, 20, 20))
        draw.text((left + 6, top + 4), text, fill=(255, 255, 0), font=font)

    path = args.output / f"sweep_{args.camera}_poses.png"
    sheet.save(path)
    print(f"\n位姿图（两个红球 = eye 与 target，白杆 = 视线）：{path}")
    print(f"  图块按行优先对应上面的顺序（共 {len(tiles)} 组）")
    for index, (label, _eye, _target) in enumerate(poses, start=1):
        print(f"    图块{index}: {label}")
    return path


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    from PIL import Image, ImageDraw, ImageFont

    from franka_kitchen_vla.cameras import (
        CameraConfig,
        CameraRig,
        attached_frame_pose,
        attached_frame_reference,
        resolve_attached_pose,
    )
    from franka_kitchen_vla.minari_utils import (
        load_dataset,
        make_replay_environment,
        set_state_from_observation,
    )

    raw = yaml.safe_load(args.camera_config.read_text(encoding="utf-8"))
    if args.camera not in raw["cameras"]:
        raise SystemExit(
            f"{args.camera!r} is not in {args.camera_config}; available: {sorted(raw['cameras'])}"
        )
    base = raw["cameras"][args.camera]
    if base.get("kind") != "attached":
        raise SystemExit(
            f"{args.camera!r} is not an 'attached' camera. Only attached cameras take "
            "eye/target/roll; free and fixed cameras are set by the values shown in the config."
        )
    for key in ("eye", "target"):
        if key not in base:
            raise SystemExit(f"{args.camera!r} has no {key!r} in {args.camera_config}.")
    frame_kind, frame_name = attached_frame_reference(base, args.camera)
    print(f"frame: {frame_kind} {frame_name!r}  (eye/target are in this frame's coordinates)")

    def axis_grid(override, base_vector, index):
        """Grid for one axis: the override if given, else the base's single component.

        Falling back to the whole base vector would silently sweep axes the caller did
        not ask for, turning a 6-row sheet into a 54-row one.
        """
        if override is not None:
            return list(override)
        return [float(base_vector[index])]

    eye_x = axis_grid(args.eye_x, base["eye"], 0)
    eye_y = axis_grid(args.eye_y, base["eye"], 1)
    eye_z = axis_grid(args.eye_z, base["eye"], 2)
    target_x = axis_grid(args.target_x, base["target"], 0)
    target_y = axis_grid(args.target_y, base["target"], 1)
    target_z = axis_grid(args.target_z, base["target"], 2)
    roll_values = list(args.roll) if args.roll is not None else [float(base.get("roll", 0.0))]

    combos = []
    for ex in eye_x:
        for ey in eye_y:
            for ez in eye_z:
                for tx in target_x:
                    for ty in target_y:
                        for tz in target_z:
                            for roll in roll_values:
                                combos.append(
                                    {
                                        "eye": [ex, ey, ez],
                                        "target": [tx, ty, tz],
                                        "roll": roll,
                                    }
                                )
    if len(combos) > args.max_combinations:
        raise SystemExit(
            f"{len(combos)} combinations exceed --max-combinations {args.max_combinations}. "
            "Narrow a range, or raise the limit if the sheet is still readable."
        )

    dataset = load_dataset(args.dataset_id, args.dataset_root)
    episodes = list(dataset.iterate_episodes(episode_indices=[args.episode]))
    if not episodes:
        raise SystemExit(f"episode {args.episode} is not in {args.dataset_id}")
    flat = np.asarray(episodes[0].observations["observation"])
    env = make_replay_environment(dataset)
    env.reset(seed=0)

    total = flat.shape[0] - 1
    state_indices = [int(round(i * total / max(1, args.states - 1))) for i in range(args.states)]

    args.output.mkdir(parents=True, exist_ok=True)

    # 整个扫描过程只创建一个 offscreen renderer。反复新建/销毁 GL 上下文既慢，在部分
    # 驱动上还会段错误，所以改成"一个 rig + 每次传入临时 CameraConfig"。
    rig = CameraRig(env.unwrapped.model, args.camera_config)
    rows = []
    table = []
    poses = []          # (label, eye_world, target_world) 供位姿图使用
    try:
        for combo in combos:
            config = copy.deepcopy(raw)
            config["cameras"][args.camera] = {
                **{
                    key: value
                    for key, value in base.items()
                    if key in ("body", "site", "reference_camera")
                },
                "kind": "attached",
                "eye": combo["eye"],
                "target": combo["target"],
                "roll": combo["roll"],
            }
            swept_values = config["cameras"][args.camera]
            swept_config = CameraConfig(name=args.camera, kind="attached", values=swept_values)
            images = []
            for state_index in state_indices:
                set_state_from_observation(env, flat[state_index])
                images.append(
                    rig.render(env.unwrapped.data, [args.camera], configs=[swept_config])[args.camera]
                )

            motion = (
                float(
                    np.mean(
                        [
                            np.abs(images[i + 1].astype(float) - images[i].astype(float)).mean()
                            for i in range(len(images) - 1)
                        ]
                    )
                )
                if len(images) > 1
                else float("nan")
            )
            label = (
                f"e({combo['eye'][0]:+.2f},{combo['eye'][1]:+.2f},{combo['eye'][2]:+.2f}) "
                f"t({combo['target'][0]:+.2f},{combo['target'][1]:+.2f},{combo['target'][2]:+.2f}) "
                f"r={combo['roll']:+.0f}"
            )
            rows.append((label, images))
            table.append({**combo, "motion": motion})
            print(f"  {label}  motion={motion:6.2f}", flush=True)

            # eye / target 的世界坐标，供位姿图使用（在同一个状态下算一次即可）
            set_state_from_observation(env, flat[state_indices[len(state_indices) // 2]])
            frame_pos, frame_rot = attached_frame_pose(
                env.unwrapped.model, env.unwrapped.data, swept_values, args.camera
            )
            poses.append(
                (
                    label,
                    frame_pos + frame_rot @ np.asarray(combo["eye"], dtype=float),
                    frame_pos + frame_rot @ np.asarray(combo["target"], dtype=float),
                )
            )

        # ---- 位姿图：用第三方视角把 eye / target 画成红色小球 -------------------
        pose_sheet_path = None
        if args.context_camera != "none" and poses:
            if args.context_camera == args.camera:
                print("note: --context-camera 与被扫描的相机相同，跳过位姿图")
            else:
                pose_sheet_path = build_pose_sheet(
                    env, flat, state_indices, poses, args, rig, Image, ImageDraw, ImageFont
                )
    finally:
        rig.close()
        env.close()

    tile_h, tile_w = rows[0][1][0].shape[:2]
    label_w, label_h = 320, 18
    sheet = Image.new(
        "RGB",
        (label_w + len(state_indices) * tile_w, label_h + len(rows) * (tile_h + label_h)),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(sheet)
    for column, state_index in enumerate(state_indices):
        draw.text((label_w + column * tile_w + 4, 4), f"frame {state_index}", fill=(0, 0, 0))
    for row_index, (label, images) in enumerate(rows):
        top = label_h + row_index * (tile_h + label_h)
        draw.text((4, top + 4), label[:52], fill=(0, 0, 0))
        for column, image in enumerate(images):
            sheet.paste(Image.fromarray(image), (label_w + column * tile_w, top + label_h))
    if args.scale > 1:
        sheet = sheet.resize((sheet.width * args.scale, sheet.height * args.scale), Image.NEAREST)
    sheet_path = args.output / f"sweep_{args.camera}.png"
    sheet.save(sheet_path)

    # 只按帧间变化排序，纯属方便定位"画面最会动"的组合，不代表哪个更好。
    # 这里刻意不给出任何"质量分"：腕部相机的画面里本来就应该有夹爪，
    # 任何把"画面中央暗/被夹爪占据"当成缺点的打分都会误导。
    ranked = sorted(table, key=lambda entry: -entry["motion"])
    report = {
        "camera": args.camera,
        "dataset_id": args.dataset_id,
        "episode": args.episode,
        "state_frames": state_indices,
        "combinations": table,
        "by_frame_to_frame_change": ranked[: args.top],
        "sheet": str(sheet_path),
    }
    (args.output / f"sweep_{args.camera}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nframe-to-frame change, largest {min(args.top, len(ranked))} "
          f"(仅便于排序定位，不代表画面更好或更差):")
    for rank, entry in enumerate(ranked[: args.top], start=1):
        print(
            f"  {rank}. eye={[round(v, 3) for v in entry['eye']]} "
            f"target={[round(v, 3) for v in entry['target']]} roll={entry['roll']:+.1f}"
            f"  motion={entry['motion']:.2f}"
        )
    best = ranked[0]
    print(f"\ncopy into {args.camera_config}:")
    print(f"  {args.camera}:")
    print("    kind: attached")
    if frame_kind == "site":
        print(f"    site: {frame_name}")
    else:
        print(f"    body: {frame_name}")
    print(f"    eye: {[round(float(v), 3) for v in best['eye']]}")
    print(f"    target: {[round(float(v), 3) for v in best['target']]}")
    print(f"    roll: {best['roll']}")
    print(f"    reference_camera: {base.get('reference_camera', 'left_cap')}")
    print(f"\nsheet: {sheet_path}")


if __name__ == "__main__":
    main()
