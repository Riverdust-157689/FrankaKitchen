from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .protocol import ControllerState, LatestControllerState
from .server import TeleopServer


# WebXR world displacement -> FrankaKitchen world displacement. This is the
# mapping used by the copied Panda controller; verify axes at a low scale on the
# actual headset before collecting production demonstrations.
VR_TO_KITCHEN = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def quaternion_to_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm < 1e-8:
        raise ValueError("quaternion has zero norm")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World-frame small-angle error used by the differential IK controller."""

    current = np.asarray(current, dtype=np.float64).reshape(3, 3)
    target = np.asarray(target, dtype=np.float64).reshape(3, 3)
    error = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        error += np.cross(current[:, axis], target[:, axis])
    return 0.5 * error


def damped_least_squares(jacobian: np.ndarray, twist: np.ndarray, damping: float) -> np.ndarray:
    jacobian = np.asarray(jacobian, dtype=np.float64)
    twist = np.asarray(twist, dtype=np.float64)
    regularizer = (float(damping) ** 2) * np.eye(jacobian.shape[0])
    return jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + regularizer, twist)


def scaled_rotation(relative_rotation: np.ndarray, scale: float) -> np.ndarray:
    """Interpolate a rotation from identity without adding SciPy as a dependency."""

    quat = np.empty(4, dtype=np.float64)  # MuJoCo order: w, x, y, z
    mujoco.mju_mat2Quat(quat, np.asarray(relative_rotation).ravel())
    if quat[0] < 0:
        quat *= -1
    vector_norm = float(np.linalg.norm(quat[1:]))
    if vector_norm < 1e-9 or scale == 0:
        return np.eye(3)
    angle = 2.0 * np.arctan2(vector_norm, float(quat[0]))
    axis = quat[1:] / vector_norm
    scaled_quat = np.r_[np.cos(scale * angle / 2), axis * np.sin(scale * angle / 2)]
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, scaled_quat)
    return matrix.reshape(3, 3)


@dataclass
class ClutchedTarget:
    vr_position: np.ndarray | None = None
    vr_rotation: np.ndarray | None = None
    ee_position: np.ndarray | None = None
    ee_rotation: np.ndarray | None = None
    active: bool = False

    def update(
        self,
        state: ControllerState,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        position_scale: float,
        rotation_scale: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not state.squeeze:
            self.active = False
            return None
        vr_rotation = quaternion_to_matrix(state.quaternion)
        if not self.active:
            self.vr_position = state.position.copy()
            self.vr_rotation = vr_rotation
            self.ee_position = current_position.copy()
            self.ee_rotation = current_rotation.copy()
            self.active = True
        assert self.vr_position is not None and self.vr_rotation is not None
        assert self.ee_position is not None and self.ee_rotation is not None
        target_position = self.ee_position + position_scale * (
            VR_TO_KITCHEN @ (state.position - self.vr_position)
        )
        relative_vr = vr_rotation @ self.vr_rotation.T
        relative_robot = VR_TO_KITCHEN @ relative_vr @ VR_TO_KITCHEN.T
        return target_position, scaled_rotation(relative_robot, rotation_scale) @ self.ee_rotation


@dataclass
class EpisodeBuffer:
    tasks: list[str]
    seed: int
    instruction: str
    values: dict[str, list[np.ndarray]] = field(default_factory=dict)
    completion_events: list[tuple[int, str]] = field(default_factory=list)

    def append(self, **values: np.ndarray) -> None:
        for key, value in values.items():
            self.values.setdefault(key, []).append(np.asarray(value).copy())

    def save(self, output_dir: Path, episode_index: int) -> Path:
        if not self.values:
            raise ValueError("cannot save an empty teleoperation episode")
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"episode_{episode_index:06d}"
        arrays = {key: np.stack(values) for key, values in self.values.items() if values}
        path = output_dir / f"{stem}.npz"
        np.savez_compressed(path, **arrays)
        metadata = {
            "format_version": 1,
            "tasks": self.tasks,
            "instruction": self.instruction,
            "seed": self.seed,
            "steps": len(next(iter(self.values.values()), [])),
            "completion_events": self.completion_events,
            "npz": path.name,
        }
        (output_dir / f"{stem}.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VR teleoperation for native FrankaKitchen-v1.")
    parser.add_argument("--tasks", nargs="+", default=["microwave", "kettle"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--position-scale", type=float, default=0.5)
    parser.add_argument("--rotation-scale", type=float, default=0.5)
    parser.add_argument("--position-gain", type=float, default=8.0)
    parser.add_argument("--rotation-gain", type=float, default=4.0)
    parser.add_argument("--damping", type=float, default=0.08)
    parser.add_argument("--max-joint-velocity", type=float, default=1.0)
    parser.add_argument("--gripper-action", type=float, default=0.5)
    parser.add_argument("--stale-seconds", type=float, default=0.25)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--websocket-port", type=int, default=8001)
    parser.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/vr/franka_kitchen"))
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate environment/Jacobian/cameras without starting network servers.",
    )
    return parser.parse_args()


def _buttons_edge(current: ControllerState, previous: ControllerState | None, name: str) -> bool:
    return name in current.buttons and (previous is None or name not in previous.buttons)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    from franka_kitchen_vla.cameras import CameraRig
    from franka_kitchen_vla.runtime import make_kitchen_env
    from franka_kitchen_vla.tasks import instruction_for

    env = make_kitchen_env(args.tasks, terminate_on_tasks_completed=False, max_episode_steps=None)
    observation, _ = env.reset(seed=args.seed)
    model, data = env.unwrapped.model, env.unwrapped.data
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    if site_id < 0:
        env.close()
        raise RuntimeError("MuJoCo site 'end_effector' was not found")
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    jacobian_rank = int(np.linalg.matrix_rank(np.vstack((jacp[:, :7], jacr[:, :7]))))
    if jacobian_rank < 5:
        env.close()
        raise RuntimeError("end_effector Jacobian is unexpectedly rank deficient")

    rig = None if args.no_images else CameraRig(model, args.camera_config)
    if args.dry_run:
        cameras = [] if rig is None else list(rig.render(data))
        print(json.dumps({
            "status": "VR_TELEOP_DRY_RUN_OK",
            "tasks": args.tasks,
            "action_shape": list(env.action_space.shape),
            "site_id": site_id,
            "jacobian_rank": jacobian_rank,
            "cameras": cameras,
        }, indent=2))
        if rig is not None:
            rig.close()
        env.close()
        return

    latest = LatestControllerState()
    server = TeleopServer(latest, args.host, args.http_port, args.websocket_port)
    server.start()
    print(
        f"WebXR: http://localhost:{args.http_port}/?ws_port={args.websocket_port}  "
        f"websocket: ws://localhost:{args.websocket_port}"
    )
    print("Hold squeeze to move; trigger closes gripper; A saves; B discards; Ctrl-C exits.")

    episode_index = 0
    seed = args.seed
    step = 0
    previous_state: ControllerState | None = None
    target = ClutchedTarget()
    buffer = EpisodeBuffer(list(args.tasks), seed, instruction_for(args.tasks))
    period = float(env.unwrapped.robot_env.dt)
    try:
        while episode_index < args.episodes:
            started = time.monotonic()
            state = latest.get()
            fresh = state is not None and started - state.received_monotonic <= args.stale_seconds
            if step >= args.max_steps and not fresh:
                delay = period - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
                continue
            if not fresh:
                action = np.zeros(9, dtype=np.float64)
                target.active = False
            else:
                assert state is not None
                if _buttons_edge(state, previous_state, "a"):
                    path = buffer.save(args.output_dir, episode_index)
                    print(f"saved {path}")
                    episode_index += 1
                    seed += 1
                    if episode_index >= args.episodes:
                        break
                    observation, _ = env.reset(seed=seed)
                    buffer = EpisodeBuffer(list(args.tasks), seed, instruction_for(args.tasks))
                    target = ClutchedTarget()
                    step = 0
                    previous_state = state
                    continue
                if _buttons_edge(state, previous_state, "b"):
                    print(f"discarded seed={seed}")
                    observation, _ = env.reset(seed=seed)
                    buffer = EpisodeBuffer(list(args.tasks), seed, instruction_for(args.tasks))
                    target = ClutchedTarget()
                    step = 0
                    previous_state = state
                    continue

                # At the episode budget the simulator stays frozen while the
                # operator decides whether A saves or B retries this seed.
                if step >= args.max_steps:
                    previous_state = state
                    delay = period - (time.monotonic() - started)
                    if delay > 0:
                        time.sleep(delay)
                    continue

                current_position = np.asarray(data.site_xpos[site_id]).copy()
                current_rotation = np.asarray(data.site_xmat[site_id]).reshape(3, 3).copy()
                desired = target.update(
                    state, current_position, current_rotation,
                    args.position_scale, args.rotation_scale,
                )
                arm_action = np.zeros(7, dtype=np.float64)
                if desired is not None:
                    desired_position, desired_rotation = desired
                    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
                    twist = np.r_[
                        args.position_gain * (desired_position - current_position),
                        args.rotation_gain * orientation_error(current_rotation, desired_rotation),
                    ]
                    qvel = damped_least_squares(
                        np.vstack((jacp[:, :7], jacr[:, :7])), twist, args.damping
                    )
                    qvel = np.clip(qvel, -args.max_joint_velocity, args.max_joint_velocity)
                    # FrankaRobot multiplies normalized actions by act_rng=2 before
                    # interpreting them as desired joint velocities.
                    arm_action = qvel / np.asarray(env.unwrapped.robot_env.act_rng[:7])
                grip = -args.gripper_action if state.trigger >= 0.5 else args.gripper_action
                action = np.r_[arm_action, grip, grip]

            frames = {} if rig is None else rig.render(data)
            state_for_record = state if fresh else None
            pre_observation = observation["observation"].copy()
            pre_qpos = data.qpos.copy()
            pre_qvel = data.qvel.copy()
            observation, _, _, _, info = env.step(action)
            for task_name in info.get("step_task_completions", []):
                if task_name not in {task for _, task in buffer.completion_events}:
                    buffer.completion_events.append((step, task_name))
            values: dict[str, np.ndarray] = {
                "observation": pre_observation,
                "qpos": pre_qpos,
                "qvel": pre_qvel,
                "action": action,
                "vr_position": np.full(3, np.nan) if state_for_record is None else state_for_record.position,
                "vr_quaternion": np.full(4, np.nan) if state_for_record is None else state_for_record.quaternion,
                "vr_squeeze": np.array(False if state_for_record is None else state_for_record.squeeze),
                "vr_trigger": np.array(0.0 if state_for_record is None else state_for_record.trigger),
            }
            values.update({f"image_{name}": frame for name, frame in frames.items()})
            buffer.append(**values)
            step += 1
            previous_state = state if fresh else previous_state
            if step == args.max_steps:
                print("max steps reached: press A to save or B to discard")
            delay = period - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        print("stopped; current unsaved episode was discarded")
    finally:
        server.close()
        if rig is not None:
            rig.close()
        env.close()


if __name__ == "__main__":
    main()
