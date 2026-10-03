from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any

import mujoco
import numpy as np
import yaml


@dataclass(frozen=True)
class CameraConfig:
    name: str
    kind: str
    values: dict[str, Any]


def load_camera_config(
    path: str | Path,
    width: int | None = None,
    height: int | None = None,
    only: list[str] | None = None,
) -> tuple[int, int, list[CameraConfig]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    resolved_width = int(raw["width"] if width is None else width)
    resolved_height = int(raw["height"] if height is None else height)
    if resolved_width <= 0 or resolved_height <= 0:
        raise ValueError("Camera width and height must be positive.")
    cameras = [
        CameraConfig(name=name, kind=str(values["kind"]), values=dict(values))
        for name, values in raw["cameras"].items()
    ]
    if not cameras:
        raise ValueError("At least one camera must be configured.")
    if only:
        available = {camera.name for camera in cameras}
        unknown = set(only).difference(available)
        if unknown:
            raise ValueError(f"Unknown camera(s): {sorted(unknown)}; available: {sorted(available)}")
        cameras = [camera for camera in cameras if camera.name in set(only)]
    return resolved_width, resolved_height, cameras


def rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation of ``vector`` about ``axis`` (right-hand rule)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / float(np.linalg.norm(axis))
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = np.eye(3) + np.sin(angle_rad) * cross + (1.0 - np.cos(angle_rad)) * (cross @ cross)
    return rotation @ np.asarray(vector, dtype=float)


# If the view direction is within this much of the roll reference, the reference is
# unusable (its perpendicular part vanishes) and the fallback axis is used instead.
_ROLL_REFERENCE_DEGENERATE_DOT = 0.9999


def _rotation_with_z(direction: np.ndarray) -> np.ndarray:
    """Rotation matrix whose third column is ``direction`` (used to orient a capsule)."""
    z = np.asarray(direction, dtype=float)
    z = z / float(np.linalg.norm(z))
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x = x / float(np.linalg.norm(x))
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def add_scene_sphere(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    radius: float,
    rgba: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
) -> bool:
    """Append a debug sphere to a MuJoCo scene. Returns False when the scene is full.

    Debug geometry must be added **after** ``mjv_updateScene`` (which resets
    ``scene.ngeom``) and before ``mjv_render``; that is what the ``annotate`` hook on
    :meth:`CameraRig.render` exists for. ``mjv_initGeom`` is always used because it
    initialises every field of the slot, including ``objtype``/``objid``/``dataid`` --
    a slot that only gets partly written can make the renderer follow a stale object
    reference.
    """
    if scene.ngeom >= scene.maxgeom:
        return False
    mujoco.mjv_initGeom(
        scene.geoms[int(scene.ngeom)],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, radius, radius], dtype=float),
        pos=np.asarray(position, dtype=float),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1
    return True


def add_scene_capsule(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> bool:
    """Append a debug capsule between two points (drawn as the view axis).

    Deliberately built from ``mjv_initGeom`` with a hand-computed frame instead of
    ``mjv_connector``: connector leaves ``objtype``/``objid``/``dataid`` untouched.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1e-9 or scene.ngeom >= scene.maxgeom:
        return False
    mujoco.mjv_initGeom(
        scene.geoms[int(scene.ngeom)],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=np.array([radius, length / 2.0, 0.0], dtype=float),
        pos=0.5 * (start + end),
        mat=_rotation_with_z(delta / length).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1
    return True


def mark_pose_in_scene(
    scene: mujoco.MjvScene,
    eye: np.ndarray,
    target: np.ndarray,
    radius: float = 0.02,
    color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
    line_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    line_width: float = 0.004,
) -> bool:
    """Draw a camera pose as two balls plus the view axis between them.

    Atomic: if the scene has no room for all three primitives nothing is drawn, so a
    tight ``maxgeom`` can never produce a half-drawn pose. Returns whether it drew.
    """
    if scene.ngeom + 3 > scene.maxgeom:
        return False
    add_scene_capsule(scene, eye, target, line_width, line_color)
    add_scene_sphere(scene, eye, radius, color)
    add_scene_sphere(scene, target, radius, color)
    return True


def attached_frame_pose(
    model: mujoco.MjModel, data: mujoco.MjData, values: Mapping[str, Any], label: str = "camera"
) -> tuple[np.ndarray, np.ndarray]:
    """World ``(position, rotation)`` of the frame an attached camera hangs off.

    Split out of :meth:`CameraRig._update_attached_scene` so that callers who only want
    to *draw* where a camera is (sweep plots, diagnostics) can get the same frame the
    renderer uses without duplicating the body/site lookup.
    """
    kind, name = attached_frame_reference(values, label)
    if kind == "body":
        obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if obj_id < 0:
            raise ValueError(f"MuJoCo body {name!r} does not exist.")
        return np.array(data.xpos[obj_id]), np.array(data.xmat[obj_id]).reshape(3, 3)
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if obj_id < 0:
        available = sorted(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)
        )
        raise ValueError(f"MuJoCo site {name!r} does not exist. Available: {available}")
    return np.array(data.site_xpos[obj_id]), np.array(data.site_xmat[obj_id]).reshape(3, 3)


def resolve_attached_pose(
    values: dict[str, Any],
    position: np.ndarray,
    rotation: np.ndarray,
    label: str = "camera",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn an attached-camera config into a world ``(eye, forward, up)`` triple.

    An attached camera is specified by exactly three things:

    ``eye``    [x, y, z] position, in the attached frame's local coordinates.
    ``target`` [x, y, z] look-at point, in the same coordinates.
    ``roll``   degrees of rotation about the view axis (optional, default 0).

    Only the *direction* ``target - eye`` matters, not the distance to it: moving
    ``target`` further along the same ray changes nothing.

    ``roll = 0`` means "the body's +y axis points up in the image (as closely as the
    view allows)". As with any look-at convention that is undefined when the view is
    along that axis, so when the view comes within ~0.8 degrees of the body's ±y the
    body's +z is used as the reference instead. Positive ``roll`` rotates the image
    content counter-clockwise (right-hand rule about the view direction).

    The attached frame is chosen by the caller (see :func:`attached_frame_reference`):
    either a MuJoCo ``body`` or a MuJoCo ``site``. Kept free of MuJoCo objects so the pose
    maths can be unit tested without a renderer.
    """
    position = np.asarray(position, dtype=float)
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    for key in ("eye", "target"):
        if key not in values:
            raise ValueError(f"Attached camera {label!r} needs {key!r}.")
        if np.asarray(values[key], dtype=float).shape != (3,):
            raise ValueError(
                f"Attached camera {label!r}: {key!r} must be three numbers, got {values[key]!r}."
            )

    eye = position + rotation @ np.asarray(values["eye"], dtype=float)
    target = position + rotation @ np.asarray(values["target"], dtype=float)

    forward = target - eye
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-8:
        raise ValueError(
            f"Attached camera {label!r}: 'eye' and 'target' coincide, so the view "
            "direction is undefined."
        )
    forward = forward / forward_norm

    # Roll reference: the body's +y, falling back to +z when the view is along +y.
    reference = rotation @ np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(reference, forward))) > _ROLL_REFERENCE_DEGENERATE_DOT:
        reference = rotation @ np.array([0.0, 0.0, 1.0])
    up = reference - forward * float(np.dot(reference, forward))
    up_norm = float(np.linalg.norm(up))
    if up_norm < 1e-8:  # pragma: no cover - both references cannot vanish at once
        raise ValueError(
            f"Attached camera {label!r}: cannot derive a roll reference for the view "
            f"direction {np.round(forward, 6)!r}."
        )
    up = up / up_norm

    roll = values.get("roll")
    if roll is not None:
        roll = float(roll)
        if not np.isfinite(roll):
            raise ValueError(f"Attached camera {label!r} has a non-finite roll: {roll!r}.")
        up = rotate_about_axis(up, forward, np.deg2rad(roll))
        up = up / float(np.linalg.norm(up))

    return eye, forward, up


def attached_frame_reference(values: Mapping[str, Any], label: str = "camera") -> tuple[str, str]:
    """Which frame an attached camera hangs off: ``("body" | "site", name)``.

    Both kinds are supported because they are genuinely different frames, even when they
    sit on the same link. On ``panda0_link7`` the model's own ``end_effector`` site is
    21 cm further along +z and rotated 45 degrees about z relative to the link frame,
    because the fingers slide along the link's (x = y) diagonal. Expressing a camera in
    the site frame is what makes "eye y = 0" mean "on the jaw's centre line".
    """
    has_body = values.get("body") is not None
    has_site = values.get("site") is not None
    if has_body and has_site:
        raise ValueError(
            f"Attached camera {label!r} sets both 'body' and 'site'; use exactly one. They "
            "are different frames even on the same link (panda0_link7 and its end_effector "
            "site are 21 cm and 45 degrees apart)."
        )
    if not has_body and not has_site:
        raise ValueError(
            f"Attached camera {label!r} needs either 'body' or 'site' to say which frame "
            "its eye/target are expressed in."
        )
    kind = "body" if has_body else "site"
    name = str(values[kind])
    if not name:
        raise ValueError(f"Attached camera {label!r}: {kind!r} must not be empty.")
    return kind, name


class CameraRig:
    """Render named, free, and body-attached views from one MuJoCo state."""

    def __init__(
        self,
        model: mujoco.MjModel,
        config_path: str | Path,
        width: int | None = None,
        height: int | None = None,
        only: list[str] | None = None,
    ):
        self.model = model
        self.width, self.height, self.configs = load_camera_config(
            config_path, width=width, height=height, only=only
        )
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)

    @property
    def names(self) -> list[str]:
        return [camera.name for camera in self.configs]

    def close(self) -> None:
        self.renderer.close()

    def render(
        self,
        data: mujoco.MjData,
        selected: list[str] | None = None,
        configs: list[CameraConfig] | None = None,
        annotate: Callable[[mujoco.MjvScene, mujoco.MjData], None] | None = None,
    ) -> dict[str, np.ndarray]:
        """Render cameras.

        ``annotate`` runs after ``update_scene`` (which resets ``scene.ngeom``) and just
        before rasterising -- the only window in which extra debug geoms survive. Use
        :func:`mark_pose_in_scene` to draw an eye/target pair.

        ``configs`` overrides the camera list for this call only, reusing the existing
        renderer. Sweeping tools should pass their per-iteration camera here instead of
        building a new :class:`CameraRig` each time: repeatedly creating and destroying
        offscreen GL contexts is slow and, on some drivers, ends in a segfault.
        """
        use_configs = list(configs) if configs is not None else self.configs
        available = [camera.name for camera in use_configs]
        selected_names = set(selected or available)
        unknown = selected_names.difference(available)
        if unknown:
            raise ValueError(f"Unknown camera(s): {sorted(unknown)}; available: {available}")
        return {
            camera.name: self._render_one(data, camera, annotate)
            for camera in use_configs
            if camera.name in selected_names
        }

    def _render_one(
        self,
        data: mujoco.MjData,
        camera: CameraConfig,
        annotate: Callable[[mujoco.MjvScene, mujoco.MjData], None] | None = None,
    ) -> np.ndarray:
        if camera.kind == "fixed":
            self.renderer.update_scene(data, camera=str(camera.values["name"]))
        elif camera.kind == "free":
            free_camera = mujoco.MjvCamera()
            free_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            free_camera.lookat[:] = np.asarray(camera.values["lookat"], dtype=float)
            free_camera.distance = float(camera.values["distance"])
            free_camera.azimuth = float(camera.values["azimuth"])
            free_camera.elevation = float(camera.values["elevation"])
            self.renderer.update_scene(data, camera=free_camera)
        elif camera.kind == "attached":
            self._update_attached_scene(data, camera)
        else:
            raise ValueError(f"Unsupported camera kind {camera.kind!r} for {camera.name!r}.")
        if annotate is not None:
            annotate(self.renderer.scene, data)
        return self.renderer.render().copy()

    def _update_attached_scene(self, data: mujoco.MjData, camera: CameraConfig) -> None:
        """Place an attached camera, i.e. glue the frame's pose to the config offsets."""
        values = camera.values
        reference = str(values.get("reference_camera", "left_cap"))
        self.renderer.update_scene(data, camera=reference)

        position, rotation = attached_frame_pose(self.model, data, values, camera.name)
        eye, forward, up = resolve_attached_pose(values, position, rotation, label=camera.name)

        # MuJoCo stores two GL cameras for stereo rendering. Keeping their poses
        # equal gives a conventional monocular frame with an arbitrary rigid pose.
        for scene_camera in self.renderer.scene.camera:
            scene_camera.pos[:] = eye
            scene_camera.forward[:] = forward
            scene_camera.up[:] = up
