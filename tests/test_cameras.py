import numpy as np
import pytest

from franka_kitchen_vla.cameras import (
    attached_frame_reference,
    load_camera_config,
    resolve_attached_pose,
    rotate_about_axis,
)

IDENTITY = np.eye(3)
ORIGIN = np.zeros(3)

# The reduced scheme: position (3) + look-at point (3) + roll (1).
BASE = {"eye": [0.0, 0.0, -0.20], "target": [0.0, 0.0, 0.35]}


def test_default_camera_config_has_global_and_wrist():
    width, height, cameras = load_camera_config("configs/cameras.yaml")
    assert (width, height) == (256, 256)
    assert [camera.name for camera in cameras] == ["global", "wrist"]
    assert [camera.kind for camera in cameras] == ["free", "attached"]


def test_attached_configs_carry_only_eye_target_roll():
    """The scheme is deliberately minimal; keep stray orientation keys out."""
    for path in ("configs/cameras.yaml", "configs/cameras_candidates.yaml"):
        _, _, cameras = load_camera_config(path)
        for camera in cameras:
            if camera.kind != "attached":
                continue
            assert "eye" in camera.values, (path, camera.name)
            assert "target" in camera.values, (path, camera.name)
            assert np.asarray(camera.values["eye"]).shape == (3,), (path, camera.name)
            assert np.asarray(camera.values["target"]).shape == (3,), (path, camera.name)
            for removed in ("up", "forward", "tilt", "swivel", "orientation"):
                assert removed not in camera.values, (path, camera.name, removed)
            # exactly one frame reference, and it must resolve
            assert ("body" in camera.values) ^ ("site" in camera.values), (path, camera.name)
            attached_frame_reference(camera.values, camera.name)


def test_arm_cameras_use_the_end_effector_site():
    """The gripper frame is 45 degrees away from link7's, so the site is the right one.

    Pinning this stops the confusion from creeping back in: in panda0_link7's frame the
    finger slide axis is the (x = y) diagonal, so `eye_y == 0` does NOT mean "on the
    gripper centre line".
    """
    for path in ("configs/cameras.yaml", "configs/cameras_candidates.yaml"):
        _, _, cameras = load_camera_config(path)
        for camera in cameras:
            if camera.kind != "attached":
                continue
            assert camera.values.get("site") == "end_effector", (path, camera.name)
            assert "body" not in camera.values, (path, camera.name)


def test_frame_reference_requires_exactly_one_of_body_or_site():
    assert attached_frame_reference({"body": "panda0_link7"}) == ("body", "panda0_link7")
    assert attached_frame_reference({"site": "end_effector"}) == ("site", "end_effector")
    for bad, pattern in (
        ({}, "needs either"),
        ({"body": "a", "site": "b"}, "exactly one"),
        ({"body": ""}, "must not be empty"),
        ({"site": None}, "needs either"),
    ):
        with pytest.raises(ValueError, match=pattern):
            attached_frame_reference(bad, "cam")


def test_pose_is_the_eye_position_and_normalised_forward():
    eye, forward, up = resolve_attached_pose(BASE, ORIGIN, IDENTITY)
    np.testing.assert_allclose(eye, BASE["eye"])
    np.testing.assert_allclose(forward, [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(forward), 1.0)
    np.testing.assert_allclose(np.linalg.norm(up), 1.0)
    np.testing.assert_allclose(float(np.dot(forward, up)), 0.0, atol=1e-12)


def test_only_the_target_direction_matters():
    """Any target on the same ray gives an identical pose -- distance is free."""
    reference = resolve_attached_pose(BASE, ORIGIN, IDENTITY)
    for scale in (0.5, 2.0, 1000.0):
        stretched = {**BASE, "target": [scale * value for value in BASE["target"]]}
        for left, right in zip(
            reference, resolve_attached_pose(stretched, ORIGIN, IDENTITY), strict=True
        ):
            np.testing.assert_allclose(left, right, atol=1e-12)


def test_roll_zero_and_absent_are_the_same():
    for left, right in zip(
        resolve_attached_pose(BASE, ORIGIN, IDENTITY),
        resolve_attached_pose({**BASE, "roll": 0.0}, ORIGIN, IDENTITY),
        strict=True,
    ):
        np.testing.assert_allclose(left, right, atol=1e-12)


def test_roll_is_a_pure_spin():
    """Roll must not move the camera or turn the view -- only spin the image."""
    eye_a, forward_a, up_a = resolve_attached_pose(BASE, ORIGIN, IDENTITY)
    eye_b, forward_b, up_b = resolve_attached_pose({**BASE, "roll": 37.0}, ORIGIN, IDENTITY)
    np.testing.assert_allclose(eye_a, eye_b)
    np.testing.assert_allclose(forward_a, forward_b)
    assert float(np.linalg.norm(up_a - up_b)) > 1e-2
    np.testing.assert_allclose(float(np.dot(forward_b, up_b)), 0.0, atol=1e-12)


def test_roll_90_rotates_up_from_y_to_minus_x():
    """Right-hand rule about forward = +z maps the body's +y up onto -x."""
    _, forward, up = resolve_attached_pose({**BASE, "roll": 90.0}, ORIGIN, IDENTITY)
    np.testing.assert_allclose(forward, [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(up, [-1.0, 0.0, 0.0], atol=1e-12)


def test_roll_is_periodic():
    first = resolve_attached_pose({**BASE, "roll": 30.0}, ORIGIN, IDENTITY)[2]
    second = resolve_attached_pose({**BASE, "roll": 390.0}, ORIGIN, IDENTITY)[2]
    np.testing.assert_allclose(first, second, atol=1e-9)


def test_roll_reference_is_the_body_y_axis():
    """roll = 0 must keep the body's +y as close to image-up as the view allows."""
    _, _, up = resolve_attached_pose(BASE, ORIGIN, IDENTITY)
    np.testing.assert_allclose(up, [0.0, 1.0, 0.0], atol=1e-12)
    # A view along the body's +x still has a usable +y reference.
    _, forward, up = resolve_attached_pose(
        {"eye": [0.0, 0.0, 0.0], "target": [1.0, 0.0, 0.0]}, ORIGIN, IDENTITY
    )
    np.testing.assert_allclose(forward, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(up, [0.0, 1.0, 0.0], atol=1e-12)


def test_roll_reference_falls_back_when_the_view_follows_the_body_y():
    """Looking along +y would make the +y reference useless; +z must take over."""
    _, forward, up = resolve_attached_pose(
        {"eye": [0.0, 0.0, 0.0], "target": [0.0, 1.0, 0.0]}, ORIGIN, IDENTITY
    )
    np.testing.assert_allclose(forward, [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(up, [0.0, 0.0, 1.0], atol=1e-12)


def test_pose_rides_along_with_the_body():
    """The same offsets under a rotated/translated body must move with it."""
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    position = np.array([1.0, 2.0, 3.0])
    eye, forward, up = resolve_attached_pose(BASE, position, rotation)
    np.testing.assert_allclose(eye, position + rotation @ np.asarray(BASE["eye"]))
    np.testing.assert_allclose(forward, rotation @ np.array([0.0, 0.0, 1.0]), atol=1e-12)
    np.testing.assert_allclose(up, rotation @ np.array([0.0, 1.0, 0.0]), atol=1e-12)


def test_rotate_about_axis_matches_hand_computed_values():
    np.testing.assert_allclose(
        rotate_about_axis([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], np.pi / 2), [0.0, 1.0, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(
        rotate_about_axis([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], np.pi / 2), [0.0, 0.0, 1.0], atol=1e-12
    )
    # rotating about its own axis changes nothing
    np.testing.assert_allclose(
        rotate_about_axis([0.0, 0.0, 1.0], [0.0, 0.0, 2.0], 1.234), [0.0, 0.0, 1.0], atol=1e-12
    )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"target": [0.0, 0.0, 1.0]}, "eye"),
        ({"eye": [0.0, 0.0, 0.0]}, "target"),
        ({"eye": [0.0, 0.0, 0.0], "target": [0.0, 0.0, 0.0]}, "coincide"),
        ({"eye": [0.0, 0.0], "target": [0.0, 0.0, 1.0]}, "three numbers"),
        ({"eye": [0.0, 0.0, 0.0], "target": [0.0, 0.0, 1.0], "roll": float("nan")}, "non-finite"),
        ({"eye": [0.0, 0.0, 0.0], "target": [0.0, 0.0, 1.0], "roll": float("inf")}, "non-finite"),
    ],
)
def test_invalid_attached_configs_are_rejected(values, message):
    with pytest.raises(ValueError, match=message):
        resolve_attached_pose(values, ORIGIN, IDENTITY)


# ------------------------------------------------- debug annotations & frame pose
# `angle="radian"` matters: MuJoCo defaults to degrees, and the real kitchen model sets
# radian so that the end_effector's `euler="0 0 -0.785398"` really is -45 degrees.
TINY_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="link" pos="0 0 0">
      <geom type="sphere" size="0.05"/>
      <site name="tip" pos="0 0 0.21" euler="0 0 -0.785398" size="0.01"/>
    </body>
  </worldbody>
</mujoco>
"""


def tiny_model_data():
    import mujoco

    model = mujoco.MjModel.from_xml_string(TINY_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_attached_frame_pose_reads_a_site():
    from franka_kitchen_vla.cameras import attached_frame_pose

    model, data = tiny_model_data()
    position, rotation = attached_frame_pose(model, data, {"site": "tip"}, "cam")
    np.testing.assert_allclose(position, [0.0, 0.0, 0.21], atol=1e-12)
    # the site carries its own -45 degree roll about z
    # Rz(-45 deg) applied to +y, i.e. the jaw axis in the parent frame
    np.testing.assert_allclose(rotation @ np.array([0.0, 1.0, 0.0]), [0.70710678, 0.70710678, 0.0], atol=1e-6)


def test_attached_frame_pose_reads_a_body():
    from franka_kitchen_vla.cameras import attached_frame_pose

    model, data = tiny_model_data()
    position, rotation = attached_frame_pose(model, data, {"body": "link"}, "cam")
    np.testing.assert_allclose(position, [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)


def test_attached_frame_pose_rejects_a_missing_site_and_lists_alternatives():
    from franka_kitchen_vla.cameras import attached_frame_pose

    model, data = tiny_model_data()
    with pytest.raises(ValueError, match="does not exist"):
        attached_frame_pose(model, data, {"site": "nope"}, "cam")
    with pytest.raises(ValueError, match="tip"):
        attached_frame_pose(model, data, {"site": "nope"}, "cam")


def test_hide_end_effector_site_clears_only_that_alpha():
    """The TCP marker is a modelling aid; it must not show up as a ball in renders."""
    import mujoco

    from franka_kitchen_vla.runtime import hide_end_effector_site

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="link">
              <geom type="sphere" size="0.05"/>
              <site name="end_effector" pos="0 0 0.21" size="0.01"/>
              <site name="kettle_site" pos="0.3 0 0" size="0.01" rgba="1 0 0 1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    assert hide_end_effector_site(model) is True
    end_effector = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    kettle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "kettle_site")
    assert model.site_rgba[end_effector, 3] == 0.0
    # colour is untouched apart from the alpha, and other sites are left alone
    np.testing.assert_allclose(model.site_rgba[end_effector, :3], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(model.site_rgba[kettle], [1.0, 0.0, 0.0, 1.0])


def test_hide_end_effector_site_reports_a_missing_site():
    import mujoco

    from franka_kitchen_vla.runtime import hide_end_effector_site

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='link'><geom type='sphere' size='0.05'/></body></worldbody></mujoco>"
    )
    assert hide_end_effector_site(model) is False
