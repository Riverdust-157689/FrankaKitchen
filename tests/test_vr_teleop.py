"""Tests for the VR teleoperation viewer plumbing.

The MuJoCo window itself needs a display, so these tests drive the two helpers with stubs:
what matters is that a closed (or never-opened) window never breaks the control loop.
"""

from VRTeleop.controller import _start_viewer, _sync_viewer


class _StubViewer:
    def __init__(self, running: bool = True) -> None:
        self.running = running
        self.syncs = 0
        self.closes = 0

    def is_running(self) -> bool:
        return self.running

    def sync(self) -> None:
        self.syncs += 1

    def close(self) -> None:
        self.closes += 1


def test_sync_viewer_is_a_noop_without_a_window():
    assert _sync_viewer(None) is None


def test_sync_viewer_pushes_state_while_the_window_is_open():
    viewer = _StubViewer()
    assert _sync_viewer(viewer) is viewer
    assert viewer.syncs == 1
    assert viewer.closes == 0


def test_sync_viewer_releases_a_closed_window_and_stops_syncing():
    viewer = _StubViewer(running=False)
    assert _sync_viewer(viewer) is None
    assert viewer.closes == 1
    assert viewer.syncs == 0
    # the loop keeps running: a second call is simply a no-op
    assert _sync_viewer(None) is None


def test_start_viewer_can_be_disabled_without_touching_gl():
    # `enabled=False` must return before importing mujoco.viewer, so this works headless.
    assert _start_viewer(object(), object(), enabled=False) is None


def _stub_viewer():
    import mujoco

    class _Viewer:
        def __init__(self) -> None:
            self.cam = mujoco.MjvCamera()

    return _Viewer()


def test_apply_default_view_copies_the_global_free_camera(tmp_path):
    """The window must start from the same view the policy is trained on, not MuJoCo's
    stat.center/extent framing."""
    import mujoco

    from VRTeleop.controller import _apply_default_view

    config = tmp_path / "cameras.yaml"
    config.write_text(
        "width: 64\nheight: 64\ncameras:\n"
        "  global:\n    kind: free\n    lookat: [-0.2, 0.5, 2.0]\n"
        "    distance: 2.2\n    azimuth: 70.0\n    elevation: -35.0\n"
        "  wrist:\n    kind: attached\n    site: end_effector\n"
        "    eye: [0.0, 0.0, -0.21]\n    target: [0.0, 0.0, 0.14]\n",
        encoding="utf-8",
    )
    viewer = _stub_viewer()
    assert _apply_default_view(viewer, config, "global") == "global"
    assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert viewer.cam.fixedcamid == -1
    assert list(viewer.cam.lookat) == [-0.2, 0.5, 2.0]
    assert viewer.cam.distance == 2.2
    assert viewer.cam.azimuth == 70.0
    assert viewer.cam.elevation == -35.0


def test_apply_default_view_falls_back_to_the_first_free_camera(tmp_path):
    from VRTeleop.controller import _apply_default_view

    config = tmp_path / "cameras.yaml"
    config.write_text(
        "width: 64\nheight: 64\ncameras:\n"
        "  wide:\n    kind: free\n    lookat: [0.0, 0.0, 1.0]\n"
        "    distance: 3.0\n    azimuth: 10.0\n    elevation: -10.0\n",
        encoding="utf-8",
    )
    viewer = _stub_viewer()
    assert _apply_default_view(viewer, config, "global") == "wide"
    assert viewer.cam.distance == 3.0


def test_apply_default_view_is_harmless_without_a_window_or_free_camera(tmp_path):
    from VRTeleop.controller import _apply_default_view

    assert _apply_default_view(None, tmp_path / "missing.yaml", "global") is None

    config = tmp_path / "attached_only.yaml"
    config.write_text(
        "width: 64\nheight: 64\ncameras:\n"
        "  wrist:\n    kind: attached\n    site: end_effector\n"
        "    eye: [0.0, 0.0, -0.21]\n    target: [0.0, 0.0, 0.14]\n",
        encoding="utf-8",
    )
    viewer = _stub_viewer()
    assert _apply_default_view(viewer, config, "global") is None
