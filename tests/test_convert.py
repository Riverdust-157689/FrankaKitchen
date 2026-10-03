import numpy as np
import pytest

from franka_kitchen_vla.cameras import load_camera_config
from franka_kitchen_vla.convert import _features, parse_exclude_sequence, plan_variants

EVENTS = [
    (56, "microwave"),
    (130, "kettle"),
    (180, "light switch"),
    (227, "slide cabinet"),
]


def test_plan_variants_without_prefixes_yields_the_full_sequence():
    assert plan_variants(EVENTS, emit_prefixes=False) == [
        (227, ("microwave", "kettle", "light switch", "slide cabinet"))
    ]


def test_plan_variants_with_prefixes_yields_one_episode_per_task():
    variants = plan_variants(EVENTS, emit_prefixes=True)
    assert [end for end, _ in variants] == [56, 130, 180, 227]
    assert [len(tasks) for _, tasks in variants] == [1, 2, 3, 4]
    assert variants[0][1] == ("microwave",)
    assert variants[-1][1] == ("microwave", "kettle", "light switch", "slide cabinet")


def test_plan_variants_with_exact_two_task_prefix_yields_one_episode():
    assert plan_variants(EVENTS, emit_prefixes=False, prefix_length=2) == [
        (130, ("microwave", "kettle"))
    ]


def test_plan_variants_exact_prefix_skips_short_sequences():
    assert plan_variants(EVENTS[:1], emit_prefixes=False, prefix_length=2) == []


def test_plan_variants_handles_no_events():
    assert plan_variants([], emit_prefixes=True) == []
    assert plan_variants([], emit_prefixes=False) == []


def test_parse_exclude_sequence_trims_and_splits():
    assert parse_exclude_sequence(" kettle , light switch ") == ("kettle", "light switch")
    with pytest.raises(ValueError):
        parse_exclude_sequence(" , ")


def test_features_match_the_smolvla_contract():
    features = _features(["global", "wrist"], 256, 256, use_videos=True)
    assert features["observation.images.global"]["dtype"] == "video"
    assert features["observation.images.global"]["shape"] == (256, 256, 3)
    assert features["observation.state"]["shape"] == (18,)
    assert features["action"]["shape"] == (9,)
    assert len(features["observation.state"]["names"]) == 18
    assert len(features["action"]["names"]) == 9

    png = _features(["global"], 128, 128, use_videos=False)
    assert png["observation.images.global"]["dtype"] == "image"


def test_camera_config_subset_and_resolution_override():
    width, height, cameras = load_camera_config("configs/cameras.yaml", width=128, height=64)
    assert (width, height) == (128, 64)
    assert [camera.name for camera in cameras] == ["global", "wrist"]

    _, _, only_wrist = load_camera_config("configs/cameras.yaml", only=["wrist"])
    assert [camera.name for camera in only_wrist] == ["wrist"]

    with pytest.raises(ValueError):
        load_camera_config("configs/cameras.yaml", only=["nope"])
