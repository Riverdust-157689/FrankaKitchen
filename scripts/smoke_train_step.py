#!/usr/bin/env python
"""One SmolVLA forward/backward/optimizer step on a converted Franka Kitchen dataset.

This is a memory and wiring check, not training. Run it on the GPU host before
starting a long fine-tune: it catches camera-key mismatches, state/action shape
mismatches, broken dataset metadata and out-of-memory conditions in about a minute.

    python scripts/smoke_train_step.py --dataset-root data/lerobot/fk_partial --device cuda
    python scripts/smoke_train_step.py --dataset-root ... --resize 224   # if 256 OOMs

It deliberately builds the policy the same way `lerobot-train` does -- through
`make_policy(config, ds_meta=...)` after setting `config.input_features = None`, which
is what `--policy.input_features=null` does on the CLI -- so a pass here means the real
training command is wired correctly too.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

# Importing this registers the ``smolvla`` config choice with draccus.
from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.datasets import LeRobotDataset, resolve_delta_timestamps
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies import make_policy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.feature_utils import dataset_to_policy_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/lerobot/fk_partial"))
    parser.add_argument("--repo-id", default=None, help="Default: read it from the conversion manifest.")
    parser.add_argument("--checkpoint", default="lerobot/smolvla_base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resize",
        type=int,
        default=256,
        help="Square size SmolVLA resizes to internally (default 256; 512 is the base config).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Real batches per step; the smoke test runs one."
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument(
        "--freeze-vision-encoder", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--train-expert-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Validate the dataset and processors without building the model.",
    )
    return parser.parse_args()


def memory_gib(num_bytes: int) -> float:
    return num_bytes / 1024**3


def resolve_checkpoint(checkpoint: str) -> str:
    """Turn a Hub repo id into its local snapshot dir when it is already cached.

    ``PreTrainedConfig.from_pretrained`` calls ``hf_hub_download`` with
    ``local_files_only=False`` and so performs a network round trip even when
    ``HF_HUB_OFFLINE=1`` is set and the checkpoint is fully cached. Resolving the
    snapshot ourselves keeps the smoke test usable on a machine whose Hub access is
    blocked, proxied, or simply offline.
    """
    if Path(checkpoint).is_dir():
        return checkpoint
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=checkpoint, local_files_only=True)
    except Exception:
        return checkpoint


def resolve_repo_id(dataset_root: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    manifest = dataset_root / "conversion_manifest.json"
    if manifest.is_file():
        repo_id = json.loads(manifest.read_text(encoding="utf-8")).get("repo_id")
        if repo_id:
            return str(repo_id)
    return f"local/{dataset_root.name}"


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not (dataset_root / "meta" / "info.json").is_file():
        raise SystemExit(f"not a LeRobot dataset (meta/info.json missing): {dataset_root}")
    repo_id = resolve_repo_id(dataset_root, args.repo_id)

    checkpoint = resolve_checkpoint(args.checkpoint)
    config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"{args.checkpoint} is not a SmolVLA checkpoint: {type(config).__name__}")
    config.device = args.device
    config.use_amp = args.amp_dtype != "none"
    config.resize_imgs_with_padding = (args.resize, args.resize)
    config.freeze_vision_encoder = args.freeze_vision_encoder
    config.train_expert_only = args.train_expert_only
    config.pretrained_path = checkpoint

    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(config, metadata)
    dataset = LeRobotDataset(
        repo_id, root=dataset_root, delta_timestamps=delta_timestamps, return_uint8=True
    )
    if len(dataset) == 0:
        raise SystemExit("dataset is empty")

    # This is what `--policy.input_features=null` makes make_policy() do: the base
    # checkpoint ships ALOHA/SO-100 features (6-D state, camera1/2/3), so they are
    # replaced wholesale by this dataset's schema before anything is built.
    policy_features = dataset_to_policy_features(dataset.meta.features)
    config.output_features = {
        key: feature for key, feature in policy_features.items() if feature.type is FeatureType.ACTION
    }
    config.input_features = {
        key: feature for key, feature in policy_features.items() if key not in config.output_features
    }

    samples = [dataset[index] for index in range(min(args.batch_size, len(dataset)))]
    batch = lerobot_collate_fn(samples)
    if batch is None:
        raise SystemExit("collating the smoke sample produced an empty batch")
    for camera_key in dataset.meta.camera_keys:
        if batch[camera_key].dtype == torch.uint8:
            batch[camera_key] = batch[camera_key].to(dtype=torch.float32) / 255.0

    preprocessor, _ = make_smolvla_pre_post_processors(config, dataset_stats=dataset.meta.stats)
    batch = preprocessor(batch)

    print(
        f"dataset      : {dataset_root}\n"
        f"  episodes={dataset.num_episodes} frames={dataset.num_frames} fps={dataset.meta.fps}\n"
        f"  cameras={dataset.meta.camera_keys}\n"
        f"  state={tuple(batch['observation.state'].shape)}"
        f" action={tuple(batch['action'].shape)}"
        f" tasks={len(dataset.meta.tasks) if hasattr(dataset.meta, 'tasks') else '?'}"
    )
    for key in dataset.meta.camera_keys:
        print(f"    {key:34s} {tuple(batch[key].shape)} {batch[key].dtype}")
    if args.preprocess_only:
        print("SMOLVLA_PREPROCESS_OK")
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run this on the GPU host, or pass --device cpu")

    policy = make_policy(config, ds_meta=dataset.meta)
    policy.train()
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    total_count = sum(parameter.numel() for parameter in policy.parameters())
    trainable_count = sum(parameter.numel() for parameter in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-10)

    device_type = torch.device(args.device).type
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": torch.float32}[args.amp_dtype]
    if device_type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free_before, total_memory = torch.cuda.mem_get_info()
        print(
            f"gpu          : total={memory_gib(total_memory):.2f}GiB "
            f"free_before={memory_gib(free_before):.2f}GiB"
        )

    try:
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=args.amp_dtype != "none"):
            loss, loss_dict = policy(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config.optimizer_grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    except torch.cuda.OutOfMemoryError as error:
        if device_type == "cuda":
            peak = memory_gib(torch.cuda.max_memory_allocated())
            torch.cuda.empty_cache()
            raise SystemExit(
                f"SMOLVLA_SMOKE_OOM peak_before_failure={peak:.2f}GiB; "
                f"retry with --resize 224 or --batch-size 1"
            ) from error
        raise

    fields = [
        f"loss={float(loss.detach()):.6f}",
        f"grad_norm={float(grad_norm):.6f}",
        f"total={total_count / 1e6:.1f}M",
        f"trainable={trainable_count / 1e6:.1f}M",
        f"batch={len(samples)}",
    ]
    if device_type == "cuda":
        fields.append(f"gpu_peak={memory_gib(torch.cuda.max_memory_allocated()):.2f}GiB")
        fields.append(f"gpu_reserved={memory_gib(torch.cuda.max_memory_reserved()):.2f}GiB")
    if isinstance(loss_dict, dict) and "loss" in loss_dict:
        fields.append(f"model_loss={float(loss_dict['loss']):.6f}")
    print("SMOLVLA_TRAIN_STEP_OK " + " ".join(fields))


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
