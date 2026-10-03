#!/usr/bin/env bash
# Fine-tune a local SmolVLA checkpoint on a converted Franka Kitchen LeRobot dataset.
#
#   scripts/train_smolvla.sh MODEL_PATH DATASET_ROOT [OUTPUT_DIR]
#
# MODEL_PATH   local SmolVLA checkpoint directory or Hub id, e.g.
#              /home/you/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/<sha>
#              (or simply "lerobot/smolvla_base" when the Hub is reachable)
# DATASET_ROOT the directory written by kitchen-to-lerobot (must contain meta/info.json)
# OUTPUT_DIR   defaults to outputs/train/franka_kitchen_smolvla
#
# Everything below can be overridden through the environment, e.g.
#   STEPS=5000 BATCH_SIZE=2 scripts/train_smolvla.sh ...
#
# ---------------------------------------------------------------------------
# Why `--policy.input_features=null` is mandatory
# ---------------------------------------------------------------------------
# smolvla_base was pretrained on 6-DoF SO-100/SO-101 data with cameras named
# observation.images.camera1/2/3 and a 6-D state/action. Our dataset uses an
# 18-D robot state (qpos + qvel) and a 9-D velocity action. The base config ships a
# non-empty `input_features`. In LeRobot's make_policy(), `input_features` is
# only inferred from the dataset when it is *empty*:
#
#     if not cfg.input_features:
#         cfg.input_features = {k: v for k, v in features.items() if k not in cfg.output_features}
#
# so without this flag the model would keep expecting camera1/2/3 plus a 6-D
# state, and the first forward pass would fail with "All image features are
# missing from the batch" (and the state normalizer would see the wrong shape).
# Passing null makes LeRobot re-derive the schema from our dataset, which also
# means the camera key names in configs/cameras.yaml are free-form.
#
# LeRobot 0.6.1 has no gradient-accumulation flag; raise --batch_size instead, or
# launch with `accelerate launch` to multiply the effective batch across GPUs.
#
# `resize_imgs_with_padding` defaults to 512x512 in the base config. Our frames
# are 256x256, so 256 keeps the aspect ratio untouched and cuts the SigLIP token
# count by 4x, which keeps the fine-tune comfortably inside consumer-GPU memory.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 MODEL_PATH DATASET_ROOT [OUTPUT_DIR]" >&2
  exit 2
fi

model_path="$1"
dataset_root="${2%/}"
output_dir="${3:-outputs/train/franka_kitchen_smolvla}"

steps="${STEPS:-20000}"
batch_size="${BATCH_SIZE:-4}"
num_workers="${NUM_WORKERS:-4}"
resize="${RESIZE:-256}"
save_freq="${SAVE_FREQ:-2000}"
log_freq="${LOG_FREQ:-50}"
eval_steps="${EVAL_STEPS:-0}"
eval_split="${EVAL_SPLIT:-0.05}"
seed="${SEED:-1000}"
lr="${LR:-}"
freeze_vision="${FREEZE_VISION_ENCODER:-true}"
train_expert_only="${TRAIN_EXPERT_ONLY:-true}"
# LeRobot 0.6.1 maps this boolean to CUDA's default fp16 autocast even though
# SmolVLA loads BF16 weights. Its GradScaler then cannot unscale BF16 gradients.
# Keep Accelerate AMP off; the frozen SmolVLM backbone is still loaded as BF16.
use_amp="${USE_AMP:-false}"
video_backend="${VIDEO_BACKEND:-pyav}"
job_name="${JOB_NAME:-franka_kitchen_smolvla}"
device="${DEVICE:-cuda}"

if [[ ! -f "${dataset_root}/meta/info.json" ]]; then
  echo "error: ${dataset_root}/meta/info.json not found." >&2
  echo "       Run kitchen-to-lerobot first; see docs/说明.md section 4." >&2
  exit 1
fi

if [[ -e "${output_dir}" ]]; then
  echo "error: output already exists: ${output_dir}" >&2
  echo "       Choose a new path or resume explicitly; runs are never overwritten." >&2
  exit 1
fi

# Recover the repo_id that the dataset was written with, so the two never drift.
repo_id=""
if [[ -f "${dataset_root}/conversion_manifest.json" ]]; then
  repo_id="$(python3 -c "
import json,sys
try:
    print(json.load(open('${dataset_root}/conversion_manifest.json')).get('repo_id','') or '')
except Exception:
    print('')
")"
fi
if [[ -z "${repo_id}" ]]; then
  repo_id="local/$(basename "${dataset_root}")"
  echo "note: no conversion_manifest.json; assuming --dataset.repo_id=${repo_id}"
fi

mkdir -p "$(dirname "${output_dir}")"
run_log="${output_dir%/}.log"

args=(
  --policy.path="${model_path}"
  --policy.input_features=null
  --policy.device="${device}"
  --policy.use_amp="${use_amp}"
  --policy.push_to_hub=false
  --policy.freeze_vision_encoder="${freeze_vision}"
  --policy.train_expert_only="${train_expert_only}"
  "--policy.resize_imgs_with_padding=[${resize},${resize}]"
  --dataset.repo_id="${repo_id}"
  --dataset.root="${dataset_root}"
  --dataset.video_backend="${video_backend}"
  --output_dir="${output_dir}"
  --job_name="${job_name}"
  --steps="${steps}"
  --batch_size="${batch_size}"
  --num_workers="${num_workers}"
  --save_freq="${save_freq}"
  --save_checkpoint=true
  --log_freq="${log_freq}"
  --env_eval_freq=0
  --seed="${seed}"
  --wandb.enable=false
)

if [[ -n "${lr}" ]]; then
  args+=(--policy.optimizer_lr="${lr}")
fi

if [[ "${eval_steps}" != "0" ]]; then
  # Eval loss needs a held-out episode split; episodes are split, not frames,
  # so sequences never leak between the two halves.
  args+=(--dataset.eval_split="${eval_split}" --eval_steps="${eval_steps}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'lerobot-train \\\n'
  printf '  %s \\\n' "${args[@]}"
  exit 0
fi

echo "repo_id        : ${repo_id}"
echo "dataset        : ${dataset_root}"
echo "output         : ${output_dir}"
echo "log            : ${run_log}"
echo "steps x batch  : ${steps} x ${batch_size}"
echo "resize/unfreeze: ${resize} / freeze_vision=${freeze_vision} train_expert_only=${train_expert_only}"
echo

# Keep the run reproducible and offline-friendly when the checkpoint is already
# in the local HuggingFace cache. Comment out the next line if you want online
# tokenizer/dataset downloads.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

lerobot-train "${args[@]}" 2>&1 | tee "${run_log}"
