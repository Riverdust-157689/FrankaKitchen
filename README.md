# FrankaKitchen × SmolVLA

本项目用于验证一条完整链路：

> 选中一个物体或有序物体序列 → 生成自然语言指令 → SmolVLA 闭环执行

当前基于 Farama `FrankaKitchen-v1` 原生 MuJoCo 环境，已经包含专家轨迹检查、双相机重建、Minari → LeRobot 转换、SmolVLA 微调、按序闭环评测和初版 Quest/WebXR 遥操作配接。

## 当前状态

- 相机：一个全局视角和一个挂在 `end_effector` site 上的腕部视角，均为 256×256。
- 策略状态：18 维机器人 `qpos + qvel`，不包含物体状态。
- 动作：9 维归一化关节速度（7 个机械臂关节、2 个夹爪关节）。
- 两任务数据：`microwave → kettle`，19 条轨迹。
- 四任务数据：`microwave → kettle → light switch → slide cabinet`，19 条轨迹。
- 四任务 16k checkpoint 的 100 次闭环结果：按序前缀为 `91/100 → 80/100 → 29/100 → 26/100`，当前主要瓶颈是第二到第三任务的切换。
- VR：已完成 WebXR → WebSocket → MuJoCo Jacobian DLS → 9D action 的初版适配；仍需真机完成坐标轴、增益和夹爪手感标定。

详细实验说明见 [docs/说明.md](docs/说明.md)，数据与 prompt 见 [docs/数据集与Prompt.md](docs/数据集与Prompt.md)，后续工作见 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 从零复刻环境

### 1. 系统依赖

推荐 Ubuntu、Python 3.12、NVIDIA 驱动和可用的 CUDA PyTorch 环境。先安装 `uv`、Git、FFmpeg；Quest USB 调试还需要 ADB。

```bash
sudo apt update
sudo apt install -y git ffmpeg libgl1 libegl1 android-tools-adb
curl -LsSf https://astral.sh/uv/install.sh | sh
```

克隆并进入仓库：

```bash
git clone <YOUR_REPOSITORY_URL> FrankaKitchen
cd FrankaKitchen
```

### 2. Python 环境

项目固定使用 Python 3.12、`gymnasium-robotics==1.4.2`、`mujoco==3.3.7` 和 `lerobot==0.6.1`。

使用锁文件复刻完整训练和 VR 环境：

```bash
uv sync --extra train --extra teleop --extra dev
source .venv/bin/activate
```

如果只运行环境和数据检查：

```bash
uv sync --extra dev
source .venv/bin/activate
```

公共环境变量：

```bash
export MUJOCO_GL=egl
export MINARI_DATASETS_PATH="$PWD/data/minari"
```

验证安装：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m VRTeleop.controller --dry-run --tasks microwave kettle
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 用于隔离系统 ROS 安装注册的 pytest 插件，避免它污染本项目测试环境。

### 3. 下载专家数据

```bash
minari download D4RL/kitchen/complete-v2
minari download D4RL/kitchen/partial-v2
```

如果 Hugging Face 因 `socks://127.0.0.1:...` 报代理 scheme 错误，可临时清除代理：

```bash
unset all_proxy ALL_PROXY
```

数据体检和专家回放：

```bash
kitchen-dataset-report \
  --dataset-id D4RL/kitchen/complete-v2 \
  --dataset-root "$PWD/data/minari" \
  --top 5

kitchen-validate \
  --dataset-id D4RL/kitchen/complete-v2 \
  --episodes 19 \
  --output outputs/reports/validate_complete.json

kitchen-replay \
  --dataset-id D4RL/kitchen/complete-v2 \
  --episode 0 \
  --mode actions
```

### 4. 相机检查

最终训练配置在 `configs/cameras.yaml`：

```bash
python scripts/preview_cameras.py \
  --camera-config configs/cameras.yaml
```

腕部相机使用 `site: end_effector`，不要改成 `body: panda0_link7`，两者坐标系相差平移和 45° 旋转。

### 5. 转换 LeRobot 数据

复刻当前“两任务一次完成”数据：

```bash
kitchen-to-lerobot \
  --dataset-id D4RL/kitchen/complete-v2 \
  --dataset-root "$PWD/data/minari" \
  --repo-id local/fk_first2 \
  --output-root "$PWD/data/lerobot/fk_first2" \
  --prefix-length 2
```

复刻固定四任务数据：

```bash
kitchen-to-lerobot \
  --dataset-id D4RL/kitchen/complete-v2 \
  --dataset-root "$PWD/data/minari" \
  --repo-id local/fk_full4 \
  --output-root "$PWD/data/lerobot/fk_full4" \
  --prefix-length 4
```

每个转换目录的 `conversion_manifest.json` 会记录相机配置 hash、状态布局、源 episode 数和 prompt，训练前应先检查它。

### 6. 准备 SmolVLA 权重并训练

可选的权重下载命令：

```bash
hf download lerobot/smolvla_base \
  --local-dir "$PWD/checkpoints/smolvla_base"
export SMOLVLA_BASE="$PWD/checkpoints/smolvla_base"
```

也可以直接指向 Hugging Face cache 中已有的 snapshot：

```bash
export SMOLVLA_BASE="$(find "$HOME/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
```

先执行单步训练体检：

```bash
python scripts/smoke_train_step.py \
  --dataset-root "$PWD/data/lerobot/fk_first2" \
  --checkpoint "$SMOLVLA_BASE" \
  --device cuda \
  --resize 256 \
  --batch-size 1
```

训练两任务模型：

```bash
STEPS=8000 BATCH_SIZE=4 SAVE_FREQ=2000 \
  scripts/train_smolvla.sh \
  "$SMOLVLA_BASE" \
  "$PWD/data/lerobot/fk_first2" \
  "$PWD/outputs/train/fk_first2_v1"
```

脚本默认关闭 Accelerate AMP，避免 BF16 权重和 FP16 GradScaler 组合触发 CUDA `unscale` 错误。

### 7. 闭环评测

```bash
export POLICY="$PWD/outputs/train/fk_first2_v1/checkpoints/008000/pretrained_model"

kitchen-eval \
  --policy-path "$POLICY" \
  --tasks microwave kettle \
  --episodes 100 \
  --seed 9000 \
  --max-steps 280 \
  --execution-horizon 10 \
  --device cuda \
  --terminate-on-success \
  --output-dir "$PWD/outputs/eval/fk_first2_seed9000"
```

任务名中有空格时必须引用：

```bash
--tasks microwave kettle "light switch" "slide cabinet"
```

汇总多个报告：

```bash
python scripts/summarize_evals.py 'outputs/eval/*/report_*.json'
```

## VR 遥操作 MVP

VR 代码位于 `VRTeleop/`，详细说明见 [VRTeleop/README.md](VRTeleop/README.md)。Quest USB 模式：

```bash
adb devices
adb reverse tcp:8000 tcp:8000
adb reverse tcp:8001 tcp:8001

kitchen-vr-teleop \
  --tasks microwave kettle \
  --episodes 20 \
  --seed 5000 \
  --position-scale 0.3 \
  --rotation-scale 0.3 \
  --output-dir "$PWD/data/vr/microwave_kettle"
```

在 Quest 浏览器打开 `http://localhost:8000`。先用低 scale 检查三个平移轴和旋转方向，再开始正式采集。

## 仓库结构

```text
configs/                    相机配置
docs/                       流程、数据说明和路线图
scripts/                    相机、训练、评测汇总工具
src/franka_kitchen_vla/     数据、运行时和评测核心代码
tests/                      不依赖 GPU 的回归测试
VRTeleop/                   WebXR 与 FrankaKitchen VR 适配
data/                       本地数据，Git 忽略
outputs/                    训练和评测产物，Git 忽略
```

## 常用命令

| 命令 | 作用 |
|---|---|
| `kitchen-dataset-report` | 查看任务词表、任务顺序和 prompt |
| `kitchen-validate` | 核验标签并回放专家动作 |
| `kitchen-replay` | 单轨迹动作/状态回放 |
| `kitchen-to-lerobot` | 转换为 LeRobot v3 |
| `kitchen-eval` | SmolVLA 闭环按序评测 |
| `kitchen-instruction-probe` | 比较同状态不同指令的动作差异 |
| `kitchen-vr-teleop` | Quest/WebXR 遥操作和原始轨迹采集 |
