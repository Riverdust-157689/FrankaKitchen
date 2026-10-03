# 数据集与 Prompt 说明

本文回答两个问题：**已下载的数据集里到底是什么数据**、**训给 SmolVLA 的 prompt 是怎么来的**。
所有数字都是在当前工作区实际扫描 HDF5 得到的，可用
`kitchen-dataset-report --json-out outputs/reports/dataset_report.json` 复现。

---

## 1. 三个数据集的定位

| 数据集 | episode | 动作步 | 任务序列数 | 场景起始状态 | 用途 |
|---|---|---|---|---|---|
| `D4RL/kitchen/complete-v2` | 19 | 4,209 | **1** | **全部完全相同** | 冒烟测试、reset 起始的单序列验证 |
| `D4RL/kitchen/partial-v2` | 621 | 156,560 | **35** | 各不相同 | 主训练集（序列指令多样性） |
| `D4RL/kitchen/mixed-v2` | 621 | 156,560 | **35** | 各不相同 | **不要用于训练**，见 §6 |

动作空间是 9 维归一化关节速度 `Box(-1,1)`，环境频率 12 Hz。
`complete-v2` 每条轨迹 208–236 步（均值 221.5），`partial-v2` 为 1–450 步（均值 252.1）。

`complete-v2` 只有一个固定顺序：

```
microwave -> kettle -> light switch -> slide cabinet       19/19 条
```

`partial-v2` 的 35 种顺序（按出现次数降序；`tok` 是该顺序拼成自然语言后的 token 数）：

| # | eps | tok | 任务顺序 |
|---|---|---|---|
| 1 | 68 | 25 | microwave -> bottom burner -> light switch -> slide cabinet |
| 2 | 29 | 26 | bottom burner -> top burner -> slide cabinet -> hinge cabinet |
| 3 | 29 | 26 | microwave -> bottom burner -> top burner -> hinge cabinet |
| 4 | 29 | 29 | microwave -> kettle -> bottom burner -> slide cabinet |
| 5 | 28 | 25 | microwave -> bottom burner -> slide cabinet -> hinge cabinet |
| 6 | 27 | 26 | bottom burner -> top burner -> light switch -> slide cabinet |
| 7 | 26 | 30 | kettle -> bottom burner -> light switch -> slide cabinet |
| 8 | 25 | 30 | kettle -> bottom burner -> slide cabinet -> hinge cabinet |
| 9 | 25 | 30 | kettle -> bottom burner -> top burner -> slide cabinet |
| 10 | 24 | 29 | microwave -> kettle -> slide cabinet -> hinge cabinet |
| 11 | 22 | 31 | kettle -> bottom burner -> light switch -> hinge cabinet |
| 12 | 22 | 30 | microwave -> kettle -> top burner -> light switch |
| 13 | 21 | 31 | kettle -> bottom burner -> top burner -> light switch |
| 14 | 21 | 25 | microwave -> bottom burner -> top burner -> slide cabinet |
| 15 | 21 | 25 | microwave -> light switch -> slide cabinet -> hinge cabinet |
| 16 | 21 | 26 | microwave -> top burner -> light switch -> hinge cabinet |
| 17 | 20 | 26 | microwave -> bottom burner -> top burner -> light switch |
| 18 | 20 | 30 | microwave -> kettle -> light switch -> hinge cabinet |
| 19 | 20 | 30 | microwave -> kettle -> top burner -> hinge cabinet |
| 20 | 19 | 31 | kettle -> bottom burner -> top burner -> hinge cabinet |
| 21 | 19 | 30 | kettle -> light switch -> slide cabinet -> hinge cabinet |
| 22 | 19 | 30 | kettle -> top burner -> light switch -> slide cabinet |
| 23 | 19 | 30 | microwave -> kettle -> bottom burner -> hinge cabinet |
| 24 | 19 | 29 | microwave -> kettle -> light switch -> slide cabinet |
| 25 | 12 | 26 | bottom burner -> hinge cabinet -> microwave -> top burner | ← 退化，见 §5 |
| 26 | 5 | 19 | bottom burner -> microwave -> top burner | ← 退化 |
| 27 | 2 | 23 | kettle -> bottom burner -> slide cabinet |
| 28 | 2 | 19 | microwave -> bottom burner -> top burner |
| 29 | 1 | 26 | bottom burner -> microwave -> top burner -> hinge cabinet | ← 退化 |
| 30–35 | 各 1 | 18–23 | （6 条零散顺序） |

**关键结论：这 35 个顺序不是检测噪声，而是脚本化演示真实执行的不同顺序。**
同一顺序内 19–68 条轨迹的完成帧非常一致（例如 #1 的完成帧均值±std = 56±3 / 129±8 / 180±20 / 227±21），
说明每条轨迹都是「按某个任务清单逐项执行」的脚本演示。这正是序列指令微调需要的监督信号。

---

## 2. 任务词表（7 个可操作任务）

`qpos idx` 是 MuJoCo **30 维 qpos**（9 机器人 + 21 物体）的下标——这就是
`KitchenEnv.compute_reward` 用的下标，**不是 59 维 observation 的下标**。
`goal` 是环境判定"完成"的目标值（阈值 0.3）。

| 任务 | qpos idx | goal | 自然语言短语 | 该任务被完成的 episode 数 |
|---|---|---|---|---|
| bottom burner | [11,12] | [-0.88, -0.01] | turn on the bottom burner | 435 |
| top burner | [15,16] | [-0.92, -0.01] | turn on the top burner | 294 |
| light switch | [17,18] | [-0.69, -0.05] | turn on the light switch | 327 |
| slide cabinet | [19] | [0.37] | open the sliding cabinet | 385 |
| hinge cabinet | [20,21] | [0.0, 1.45] | open the hinged cabinet | 310 |
| microwave | [22] | [-0.75] | open the microwave door | 385 |
| kettle | [23..29] | [-0.23, 0.75, 1.62, 0.99, 0, 0, -0.06] | move the kettle to the top-left burner | 333 |

（统计基于 `partial-v2`，每个任务出现于约一半到三分之二的 episode。）

**为什么有 7 个任务但 `partial-v2` 只有 4 个进了 goal 字典？**
环境用 `tasks_to_complete` 决定 `achieved_goal`/`desired_goal` 里放哪些任务，
`partial-v2` 的 spec 是 `[microwave, kettle, light switch, slide cabinet]`，
所以另外 3 个任务（bottom burner / top burner / hinge cabinet）在数据里**没有 goal 字段**，
只能从 59 维 observation 的几何位置还原。见 §4 的一致性验证。

---

## 3. Prompt 是怎么产生的

### 3.1 生成方式：事后标注（hindsight labeling）

演示数据里**没有人说过话**。prompt 是转换时由"这条轨迹实际完成了什么、按什么顺序"
反推出来的：

1. 对每条 episode 逐帧计算每个任务是否达到 goal（阈值 0.3），取**首次达标帧**；
2. 按首次达标帧排序，得到该 episode 的有序任务列表，例如 `[microwave, kettle, light switch, slide cabinet]`；
3. 用固定模板把这个列表渲染成一句英文。

### 3.2 模板

```python
TASK_PHRASES = {
    "bottom burner": "turn on the bottom burner",
    "top burner":    "turn on the top burner",
    "light switch":  "turn on the light switch",
    "slide cabinet": "open the sliding cabinet",
    "hinge cabinet": "open the hinged cabinet",
    "microwave":     "open the microwave door",
    "kettle":        "move the kettle to the top-left burner",
}
```

* 单任务：`短语.capitalize() + "."` → `Turn on the light switch.`
* 多任务：`", then ".join(短语).capitalize() + "."` → 见下例
* `--prompt-style compact` 时改为对象名逗号列表（贴近"选中物体"接口）：
  `microwave, kettle, light switch, slide cabinet`

真实样例（`--emit-prefixes` 会为同一条源轨迹生成 4 条带不同 prompt 的 episode）：

```
Open the microwave door.
Open the microwave door, then move the kettle to the top-left burner.
Open the microwave door, then move the kettle to the top-left burner, then turn on the light switch.
Open the microwave door, then move the kettle to the top-left burner, then turn on the light switch, then open the sliding cabinet.
```

### 3.3 Token 预算（重要但没问题）

smolvla_base 的 `tokenizer_max_length = 48`，超长会被**截断**，后半段指令模型根本看不到。
实测 35 个四任务 prompt 的 token 数为 **18–31**，最长 31 < 48，**全部安全**，无需改动。

### 3.4 Prompt 在 LeRobot 数据集里怎么落盘

* 每帧写入一个 `task` 字段（字符串）；
* LeRobot 自动去重，写入 `meta/tasks.jsonl`，episode 元数据里记录 `task` 索引；
* 训练时预处理链 `rename_observations → add_batch_dim → smolvla_new_line → tokenizer → device → normalize`
  会补一个换行、按 `tokenizer_max_length` padding 到 48、再把 token 送进 VLM；
* **推理时必须用与训练完全相同的措辞**。`kitchen-eval --tasks microwave kettle` 会调用同一个
  `instruction_for()` 生成 prompt，保证一致；`--prompt-style` 要和转换时一致，转换清单里也记了这个字段。

> 注意：同一条 episode 的**所有帧共用同一句 prompt**（整段序列指令），而不是逐任务推进的。
> 也就是说"当前该做第几个任务"要靠模型自己从图像判断，语言只给整段目标序列。

---

## 4. 标签正确性验证

转换时写进 `task` 的标签必须等于环境真正的完成判定，否则模型学的是错的目标。
用两种独立方式交叉验证：

**(a) 与数据自带的 goal 字典对比。** `achieved_goal`/`desired_goal` 就是
`KitchenEnv.compute_reward` 比较的那两个数，用它算出的完成事件是"环境权威答案"。
几何检测（在带噪 observation 上取阈值）与环境权威答案的对比：

| 数据集 | 一致 | 一致率 |
|---|---|---|
| complete-v2 | 19 / 19 | **100%** |
| partial-v2 | 621 / 621 | **100%** |
| mixed-v2 | 621 / 621 | **100%** |

早期把 59 维 observation 下标当成 qpos 下标时会得到 620/621（唯一差异是 kettle 完成帧差 1 帧），
`tests/test_tasks.py::test_completion_indices_are_qpos_indices_not_observation_indices` 固定了这个坑。

**(b) 在真实环境里回放专家动作。** 用 `kitchen-validate --replay-goal-set all`：

```
exact task-order reproduction: 6/8 (75.0%)
per-task recall: bottom burner 83.3% | top burner 100% | light switch 100% |
                 slide cabinet 80.0% | hinge cabinet 66.7% | microwave 100% | kettle 100%
```

回放必须在**该 episode 自己的初始状态**下开始（`--replay-start episode-init`，新默认值），
因为这些 episode 并不都从 `env.reset()` 的状态开始；直接从 `env.reset()` 回放是另一件
（更苛刻的）事，见 §5。

---

## 5. 数据质量：三个必须知道的坑

### 5.1 有 18 条 1 步退化 episode（会被默认过滤）

`partial-v2` / `mixed-v2` 里各有 18 条只有 **1 个动作步**的 episode。看
`episode_311`：`qpos[0][11,12] = [-0.889, -0.005]` 已经正好是 bottom burner 的 goal，
`qpos[0][15,16] ≈ [-0.897, -0.005]` 是 top burner 的 goal，微波炉也已经开到判定阈值内，
且 `truncations[-1] = True`。

它们是**一条被切分的演示剩下的尾巴**：状态已经是"活都干完了"，却会被标注成
`Turn on the bottom burner, then open the hinged cabinet, then open the microwave door,
then turn on the top burner.`（完成帧全是 0）。这类样本只教模型"什么都不用做"，是纯噪声。

`kitchen-to-lerobot` 默认已过滤，两个开关：

* `--min-episode-steps 30`（默认）：丢掉过短的 episode；
* `--min-first-completion-frame 1`（默认）：丢掉"起始状态就已经完成任务"的 episode。

传 `--min-episode-steps 1 --min-first-completion-frame 0` 可以关掉过滤来复现问题。

### 5.2 `partial-v2` 的 episode 不都从 reset 开始

`complete-v2` 的 19 条轨迹起始机器人关节**完全相同**（逐维 std = 0），是规范 reset 状态。
但 `partial-v2` 的起始机器人关节逐维 std 最大到 **0.318**（joint7）。

含义：

* 用 `--emit-prefixes` 生成的"前缀"是**该段轨迹内部**的前缀，不是"从 reset 开始做前 k 个任务"。
  对行为克隆仍然完全有效（指令在起始状态下是真的），而且顺带覆盖了大量中间起始状态——
  这对"选中物体→执行单任务"的上游接口反而是好事；
* 但如果要做"从 reset 出发、按顺序做前 k 个任务"的干净评测，用 `complete-v2` 更合适，
  或者评测时固定 seed 后由 `kitchen-eval` 从同一 reset 状态出发。

### 5.3 `mixed-v2` 与 `partial-v2` 的观测和动作逐帧完全相同

实测 **621/621 条 episode 的 `observation` 与 `action` 逐字节相等**。
两者唯一区别是 env spec 的 `tasks_to_complete`（因此 `achieved_goal` 键、奖励定义不同）：

* `partial-v2`: `[microwave, kettle, light switch, slide cabinet]`
* `mixed-v2`:   `[microwave, kettle, bottom burner, light switch]`

所以**绝不能把两者同时放进训练集**，否则同一批轨迹被重复加权。训练只用 `partial-v2`。

---

## 6. 喂给模型的观测/动作契约

转换出的 LeRobot 数据集字段：

| 字段 | 内容 | 形状 |
|---|---|---|
| `observation.images.global` | 全局视角（默认 MuJoCo free camera） | 256×256×3 |
| `observation.images.wrist` | 末端随动视角（挂在 `panda0_link7` 的 `end_effector` site 上） | 256×256×3 |
| `observation.state` | **仅机器人**：9 维 qpos + 9 维 qvel，即 `obs[:18]` | (18,) |
| `action` | 9 维归一化关节速度 | (9,) |
| `task` | 上面那句英文指令 | 字符串 |

* **物体状态没有喂给模型**（59 维里物体那部分只用于离线重建画面），模型必须从图像里认目标；
* 相机键名会成为 `observation.images.<键名>` 并写进 checkpoint，**训练开始后不能改名**；
* smolvla_base 预训练用的是 `camera1/2/3` + 6 维 state/action，微调时必须用
  `--policy.input_features=null` 让 LeRobot 从数据集重新推断特征（详见 `docs/说明.md` §5）。
