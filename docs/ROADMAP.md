# 后续规划

本文只记录实施顺序、接口和验收标准，不在当前阶段展开算法调研。

## 总体原则

- 近期继续使用原生 `FrankaKitchen-v1`，不迁移仿真器。
- 策略部署输入保持为双相机、机器人 qpos/qvel 和语言指令。
- 物体状态、任务阶段和完成标志属于 privileged state，只能进入数据分析、reward、teacher 或 critic，不能隐式进入最终部署策略。
- 所有方法使用同一组闭环 seeds、相机配置和按序评测器比较。

## 阶段 1：VR 遥操作与自定义数据

### 已完成的初版

- 裁剪 copied `VRTeleop/`，移除 ManiSkill/Pink/双臂串口依赖。
- WebXR 右手柄通过 WebSocket 发送绝对位姿和按键。
- squeeze 实现无跳变离合，trigger 控制夹爪。
- MuJoCo site Jacobian + 阻尼最小二乘生成 7 维机械臂速度。
- 保存双相机、action、完整 qpos/qvel、原始 observation 和 VR 输入。

### 下一步

1. Quest 真机校准坐标轴、position/rotation scale、DLS 增益和夹爪速度。
2. 增加实时仿真预览和目标/实际末端位姿诊断。
3. 检查保存轨迹的动作回放是否能精确复现。
4. 实现 VR NPZ → LeRobot v3 转换器，并记录转换 manifest。
5. 先采集单任务，再采集任务组合；失败样本默认丢弃，但保留明确标注的恢复/接管片段。
6. 在当前 SmolVLA rollout 上加入 VR takeover，形成 DAgger 数据聚合循环。

### 验收标准

- 离合按下和重新按下没有明显跳变。
- 控制丢包后 action 自动归零。
- 保存轨迹的图像、状态和动作严格等长。
- 从保存状态和 action 回放能复现任务完成事件。
- 至少完成一组单任务和一组双任务的稳定人工示教。

## 阶段 2：验证多任务学习能力

### 问题拆分

需要分别回答：

1. 模型是否掌握每个单任务技能；
2. 是否能执行训练中见过的任务组合；
3. 是否能在任务相同但顺序变化时服从顺序；
4. 是否能组合训练中分别见过、但从未共同出现的任务；
5. 是否能从较短序列泛化到较长序列；
6. 是否真正响应语言，而不是只复现固定轨迹。

### 数据划分

- `single`：单任务技能。
- `seen-composition`：训练中见过的组合与顺序。
- `held-out-order`：任务集合见过，但完整顺序未见。
- `held-out-combination`：各任务分别见过，但组合未见。
- `length-generalization`：训练 1～3 个任务，测试 4 个任务。
- `paraphrase`：序列相同，使用未见语言表达。

先围绕 microwave、kettle、light switch、slide cabinet 四项建立完整矩阵，再决定是否扩展到其余三项任务。

### 评测指标

- 严格有序序列成功率。
- ordered-prefix 阶梯。
- 单任务完成率和相邻阶段转移成功率。
- 完成未请求任务、做错顺序和超时的次数。
- 每阶段耗时与首次失败阶段。
- 相同初始状态、不同指令下的动作差异。

评测器观察所有任务状态用于计分，但这些信号不能进入 SmolVLA policy input。

## 阶段 3：RL 后训练

RL 按从低风险到高成本的顺序推进，首先保持 SmolVLA 主体冻结。

### 3.1 Asymmetric Actor-Critic

- Actor：只看部署时可用的图像、机器人 qpos/qvel 和指令。
- Critic：训练时额外看物体 qpos/qvel、接触、当前序列阶段、目标距离和完成标志。
- Reward：当前子目标进度、按序完成奖励、整序列奖励、错误任务惩罚和动作平滑项。
- 首先训练小 residual actor，而不是直接在线更新完整 SmolVLA。

### 3.2 Teacher → Student

- 使用完整仿真状态训练 state-based teacher。
- 利用中间状态 reset 和课程学习集中训练困难阶段。
- teacher 成功 rollout 转成带双相机的示教数据。
- student 使用图像、机器人状态和语言进行行为克隆或蒸馏。
- 最终闭环评测只运行 student，不向其提供 privileged state。

### 3.3 视觉 encoder 预测 privileged state

- 在视觉 encoder 上增加辅助预测头，预测与操作相关的物体状态、任务完成概率或当前阶段。
- 预测目标只作为训练监督，不把真实 privileged state 拼进 policy observation。
- 比较三种设置：无辅助头、共享 encoder 多任务头、冻结 encoder 后独立 probe。
- 检查辅助任务是否提升闭环成功率，而不只看 privileged-state 预测误差。

### 3.4 后训练顺序

1. 原始专家数据 + VR 专家数据重新 BC。
2. 加入 VR takeover 的 DAgger 数据。
3. 训练 state teacher 并生成更丰富的成功轨迹。
4. Teacher → Student 蒸馏。
5. Asymmetric critic + 小型 residual actor 在线优化。
6. 视觉 privileged-state 辅助目标联合训练。
7. 只有上述方法达到平台期后，才考虑解冻更多 SmolVLA 参数。

## 阶段门槛

进入下一阶段前必须固定：数据版本、相机 hash、训练配置、评测 seed、reset noise、execution horizon 和 checkpoint 选择规则。所有改进以闭环成功率及其置信区间为准，不以训练 loss 单独决定。
