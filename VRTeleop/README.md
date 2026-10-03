# FrankaKitchen VR Teleoperation

这是从原 XLeVR/ManiSkill 工程裁剪并适配到本项目的最小实现。保留的部分只有：

- 本地 A-Frame/WebXR 页面；
- HTTP 8000 和 WebSocket 8001 服务；
- 右手柄位姿、squeeze、trigger、A/B 按钮协议；
- FrankaKitchen `end_effector` Jacobian 阻尼最小二乘控制；
- 全局/腕部图像、动作、qpos/qvel 和原始 VR 输入采集。

原工程的 ManiSkill、Pink、双臂串口、HTTPS 私钥和任务截图已经移除。上游许可保留在 `LICENSE`。

## 控制映射

| 输入 | 行为 |
|---|---|
| 按住右手柄 squeeze | 建立离合参考并控制末端位姿 |
| 松开 squeeze | 停止机械臂并允许手柄重新摆位 |
| 按住 trigger | 闭合夹爪 |
| A | 保存当前 episode 并进入下一个 seed |
| B | 丢弃当前 episode，用相同 seed 重试 |
| Ctrl+C | 退出并丢弃尚未保存的 episode |

控制器输出为 9 维 FrankaKitchen 归一化速度 action。机械臂部分通过 MuJoCo `mj_jacSite` 计算 `end_effector` 的 6×7 Jacobian，再以阻尼最小二乘求关节速度；夹爪两个关节使用同向速度。

## 安装和自检

```bash
uv sync --extra teleop --extra dev
source .venv/bin/activate
export MUJOCO_GL=egl

python -m VRTeleop.controller --dry-run --tasks microwave kettle
```

成功时应输出 `VR_TELEOP_DRY_RUN_OK`、9 维 action、有效 Jacobian rank 以及 `global/wrist` 两个相机名。

## Quest USB 运行

```bash
adb devices
adb reverse tcp:8000 tcp:8000
adb reverse tcp:8001 tcp:8001

kitchen-vr-teleop \
  --tasks microwave kettle \
  --episodes 2 \
  --seed 5000 \
  --position-scale 0.3 \
  --rotation-scale 0.3 \
  --output-dir data/vr/pilot
```

在 Quest 浏览器访问 `http://localhost:8000`，点击 **Start controller tracking**。

## 第一次真机必须检查

1. 保持 `--position-scale 0.3`，分别沿手柄 X/Y/Z 小幅移动，确认机器人方向。
2. squeeze 第一次按下时机器人不应跳变。
3. 松开 squeeze、移动手柄、再次按下，机器人仍不应跳变。
4. trigger 按下应闭合两个手指，松开应张开。
5. 手柄停止发送超过 0.25 秒时，机械臂 action 应自动归零。
6. 检查保存的 NPZ 中 `action/qpos/qvel/observation/image_global/image_wrist` 长度一致。

当前 `VR_TO_KITCHEN` 坐标变换沿用了复制工程的 Panda 设置，只有完成上述真机检查后才能用于正式数据采集。

## 原始采集格式

每个保留 episode 生成：

```text
episode_000000.npz
episode_000000.json
```

NPZ 包含：

- `observation`：环境 59 维原始 observation；
- `qpos` / `qvel`：完整 MuJoCo 状态，用于精确回放和未来 privileged-state 训练；
- `action`：实际送入环境的 9 维 action；
- `image_global` / `image_wrist`：与 action 对齐的策略输入图像；
- `vr_position` / `vr_quaternion`：原始右手柄位姿；
- `vr_squeeze` / `vr_trigger`：控制状态。

JSON 包含任务序列、自然语言指令、seed、步数和任务首次完成事件。下一阶段会增加该格式到 LeRobot v3 的正式转换器，在转换前保留原始文件不变。
