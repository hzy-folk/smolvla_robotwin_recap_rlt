# LIBERO SmolVLA 闭环评测

脚本：`eval_smol_libero_osc7.py`。默认使用当前数据集中 `seed_00` 到 `seed_39` 的40个训练场景。

```bash
cd /home/hzy/git/learn/project2/smolvla
/home/hzy/git/learn/feasibility_audit/libero_venv/bin/python \
  collection/eval_smol_libero_osc7.py \
  --checkpoint /你的训练输出目录/checkpoint_020000.pt \
  --output runs/libero_eval_train40_step20000 \
  --device cuda
```

使用现有 `libero_venv` 的 NumPy 1.26 / MuJoCo 2.3.7，并引用现有 SmolVLA 依赖，无须安装新环境。CPU 推理可改 `--device cpu`。输出目录必须为空。

## 实际执行过程

1. 从每条 NPZ 读取 `initial_sim_state`，恢复采集时已完成静置的场景，包括沙拉酱的 x/y 扰动。再核对机器人、沙拉酱和篮子的初始状态。不再次随机采样，不重放专家动作。
2. 当前全景、腕部图像各旋转180度，组成 `(1,256,256,3) uint8`；原始 state 为 `(1,8) float32`；指令为 `pick up the salad dressing`。
3. 调用实际 `SmolVLABackend`，复用部署 profile、图像通道转换、像素缩放及 checkpoint 中的 state/action 统计量。模型10步去噪，预测16步；policy 反归一化后，由 bridge 裁剪到 `[-1,1]`，取前5步，返回 `(1,5,7)`。
4. 按顺序将5个 OSC 动作发给20Hz仿真器，再获取当前画面继续预测。每条最多400步，即20秒仿真时间，可用 `--max-steps` 调整。计算耗时不改变仿真控制频率。
5. 官方成功条件首次成立后，允许模型再执行20步，观察松爪与物体稳定情况；全部动作仍来自模型。每个 episode 都重置 policy 缓存及随机数种子。

视频采用人眼正立的显示朝向；模型输入仍采用 RPent 的180度约定。视频包含初始帧和每个动作执行后的帧，左右分别为全景、腕部。

## 结果

- `summary.json`：逐条结果和成功率。运行错误单独统计，并在成功率分母中作为失败计入。
- `seed_XX/result.json`：场景恢复误差、步数、推理耗时和成功标记。
- `seed_XX/rollout.mp4`：可供浏览器播放的 H.264 双视角视频。
- `seed_XX/rollout.npz`：动作前 state、实际执行的 actions，以及动作后的物体位置、夹爪接触和官方成功状态。不是重新采集的训练数据（不包含全部相机帧）。
- `index.html`：40条视频与结果汇总，运行中逐条更新。

抓取成功：robosuite 判断左右指垫接触沙拉酱，且物体比初始位置升高至少3cm，连续保持3步。任务成功：LIBERO 官方 `check_success()` 曾成立。另记“松爪入篮”：官方成功且夹爪不再抓住物体，连续5步。这三个标记分别报告。

这是训练场景上的 **SmolVLA + bridge + LIBERO** 闭环测试，不启动 GPT 规划器或 SAM3，也不代表 RPent 完整规划流程的成功率。它从整条训练 episode 的起始姿态开始，未加入 RPent 的接近动作或接管逻辑。

## 查看网页

```bash
/home/hzy/git/learn/feasibility_audit/libero_venv/bin/python -m http.server \
  8766 --bind 127.0.0.1 --directory runs/libero_eval_train40_step20000
```

浏览器打开 `http://127.0.0.1:8766/`。远程 IDE 需转发8766端口。

## 没有 checkpoint 时检查场景

```bash
/home/hzy/git/learn/feasibility_audit/libero_venv/bin/python \
  collection/eval_smol_libero_osc7.py \
  --check-scenes --output runs/libero_eval_train40_scene_check
```

该模式仅恢复和核对40个场景，不产生模型成功率。先测试一条可增加 `--seed-start 0 --seed-end 0`；评测后10条验证场景可改为 `--seed-start 40 --seed-end 49`。
