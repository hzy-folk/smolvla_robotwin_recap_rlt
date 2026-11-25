# RLT：先训练 z，再做三阶段 Actor-Critic 训练

这里的 **z / RL Token 训练是独立的前置过程**。它不属于后面 Actor-Critic 的三个阶段；`train_token.py`、`model.py` 及全部运行依赖均保留。

```text
SmolVLA 检查点 + 成功数据
→ 动作/状态分位数统计
→ 训练 RL Token 编解码器，得到 z 模型
→ Stage 1：冻结策略和 z 模型，采集经验，不更新 Actor-Critic
→ Stage 2：固定经验池，训练 Actor-Critic
→ Stage 3：加载 Stage 2 的 learner/Actor，在线执行和继续学习
→ Actor 评估与视频画廊
```

## 保留入口

| 用途 | 脚本 |
| --- | --- |
| 归一化统计 | `tools/make_stats.py` |
| **z 训练/续训** | **`train_token.py`**（`--resume`、`--epochs`、`--cache-prefixes` 均保留） |
| 三阶段调度 | `run_phased_training.py` |
| Stage 1 实现 | `run_online.py --phase collect`、`online.py` |
| Stage 2 实现 | `run_fixed_buffer.py` |
| Stage 3 实现 | `legacy_collect/run_actor_online.py` |
| 基准评估、可选经验池生成 | `legacy_collect/run_rollout_split.py` |
| Stage 2 检查点评估 | `evaluate_stage2_checkpoints.py` |
| 最终 Actor 评估 | `evaluate_actor_100.py` |
| CPU 链路自检 | `tools/smoke_test.py` |

`legacy_collect/scripts/` 仍是 Stage 3 和最终评估的实际依赖，不能因 legacy 名称删除。`serve.py` 同时实现本地特征提取和远程服务，`wire.py` 也是运行/自检依赖，均保留。

## 调用顺序

在项目根目录，使用准备好的解释器和数据。下面变量需替换成真实路径；三个阶段保持同一份配置，尤其是 n-step、归一化、z 维度和动作定义。

```bash
CONFIG=rlt/configs/robotwin_piper_nstep3_bc10warmup5000.yaml
SFT=/path/to/smolvla_checkpoint.pt
DATA=/path/to/robotwin_success_qpos16_formal100
TOKEN=/path/to/rlt_token.pt
RUN=/path/to/new_three_stage_run

# 前置：统计和 z 训练。统计输出路径应与 CONFIG 的 data.stats_path 对应。
python rlt/tools/make_stats.py --dataset "$DATA" --config "$CONFIG" --out /path/to/stats.json
python rlt/train_token.py --config "$CONFIG" --checkpoint "$SFT" \
  --dataset "$DATA" --out "$TOKEN" --steps 5000 --batch-size 8

# z 已训练好后，才进入以下三阶段。
python rlt/run_phased_training.py --stage 1 --config "$CONFIG" --run-dir "$RUN" \
  --checkpoint "$SFT" --token "$TOKEN" --seed-manifest "$DATA/splits/train.json"
python rlt/run_phased_training.py --stage 2 --config "$CONFIG" --run-dir "$RUN"
python rlt/run_phased_training.py --stage 3 --config "$CONFIG" --run-dir "$RUN" \
  --checkpoint "$SFT" --token "$TOKEN" --seed-manifest "$DATA/splits/train.json" --resume
```

`Stage 1` 保存 `replay_buffer.npz`；`Stage 2` 保存 `checkpoints/stage2_final.pt` 和 Actor 快照；`Stage 3` 接续这些状态。新实验使用新的 RUN。暂停后的恢复参数见各入口 `--help`，不能把加载 Actor 快照当作完整 learner/经验池续训。

当前正式评估器要求 100 集数据（80 train / 20 validation），并需要同一训练 split 的 80 集基础策略基准报告。`run_online.py` 的 `online_summary.json` 不能直接当成该报告，因此保留下面的基准入口：

```bash
python rlt/legacy_collect/run_rollout_split.py --checkpoint "$SFT" --data "$DATA" \
  --split train --output /path/to/base_train80
python rlt/evaluate_actor_100.py --checkpoint "$SFT" --token "$TOKEN" \
  --config "$CONFIG" --data "$DATA" --actor-snapshot "$RUN/actor_snapshot.pkl" \
  --baseline /path/to/base_train80/summary.json --output /path/to/new_actor_eval \
  --actor-deterministic
```

`evaluate_actor_100.py` 调用 `recap/generate_rlt_actor_100_gallery.py`，后者和共享画廊模块仍保留。模型和 SAPIEN 使用统一 `runtime.local.json` 中配置的独立解释器；安装步骤见 [DEPENDENCIES.md](../DEPENDENCIES.md)。

## 清理与检查

```bash
python rlt/tools/smoke_test.py
```

该自检使用 CPU/假环境验证 token、Actor-Critic、经验池、协议和主循环；不代表重新完成了真实仿真或训练实验。
