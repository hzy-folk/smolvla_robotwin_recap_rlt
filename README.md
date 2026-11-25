# SmolVLA

本项目包含两个任务的数据采集、SmolVLA 微调、RoboTwin RECAP 一次迭代，以及 z / RL Token 训练和三阶段 Actor-Critic 流程。

代码和配置按流程放在三个目录：

```text
collection/       采集、审计、普通微调、共享模型模块
  configs/        RoboTwin / LIBERO 的采集和微调配置
recap/            RECAP 八阶段及价值模型
  configs/        RECAP 流水线配置
rlt/              z 训练及三阶段 Actor-Critic
  configs/        RLT 配置和归一化统计
```

采集及普通微调的使用方法见 [collection/README.md](collection/README.md)。

```text
LIBERO-Pro 采集 → OSC7 微调 → 闭环评估
RoboTwin 采集 → 审计、划分 → qpos16 微调
                              ├→ RECAP 八阶段
                              └→ z 训练 → RLT 三阶段 → Actor 评估
```

| 用途 | 入口 |
| --- | --- |
| RoboTwin 成功示范采集 | `collection/collect_robotwin_success_qpos16.py` |
| 数据审计、划分 | `collection/audit_robotwin_success_dataset.py`、`collection/finalize_robotwin_success_dataset.py` |
| RoboTwin 普通微调、推理 | `collection/train_smol_robotwin_qpos16.py`、`collection/infer_smol_robotwin_qpos16.py` |
| LIBERO / LIBERO-Pro 采集 | `collection/collect_libero_salad_dressing_delta.py`，Pro 使用 `--benchmark pro` |
| LIBERO 普通微调、评估 | `collection/train_smol_libero_osc7.py`、`collection/eval_smol_libero_osc7.py` |
| RECAP | `recap/run_pipeline.py` |
| z、三阶段 RLT | `rlt/train_token.py`、`rlt/run_phased_training.py` |

RoboTwin 使用三相机、25 Hz、16维状态和绝对动作、预测16帧执行8帧。LIBERO 使用双相机、20 Hz、8维状态和7维 OSC 增量动作、预测16帧执行5帧。两种动作数据不能混用。


## RECAP

恢复并整理后的八阶段流程、配置方法和小步验证见 [recap/README.md](recap/README.md)。统一入口为 `recap/run_pipeline.py`，配置示例为 `recap/configs/pipeline.example.json`。

## RLT

保留 z / RL Token 训练及其后的三阶段 Actor-Critic 训练，入口和顺序见 [rlt/README.md](rlt/README.md)。脚本清理记录见 [SCRIPT_AUDIT.md](recap/SCRIPT_AUDIT.md)。

## 验证

2026-11-25 已用真实仿真和模型完成小步全流程验证，含两个采集器、普通微调、RECAP 真实价值训练/完整验证/推理/最终策略评估，以及 z 和 RLT 三阶段。详见 [验证记录](recap/README.md#2026-11-25-真实流程验证)。这是执行链验证，不代表完整训练效果或任务成功率。

结果摘要见 [全流程验证](tests/full_flow_validation.json)。安装与外部资产见 [DEPENDENCIES.md](DEPENDENCIES.md)，完整命令顺序见 [RUNBOOK.md](RUNBOOK.md)，文件管理约定见 [REPOSITORY_CONTENTS.md](REPOSITORY_CONTENTS.md)。
