# SmolVLA

本项目包含两个任务的数据采集、SmolVLA 微调、RoboTwin RECAP 一次迭代，以及 z / RL Token 训练和三阶段 Actor-Critic 流程。

```text
LIBERO-Pro 采集 → OSC7 微调 → 闭环评估
RoboTwin 采集 → 审计、划分 → qpos16 微调
                              ├→ RECAP 八阶段
                              └→ z 训练 → RLT 三阶段 → Actor 评估
```

| 用途 | 入口 |
| --- | --- |
| RoboTwin 成功示范采集 | `scripts/collect_robotwin_success_qpos16.py` |
| 数据审计、划分 | `scripts/audit_robotwin_success_dataset.py`、`scripts/finalize_robotwin_success_dataset.py` |
| RoboTwin 普通微调、推理 | `scripts/train_smol_robotwin_qpos16.py`、`scripts/infer_smol_robotwin_qpos16.py` |
| LIBERO / LIBERO-Pro 采集 | `scripts/collect_libero_salad_dressing_delta.py`，Pro 使用 `--benchmark pro` |
| LIBERO 普通微调、评估 | `scripts/train_smol_libero_osc7.py`、`scripts/eval_smol_libero_osc7.py` |
| RECAP | `scripts/recap/run_pipeline.py` |
| z、三阶段 RLT | `rlt/train_token.py`、`rlt/run_phased_training.py` |

RoboTwin 使用三相机、25 Hz、16维状态和绝对动作、预测16帧执行8帧。LIBERO 使用双相机、20 Hz、8维状态和7维 OSC 增量动作、预测16帧执行5帧。两种动作数据不能混用。


## RECAP

恢复并整理后的八阶段流程、配置方法和小步验证见 [scripts/recap/README.md](scripts/recap/README.md)。统一入口为 `scripts/recap/run_pipeline.py`，配置示例为 `configs/recap/pipeline.example.json`。

## RLT

保留 z / RL Token 训练及其后的三阶段 Actor-Critic 训练，入口和顺序见 [rlt/README.md](rlt/README.md)。脚本清理记录见 [SCRIPT_AUDIT.md](scripts/recap/SCRIPT_AUDIT.md)。

## 验证

2026-11-25 已用真实仿真和模型完成小步全流程验证，含两个采集器、普通微调、RECAP 真实价值训练/完整验证/推理/最终策略评估，以及 z 和 RLT 三阶段。详见 [验证记录](scripts/recap/README.md#2026-11-25-真实流程验证)。这是执行链验证，不代表完整训练效果或任务成功率。

结果摘要见 [全流程验证](tests/full_flow_validation.json)。环境配置见 [DEPENDENCIES.md](DEPENDENCIES.md)，文件管理约定见 [REPOSITORY_CONTENTS.md](REPOSITORY_CONTENTS.md)。
