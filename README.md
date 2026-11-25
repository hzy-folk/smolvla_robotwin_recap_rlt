# SmolVLA：数据采集、微调、RECAP 与 RLT

本项目基于 SmolVLA，包含 RoboTwin 和 LIBERO / LIBERO-Pro 数据采集、策略微调，以及 RoboTwin 上的 RECAP 和 RLT 训练流程。

```text
LIBERO / LIBERO-Pro 采集 → 微调 → 闭环评估
RoboTwin 采集 → 审计与划分 → 微调
                            ├→ RECAP：价值学习 → Advantage 标注 → 条件策略微调 → 评估
                            └→ RLT：训练 z → 三阶段 Actor-Critic → 评估
```

## 1. 数据采集

- **RoboTwin**：采集堆叠积木的成功示范，再审计数据并划分训练集、验证集。使用三相机、25 Hz、16 维状态和 16 维绝对关节动作。
- **LIBERO / LIBERO-Pro**：采集沙拉酱入篮任务，按 seed 保存示范；Pro 通过 `--benchmark pro` 选择。使用双相机、20 Hz、8 维状态和 7 维 OSC 增量动作。

两个采集入口为 `collection/collect_robotwin_success_qpos16.py` 和 `collection/collect_libero_salad_dressing_delta.py`，配置位于 `collection/configs/`。两种任务的动作定义不同，数据不能混用。

## 2. SmolVLA 微调

使用对应任务的数据微调 SmolVLA，训练集用于计算归一化统计，验证集用于检查模型表现。RoboTwin 预测 16 帧、每次执行 8 帧；LIBERO 预测 16 帧、每次执行 5 帧。

- RoboTwin：`collection/train_smol_robotwin_qpos16.py`，训练后通过 `collection/infer_smol_robotwin_qpos16.py` 闭环评估。
- LIBERO：`collection/train_smol_libero_osc7.py`，训练后通过 `collection/eval_smol_libero_osc7.py` 闭环评估。支持按 seed 00～39 训练、40～49 验证。

采集、审计、划分和微调命令见 [采集与微调说明](collection/README.md)。

## 3. RECAP

RECAP 在 RoboTwin 上执行一次完整迭代：

1. 用成功示范训练初始策略，在训练场景中 rollout。
2. 合并原始成功/失败示范和 rollout，训练 PiStar 视觉价值模型。
3. 对全部数据预测价值，计算 N=8 的 Advantage，按全体帧的中位数划分 positive / negative 标签。
4. 将标签加入语言条件，微调最终策略，再到验证场景闭环评估。

统一入口为 `recap/run_pipeline.py`，支持分阶段执行和恢复。配置示例 `recap/configs/pipeline.example.json` 使用初始策略 20000 步、价值模型 40000 步、最终策略 24000 步，策略 batch 56、价值 batch 8；可按自己的数据和资源调整。rollout 来自训练场景，最终评估使用验证场景。

具体八阶段及输入约定见 [RECAP 说明](recap/README.md)。

## 4. RLT

RLT 先训练 **z / RL Token 编解码器**，再进行三阶段 Actor-Critic 训练；z 训练是独立的前置步骤。

1. 根据成功数据生成归一化统计，用 SmolVLA 特征训练 z 模型。
2. **Stage 1**：冻结基础策略和 z 模型，采集经验池，不更新 Actor-Critic。
3. **Stage 2**：使用固定经验池训练 Actor-Critic。
4. **Stage 3**：接续 Stage 2 的训练状态，在线采集并继续学习，最后评估 Actor。

z 训练入口为 `rlt/train_token.py`，三阶段入口为 `rlt/run_phased_training.py`，配置位于 `rlt/configs/`。各阶段需保持相同的动作定义、归一化统计和 z 配置。`rlt/legacy_collect/` 是第三阶段和评估的实际依赖。

完整顺序及评估命令见 [RLT 说明](rlt/README.md)。

## 使用文档

- [完整运行命令](RUNBOOK.md)
- [环境、权重与外部资源准备](DEPENDENCIES.md)
- [仓库文件说明](REPOSITORY_CONTENTS.md)

现有环境已完成小步实际流程验证，覆盖采集、微调、RECAP、z 训练和 RLT 三阶段；这不代表完整训练效果或任务成功率，尚未进行全新环境安装验证。
