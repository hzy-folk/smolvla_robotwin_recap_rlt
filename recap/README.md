# SmolVLA RECAP：一次完整迭代

正式流程：成功示范 SFT → train 场景 rollout → 合并成功/失败数据 → PiStar 价值训练 → 全数据价值推理 → N=8 Advantage 标注 → 条件策略 SFT → validation 场景评估。

当前代码已解除旧实验目录、80/160 条 episode 和固定 50 条 rollout 的限制；仍使用 16 维状态/绝对动作、三相机、预测 16 帧/执行 8 帧，以及全体 Advantage 的 top 50% 条件。早期方案及多次覆盖的实验过程记录已归档，不作为当前启动说明。

当前奖励设置为 H=375，失败终止惩罚93.75，过程奖励-1、成功终止奖励0，统一除以468.75；完整回报裁剪到[-1,0]。N=8 Advantage 为后续最多8帧奖励之和加 V(t+8)-V(t)，终止后价值为0。所有来源统一计算中位数阈值，达到阈值的帧标 positive。

保留正式记录：初始 SFT 使用60条成功训练、20条验证，batch56、20000步；PiStar 使用80原始成功、30原始失败、41成功rollout、9失败rollout，共56730帧，30条原始episode留作验证，batch8、40000步、冻结backbone、关闭EMA；最终策略使用46322训练帧、10408验证帧，batch56、24000步，学习率1e-4、warmup1000、原21000步衰减周期不变。最终策略目录仍叫 `final_policy_batch56_21000`，实际检查点为 `checkpoint_024000.pt`。这些历史记录不表示当前机器仍保有全部原始输入。

## 统一启动

从项目根目录运行。复制 `recap/configs/pipeline.example.json` 为自己的配置，填写三个解释器、数据目录、PiStar checkout 和预训练资产路径。配置中的相对路径均相对项目根目录。

```bash
# 只生成各阶段命令，不加载模型、不创建运行目录。
python3 recap/run_pipeline.py --config recap/configs/my_pipeline.json --plan

# 真正执行；首次使用新的 output 目录。
python3 recap/run_pipeline.py --config recap/configs/my_pipeline.json

# 继续同一个配置：跳过完成阶段，训练阶段从本阶段检查点恢复。
python3 recap/run_pipeline.py --config recap/configs/my_pipeline.json --resume
```

可用 `--from-stage` / `--through-stage` 限定阶段，名称以 `--plan` 输出为准。输出目录保存每阶段日志和 `pipeline_status.json`；流水线同时校验退出码和关键产物，不把“提前暂停”判为训练完成。修改配置后应使用新输出目录。

| 阶段 | 入口 |
| --- | --- |
| 初始 SFT | `train_recap_initial_sft.py` / `initial_sft_trainer.py` / `initial_sft_data.py` |
| 策略 rollout、最终评估 | `run_rollout50.py`（数量可配置） |
| 价值数据准备 | `prepare_recap_value_dataset.py` |
| 价值训练、推理 | `train_recap_value_vision.py`、`predict_recap_values.py` |
| Advantage 标注 | `label_recap_advantage.py` |
| 最终策略训练 | `train_recap_policy.py` / `final_policy_data.py` |

`prepare_recap_value_dataset.py` 会在 manifest 中记录原始数据和 rollout 来源；最终训练通过 `--value-data` 读取这份 manifest，不要求修改 Python 常量。对旧 manifest，适配器保留历史路径回退。

## 输入约定

- 成功/失败原始数据分别提供 `manifest.json` 和 NPZ；episode 标记 `train` 或 `validation`，两种 split 均应存在。成功数据可由项目的 RoboTwin 采集、审计和整理入口准备。
- 原始 NPZ 包含 float32 `state/action`（N×16）、三路 uint8 RGB 图像 `images/left_wrist/right_wrist`（N×256×256×3），以及 `task`、`joint_names`、`joint_units`、`sample_hz`、`timestamp` 和标量 `success`。如提供逐帧 `official_success`，额外校验其与最终结果一致；新采集脚本只有标量结果时也可直接使用。
- `derivation.timing_rule` 声明 inward3mm 契约时校验修正动作；未声明时按 NPZ 记录的绝对动作训练，不重复施加夹爪修正。
- rollout 使用真实的 `applied_targets`，满足 `len(applied_targets) == len(state)-1`。适配器转换未来 +1 索引，避免错一帧；episode 尾部重复最后动作。
- 价值数据验证集沿用原始 split；rollout 仅来自训练场景。最终评估只从 validation 场景抽取，场景数须够用。
- 最终策略输入逐样本追加 `Advantage: positive/negative`，原始成功数据也使用实际计算标签。训练集独立计算归一化统计。

安装、依赖版本、权重和统一路径配置见 [环境文档](../DEPENDENCIES.md)。

## 验证

```bash
# 使用具备 numpy/opencv/torch 的解释器：临时数据走真实数据准备和标注入口。
python tests/check_recap_flow.py

# 增加真实 SmolVLA 更新、保存、暂停恢复和最终条件训练；默认 CPU，避免占用仿真 GPU。
python tests/check_recap_flow.py --with-policy --policy-device cpu
```

测试覆盖原始成功/失败、rollout 成功/失败、JPEG/PNG 缓存、正负语言、动作时间对齐、episode 边界、训练集统计，以及保留历史标签的重算。临时价值预测是测试数据，不冒充 PiStar 前向；小步测试也不代表任务成功率。

价值模型完整前向/训练和真实 RoboTwin rollout 需要对应环境和资源。现有环境已完成小步流程验证，详细运行记录保留在本机，不随源码上传；尚未进行全新环境安装验证。
