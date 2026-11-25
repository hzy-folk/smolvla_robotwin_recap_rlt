# SmolVLA RECAP：一次完整迭代

正式流程：成功示范 SFT → train 场景 rollout → 合并成功/失败数据 → PiStar 价值训练 → 全数据价值推理 → N=8 Advantage 标注 → 条件策略 SFT → validation 场景评估。

代码恢复来源见 [RECOVERY.md](RECOVERY.md)。当前代码已解除旧实验目录、80/160 条 episode 和固定 50 条 rollout 的限制；仍使用 16 维状态/绝对动作、三相机、预测 16 帧/执行 8 帧，以及全体 Advantage 的 top 50% 条件。早期方案及多次覆盖的实验过程记录已归档，不作为当前启动说明。

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

## 环境

模型解释器需要项目当前兼容的 LeRobot、PyTorch、Transformers；仿真解释器需要 RoboTwin/SAPIEN、NumPy、OpenCV；价值解释器需要 PiStar 的 JAX/Flax/Optax 环境，以及 SigLIP、Gemma 和 Gemma tokenizer。

已有完整 PiStar 价值检查点时，可在流水线配置设置 `value_init_checkpoint`，替代 `siglip`/`gemma` 两项；此方式载入已有权重并重新初始化优化器、从第 0 步开始。它与 `--resume` 的继续训练语义不同，tokenizer 仍必需。检查点加载明确映射到当前 JAX 设备，支持把旧 GPU 权重用于 CPU 验证。

本机 2026-11-25 的 PiStar 验证使用 Python 3.11、JAX/JAXlib 0.5.3、Flax 0.10.2、Optax 0.2.4、Orbax 0.11.13、Kauldron 1.3.0、Grain 0.2.11、NumPy 1.26.4、Jaxtyping 0.2.36、Beartype 0.19.0、Transformers 4.53.2、ML Collections 1.0.0。PiStar 导入训练入口还需要旧 LeRobot commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`。CPU 环境使用 TensorFlow CPU 2.19.1，配套 protobuf 5.29.5、tensorflow-metadata 1.16.1、sentencepiece 0.2.0、PyArrow 19.0.1 和 datasets 3.6.0，避免混装最新版本产生导入错误。这是本机验证组合，尚不是独立的安装包。

当前模型封装仍使用本机 `feasibility_audit` 中的 LeRobot 源码和基础权重；这是本地流程恢复，不代表已经完成面向任意机器的打包。不要直接用最新版 LeRobot 替换本地兼容版本。测试环境的路径注入方式为：

```bash
export PYTHONPATH=/home/hzy/git/learn/feasibility_audit/venv/lib/python3.10/site-packages:/home/hzy/git/learn/feasibility_audit/lerobot_stable_src:$PWD/collection:$PWD/recap
```

## 验证

```bash
# 使用具备 numpy/opencv/torch 的解释器：临时数据走真实数据准备和标注入口。
python tests/check_recap_flow.py

# 增加真实 SmolVLA 更新、保存、暂停恢复和最终条件训练；默认 CPU，避免占用仿真 GPU。
python tests/check_recap_flow.py --with-policy --policy-device cpu
```

测试覆盖原始成功/失败、rollout 成功/失败、JPEG/PNG 缓存、正负语言、动作时间对齐、episode 边界、训练集统计，以及保留历史标签的重算。临时价值预测是测试数据，不冒充 PiStar 前向；小步测试也不代表任务成功率。

价值模型完整前向/训练和真实 RoboTwin rollout 需要对应环境和资源。下面记录本机小步实测，不是重跑整轮正式实验。早期占位、废弃 critic 和一次性脚本已移至 `回收站/20261124_recap_cleanup/`，正式流程不依赖它们。RLT 使用的画廊脚本保留。

### 2026-11-25 真实流程验证

在本机现有环境中逐阶段执行，并用流水线的 `verify_stage` 校验全部八阶段实际产物，已全部通过。详细记录位于项目根目录的 `validation_runs/20261124_full_flow/full_flow_validation.json`，同目录保留配置、日志、权重和闭环视频；该目录被 Git 忽略。

| 环节 | 实测结果 |
| --- | --- |
| RoboTwin 采集、审计、划分 | 新采集成功 seed 940000，344 帧；与另一个保留的真实 episode 组成 train/validation 各 1 条的数据集，审计通过 |
| LIBERO-Pro 采集、微调、评估 | seed 0 成功示范 151 帧；真实更新并保存权重，再经实际 bridge 执行 5 步闭环，无运行错误 |
| 普通 RoboTwin 微调 | 真实 SmolVLA 前向、反向和优化器更新通过 |
| RECAP 初始 SFT、rollout | 2 次更新保存检查点；加载该检查点执行 16 步真实闭环 |
| PiStar 价值准备、训练 | 成功示范 2 条 + 真实失败 rollout 1 条，共 670 帧；从保留的 40000 步完整检查点初始化，真实更新 1 步并保存 |
| PiStar 验证、推理 | 默认完整验证覆盖 309 帧；新 CPU 训练检查点在 GPU 推理全部 670 帧，结果均有限 |
| Advantage、最终条件 SFT | 用实际 PiStar 预测生成正负各 335 个标签；更新 1 步、保存权重，完整验证 309 帧通过 |
| 最终 RECAP 评估 | 加载新条件策略，在 validation 场景执行 16 步真实闭环，无运行错误 |
| RLT | 真实特征训练 z 1 步；三阶段真实采集/更新完成；加载 Actor 执行 16 步评估；第三阶段配置步数修复后复测通过 |

原始失败示范数据缺失，按要求跳过该来源，空 manifest 明确标记缺失原因，没有伪造失败示范。失败 rollout 来自本次实际执行。独立的 SigLIP/Gemma 预训练文件未提供，本次使用完整历史价值检查点初始化，优化器和步数重新开始。JAX CPU 训练、GPU 推理均已实测；GPU 分支另测了真实更新和默认完整验证。

短训练和短闭环只证明流程可执行，不能用于判断任务成功率。RLT 评估借用了历史 baseline 的场景列表，历史教师与本次微调教师不同，因此不能把它作为公平的性能对比。当前验证依赖本机 PiStar、LeRobot、仿真环境、基础模型和 bridge，尚未验证另一台机器从零安装。


## 本次脚本审查

当前只保留 RECAP 八阶段主线、可视化和 RLT 共享依赖。独立价值监控调度器、旧普通 SFT 调度器、重复评估/画廊和一次性 JAX 检查脚本已移入回收站。数值验证运行模块 `recap_value_validation.py`、暂停恢复运行模块 `recap_training_control.py` 仍保留。详见 [SCRIPT_AUDIT.md](SCRIPT_AUDIT.md)。
