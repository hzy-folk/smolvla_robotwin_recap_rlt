# RECAP / RLT 脚本审查（2026-11-24）

2026-11-25追加整理：旧配对实验的两个入口、两份早期RECAP方案、一份LIBERO启动过程记录、两份早期检查快照和被替代的调试日志移至 `回收站/20261125_github_cleanup/`，逐文件原因及SHA256见其 `manifest.json`。当前数据、模型、经验池、最终评估及成功验证日志保留原路径；所有现行运行代码和配置通过哈希核对。2026-11-25真实小步验证结果见 [README.md](README.md)，下文的“未重跑”仅描述上一次审查。

保留范围：RECAP 八阶段完整迭代；RLT 前置归一化统计和 **z / RL Token 训练**；z 参与的 Actor-Critic 三阶段训练；基准评估、Stage 2 检查点评估及最终 Actor 评估。训练算法和配置没有修改，全部 YAML 配置保留。

## 保留的关键依赖

- z：`rlt/train_token.py`、`rlt/model.py`、`rlt/tools/make_stats.py`，以及 config/bootstrap/模型和归一化依赖。
- 三阶段调度：`rlt/run_phased_training.py`；Stage 1 用 `run_online.py`，Stage 2 用 `run_fixed_buffer.py`，Stage 3 用 `legacy_collect/run_actor_online.py`。
- `rlt/legacy_collect/scripts/` 整条推理/worker/数据/模型/场景依赖链继续保留，不能按 legacy 名称移除。
- `rlt/legacy_collect/run_rollout_split.py` 保留，用于生成正式评估器要求的基础策略 80 集基准报告（也支持带 z 的经验导出）。
- `rlt/evaluate_actor_100.py`、`evaluate_stage2_checkpoints.py` 保留；`scripts/recap/generate_rlt_actor_100_gallery.py` 和共享画廊模块继续保留。
- `serve.py` 里也有本地特征提取类，不只是服务器入口；它和 `wire.py` 是运行/自检依赖，继续保留。
- RECAP 的 `recap_value_validation.py`、`recap_training_control.py` 是运行时模块，移出的只是一次性检查入口。

## 移入回收站

没有直接删除任何脚本、数据或结果。文件及对应字节码缓存均保留在本次回收站目录，逐文件原路径和 SHA256 见其中 manifest.json。

回收站：`回收站/20261124_225729_recap_rlt_flow_cleanup`。

| 脚本 | 移动原因 |
| --- | --- |
| `scripts/recap/check_training_pause_resume.py` | 一次性 PiStar 暂停恢复回归脚本；正式训练使用 recap_training_control.py，保留其运行依赖。 |
| `scripts/recap/check_value_validation.py` | 一次性 PiStar 数值回归脚本；正式验证使用 recap_value_validation.py。 |
| `scripts/recap/finish_training_rollout_pipeline.py` | 普通 SFT 单次实验的等待/rollout 调度器；RECAP 八阶段已由 run_pipeline.py 串联。 |
| `scripts/recap/generate_custom_rollout_gallery.py` | 上述单次调度器的专用画廊；正式 RECAP 使用 generate_current_rollout_gallery.py。 |
| `scripts/recap/generate_rlt_rollout_gallery.py` | 旧 RLT 在线训练流水线的画廊；三阶段评估使用 generate_rlt_actor_100_gallery.py。 |
| `scripts/recap/run_monitored_value_training.py` | 独立价值阶段调度器；正式 RECAP 已由 run_pipeline.py 调用训练、推理和标注。 |
| `scripts/recap/run_rollout_split.py` | 普通 SmolVLA split 评估副本；RECAP 用 run_rollout50，RLT 基准用 legacy_collect/run_rollout_split。 |
| `rlt/run_full_pipeline.py` | 固定 16525 步检查点的单次实验续训器；通用三阶段使用 run_phased_training.py。 |
| `rlt/run_training_pipeline.py` | 旧的在线训练到评估流水线；没有固定经验池 Stage 2。 |
| `rlt/run_bc_schedule_comparison.py` | BC warmup 对比实验调度器，不属于选定的三阶段主线。 |
| `rlt/watch_training_report_policy.py` | 单次训练监控和报告工具，不参与模型/经验池的生成或执行。 |
| `rlt/eval_rlt.py` | 早期在线 Actor 评估入口；三阶段评估保留 evaluate_actor_100.py。 |
| `rlt/eval_rollout50.py` | 旧在线流水线的评估入口；三阶段评估保留 evaluate_actor_100.py。 |
| `rlt/tools/watch_nstep_eval_sequence.py` | n-step 对比实验的监控调度器。 |
| `rlt/tools/schedule_token_training.py` | 按既有训练产物启动 token 训练的一次性调度器；z 正式入口 train_token.py 保留。 |
| `rlt/tools/rebuild_nstep_buffer.py` | 跨 n-step 实验的经验池重建工具；正式 Stage 1 按所选配置直接生成经验池。 |

## 验证

静态审查解析清理后 50 个活动 Python 文件，核对本地 RLT import 和脚本路径常量，没有发现仍引用本次移出文件的活动代码。RECAP 临时数据端到端检查通过。

RLT 自检首次在协议步骤发现解释器缺少 websockets；借用的本机版本要求 Python 3.11，不能用于当前 3.10 测试解释器；最终在 /tmp 隔离目录安装 websockets==12.0 后重测，没有安装或修改现有环境。

这些检查不启动真实仿真或整轮训练。RLT 的三阶段实现、checkpoint 和数据契约没有改动。


最终检查结果：

- 16 个脚本移动后 SHA256 与移动前一致，未永久删除。
- RECAP 临时数据检查通过（6 条 episode、24 帧），实际数据准备、标注和 batch 构造正常。
- `train_token.py --synthetic --dim 32 --steps 2 --batch-size 2 --device cpu` 通过，并保存约 49 MB 的临时 z 检查点。
- RLT CPU/假环境完整自检输出“全部自检通过”，包括 token 重建、Actor-Critic 更新、经验池、特征协议、在线主循环和暂停恢复。
- z 训练、三阶段调度、固定经验池训练、Stage 3 在线入口、Actor 评估五个 CLI 的 `--help` 均通过。
- 本次未重新跑真实 RoboTwin 或完整训练；不能把合成 z 输入/假环境自检当作任务成功率实验。
