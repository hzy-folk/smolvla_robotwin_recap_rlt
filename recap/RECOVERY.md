# RECAP 代码恢复记录（2026-11-24）

本文是恢复过程的历史记录，其中“未重跑”等结论对应2026-11-24当时的检查。2026-11-25已补做真实小步全流程验证，当前验证范围和入口以 [README.md](README.md) 为准。

目录重组后，当前入口为顶层 `recap/`，其配置在 `recap/configs/`，共享模型模块在 `collection/`。本文历史来源路径与原始哈希保留；`recovery_manifest.json` 的 `current_path` 已更新到当前位置。

## 来源和范围

从 2026-11-07 开始的原实验本地 Codex 会话重建：完整文件写入、按时间排序的补丁，以及创建最终训练入口时读取的原训练器快照。不执行历史 shell 命令、下载、进程控制或训练命令，只提取代码并在临时目录重放补丁。

恢复 32 个缺失历史文件；6 个已有文件保持现状；另将历史初始 SFT 训练循环隔离为 `initial_sft_trainer.py`。原 `scripts/train_smol_robotwin_qpos16.py` 已经变化，因此初始 SFT wrapper 改为调用这个隔离副本，保留暂停/恢复等原功能。对应源码哈希路径作了必要调整。

价值训练器补齐 JPEG 数据路径所需的 `cv2` 导入，去掉对 JPEG 数据不存在的 `images.npy` 的提前加载，并检查损坏 JPEG。这些兼容修补与历史逐字恢复有区别。

`recovery_manifest.json` 记录恢复/保留文件、重建与安装哈希及兼容修改。

## 与真实实验对照

读取保留的 `recap_runs/final_policy_batch56_21000/checkpoint_024000.pt`（CPU、mmap，不载入 GPU）中的 `config.source_sha256`，确认刚恢复时以下两个文件与训练时逐字一致（随后为可配置流程做了接口修改，当前文件不再宣称逐字相同）：

- `train_recap_policy.py`：`ab4563ec716c9b4c0bc721b467c182b77c47f08ab4dd6b85e005377a292a4f02`
- `final_policy_data.py`：`fc690761d6377a6ab9d0fb0be667570e2bafd90bdfefc726820bcd1d98656ab4`

用保留的 `predicted_values.json` 重算 160 条 episode、56,730 帧 Advantage，与历史 `advantages.json` 对比：最大数值误差约 `1.39e-17`，阈值误差约 `3.47e-18`，positive/negative 标签差异为 0。

40 个 Python 文件通过 AST 语法检查；历史 shell 脚本通过 `bash -n`。临时成功/失败样例通过完整数据流 dry run；最终训练采样器保存恢复后连续 10 个 batch 与未中断序列一致。

## 尚未恢复的实验输入

本次恢复的是代码，不伪造缺失的原始实验数据。当前仍缺：

- `data/robotwin_success_80_inward3mm_v3/`
- `data/robotwin_failure_30_gpu_final_inward3mm/`
- `recap_runs/rollout50_checkpoint020000_adv_positive/`
- `recap_runs/value_round1_pistar/data/`
- `assets/pistar_vlm_ckpt/`

部分历史运行的 `launch.json`、优化器续训检查点和虚拟环境也未随本次代码恢复重建。不能用其他任务数据或重采数据冒充原实验输入。

未重跑 GPU 训练、PiStar 前向或真实仿真；静态/数值检查不等于当前机器能够重新完成所有实验。完整再现需要先准备数据、预训练资产、第三方源码和对应 Python 环境。


## 正式流程整理

用户确认无需恢复原始数据后，增加 `run_pipeline.py` 和配置示例，串联八阶段。最终策略从价值 manifest 读取数据来源；移除固定 episode 数量和旧运行目录依赖；初始/最终适配器同时接受明确声明的 inward3mm 数据和按记录动作训练的数据。原始数据没有声明修正规则时不追加修正。

15 个占位、旧 critic、一次性检查和旧实验专用脚本移至 `回收站/20261124_recap_cleanup/`，未永久删除。RLT 相关画廊仍在原位置。

CPU 集成测试使用临时成功/失败数据和合成 rollout，调用真实数据准备、标注和 batch 构造。真实 SmolVLA 额外完成初始 SFT 第 1 步保存暂停、恢复到第 2 步，以及最终条件 SFT 两步更新、检查点保存和验证。首次 GPU 测试由于另一个任务的桥接服务和 SAM3 并发占用显存而 OOM，未停止它们，改用 CPU 验证通过。

PiStar 侧已核对本地源码的 tokenizer、create/load/save train state、C51 损失和推理接口；当前测试环境未装 JAX/Flax/Optax，未重跑价值 VLM 前向/训练或真实 RoboTwin rollout。这是验证边界，不把合成价值预测写成真实 PiStar 实测。
