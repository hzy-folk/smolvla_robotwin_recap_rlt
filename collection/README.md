# 数据采集与普通微调

本目录包含两个任务的采集、数据审计、普通微调和评估入口。`robotwin_smol_*`、`smol_model.py`、`dataset.py` 等模块也供 RECAP 和 RLT 使用。

| 任务 | 配置 | 采集 | 微调 |
| --- | --- | --- | --- |
| RoboTwin | `configs/robotwin.json` | `collect_robotwin_success_qpos16.py` | `train_smol_robotwin_qpos16.py` |
| LIBERO / LIBERO-Pro | `configs/libero.json` | `collect_libero_salad_dressing_delta.py` | `train_smol_libero_osc7.py` |

每份配置中的 `collection` 和 `finetune` 分别对应采集与普通微调参数。脚本默认读取本任务配置；`--config` 可以指定另一份JSON；显式命令行参数优先。配置中的路径相对项目根目录，命令行中的相对路径沿用当前工作目录。数据与输出目录仍须按入口要求指定。

配置保留原脚本默认值：RoboTwin默认采集3条；LIBERO默认standard、普通微调batch32和随机划分。此前LIBERO-Pro正式实验通过参数覆盖为Pro、batch64及seed 00～39训练/40～49验证，不将这些历史覆盖悄悄改成所有任务的默认值。

从项目根目录运行，Python解释器选择见 [环境说明](../DEPENDENCIES.md)：

```bash
python collection/collect_robotwin_success_qpos16.py \
  --count 100 --first-seed 930000 --output data/robotwin_new
python collection/audit_robotwin_success_dataset.py --dataset data/robotwin_new
python collection/finalize_robotwin_success_dataset.py \
  --dataset data/robotwin_new --validation-count 20
python collection/train_smol_robotwin_qpos16.py \
  --data data/robotwin_new --output runs/robotwin_new

# 每个seed单独采集；批量采集时为每个seed使用独立输出目录。
python collection/collect_libero_salad_dressing_delta.py \
  --benchmark pro --seed 0 --output data/libero_new/seed_00
python collection/train_smol_libero_osc7.py \
  --data data/libero_new --output runs/libero_new \
  --batch-size 64 --split-mode seed-00-39
```

LIBERO按上述seed划分微调前，需准备完整的训练及验证episode。RoboTwin审计和划分先于普通微调；数据契约要求25Hz、三相机、16维绝对动作。LIBERO使用20Hz、双相机、8维状态和7维OSC增量动作。

推理入口为 `infer_smol_robotwin_qpos16.py` 和 `eval_smol_libero_osc7.py`。重组没有移动已有 `data/`、`runs/`、`recap_runs/` 或 `rlt_runs/`；旧日志中的 `scripts/...` 是历史入口路径，运行新命令时换为 `collection/...`，原 `scripts/recap/...` 换为 `recap/...`。
