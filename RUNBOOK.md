# 全流程运行手册

先完成 [安装与资产准备](DEPENDENCIES.md)，填写`runtime.local.json`。以下命令从仓库根目录执行，输出目录须使用新名称。提供的数据应符合各入口的manifest/NPZ契约。

## 采集与普通微调

```bash
.venvs/robotwin/bin/python collection/collect_robotwin_success_qpos16.py \
  --count 100 --first-seed 930000 --output data/robotwin_new
.venvs/robotwin/bin/python collection/audit_robotwin_success_dataset.py --dataset data/robotwin_new
.venvs/robotwin/bin/python collection/finalize_robotwin_success_dataset.py \
  --dataset data/robotwin_new --validation-count 20
.venvs/model/bin/python collection/train_smol_robotwin_qpos16.py \
  --data data/robotwin_new --output runs/robotwin_new
```

采集和普通微调配置在`collection/configs/`；`--config`指定其他JSON，显式CLI参数覆盖默认值。RoboTwin默认24000次更新、batch56；采集默认只取3条，因此正式采集必须明确count。

LIBERO-Pro每个seed独立输出；以下循环准备seed00～49，然后按40训练/10验证微调：

```bash
for seed in $(seq 0 49); do
  episode_dir=$(printf 'data/libero_new/seed_%02d' "$seed")
  .venvs/libero/bin/python collection/collect_libero_salad_dressing_delta.py \
    --benchmark pro --seed "$seed" --output "$episode_dir"
done
.venvs/model/bin/python collection/train_smol_libero_osc7.py \
  --data data/libero_new --output runs/libero_new --batch-size 64 --split-mode seed-00-39
.venvs/libero/bin/python collection/eval_smol_libero_osc7.py \
  --checkpoint runs/libero_new/checkpoint_020000.pt \
  --data data/libero_new --output runs/libero_eval_new --seed-start 40 --seed-end 49
```

LIBERO默认配置是standard、batch32和随机划分，上面的Pro参数是显式覆盖。两平台状态与动作不同，不能混用数据或检查点。

## RECAP一次完整迭代

成功示范SFT → 训练场景rollout → 合并成功/失败/rollout → PiStar价值训练 → 逐帧价值推理 → N=8 Advantage → 条件策略SFT → 验证场景闭环。

```bash
cp recap/configs/pipeline.example.json recap/configs/my_pipeline.json
# 编辑success_data、failure_data、scene_data、output，以及价值模型资产路径。
python3 recap/run_pipeline.py --config recap/configs/my_pipeline.json --plan
python3 recap/run_pipeline.py --config recap/configs/my_pipeline.json
# 中断后用原配置继续：
python3 recap/run_pipeline.py --config recap/configs/my_pipeline.json --resume
```

示例已经指向四环境布局中的三个所需解释器。RECAP初始SFT从官方基础模型开始，所有条件positive；最终SFT使用所有来源实际计算的positive/negative。`failure_data`为独立manifest；缺失该来源的链路检查可显式提供`{"episodes": []}`，但这不是完整失败示范实验。rollout和正式评估仍需真实场景配置。成功示范必须有train/validation两种split，场景数须覆盖配置的rollout_count/eval_count。

默认示例保留batch56、初始20000步、价值batch8/40000步、最终24000步。显存不足可降低micro-batch，保持需要的有效batch；这与原物理batch实验配置有区别。阶段名和独立入口见[RECAP说明](recap/README.md)。

## RLT：先z，再三阶段

```bash
CONFIG=rlt/configs/robotwin_piper_nstep3_bc10warmup5000.yaml
DATA=data/robotwin_new
SFT=runs/robotwin_new/checkpoint_024000.pt
TOKEN=rlt_runs/new/rlt_token.pt
RUN=rlt_runs/new/three_stage

# 编辑CONFIG的env.scene_manifest，使其指向新数据的train.json；
# data.stats_path与下面的out一致，YAML相对路径以YAML所在目录解析。
.venvs/model/bin/python rlt/tools/make_stats.py \
  --dataset "$DATA" --config "$CONFIG" --out rlt/configs/stats/new_stats.json
.venvs/model/bin/python rlt/train_token.py \
  --config "$CONFIG" --checkpoint "$SFT" --dataset "$DATA" \
  --steps 25000 --batch-size 300 --learning-rate 1e-4 --out "$TOKEN"
.venvs/model/bin/python rlt/run_phased_training.py --stage 1 \
  --config "$CONFIG" --run-dir "$RUN" --checkpoint "$SFT" --token "$TOKEN" \
  --seed-manifest "$DATA/splits/train.json"
.venvs/model/bin/python rlt/run_phased_training.py --stage 2 --config "$CONFIG" --run-dir "$RUN"
.venvs/model/bin/python rlt/run_phased_training.py --stage 3 \
  --config "$CONFIG" --run-dir "$RUN" --checkpoint "$SFT" --token "$TOKEN" \
  --seed-manifest "$DATA/splits/train.json" --resume
```

z单独训练，三阶段冻结基础策略及z。Stage1只采集，Stage2固定经验池更新，Stage3在线执行并继续学习。正式配置的n-step=3、warmup5000次更新、batch256和BC权重10→5均在同一YAML内。更换数据必须重新生成归一化统计，不要沿用仓库内历史任务统计。

```bash
.venvs/robotwin/bin/python rlt/legacy_collect/run_rollout_split.py \
  --checkpoint "$SFT" --data "$DATA" --split train --output rlt_runs/new/base_train80
.venvs/model/bin/python rlt/evaluate_actor_100.py \
  --checkpoint "$SFT" --token "$TOKEN" --config "$CONFIG" --data "$DATA" \
  --actor-snapshot "$RUN/actor_snapshot.pkl" \
  --baseline rlt_runs/new/base_train80/summary.json --output rlt_runs/new/actor_eval \
  --actor-deterministic
```

该正式评估器采用100集、80训练/20验证的数据契约，基准必须与当前基础策略和场景一致。阶段及评估产物说明见[RLT说明](rlt/README.md)。

## 本地检查

```bash
python3 tests/check_layout.py
python3 tests/check_portability.py
.venvs/model/bin/python tests/check_recap_flow.py --with-policy --policy-device cpu
.venvs/model/bin/python rlt/tools/smoke_test.py
```

前两项不需要模型、仿真或网络。小步验证不等于正式训练效果；本次按要求没有新建环境执行从零安装。
