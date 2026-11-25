# 运行环境与外部资产

当前已在本机现有环境完成真实小步全流程验证，尚未验证另一台机器从零安装。代码中部分本机路径仍需按部署环境调整。

| 环节 | 环境与资产 |
| --- | --- |
| SmolVLA 微调、推理 | 兼容 LeRobot 源码、PyTorch、Transformers、官方 SmolVLA 和 SmolVLM 权重 |
| RoboTwin 采集、闭环 | RoboTwin、SAPIEN、NumPy、OpenCV及机器人/场景资产 |
| LIBERO 采集、闭环 | LIBERO、LIBERO-Pro任务文件、MuJoCo、robosuite及官方初始状态 |
| LIBERO 策略评估 | 相邻目录 `../smolvla_rpent_bridge/` 的 backend 和 profile |
| RECAP 价值模型 | PiStar checkout、JAX/Flax/Optax、SigLIP/Gemma权重和Gemma tokenizer |
| RLT | 模型与仿真环境、PyYAML、websockets，以及训练得到的z和归一化统计 |

本机兼容 LeRobot、基础权重和仿真环境主要位于 `/home/hzy/git/learn/feasibility_audit/`。部分脚本直接引用此路径；RLT支持 `RLT_AUDIT_DIR` 和 `RLT_EXTRA_SITE`，但这不代表所有路径都已参数化。

RECAP 的已验证依赖版本、三种解释器的用途和预训练资产配置见 [RECAP说明](recap/README.md#环境)。流水线配置示例在 `recap/configs/pipeline.example.json`。已有完整 PiStar 检查点时可以通过 `value_init_checkpoint` 初始化，tokenizer仍必需。

RoboTwin普通微调使用 `collection/robotwin_smol_model.py`，LIBERO使用 `collection/smol_model.py`。两者动作维度、控制频率和图像处理不同，不可交换检查点。

第三方源码、模型、数据和论文保留各自授权；本仓库未另行替第三方授予许可证。
