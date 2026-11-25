# 安装、依赖和外部资产

本仓库不提交数据、模型权重或虚拟环境。它提供固定依赖、第三方源码版本与补丁、安装入口和统一运行路径。下面步骤由使用者在自己的机器执行；本次没有执行从零安装验证。

## 1. 系统与安装入口

使用Linux x86_64。模型、RoboTwin和LIBERO使用Python 3.10，PiStar价值模型单独使用Python 3.11，避免两个LeRobot版本冲突。准备Git、编译工具、CMake、FFmpeg、OpenGL/EGL/Vulkan运行库；GPU执行还需匹配PyTorch CUDA后端的NVIDIA驱动。无显示器运行使用EGL；Vulkan ICD由系统或`VK_ICD_FILENAMES`指定。

```bash
python3 -m pip install uv
# 默认只打印计划，不安装或下载：
python3 setup_env.py --profile model

# 构建环境；源代码按 sources.lock.json 固定revision并应用 patches/：
python3 setup_env.py --profile model --install
python3 setup_env.py --profile robotwin --fetch-sources --install
python3 setup_env.py --profile libero --fetch-sources --install
python3 setup_env.py --profile value --fetch-sources --install
```

默认Torch后端为cu128；可传`--torch-backend cpu`，但这不替代真实仿真所需的图形驱动。已有源码目录时安装器只接受锁定revision，不重置用户修改；补丁已应用时跳过。安装器不下载数据或权重。

## 2. 官方SmolVLA代码在哪里

`requirements/model.txt`固定`lerobot[smolvla]==0.3.3`、Torch2.7.1、Transformers4.51.3。安装时会获取官方LeRobot代码，其中包含`lerobot/policies/smolvla/`；本项目的模型封装从这里导入，不依赖作者电脑上的LeRobot目录。可查看实际文件：

```bash
.venvs/model/bin/python -c 'import lerobot.policies.smolvla.modeling_smolvla as m; print(m.__file__)'
```

PiStar需要另一版LeRobot，固定commit为`0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`，只装在value环境。`requirements/value.txt`记录已使用的包版本组合；安装器按完整包列表使用`--no-deps`，其中`tensorflow-cpu`提供Kauldron需要的TensorFlow模块，避免再次混入另一个TensorFlow发行包。PiStar客户端从同一固定checkout安装。

RoboTwin固定main revision，并附本实验所需的MPlib、相机与干净场景补丁。PiStar附VALUE类型、backbone重计算和可配置预训练路径补丁。原始第三方授权仍适用，出处和revision见`sources.lock.json`，补丁文件清单及SHA256见`patches/manifest.json`。

## 3. 获取权重和仿真资产

使用者自行下载；不要只复制一个safetensors文件，模型目录还需配置、tokenizer和processor文件。

```bash
.venvs/model/bin/hf download lerobot/smolvla_base \
  --revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 --local-dir assets/smolvla
.venvs/model/bin/hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --revision 4cde9d208e19519c36917aa170fffc7e381aa0ad --local-dir assets/smolvlm
```

基础SmolVLA权重SHA256应为`7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`。不要用当前Hub的浮动main代替此revision来加载已有任务检查点；任务检查点会核对基础权重哈希。

| 资产 | 获取与放置 |
| --- | --- |
| RoboTwin机器人 | [官方RoboTwin2.0资产](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)的embodiments.zip，解压到`third_party/RoboTwin/assets/embodiments/`，至少包含piper和配置中引用的文件 |
| LIBERO | 固定LIBERO源码对应的assets、bddl_files和init_files；保持官方目录结构 |
| LIBERO-Pro任务 | 来自[官方LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO)的task perturbation资源。将`bddl_files/`、`init_files/`放在配置的`libero_pro_root`下；本任务所需两个文件及哈希见`assets.required.json` |
| PiStar预训练 | [PiStar说明](https://github.com/ybpy/pistar)与[ybpy/vlm_ckpt](https://huggingface.co/ybpy/vlm_ckpt)：SigLIP2 So400m/14 224的NPZ、Gemma3-270M Orbax目录、tokenizer.model，默认放在`assets/pistar_vlm_ckpt/` |

PiStar必须使用其加载器支持的JAX/Orbax格式，不能直接用任意HF PyTorch Gemma目录替代。已有完整PiStar价值模型检查点时，可在RECAP配置设置`value_init_checkpoint`替代SigLIP/Gemma资产，tokenizer仍必需；这是重置优化器的初始化，不是断点续训。

## 4. 统一路径配置

```bash
cp runtime.example.json runtime.local.json
python3 check_runtime.py --profile model --verify-hashes
python3 check_runtime.py --profile robotwin
python3 check_runtime.py --profile libero --verify-hashes
python3 check_runtime.py --profile value
```

`runtime.local.json`被Git忽略。目录可放在任意磁盘，配置的相对路径统一相对仓库根目录；可用`SMOLVLA_RUNTIME_CONFIG=/绝对路径/runtime.json`选择配置，也可用`SMOLVLA_MODEL_DIR`、`SMOLVLA_ROBOTWIN_ROOT`等同名大写环境变量覆盖单项。`lerobot_src`留空表示使用环境中安装的LeRobot；只在使用自有源码checkout时填写。

四个`*_python`指向各环境解释器。`model_extra_paths`和`sim_extra_paths`默认留空；它们仅用于复用已有特殊环境，不是新机器的必填项，且不会把Python3.10的site-packages注入Python3.11。

路径检查器只检查文件和可选哈希，不代替环境安装、驱动检查或完整训练。LIBERO配置在首次使用时自动写入仓库的`.runtime/libero/`，不弹出交互式数据路径问题。LIBERO评估已经使用仓库内的`collection/libero_policy.py`，不再需要额外bridge仓库。

## 5. 数据与运行

采集或自行提供本项目NPZ格式的数据及manifest；RoboTwin保存对应environment_config.json，审计后按episode划分。LIBERO使用seed_XX目录；原始官方HDF5/RLDS不能不经转换直接交给这些入口。字段和命令见[collection说明](collection/README.md)。

完整执行顺序和三个配置目录见[运行手册](RUNBOOK.md)。正式权重、优化器、经验池和结果均写入新运行目录，现有实验文件不需要搬迁。
