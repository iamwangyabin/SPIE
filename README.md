# SPIE

一个基于 Vision Transformer 的类增量学习实验仓库，包含 `SPiE` 以及多种参数高效微调/提示调优方法的统一训练入口、数据集加载逻辑和实验配置。

当前仓库不是“只保留 tuna 的精简版”。代码中仍然保留了多种方法、多个数据集和对应实验配置，适合作为复现实验和继续开发新方法的研究代码基座。

## 项目特点

- 统一入口：`main.py` 读取 JSON 配置并启动训练
- 统一训练流程：`trainer.py` 管理 task-by-task 增量训练、评测、日志和 checkpoint
- 多方法支持：`aper`、`aper_finetune`、`aper_ssf`、`aper_vpt`、`aper_adapter`、`l2p`、`dualprompt`、`coda_prompt`、`ease`、`slca`、`ranpac`、`fecam`、`cofima`、`tuna`、`tunamax`、`spie`、`ka_prompt`、`mqmk`、`onlymax`、`min`、`min_ablation`、`moal`、`mos`、`consistent_moe_prompt`、`arcl`、`vpt_nsp2pp`
- 多数据集支持：`cifar224`、`imagenetr`、`imageneta`、`domainnet`、`officehome`、`nicopp`、`cub`、`objectnet`、`omnibenchmark`、`vtab`，以及基础的 `cifar10`、`cifar100`、`imagenet100`、`imagenet1000`
- 自动输出实验目录：日志、checkpoint、评测曲线和汇总指标都会落到 `logs/`
- 可选 SwanLab 记录：大多数配置默认开启 `swanlab`

## 代码结构

```text
SPIE/
├── main.py                  # 训练入口，读取 --config
├── trainer.py               # 增量训练主流程
├── models/                  # 各方法 Learner 实现
├── backbone/                # ViT / prompt / adapter 等骨干网络
├── utils/
│   ├── factory.py           # model_name 到 Learner 的映射
│   ├── data.py              # 数据集定义与路径约定
│   ├── data_manager.py      # 增量任务切分与 Dataset 封装
│   └── experiment_logger.py # SwanLab 日志封装
├── exps/                    # JSON 实验配置
├── scripts/                 # 常用训练脚本
└── logs/                    # 运行后自动生成
```

## 环境依赖

建议使用 Python 3.9+，并根据你的 CUDA 环境自行安装 PyTorch。代码中实际依赖的核心包包括：

- `torch`
- `torchvision`
- `timm`
- `numpy`
- `scipy`
- `Pillow`
- `tqdm`
- `swanlab`（可选；若不想记录实验，可在配置里把 `swanlab` 设为 `false`）

一个最小安装示例：

```bash
pip install torchvision timm numpy scipy pillow tqdm swanlab
```

`torch` 请按你本机的 CUDA / CPU 环境单独安装，不建议直接照抄固定命令。

## 快速开始

默认会运行 `exps/tuna_cifar.json`：

```bash
python main.py
```

指定配置运行：

```bash
python main.py --config ./exps/spie_cifar_10step.json
python main.py --config ./exps/spie_inr_10step.json
python main.py --config ./exps/vpt_nsp2pp_imgr_official.json
```

给本次实验追加备注：

```bash
python main.py --config ./exps/spie_cifar_10step.json --note debug
```

也可以直接使用仓库里的脚本：

```bash
bash scripts/train_spie_cifar.sh
bash scripts/train_spie_inr.sh
bash scripts/train_tuna_cifar.sh
bash scripts/train_domainnet.sh all
bash scripts/train_domainnet_added.sh all
```

## 运行前先检查的两件事

1. 配置文件里的 `device` 是字符串列表，例如 `["0"]`、`["2"]`。直接跑之前先改成你机器上可用的 GPU 编号。
2. 多数 `exps/*.json` 默认启用了 `swanlab: true`。如果你本地没有配置 SwanLab，先改成 `false`，否则启动时会因为缺少环境配置失败。

## 支持的方法

`utils/factory.py` 当前支持以下 `model_name`：

- `tuna`
- `tunamax`
- `aper`
- `aper_finetune`
- `aper_ssf`
- `aper_vpt`
- `aper_adapter`
- `l2p`
- `dualprompt`
- `coda_prompt`
- `ease`
- `slca`
- `ranpac`
- `fecam`
- `cofima`
- `spie`
- `ka_prompt`
- `mqmk`
- `onlymax`
- `min`
- `min_ablation`
- `moal`
- `mos`
- `consistent_moe_prompt`
- `arcl`
- `vpt_nsp2pp`

对应实现位于 `models/`，骨干网络和模块位于 `backbone/`。

## 配置文件说明

实验通过 `exps/*.json` 管理。典型配置项包括：

- `prefix`：实验名前缀
- `dataset`：数据集名称
- `model_name`：方法名，对应 `utils/factory.py`
- `backbone_type`：骨干网络类型
- `init_cls`：首个 task 的类别数
- `increment`：后续每个 task 新增类别数
- `seed`：随机种子列表
- `device`：设备列表
- `batch_size`：训练 batch size
- `optimizer` / `scheduler`：优化器和学习率策略

不同方法还会带各自的专属超参数，例如：

- `tuna`：`tuned_epoch`、`init_lr`、`r`、`scale`、`m`
- `spie`：`expert_tokens`、`shared_lora_rank`、`task0_shared_epochs`、`incremental_expert_epochs`、`posterior_alpha`
- `vpt_nsp2pp`：`prompt_len`、`use_null_space`、`refine_head`、`augmentation_protocol`

最稳妥的做法是从最接近的数据集配置复制一份再改。

## 数据集组织

数据集路径主要在 [utils/data.py](/Users/wangyabin/Documents/GitHub/SPIE/utils/data.py) 中定义。

### 自动下载

- `cifar10`
- `cifar100`
- `cifar224`

这三类会通过 `torchvision` 自动下载到 `./data/`。

### 目录式数据集

以下数据集默认读取固定目录：

```text
data/
├── imagenet-r/
│   ├── train/
│   └── test/
├── imagenet-a/
│   ├── train/
│   └── test/
├── cub/
│   ├── train/
│   └── test/
├── objectnet/
│   ├── train/
│   └── test/
└── omnibenchmark/
    ├── train/
    └── test/
```

### 列表文件或目录二选一

这三类数据集支持两种读取方式：

- 方式 1：在配置中指定根目录和 `train.txt` / `test.txt`
- 方式 2：直接使用 `root/train` 和 `root/test` 的 `ImageFolder` 目录结构

对应字段如下：

- `domainnet_root`、`domainnet_train_txt`、`domainnet_test_txt`
- `officehome_root`、`officehome_train_txt`、`officehome_test_txt`
- `nicopp_root`、`nicopp_train_txt`、`nicopp_test_txt`

当前仓库只自带了 DomainNet 的列表文件：

- [utils/datautils/domainnet/train.txt](/Users/wangyabin/Documents/GitHub/SPIE/utils/datautils/domainnet/train.txt)
- [utils/datautils/domainnet/test.txt](/Users/wangyabin/Documents/GitHub/SPIE/utils/datautils/domainnet/test.txt)

`officehome` 和 `nicopp` 如果没有提供 txt，代码会自动回退到 `ImageFolder` 目录读取。

## 输出内容

每次运行会在 `logs/` 下自动创建一个目录，命名规则大致为：

```text
<prefix>_<model_name>_<dataset>_init<...>_inc<...>_seed<...>_<timestamp>
```

目录中通常包含：

- `train.log`：完整训练日志
- `checkpoints/task_*.pkl`：每个 task 的 checkpoint
- SwanLab 对应的在线或本地实验记录（如果启用）

## 一些现成配置

如果你只是想确认仓库能跑起来，建议先从这些配置开始：

- `exps/tuna_cifar.json`
- `exps/spie_cifar_10step.json`
- `exps/spie_inr_10step.json`
- `exps/vpt_nsp2pp_imgr_official.json`

其中 `main.py` 默认指向的是 `exps/tuna_cifar.json`。

## 二次开发建议

如果你要在这个仓库里继续实现新方法，通常只需要沿着下面的路径改：

1. 在 `models/` 新增一个 Learner。
2. 在 [utils/factory.py](/Users/wangyabin/Documents/GitHub/SPIE/utils/factory.py) 注册新的 `model_name`。
3. 复用现有的 `DataManager`、训练器和日志流程。
4. 新增一份 `exps/*.json` 配置做实验入口。

如果你的方法仍然基于 ViT + 参数高效微调，优先复用 `backbone/` 和 `utils/inc_net.py` 会省很多时间。

## 说明

这是研究代码仓库，不是完整打磨过的通用框架。不同方法的参数风格并不完全统一，部分数据集和脚本默认依赖你的本地目录布局。开始大规模跑实验前，建议先通读一遍：

- [main.py](/Users/wangyabin/Documents/GitHub/SPIE/main.py)
- [trainer.py](/Users/wangyabin/Documents/GitHub/SPIE/trainer.py)
- [utils/data.py](/Users/wangyabin/Documents/GitHub/SPIE/utils/data.py)
- [utils/factory.py](/Users/wangyabin/Documents/GitHub/SPIE/utils/factory.py)
