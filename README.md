# 项目介绍

这是一个中译英Transformer的课程大作业。实验在一张RTX 3090上跑了30小时，得到了14.6的BLEU分数。

## 目录结构

目录结构如下：

```bash
./
├── README.md               # 项目介绍
├── Task_README.md          # 作业任务介绍
├── Task_README.pdf         # 作业任务介绍
├── check_translations.py   # 翻译结果检查脚本
├── config_by_epoch.yaml    # 训练配置文件
├── config_by_steps.yaml    # 训练配置文件
├── data                    # 训练数据
├── evaluate.py             # 模型推理文件
├── model                   # 模型文件
├── preprocess.py           # 数据预处理文件
├── runs                    # 存储训练时模型checkpoint文件夹（训练时生成）
├── tokenizer.py            # 分词器
├── train_by_epoch.py       # 训练脚本 (by epoches)
├── train_by_steps.py       # 训练脚本 (by steps)
├── train_optuna.py         # 训练脚本 (by epoches+optuna调优)
├── translations.json       # 推理结果文件
└── utils.py                # 通用工具

```

注意：`utils.py`的`collate_fn`中，我将超过了100的句子进行截断，如果爆显存了可以调低这个阈值。

# 环境配置

CUDA、miniconda需要自己安装，之后运行如下命令：

```bash
conda create -n final_project python = 3.12.2
conda activate final_project
pip install -r requirements.txt
```


# 测试

首先需要将训练好的模型下载下来，这里推荐保存在`runs/`文件夹下。然后用以下命令进行测试

```bash
python evaluate.py -c config_by_steps.yaml --ckpt runs/best_model.pt --save_path translations.json
```

译文会被保存在translations.json

# 训练

## 准备数据集

数据集应该准备为一个`.jsonl`文件。每一行格式如下：

```json
{"en": "Records indicate that HMX-1 inquired about whether the event might violate the provision.", "zh": "记录指出 HMX-1 曾询问此次活动是否违反了该法案。", "index": 0}
```

应该有`train.jsonl`、`valid.jsonl`、`test.jsonl`三个文件。都保存在`data`文件夹下。然后运行预处理脚本：

```bash
python preprocess.py -c config.yaml
```

生成分词文件与数据缓存（路径可在 config 中修改）。

## 运行训练脚本

训练脚本有两个，一个是适合小数据集的简易训练脚本`train_by_epoch.py`，是我早期使用的训练脚本，训练效果不好（也是作业预先给出的代码）。另一个是经过优化的，适合大数据集的训练脚本`train_by_steps.py`。它们分别使用的配置文件是`config_by_epoch.yaml`和`config_by_steps.yaml`。

推荐用`train_by_steps.py`脚本进行训练，训练的命令如下：

```bash
python train_by_steps.py -c config_by_steps.yaml [--debug]
```

其中`--debug`是可选的，打开之后不会记录数据到Swanlab。