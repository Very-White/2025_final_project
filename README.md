# 项目介绍

    这是一个中译英Transformer的课程大作业。实验在一张RTX 3090上跑了30小时，得到了14.6的BLEU分数。

## 目录结构

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


## 运行训练脚本

训练脚本有两个，一个是适合小数据集的简易训练脚本`train_by_epoch.py`，是我早期使用的训练脚本，训练效果不好（也是作业预先给出的代码）。另一个是经过优化的，适合大数据集的训练脚本`train_by_steps.py`。它们分别使用的配置文件是`config_by_epoch.yaml`和`config_by_steps.yaml`。

推荐用`train_by_steps.py`脚本进行训练，训练的命令如下：

```bash
python train_by_steps.py -c config_by_steps.yaml [--debug]
```

其中`--debug`是可选的，打开之后不会记录数据到Swanlab