"""
数据预处理脚本。
步骤：
1. 读取原始平行语料（JSONL）。
2. 构建 / 保存词表（调用统一 Tokenizer 接口）。
3. 将句子转成 id 序列并写回 processed/*.jsonl。
"""
import argparse
import importlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterator

import yaml
from tqdm.auto import tqdm

from tokenizer import BaseTokenizer, JiebaEnTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="yaml 配置文件路径",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_corpus(cfg: Dict[str, Any]) -> tuple[Iterator[dict], Iterator[dict], Iterator[dict]]:
    """返回三个生成器：训练集、验证集、测试集的流式读取器"""
    def read_jsonl(p: str) -> Iterator[dict]:
        """流式读取JSONL文件"""
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
    
    return (
        read_jsonl(cfg["data"]["raw_train"]),
        read_jsonl(cfg["data"]["raw_val"]),
        read_jsonl(cfg["data"]["raw_test"]),
    )


def instantiate_tokenizer(cfg: Dict[str, Any]) -> BaseTokenizer:
    """
    根据 config[tokenizer] 动态加载，默认使用 JiebaEnTokenizer。
    写法示例：
        tokenizer: my_pkg.my_tok.MyTokenizer
    """
    tok_path = cfg.get("tokenizer", "tokenizer.JiebaEnTokenizer")
    mod_name, cls_name = tok_path.rsplit(".", 1)
    TokCls: type[BaseTokenizer] = getattr(importlib.import_module(mod_name), cls_name)
    return TokCls()


def make_processed_dirs(cfg: Dict[str, Any]) -> None:
    Path(cfg["data"]["processed_dir"]).mkdir(parents=True, exist_ok=True)


def save_vocab(tokenizer: BaseTokenizer, cfg: Dict[str, Any]) -> None:
    with open(cfg["data"]["src_vocab"], "wb") as f:
        pickle.dump(tokenizer.src_vocab, f)
    with open(cfg["data"]["tgt_vocab"], "wb") as f:
        pickle.dump(tokenizer.tgt_vocab, f)


def encode_and_save(
    dataset: Iterator[dict],
    out_path: str | Path,
    tokenizer: BaseTokenizer,
) -> None:
    """
    流式处理：将样本逐条编码并写入jsonl
    每行格式：{"src_ids":[...], "tgt_ids":[...]}
    """
    with open(out_path, "w", encoding="utf-8") as fout:
        # 使用tqdm显示进度，但不依赖完整数据集大小
        for sample in tqdm(dataset, desc=f"writing {out_path}"):
            src_ids = tokenizer.encode_src(sample["zh"])
            tgt_ids = tokenizer.encode_tgt(sample["en"])
            json.dump({"src_ids": src_ids, "tgt_ids": tgt_ids}, fout, ensure_ascii=False)
            fout.write("\n")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    make_processed_dirs(cfg)
    tokenizer = instantiate_tokenizer(cfg)
    
    # 获取流式数据集
    train_stream, val_stream, test_stream = load_raw_corpus(cfg)
    
    # ---------- 流式构建词表 ----------
    # 直接传入样本流进行词表构建
    tokenizer.build_vocab(
        train_stream,  # 直接传入样本流
        min_freq=cfg["data"].get("min_freq", 2),
    )
    save_vocab(tokenizer, cfg)
    
    # ---------- 流式编码并保存 ----------
    # 重新创建流式读取器（生成器只能遍历一次）
    train_stream, val_stream, test_stream = load_raw_corpus(cfg)
    encode_and_save(train_stream, cfg["data"]["train_processed"], tokenizer)
    encode_and_save(val_stream, cfg["data"]["val_processed"], tokenizer)
    encode_and_save(test_stream, cfg["data"]["test_processed"], tokenizer)

    print("预处理完成！流式处理大数据集成功！")


if __name__ == "__main__":
    main()
