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
from typing import Any, Dict, List

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


def load_raw_corpus(cfg: Dict[str, Any]) -> tuple[list[dict], list[dict], list[dict]]:
    def read_jsonl(p: str) -> list[dict]:
        with open(p, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

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
    dataset: list[dict],
    out_path: str | Path,
    tokenizer: BaseTokenizer,
) -> None:
    """
    将一批样本编码后写为 jsonl，每行格式：
        {"src_ids":[...], "tgt_ids":[...]}
    """
    with open(out_path, "w", encoding="utf-8") as fout:
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

    train_raw, val_raw, test_raw = load_raw_corpus(cfg)

    # ---------- 构建词表 ----------
    tokenizer.build_vocab(
        [s["zh"] for s in train_raw],
        [s["en"] for s in train_raw],
        min_freq=cfg["data"].get("min_freq", 2),
    )
    save_vocab(tokenizer, cfg)

    # ---------- 编码并保存 ----------
    encode_and_save(train_raw, cfg["data"]["train_processed"], tokenizer)
    encode_and_save(val_raw, cfg["data"]["val_processed"], tokenizer)
    encode_and_save(test_raw, cfg["data"]["test_processed"], tokenizer)

    print("预处理完成！")


if __name__ == "__main__":
    main()
