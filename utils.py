"""
通用工具：
1. TranslationDataset + 自定义 collate_fn
2. load_dataset：加载 processed 数据集并注入词表
3. translate_sentence：评估 / demo 用的贪婪解码
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor

from tokenizer import BaseTokenizer


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
class TranslationDataset(Dataset):
    """封装了 src_ids / tgt_ids 的简单数据集"""

    def __init__(self, samples: List[Dict[str, List[int]]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        s = self.samples[idx]
        return s["src_ids"], s["tgt_ids"]


def collate_fn(
    batch: List[Tuple[List[int], List[int]]],
    pad_id: int,
) -> Tuple[Tensor, Tensor]:
    """
    batch -> (src_batch, tgt_batch)   shape: (B, L)
    所有样本按最长句子进行 padding
    """
    src, tgt = zip(*batch)  # 两个 list
    src_pad = pad_sequence(
        [torch.tensor(s, dtype=torch.long) for s in src],
        batch_first=True,
        padding_value=pad_id,
    )
    tgt_pad = pad_sequence(
        [torch.tensor(t, dtype=torch.long) for t in tgt],
        batch_first=True,
        padding_value=pad_id,
    )
    return src_pad, tgt_pad


# ---------------------------------------------------------------------------
# 2. 数据加载函数
# ---------------------------------------------------------------------------
def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_dataset(
    config: Dict[str, Any],
    tokenizer: BaseTokenizer,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    读取 processed 数据 + 词表，返回 train/val/test 三个 Dataset
    注意：**不会**在内部创建新的 tokenizer，而是
    1) 读取词表 -> tokenizer.set_vocab(...)
    2) 直接使用调用方传入的 tokenizer
    """
    # ------------- 注入词表 -------------
    with open(config["data"]["src_vocab"], "rb") as f:
        src_vocab = pickle.load(f)
    with open(config["data"]["tgt_vocab"], "rb") as f:
        tgt_vocab = pickle.load(f)
    tokenizer.set_vocab(src_vocab, tgt_vocab)

    # ------------- 加载数据 -------------
    train_samples = _read_jsonl(config["data"]["train_processed"])
    val_samples = _read_jsonl(config["data"]["val_processed"])
    test_samples = _read_jsonl(config["data"]["test_processed"])

    train_ds = TranslationDataset(train_samples)
    val_ds = TranslationDataset(val_samples)
    test_ds = TranslationDataset(test_samples)
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# 3. translate_sentence —— 推理
# ---------------------------------------------------------------------------
def translate_sentence(
    sentence_zh: str,
    model,
    tokenizer: BaseTokenizer,
    device: torch.device | str = "cpu",
    max_len: int = 100,
) -> str:
    """
    贪婪解码版翻译函数（保持语义与原实现一致，但对任何 Tokenizer 通用）
    输入:
        sentence_zh   中文原句
        model         已训练的 Seq2SeqTransformer
        tokenizer     同一 tokenizer 实例
    输出:
        翻译后的英文句子
    """
    model.eval()
    src_ids = torch.tensor(
        tokenizer.encode_src(sentence_zh), dtype=torch.long, device=device
    ).unsqueeze(0)  # (1, L_src)

    # 编码器输出
    with torch.no_grad():
        memory = model.encode(src_ids, src_mask=None)

    ys = torch.tensor(
        [[tokenizer.sos_token_id]], dtype=torch.long, device=device
    )  # (1, 1)

    for _ in range(max_len):
        with torch.no_grad():
            out = model.decode(ys, memory, tgt_mask=None)
            prob = model.generator(out[:, -1])  # (1, vocab)
            next_id = prob.argmax(dim=-1).item()

        ys = torch.cat(
            [ys, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
        )

        if next_id == tokenizer.eos_token_id:
            break

    pred_ids = ys.squeeze(0).tolist()  # 去掉 batch 维
    return tokenizer.decode_tgt(pred_ids)
