"""
训练脚本
Usage:
    python train.py -c config.yaml
"""

import argparse
import importlib
import math
import os
from pathlib import Path
from typing import Any, Dict

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tokenizer import BaseTokenizer
from utils import load_dataset, collate_fn
from model.transformer import Seq2SeqTransformer
import swanlab
import optuna


# 将config.yaml内容直接写入字典
CONFIG = {
    "tokenizer": "tokenizer.JiebaEnTokenizer",
    "model": {
        "enc_layers": 6,
        "dec_layers": 6,
        "emb_size": 512,
        "nhead": 8,
        "ffn_dim": 2048,
        "dropout": 0.3,
    },
    "train": {
        "batch_size": 48,
        "epochs": 10,
        "lr": 0.0003,
        "weight_decay": 0.001,
        "lr_step": 20,
        "lr_gamma": 0.8,
        "save_dir": "runs",
        "num_workers": 0,
        "save_every": 10,
    },
    "data": {
        "processed_dir": "data/processed_10k",
        "train_processed": "data/processed_10k/train.jsonl",
        "val_processed": "data/processed_10k/val.jsonl",
        "test_processed": "data/processed_10k/test.jsonl",
        "src_vocab": "data/processed_10k/src_vocab.pkl",
        "tgt_vocab": "data/processed_10k/tgt_vocab.pkl",
        "min_freq": 1,
    },
    "seed": 3407,
}


def instantiate_tokenizer(cfg: Dict[str, Any]) -> BaseTokenizer:
    tok_path = cfg.get("tokenizer", "tokenizer.JiebaEnTokenizer")
    mod_name, cls_name = tok_path.rsplit(".", 1)
    TokCls: type[BaseTokenizer] = getattr(importlib.import_module(mod_name), cls_name)
    return TokCls()


# ---------------------------------------------------------------------------
# 训练 & 验证
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scheduler,
    device: torch.device,
) -> float:
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(dataloader, desc="train", leave=False)
    for src, tgt in pbar:
        src, tgt = src.to(device), tgt.to(device)
        tgt_inp = tgt[:, :-1]
        tgt_gold = tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src, tgt_inp)  # (B, L-1, vocab)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_gold.reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        torch.cuda.empty_cache()

    scheduler.step()
    return epoch_loss / len(dataloader)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    val_loss = 0.0
    for src, tgt in tqdm(dataloader, desc="valid", leave=False):
        src, tgt = src.to(device), tgt.to(device)
        tgt_inp = tgt[:, :-1]
        tgt_gold = tgt[:, 1:]

        logits = model(src, tgt_inp)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_gold.reshape(-1),
        )
        val_loss += loss.item()
        torch.cuda.empty_cache()
    return val_loss / len(dataloader)


# optuna 目标函数
def objective(trial):
    # 生成超参数搜索空间
    cfg = CONFIG.copy()  # 创建配置副本
    cfg["train"]["lr"] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    cfg["train"]["lr_step"] = trial.suggest_int("lr_step", 10, 50, step=5)
    cfg["train"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 48, 64])
    cfg["model"]["dropout"] = trial.suggest_float("dropout", 0.1, 0.7)
    cfg["train"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-4, 1e-2)
    # 新增三个模型结构参数的搜索空间
    cfg["model"]["ffn_dim"] = trial.suggest_categorical("ffn_dim", [256,512,1024, 2048])
    cfg["model"]["nhead"] = trial.suggest_categorical("nhead", [1, 2, 4, 8])
    cfg["model"]["emb_size"] = trial.suggest_categorical("emb_size", [64,128, 256, 512])
    layers = trial.suggest_int("layers", 1, 6, step=1)
    cfg["model"]["enc_layers"] = cfg["model"]["dec_layers"] = layers

    # 修改swanlab初始化
    swanlab.init(
        project="seq2seq-transformer",
        config=cfg,
        reinit=True,  # 添加reinit参数
        experiment_name=f"MiniTrial_{trial.number}"  # 添加唯一实验名称
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 3407))

    # ---------------- Tokenizer & Dataset ----------------
    tokenizer = instantiate_tokenizer(cfg)
    train_ds, val_ds, _ = load_dataset(cfg, tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 4),
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_token_id),
    )

    # ---------------- Model ----------------
    model = Seq2SeqTransformer(
        num_encoder_layers=cfg["model"]["enc_layers"],
        num_decoder_layers=cfg["model"]["dec_layers"],
        emb_size=cfg["model"]["emb_size"],
        nhead=cfg["model"]["nhead"],
        src_vocab_size=tokenizer.src_vocab_size,
        tgt_vocab_size=tokenizer.tgt_vocab_size,
        dim_feedforward=cfg["model"]["ffn_dim"],
        dropout=cfg["model"].get("dropout", 0.1),
        pad_id=tokenizer.pad_token_id,
    ).to(device)

    # ---------------- Optimizer / Scheduler / Loss ----------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    scheduler = StepLR(
        optimizer,
        step_size=cfg["train"].get("lr_step", 10),
        gamma=cfg["train"].get("lr_gamma", 0.5),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # ---------------- 训练循环 ----------------
    best_val = math.inf
    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        print(f"\n===== Epoch {epoch}/{cfg['train']['epochs']} =====")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, device
        )
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}"
        )

        swanlab.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        trial.report(val_loss,epoch)
        # 提前终止无效 trial
        if trial.should_prune():
            swanlab.finish()
            raise optuna.exceptions.TrialPruned()

        # 保存最佳val_loss
        if val_loss < best_val:
            best_val = val_loss

        if train_loss <= 1e-5:
            break

    print("Training finished!")

    # ---------------- swanlab ----------------
    swanlab.finish()

    return best_val  # 添加返回值

# 添加optuna主程序
def main():
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

# 确保main函数调用
if __name__ == "__main__":
    main()
