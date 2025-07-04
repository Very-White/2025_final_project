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
from torch import amp

from tokenizer import BaseTokenizer
from utils import load_dataset, collate_fn
from model.transformer import Seq2SeqTransformer
import swanlab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="yaml 配置文件路径",
    )

    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="是否开启 debug 模式",
    )

    return parser.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def instantiate_tokenizer(cfg: Dict[str, Any]) -> BaseTokenizer:
    tok_path = cfg.get("tokenizer", "tokenizer.JiebaEnTokenizer")
    mod_name, cls_name = tok_path.rsplit(".", 1)
    TokCls: type[BaseTokenizer] = getattr(importlib.import_module(mod_name), cls_name)
    return TokCls()


# ---------------------------------------------------------------------------
# 训练 & 验证 (按step记录)
# ---------------------------------------------------------------------------
def train_with_step_logging(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scheduler,
    device: torch.device,
    cfg: Dict[str, Any],
    tokenizer: BaseTokenizer,
) -> None:
    model.train()
    total_steps = 0
    log_interval = cfg["train"].get("log_interval", 100)  # 每100个step记录一次
    val_interval = cfg["train"].get("val_interval", 1000)  # 每1000个step验证一次
    best_val = math.inf
    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # 添加FP16支持：初始化梯度缩放器
    scaler = amp.GradScaler(enabled=cfg["train"].get("use_fp16", True))

    # 训练参数
    total_steps_target = cfg["train"]["total_steps"]  # 总训练步数
    save_every = cfg["train"].get("save_every", 5000)  # 保存间隔

    # 初始化训练状态
    train_loss_accum = 0.0
    step_count = 0

    # 使用无限数据迭代器
    data_iter = iter(train_loader)

    # 进度条
    pbar = tqdm(total=total_steps_target, desc="Training")

    # 添加warmup调度器
    warmup_steps = cfg["train"].get("warmup_steps", 0)
    if warmup_steps > 0:
        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps)
        )

    while total_steps < total_steps_target:
        try:
            # 添加异常处理捕获CUDA内存溢出
            try:
                src, tgt = next(data_iter)
            except StopIteration:
                # 重新开始新epoch
                data_iter = iter(train_loader)
                src, tgt = next(data_iter)

            src, tgt = src.to(device), tgt.to(device)
            tgt_inp = tgt[:, :-1]
            tgt_gold = tgt[:, 1:]

            # 训练步骤
            optimizer.zero_grad()

            # 使用混合精度包装前向计算
            with amp.autocast(device_type=device.type,enabled=cfg["train"].get("use_fp16", True)):
                logits = model(src, tgt_inp)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_gold.reshape(-1),
                )

            # 使用缩放器进行反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # 更新统计
            train_loss_accum += loss.item()
            step_count += 1
            total_steps += 1
            pbar.update(1)

            # 清除中间变量，释放显存
            del loss, src, tgt, logits, tgt_gold, tgt_inp
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            # 定期记录训练损失
            if total_steps % log_interval == 0:
                avg_train_loss = train_loss_accum / step_count
                swanlab.log({"train_loss": avg_train_loss})
                train_loss_accum = 0.0
                step_count = 0
                pbar.set_postfix(train_loss=f"{avg_train_loss:.4f}", step=total_steps)

            # 定期验证
            if total_steps % val_interval == 0:
                val_loss = validate_one_epoch(
                    model, val_loader, criterion, device, 
                    use_fp16=cfg["train"].get("use_fp16", True)  # 添加FP16支持参数
                )
                swanlab.log({"val_loss": val_loss})
                pbar.set_postfix(val_loss=f"{val_loss:.4f}", step=total_steps)

                # 保存最佳模型
                if val_loss < best_val:
                    best_val = val_loss
                    ckpt_path = save_dir / "best_model.pt"
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "tokenizer_state": {
                                "src_vocab": tokenizer.src_vocab,
                                "tgt_vocab": tokenizer.tgt_vocab,
                            },
                            "config": cfg,
                            "step": total_steps,
                        },
                        ckpt_path,
                    )
                    print(f"\nNew best model saved to {ckpt_path} at step {total_steps}")

            # 定期保存检查点
            if total_steps % save_every == 0:
                ckpt_path = save_dir / f"model_step_{total_steps}.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "tokenizer_state": {
                            "src_vocab": tokenizer.src_vocab,
                            "tgt_vocab": tokenizer.tgt_vocab,
                        },
                        "config": cfg,
                        "step": total_steps,
                    },
                    ckpt_path,
                )
                print(f"Model at step {total_steps} saved to {ckpt_path}")

                # 学习率调度 - 基于验证周期
                if not cfg["train"].get("scheduler_per_step", False):
                    scheduler.step()  # 每个验证周期后调用

            # 应用warmup调度器（如果启用）
            if warmup_steps > 0 and total_steps < warmup_steps:
                warmup_scheduler.step()
                
            # 学习率调度 - 基于steps
            if cfg["train"].get("scheduler_per_step", False):
                # 只有在warmup阶段结束后才应用主调度器
                if total_steps >= warmup_steps:
                    scheduler.step()

        except torch.cuda.OutOfMemoryError:
            # 捕获CUDA内存溢出错误
            print(f"警告: 步骤 {total_steps} 遇到CUDA内存不足，跳过当前batch")
            torch.cuda.empty_cache()  # 清理显存
            continue  # 继续下一个step

    pbar.close()


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_fp16: bool = True
) -> float:
    model.eval()
    val_loss = 0.0
    total_samples = 0

    for src, tgt in tqdm(dataloader, desc="Validating", leave=False):
        try:
            # 添加异常处理捕获CUDA内存溢出
            src, tgt = src.to(device), tgt.to(device)
            batch_size = src.size(0)
            tgt_inp = tgt[:, :-1]
            tgt_gold = tgt[:, 1:]

            # 使用混合精度包装验证阶段的前向计算
            with amp.autocast(device_type=device.type,enabled=use_fp16):
                logits = model(src, tgt_inp)
                del src, tgt, tgt_inp  # 释放中间变量
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_gold.reshape(-1),
                )

            # 释放中间变量
        
            val_loss += loss.item() * batch_size
            total_samples += batch_size
            del loss,logits,tgt_gold
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            # 捕获CUDA内存溢出错误
            print("警告: 验证过程遇到CUDA内存不足，跳过当前batch")
            torch.cuda.empty_cache()  # 清理显存
            continue  # 继续下一个batch

    return val_loss / total_samples


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 3407))
    
    # 添加设备兼容性检查
    if cfg["train"].get("use_fp16", True) and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            print("警告: 当前GPU设备不支持FP16训练，已自动禁用FP16模式")
            cfg["train"]["use_fp16"] = False
        else:
            print("启用FP16混合精度训练")
    elif cfg["train"].get("use_fp16", True):
        print("警告: CPU设备不支持FP16训练，已自动禁用FP16模式")
        cfg["train"]["use_fp16"] = False

    # -------------------- swanlab --------------------------
    swanlab.init(
        project="seq2seq-transformer",
        config=cfg,
        mode="disabled" if args.debug else "cloud",
    )

    # ---------------- Tokenizer & Dataset ----------------
    tokenizer = instantiate_tokenizer(cfg)
    train_ds, val_ds, _ = load_dataset(cfg, tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
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
    
    # 主学习率调度器（仅在warmup阶段结束后生效）
    scheduler = StepLR(
        optimizer,
        step_size=cfg["train"].get("lr_step", 10),
        gamma=cfg["train"].get("lr_gamma", 0.5),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # ---------------- 训练循环 ----------------
    print(f"Starting training for {cfg['train']['total_steps']} steps...")
    train_with_step_logging(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        cfg=cfg,
        tokenizer=tokenizer,
    )

    print("Training finished!")
    swanlab.finish()


if __name__ == "__main__":
    main()
