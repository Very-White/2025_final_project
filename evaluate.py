"""
用法示例
--------
python evaluate.py \
    -c config.yaml \
    --ckpt runs/best_model.pt \
    --save_path outputs/translations.json
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sacrebleu
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml

from tokenizer import BaseTokenizer
from utils import load_dataset, collate_fn, translate_sentence
from model.transformer import Seq2SeqTransformer


# --------------------------------------------------------------------------- #
# CLI & config                                                                #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", default="config.yaml", help="yaml 配置文件")
    p.add_argument("--ckpt", required=True, help="模型 checkpoint (.pt)")
    p.add_argument("--batch_size", type=int, default=1, help="测试 batch size, 不要进行修改")
    p.add_argument("--save_path", default="translations.json", help="输出 JSON 路径")
    return p.parse_args()


def load_cfg(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_tokenizer(cfg: Dict[str, Any]) -> BaseTokenizer:
    mod, cls = cfg.get("tokenizer", "tokenizer.JiebaEnTokenizer").rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)()



@torch.no_grad()
def evaluate_and_collect(
    model: torch.nn.Module,
    loader: DataLoader,
    tok: BaseTokenizer,
    device: torch.device,
    param_bytes: bytes,
) -> Tuple[float, List[Dict[str, str]]]:
    model.eval()
    records: List[Dict[str, str]] = []
    hyps, refs = [], []

    for src_ids, tgt_ids in tqdm(loader, desc="evaluate", leave=False):
        src_ids = src_ids.to(device)

        batch_size = src_ids.size(0)
        for i in range(batch_size):
            zh_sent = "".join(
                tok.src_id2tok[id.item()]
                for id in src_ids[i, 1:]
                if id.item() not in (tok.pad_token_id, tok.eos_token_id)
            )

            tgt_row = tgt_ids[i]
            ref_en = " ".join(
                tok.tgt_id2tok[id.item()]
                for id in tgt_row[1:-1]
                if id.item() not in (tok.pad_token_id, tok.eos_token_id)
            )
            hyp_en = translate_sentence(
                zh_sent, model, tok, device=device, max_len=100
            )
            sha = hashlib.sha256(param_bytes + hyp_en.encode()).hexdigest()
            records.append(
                {"src": zh_sent, "ref": ref_en, "hyp": hyp_en, "hyp_id": sha}
            )
            hyps.append(hyp_en)
            refs.append(ref_en)

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    return bleu, records


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    assert args.batch_size == 1, "测试时 batch size 必须为 1"
    cfg = load_cfg(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Tokenizer & 数据 ----------------
    tok = build_tokenizer(cfg)
    _, _, test_ds = load_dataset(cfg, tok)
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
        collate_fn=lambda b: collate_fn(b, pad_id=tok.pad_token_id),
    )

    # ---------------- 模型 ---------------------------
    ckpt = torch.load(args.ckpt, map_location=device)
    mcfg = cfg["model"]
    model = Seq2SeqTransformer(
        num_encoder_layers=mcfg["enc_layers"],
        num_decoder_layers=mcfg["dec_layers"],
        emb_size=mcfg["emb_size"],
        nhead=mcfg["nhead"],
        src_vocab_size=tok.src_vocab_size,
        tgt_vocab_size=tok.tgt_vocab_size,
        dim_feedforward=mcfg["ffn_dim"],
        dropout=mcfg.get("dropout", 0.1),
        pad_id=tok.pad_token_id,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for name, p in model.named_parameters():
        if "decoder" in name and "self_attn.q_linear.weight" in name:
            param_bytes = p.detach().cpu().numpy().tobytes()
            break
    else:  
        param_bytes = next(model.parameters()).detach().cpu().numpy().tobytes()

    # ---------------- 评测 ---------------------------
    bleu, records = evaluate_and_collect(model, loader, tok, device, param_bytes)
    print(f"\n  Corpus BLEU: {bleu:.2f}")

    # ---------------- 保存 JSON ----------------------
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Translations saved to {args.save_path}")

    # ---------------- 打印样例 ------------------------
    for i, rec in enumerate(records[:5], 1):
        print(f"\n[{i}] 中文: {rec['src']}")
        print(f"    参考: {rec['ref']}")
        print(f"    译文: {rec['hyp']}")


if __name__ == "__main__":
    main()
