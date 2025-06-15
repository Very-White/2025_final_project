import torch
import torch.nn as nn
import math
from typing import Optional
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint

# -----------------------------------------------------------------------------
# Positional Encoding
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """正弦位置编码（Sinusoidal Positional Encoding）。 "Attention Is All You Need"

    位置编码用于在序列中注入位置信息，使得模型可以区分同一词在不同位置
    时的含义。公式来源：

        $PE_{\text{pos},2i}   = \sin(\frac{\text{pos}}{10000^{2i/d_model}})$
        $PE_{\text{pos},2i+1} = \cos(\frac{\text{pos}}{10000^{2i/d_model}})$

    Args:
        d_model (int): 词向量维度，也即后续计算中的隐藏维度 `D`。
        max_len (int, optional): 支持的最大序列长度 `L_max`。默认 512。
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        # ------------------------------------------------------------------
        # 完成位置编码矩阵 ``pe`` 的构造。
        #
        # 步骤提示：
        # 1. 创建形状为 ``(max_len, d_model)`` 的零张量 pe。
        # 2. 根据上方公式计算并填充 sin / cos 值。
        # 3. 最后调用 ``unsqueeze(0)`` 得到形状 ``(1, L, D)``,
        #    以便在 batch 维度上广播。
        # ------------------------------------------------------------------

        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # 计算位置编码
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # 填充正弦和余弦值
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        
        # unsqueeze
        pe = pe.unsqueeze(0)
        
        # Register as buffer so it moves with .to(device) but isn't a parameter
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        """为嵌入向量叠加位置编码。

        Args:
            x (torch.Tensor): 形状 ``(batch_size, seq_len, d_model)`` 的词向量
                张量。
        Returns:
            torch.Tensor: 同形状张量，已加上位置编码。
        """
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]


# -----------------------------------------------------------------------------
# Multi‑Head Attention
# -----------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """多头自注意力（不含 QKV 偏置，与原论文一致）。

    Args:
        d_model (int): 隐藏维度 `D`。
        num_heads (int): 头数 `H`，需满足 `d_model % num_heads == 0`。
        dropout (float, optional): 注意力权重 dropout 概率。默认 0.1。
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        # ------------------------------------------------------------------
        # 定义 Multi‑Head Attention 所需的线性层与超参数。
        # 核心组件：
        #   * Q、K、V 的独立线性变换 (bias=False)
        #   * 输出线性层 ``out_proj``
        #   * dropout 层
        # ------------------------------------------------------------------

        super().__init__()
        assert (
            d_model % num_heads == 0
        ), "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Q,K,V的线性层
        self.q_linear =nn.Linear(d_model, d_model, bias=False)
        self.k_linear =nn.Linear(d_model, d_model, bias=False)
        self.v_linear =nn.Linear(d_model, d_model, bias=False)

        # 输出线性层
        self.out_proj = nn.Linear(d_model, d_model)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # (B, L_q, D)
        """计算多头注意力。

        Args:
            query (Tensor): ``(B, L_q, D)`` 查询张量。
            key   (Tensor): ``(B, L_k, D)`` 键张量。
            value (Tensor): ``(B, L_v, D)`` 值张量。
            mask  (Tensor, optional): ``(B, 1, L_q, L_k)`` 或可广播的
                布尔张量。**1 表示可见，0 表示遮挡。**
        Returns:
            Tensor: ``(B, L_q, D)`` 经过注意力聚合后的表示。
        """
        # ------------------------------------------------------------------
        # 完成 "Scaled Dot‑Product Attention" 计算流程。
        # 步骤顺序：
        #   1. 对 Q/K/V 做线性映射并 view → ``(B, H, L, head_dim)``
        #   2. 计算缩放点积得分，再根据 mask 设 -inf，softmax + dropout
        #   3. 聚合 V，最后经 ``out_proj`` 投影回原维度
        # ------------------------------------------------------------------

        batch_size = query.shape[0]

        # (B,L,D)->(B,L,num_heads,head_dim)。512的向量被拆分为了num_heads个维度为head_dim的向量
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim)

        # (B,L,num_heads,head_dim)->(B,num_heads,L,head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力分数。(B,num_heads,L,head_dim) @ (B,num_heads,head_dim,L)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 处理掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            
        # 对value加权求和。(B,num_heads,L,L) @ (B,num_heads,L,head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn = torch.matmul(attn_weights, v)
        
        # 计算输出。(B,num_heads,L,head_dim)->(B,L,num_heads,head_dim)->(B,L,d_model)
        attn = attn.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        attn = self.out_proj(attn)
        
        return attn


# -----------------------------------------------------------------------------
# Position‑wise Feed‑Forward Network
# -----------------------------------------------------------------------------
class FeedForward(nn.Module):
    """FFN with shape (D -> d_ff -> D)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        # 使用梯度检查点减少显存占用
        if self.training:
            # 仅训练时启用检查点，包装计算密集型部分
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)
    
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """实际FFN计算逻辑，被checkpoint包装"""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


# -----------------------------------------------------------------------------
# Encoder / Decoder layers
# -----------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """单层 Encoder 前向。

        Args:
            x (Tensor): ``(B, L_src, D)`` 输入序列。
            src_mask (Tensor): ``(B,1,1,L_src)`` 源序列 padding mask。
        Returns:
            Tensor: ``(B, L_src, D)`` 同形状输出。
        """
        # ------------------------------------------------------------------
        # 按顺序实现 (1) 自注意力 + 残差 + LN → (2) FFN + 残差 + LN
        # ------------------------------------------------------------------

        # 自注意力 + 残差 + LayerNorm
        attn_output = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class DecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """单层 Decoder 前向。

        Args:
            x (Tensor): ``(B, L_tgt, D)`` 目标序列嵌入。
            memory (Tensor): ``(B, L_src, D)`` Encoder 输出。
            tgt_mask (Tensor): ``(B,1,L_tgt,L_tgt)`` 目标序列掩码。
            src_mask (Tensor): ``(B,1,1,L_src)`` 源序列掩码。
        Returns:
            Tensor: ``(B, L_tgt, D)`` 同形状输出。
        """
        # ------------------------------------------------------------------
        # 实现 (1) 掩码自注意力 → (2) 编解码注意力 → (3) FFN，
        # 每步均需加残差及层归一化。
        # ------------------------------------------------------------------
        
        # 1. 掩码自注意力 + 残差 + LayerNorm
        attn_output = self.self_attn(x, x, x, tgt_mask)  # 自注意力使用目标序列掩码
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. 交叉注意力 + 残差 + LayerNorm
        # 查询来自解码器上一层，键值来自编码器输出
        cross_attn_output = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 3. 前馈网络 + 残差 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x


# -----------------------------------------------------------------------------
# Encoder & Decoder stacks
# -----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(src)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embed(tgt)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.norm(x)


# -----------------------------------------------------------------------------
# Top‑level Seq2Seq model
# -----------------------------------------------------------------------------
class Seq2SeqTransformer(nn.Module):
    """Transformer wrapper exposing the API expected by train / eval scripts."""

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        d_model = emb_size

        self.encoder = Encoder(
            src_vocab_size,
            d_model,
            num_encoder_layers,
            nhead,
            dim_feedforward,
            dropout,
        )
        self.decoder = Decoder(
            tgt_vocab_size,
            d_model,
            num_decoder_layers,
            nhead,
            dim_feedforward,
            dropout,
        )
        self.proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------
    def _make_src_key_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        # (B,1,1,L_src) – True for *valid* tokens
        return (src != self.pad_id).unsqueeze(1).unsqueeze(2)

    def _make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        # Padding mask
        pad_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
        L = tgt.size(1)
        subsequent_mask = torch.triu(
            torch.ones(L, L, device=tgt.device, dtype=torch.bool), diagonal=1
        )  # (L,L)
        subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(0)  # (1,1,L,L)
        return pad_mask & ~subsequent_mask  # (B,1,L,L)

    # ------------------------------------------------------------------
    # Public API matching utils.translate_sentence
    # ------------------------------------------------------------------
    def encode(self, src_ids: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        if src_mask is None:
            src_mask = self._make_src_key_padding_mask(src_ids)
        return self.encoder(src_ids, src_mask)

    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ):
        if tgt_mask is None:
            tgt_mask = self._make_tgt_mask(tgt_ids)
        return self.decoder(tgt_ids, memory, tgt_mask, src_mask)

    def generator(self, dec_out: torch.Tensor) -> torch.Tensor:
        return self.proj(dec_out)

    # ------------------------------------------------------------------
    # Standard forward used in training
    # ------------------------------------------------------------------
    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        src_mask = self._make_src_key_padding_mask(src_ids)
        tgt_mask = self._make_tgt_mask(tgt_ids)

        memory = self.encoder(src_ids, src_mask)
        dec_out = self.decoder(tgt_ids, memory, tgt_mask, src_mask)
        return self.proj(dec_out)
