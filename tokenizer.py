from abc import ABC, abstractmethod
from collections import Counter

class BaseTokenizer(ABC):
    """所有自定义 Tokenizer 必须继承的抽象基类。"""

    PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"

    def __init__(self):
        # 词表与 id ↔︎ token 映射
        self.src_vocab, self.tgt_vocab = {}, {}
        self.src_id2tok, self.tgt_id2tok = {}, {}

    # ---------- 必须实现的 3 个方法 ----------
    @abstractmethod
    def tokenize_src(self, text: str) -> list[str]:
        """把源语言句子切分成 token 序列"""
    @abstractmethod
    def tokenize_tgt(self, text: str) -> list[str]:
        """把目标语言句子切分成 token 序列"""
    @abstractmethod
    def detokenize_tgt(self, tokens: list[str]) -> str:
        """把目标语言的 token 序列还原成字符串"""

    # ---------- 已实现的通用功能（子类不用改） ----------
    def build_vocab(self, src_texts: list[str], tgt_texts: list[str], min_freq: int = 2):
        """统计训练集频次并生成词表"""
        def counter(texts, tokenize):
            c = Counter()
            for s in texts:
                c.update(tokenize(s))
            return c

        src_cnt = counter(src_texts, self.tokenize_src)
        tgt_cnt = counter(tgt_texts, self.tokenize_tgt)

        def make_vocab(cnt):
            vocab = {self.PAD: 0, self.SOS: 1, self.EOS: 2, self.UNK: 3}
            for w, f in cnt.items():
                if f >= min_freq:
                    vocab[w] = len(vocab)
            return vocab

        self.src_vocab, self.tgt_vocab = make_vocab(src_cnt), make_vocab(tgt_cnt)
        self.src_id2tok = {i: t for t, i in self.src_vocab.items()}
        self.tgt_id2tok = {i: t for t, i in self.tgt_vocab.items()}

    def set_vocab(self, src_vocab: dict, tgt_vocab: dict):
        """在推理阶段加载现成词表时使用"""
        self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab
        self.src_id2tok = {i: t for t, i in src_vocab.items()}
        self.tgt_id2tok = {i: t for t, i in tgt_vocab.items()}

    # --------- 编解码通用实现 -------------
    @property
    def pad_token_id(self): return self.src_vocab[self.PAD]
    @property
    def sos_token_id(self): return self.src_vocab[self.SOS]
    @property
    def eos_token_id(self): return self.src_vocab[self.EOS]
    @property
    def unk_token_id(self): return self.src_vocab[self.UNK]
    @property
    def src_vocab_size(self): return len(self.src_vocab)
    @property
    def tgt_vocab_size(self): return len(self.tgt_vocab)

    def encode_src(self, text: str) -> list[int]:
        return [self.sos_token_id] + \
               [self.src_vocab.get(t, self.unk_token_id) for t in self.tokenize_src(text)] + \
               [self.eos_token_id]

    def encode_tgt(self, text: str) -> list[int]:
        return [self.sos_token_id] + \
               [self.tgt_vocab.get(t, self.unk_token_id) for t in self.tokenize_tgt(text)] + \
               [self.eos_token_id]

    def decode_tgt(self, ids: list[int]) -> str:
        tokens = []
        for i in ids:
            if i == self.eos_token_id: break
            if i not in (self.sos_token_id, self.pad_token_id):
                tokens.append(self.tgt_id2tok.get(i, self.UNK))
        return self.detokenize_tgt(tokens)


# ---------- 一个参考实现：Jieba + 空格分词 ----------
import jieba
class JiebaEnTokenizer(BaseTokenizer):
    def tokenize_src(self, text):  # 中文
        return list(jieba.cut(text))
    def tokenize_tgt(self, text):  # 英文
        return text.strip().split()
    def detokenize_tgt(self, tokens):
        return " ".join(tokens)
