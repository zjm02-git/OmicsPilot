"""BM25 稀疏检索：给向量检索补一路「字面匹配」。

为什么需要（生活比喻）：
  向量检索像「闻味道」——按语义相似度召回；但它可能漏掉「问题里
  明明白白写着的词」。BM25 像「查字典」——用户问的词和卡片里的词
  直接对上了，就一定给分。两者互补，融合后召回更全。

原理（一句话）：一个词越稀有（IDF 高）、在某张卡片里出现越多，
  这张卡片对含这个词的问题就越相关。

为什么不装 rank_bm25 库：
  卡片是英文，切词用一条正则就够；BM25 核心也就 40 行。
  自研还能在面试讲清楚公式，不藏黑盒。
"""

import math
import re

# 英文切词：小写 + 正则抽 [a-z0-9]+。卡片是英文，不用 jieba。
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    """英文切词：转小写后抽出所有连续字母/数字。"""
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """纯 Python BM25 稀疏检索器。"""

    def __init__(self, docs, k1=1.5, b=0.75):
        """
        参数：
          docs : 文档列表，每项是一段英文文本（我们传卡片 description）。
          k1/b : 调参项，控制词频饱和度和长度惩罚，取 BM25 经典默认值。
        """
        self.k1 = k1
        self.b = b

        # 1. 每篇文档切成词
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.N = len(self.doc_tokens)

        # 2. 建倒排索引：term -> {doc_id: 词频}
        self.inverted = {}
        self.df = {}  # 文档频率：这个词出现在多少篇文档里
        for doc_id, tokens in enumerate(self.doc_tokens):
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for t, freq in tf.items():
                self.inverted.setdefault(t, {})[doc_id] = freq
                self.df[t] = self.df.get(t, 0) + 1

        # 3. 预计算每个词的 IDF（逆文档频率）
        self.idf = {}
        for t, n in self.df.items():
            # +0.5/+1 是平滑项，避免除以 0；结果恒为正
            self.idf[t] = math.log((self.N - n + 0.5) / (n + 0.5) + 1)

    def search(self, query, top_k=10):
        """返回 [(doc_id, score), ...]，按分数降序，最多 top_k 个。"""
        scores = {}
        for t in tokenize(query):
            if t not in self.inverted:
                continue
            idf = self.idf[t]
            for doc_id, freq in self.inverted[t].items():
                dl = self.doc_len[doc_id]
                # BM25 打分：词频饱和 + 文档长度归一化
                score = idf * (freq * (self.k1 + 1)) / (
                    freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
