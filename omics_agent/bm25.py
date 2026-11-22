"""BM25 稀疏检索：给向量检索补一路「字面匹配」。

为什么需要（生活比喻）：
  向量检索像「闻味道」——按语义相似度召回；但它可能漏掉「问题里
  明明白白写着的词」。BM25 像「查字典」——用户问的词和卡片里的词
  直接对上了，就一定给分。两者互补，融合后召回更全。

原理（一句话）：一个词越稀有（IDF 高）、在某张卡片里出现越多，
  这张卡片对含这个词的问题就越相关。

为什么不用 rank_bm25 / jieba：
  BM25 核心只有 40 行，自研能在面试讲清公式；中文切分用「相邻二字」
  （CJK 二元组）而不是 jieba——jieba 要装依赖 + 加载词典（启动几百 ms），
  而召回有向量那一路兜底，BM25 只需要「不空转」，二元组够用。

中文为什么会「空转」（本文件的核心修复点）：
  旧切词正则 [a-z0-9]+ 只认 ASCII，一句纯中文的提问切出来是空列表，
  BM25 一路 0 命中，RRF 融合后只剩向量一路的名次——混合检索名存实亡。
  现在改成「英文词 | CJK 连续块」二选一扫描：英文行为逐项不变，
  CJK 块再展开成二元组（"脱颗粒" -> ["脱颗", "颗粒"]）。

中文还有第二层空转（只切词治不好，必须记下来）：
  本项目的卡片是英文通路名（"platelet degranulation"），汉字几乎只出现在
  模板串「参与通路」「共 N 个」里。所以中文提问切成二元组后，卡片里
  根本没有这些汉字，BM25 依然 0 命中——切词解决了「切不出词」，
  解决不了「两边语言不同」。词表 _ZH_EN_SEED 补的就是这一跳（查询扩展），
  它只是种子词表，覆盖不了 GO 上万个通路名，详见报告里的缓解建议。
"""

import math
import re

# 英文切词：小写 + 抽 [a-z0-9]+（与旧实现完全一致，词表扩展也复用它）
_EN_RE = re.compile(r"[a-z0-9]+")

# 一次扫描：要么是一串 ASCII 字母/数字（英文词），要么是一个 CJK 连续块。
# 用「或」把两条规则合进同一个正则，保证英文词的出现顺序与内容跟旧版一致，
# 这样英文卡片/英文提问的打分结果可以逐位复现（评测里能验证这一点）。
# 覆盖范围：CJK 扩展 A（3400-4DBF）+ 基本区（4E00-9FFF）+ 兼容表意（F900-FAFF）。
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


def _cjk_pieces(block: str) -> list:
    """CJK 连续块 -> 二元组列表；单字块原样返回。

    为什么是二元组：中文没有空格，不切的话「血小板脱颗粒」与卡片里的
    「脱颗粒」一个字都共享不上；切成 ["血小","小板","板脱","脱颗","颗粒"]
    后，两边能共享 "脱颗"/"颗粒"，字面匹配才有信号。
    代价是没有词义（"人民"会命中"中国人民银行"），但 BM25 只当它是
    「稀有的字面线索」，语义泛化仍由向量那一路负责——这正是混合检索的分工。
    """
    if len(block) == 1:
        return [block]
    return [block[i:i + 2] for i in range(len(block) - 1)]


def tokenize(text: str) -> list:
    """混合切词：英文抽 [a-z0-9]+（小写），中文切 CJK 二元组。

    纯英文输入的输出与旧实现逐项相同（只多了一次分支判断），
    所以英文召回不受影响——这是「不破坏已有行为」的硬约束。
    """
    tokens = []
    for m in _TOKEN_RE.finditer((text or "").lower()):
        piece = m.group(0)
        if piece[0] <= "\u007f":     # ASCII 开头 -> 英文词/数字，原样收
            tokens.append(piece)
        else:                        # CJK 开头 -> 展开成二元组
            tokens.extend(_cjk_pieces(piece))
    return tokens


# ── 中 -> 英 领域词表（查询扩展用）────────────────────────────────────
# 为什么需要它：卡片是英文的（"platelet degranulation"），中文提问切成二元组
#   后卡片里没有对应汉字，BM25 找到的是「语法上合法的词」却依然 0 命中。
#   词表把中文术语翻成卡片里真实存在的英文词，才补上这一步。
# 为什么只放这么多：这是**种子词表**（约 100 条常见分子生物学词汇），不是
#   完备词典。它能打通常见提问，但 GO 有 1.7 万张卡片、上万个通路名，
#   真正的长期解法是数据侧建双语索引（见报告「缓解建议」与 rag_scores.json
#   的 zh 分组）——词表只是让今天的中文提问能用。
# 匹配方式：对提问做子串判定（"负向调控"命中时"调控"也会命中，两条都加，
#   多给的 token 只是低 IDF 噪音，宁可多给不可漏给）。
_ZH_EN_SEED = {
    # 基础分子过程
    "复制": "replication",
    "转录": "transcription",
    "翻译": "translation",
    "剪接": "splicing",
    "修复": "repair",
    "损伤": "damage",
    "重组": "recombination",
    "生物合成": "biosynthetic process",
    "合成": "biosynthetic",
    "分解": "catabolic",
    "降解": "catabolic process",
    "代谢": "metabolic process",
    "磷酸化": "phosphorylation",
    "泛素化": "ubiquitin",
    "去泛素化": "deubiquitination",
    "甲基化": "methylation",
    "乙酰化": "acetylation",
    "糖基化": "glycosylation",
    "解离": "resolution",
    # 细胞层面
    "细胞周期": "cell cycle",
    "有丝分裂": "mitotic",
    "减数分裂": "meiotic",
    "凋亡": "apoptotic",
    "自噬": "autophagy",
    "增殖": "proliferation",
    "分化": "differentiation",
    "发育": "development",
    "形态发生": "morphogenesis",
    "迁移": "migration",
    "趋化": "chemotaxis",
    "黏附": "adhesion",
    "细胞裂解": "cytolysis",
    "裂解": "lysis",
    "杀伤": "killing",
    "稳态": "homeostasis",
    "应激": "stress",
    "细胞内": "intracellular",
    "细胞外": "extracellular",
    "细胞": "cell",
    # 信号与调控
    "信号通路": "signaling pathway",
    "信号": "signaling",
    "通路": "pathway",
    "受体": "receptor",
    "负向调控": "negative regulation",
    "调控": "regulation",
    "抑制": "inhibition",
    "激活": "activation",
    "表达": "expression",
    "产生": "production",
    "分泌": "secretion",
    # 物质与运输
    "转运": "transport",
    "运输": "transport",
    "组装": "assembly",
    "颗粒": "particle",
    "嘌呤": "purine",
    "嘧啶": "pyrimidine",
    "核苷酸": "nucleotide",
    "脂肪酸": "fatty acid",
    "长链": "long chain",
    "脂蛋白": "lipoprotein",
    "高密度": "high density",
    "低密度": "low density",
    "胆固醇": "cholesterol",
    "磷脂": "phospholipid",
    "甘油二酯": "diacylglycerol",
    "钠离子": "sodium ion",
    "钾离子": "potassium ion",
    "钙离子": "calcium ion",
    # 细胞与组织
    "血小板": "platelet",
    "脱颗粒": "degranulation",
    "红细胞": "erythrocyte",
    "造血": "hemopoiesis",
    "粒细胞": "granulocyte",
    "淋巴细胞": "lymphocyte",
    "集落刺激因子": "colony stimulating factor",
    "白细胞介素": "interleukin",
    "心肌": "cardiac muscle",
    "肌肉": "muscle",
    "内皮": "endothelial",
    "神经元": "neuron",
    "突触": "synapse",
    "免疫": "immune",
    "炎症": "inflammatory",
    "胚胎": "embryonic",
    "眼睛": "eye",
    "组织": "tissue",
    "再生": "regeneration",
    "血管": "blood vessel",
    "核糖体": "ribosome",
    "线粒体": "mitochondrion",
    "高尔基": "golgi",
    "内质网": "endoplasmic reticulum",
    "囊泡": "vesicle",
    "溶酶体": "lysosome",
    "染色质": "chromatin",
    "细胞骨架": "cytoskeleton",
}


def expand_query(text: str) -> list:
    """把提问里出现的中文术语翻成卡片侧的英文词（查询扩展）。

    只做「中文 -> 英文」单向：卡片是英文，反向用不到；单向扩展不碰索引，
    可以随时关掉做消融（评测里 bm25_nogloss / hybrid_nogloss 两列就是它）。
    英文提问不会被扩展（没有中文术语可命中），行为与旧版一致。
    """
    low = (text or "").lower()
    extra = []
    for zh, en in _ZH_EN_SEED.items():
        if zh in low:
            extra.extend(_EN_RE.findall(en))   # 复用英文正则，保证与索引侧切词同口径
    return list(dict.fromkeys(extra))          # 去重、保序


class BM25:
    """纯 Python BM25 稀疏检索器。"""

    def __init__(self, docs, k1=1.5, b=0.75):
        """
        参数：
          docs : 文档列表，每项是一段卡片文本（中英混排都可以）。
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

    def search(self, query, top_k=10, expand_zh=True):
        """返回 [(doc_id, score), ...]，按分数降序，最多 top_k 个。

        expand_zh=True 时除切词结果外再并入中文->英文词表扩展项；
        纯英文提问不受影响（命不中文术语，扩展项为空）。
        """
        terms = tokenize(query)
        if expand_zh:
            # 已经在切词结果里的不重复加，避免同一个词被算两遍分数
            terms.extend(t for t in expand_query(query) if t not in terms)

        scores = {}
        for t in terms:
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


if __name__ == "__main__":
    # 自测：纯标准库，不需要 numpy/pandas，也不需要联网。
    #   .venv\Scripts\python.exe -m omics_agent.bm25
    print("英文切词 :", tokenize("Which genes are involved in DNA repair?"))
    print("中文切词 :", tokenize("哪些基因参与血小板脱颗粒？"))
    print("中文扩展 :", expand_query("哪些基因参与血小板脱颗粒？"))

    demo_docs = [
        "P1: 参与通路 platelet degranulation , wound healing（共 2 个）",
        "P2: 参与通路 protein phosphorylation（共 1 个）",
    ]
    bm = BM25(demo_docs)

    en_q = "which genes are involved in platelet degranulation"
    print("英文提问 top2（不扩展）:", bm.search(en_q, top_k=2, expand_zh=False))
    print("英文提问 top2（默认扩展）:", bm.search(en_q, top_k=2))

    zh_q = "哪些基因参与血小板脱颗粒"
    print("中文提问 top2（不扩展，= 修复前的空转）:", bm.search(zh_q, top_k=2, expand_zh=False))
    print("中文提问 top2（默认扩展，= 修复后）:", bm.search(zh_q, top_k=2))
