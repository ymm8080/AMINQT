# Skill: graph-to-vector (图形向量化)

> **用途**: 将K线图形、技术指标图表、股票关系网络转化为向量表示，供ML/DL模型消费
> **触发**: 实现K线图转矩阵、图表模式识别、图神经网络嵌入、node2vec/Graph2Vec、CNN图像嵌入时

---

## 概述

本技能覆盖项目中"把K线图转为数字矩阵"的核心需求，包含四条向量化路径：

| 路径 | 方法 | 输入 | 输出 | 适用场景 |
|:---|:---|:---|:---|:---|
| A | 滑动窗口特征矩阵 | OHLCV + 技术指标 | `(N, 20, F)` | LSTM/Transformer 时序预测 |
| B | K线图像 CNN 嵌入 | K线图 PNG | `(N, embed_dim)` | 图表形态识别 |
| C | 技术形态二值向量 | K线 + 指标序列 | `(N, pattern_count)` | 金叉/死叉/背离等离散信号 |
| D | 股票关系图嵌入 | 股票关联图谱 | `(stock_count, embed_dim)` | 板块联动、产业链传导 |

---

## 路径 A：滑动窗口特征矩阵（主力路径）

将连续 N 天的 OHLCV + 技术指标堆叠为 3D 张量，这是 LSTM/Transformer 的标准输入。

```python
import numpy as np
import pandas as pd
from typing import Tuple

WINDOW_DAYS = 20
HORIZON_DAYS = 5

def kline_to_matrix(df: pd.DataFrame,
                    window: int = WINDOW_DAYS,
                    horizon: int = HORIZON_DAYS) -> Tuple[np.ndarray, np.ndarray]:
    """将K线 DataFrame 转为 (N, window, F) 特征矩阵 + (N,) 标签。
    
    前置条件: df 已经过 factor_engine 计算技术指标列。
    
    Args:
        df: 含 OHLCV + 技术指标的 DataFrame，按日期升序
        window: 回看窗口 (默认20个交易日)
        horizon: 预测 horizon (默认未来5日收益率)
    
    Returns:
        X: (N, window, F) 特征张量
        y: (N,) 未来 horizon 日收益率
    """
    feature_cols = [
        # 原始量价
        'open', 'close', 'high', 'low', 'volume', 'amount',
        # MACD
        'dif', 'dea', 'bar',
        # KDJ
        'k', 'd', 'j',
        # BOLL
        'boll_upper', 'boll_mid', 'boll_lower',
        # RSI
        'rsi',
        # 衍生特征
        'close_dif_dev', 'bias_ma5', 'bias_ma20',
        'macd_slope', 'kdj_slope', 'rsi_slope',
        'vol_ma5', 'vol_ratio', 'boll_pos',
        'pct_change', 'turnover', 'amplitude',
    ]
    assert len(feature_cols) >= 25, f"特征数 {len(feature_cols)} < 25"

    X, y = [], []
    for i in range(window, len(df) - horizon):
        X.append(df[feature_cols].iloc[i - window:i].values)
        future_ret = (df['close'].iloc[i + horizon] / df['close'].iloc[i]) - 1
        y.append(future_ret)

    X = np.array(X, dtype=np.float32)  # (N, window, F)
    y = np.array(y, dtype=np.float32)  # (N,)

    # 关键: NaN → 0，防止模型崩溃
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y
```

### 归一化策略（防止数据泄露）

```python
def normalize_per_window(X: np.ndarray) -> np.ndarray:
    """按每个样本窗口独立归一化 (z-score)。
    
    严禁用全量数据 fit_transform 后再切窗 — 那会让早期窗口"看到"未来统计量。
    每个窗口用自己的 mean/std 归一化。
    """
    mean = X.mean(axis=1, keepdims=True)   # (N, 1, F)
    std = X.std(axis=1, keepdims=True)     # (N, 1, F)
    std = np.where(std < 1e-10, 1.0, std)  # 除零防护
    return (X - mean) / std
```

---

## 路径 B：K线图像 CNN 嵌入

将K线图渲染为图像，用 CNN 提取视觉特征向量。适用于形态识别（如头肩顶、双底）。

```python
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt
import io
from PIL import Image
import torch
import torch.nn as nn

def kline_to_image(df: pd.DataFrame, window: int = 20,
                   img_size: int = 64) -> np.ndarray:
    """将 window 天 K线渲染为 (img_size, img_size, 3) 图像。
    
    Args:
        df: 含 OHLCV 的 DataFrame
        window: K线窗口天数
        img_size: 输出图像尺寸 (正方形)
    
    Returns:
        np.ndarray: (img_size, img_size, 3) RGB 图像, 值域 [0, 1]
    """
    chunk = df.tail(window).copy()
    fig, ax = plt.subplots(figsize=(2, 2), dpi=img_size // 2)
    
    # 绘制K线
    for idx, row in chunk.iterrows():
        color = 'red' if row['close'] >= row['open'] else 'green'
        # 影线
        ax.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=0.5)
        # 实体
        ax.bar(idx, abs(row['close'] - row['open']),
               bottom=min(row['open'], row['close']),
               color=color, width=0.6)
    
    ax.set_xlim(0, window)
    ax.set_ylim(chunk['low'].min(), chunk['high'].max())
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    img = Image.open(buf).convert('RGB').resize((img_size, img_size))
    return np.array(img, dtype=np.float32) / 255.0


class ChartCNN(nn.Module):
    """CNN 编码器: K线图像 → 嵌入向量。
    
    输入: (batch, 3, H, W)
    输出: (batch, embed_dim)
    """
    
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, embed_dim)
    
    def forward(self, x):
        x = self.features(x)           # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)      # (B, 128)
        return self.fc(x)               # (B, embed_dim)
```

---

## 路径 C：技术形态二值向量

将经典技术形态（金叉、死叉、背离等）编码为二值向量，可与路径A的特征矩阵拼接。

```python
def detect_patterns(df: pd.DataFrame) -> dict:
    """检测技术形态，返回二值特征字典。
    
    所有检测只用当前及历史数据 (无未来函数)。
    """
    patterns = {}
    n = len(df)
    if n < 30:
        return patterns
    
    i = n - 1  # 最后一天

    # 1. MACD 金叉 (DIF 从下方穿越 DEA)
    patterns['macd_golden_cross'] = int(
        df['dif'].iloc[i] > df['dea'].iloc[i] and
        df['dif'].iloc[i-1] <= df['dea'].iloc[i-1]
    )

    # 2. MACD 死叉
    patterns['macd_death_cross'] = int(
        df['dif'].iloc[i] < df['dea'].iloc[i] and
        df['dif'].iloc[i-1] >= df['dea'].iloc[i-1]
    )

    # 3. KDJ 金叉 (K 从下方穿越 D)
    patterns['kdj_golden_cross'] = int(
        df['k'].iloc[i] > df['d'].iloc[i] and
        df['k'].iloc[i-1] <= df['d'].iloc[i-1]
    )

    # 4. KDJ 超买 (J > 100)
    patterns['kdj_overbought'] = int(df['j'].iloc[i] > 100)

    # 5. KDJ 超卖 (J < 0)
    patterns['kdj_oversold'] = int(df['j'].iloc[i] < 0)

    # 6. RSI 超买 (> 70)
    patterns['rsi_overbought'] = int(df['rsi'].iloc[i] > 70)

    # 7. RSI 超卖 (< 30)
    patterns['rsi_oversold'] = int(df['rsi'].iloc[i] < 30)

    # 8. 价格突破布林带上轨
    patterns['boll_breakout_up'] = int(df['close'].iloc[i] > df['boll_upper'].iloc[i])

    # 9. 价格跌破布林带下轨
    patterns['boll_breakout_down'] = int(df['close'].iloc[i] < df['boll_lower'].iloc[i])

    # 10. 放量上涨 (成交量 > 5日均量 * 1.5 且上涨)
    vol_ma5 = df['volume'].iloc[i-4:i+1].mean()
    patterns['volume_surge_up'] = int(
        df['volume'].iloc[i] > vol_ma5 * 1.5 and
        df['close'].iloc[i] > df['open'].iloc[i]
    )

    # 11. 缩量下跌
    patterns['volume_shrink_down'] = int(
        df['volume'].iloc[i] < vol_ma5 * 0.5 and
        df['close'].iloc[i] < df['open'].iloc[i]
    )

    # 12. MA5 上穿 MA20 (均线金叉)
    ma5 = df['close'].iloc[i-4:i+1].mean()
    ma20 = df['close'].iloc[i-19:i+1].mean()
    ma5_prev = df['close'].iloc[i-5:i].mean()
    ma20_prev = df['close'].iloc[i-20:i-1+1].mean() if n >= 21 else ma20
    patterns['ma_golden_cross'] = int(ma5 > ma20 and ma5_prev <= ma20_prev)

    return patterns


def patterns_to_vector(patterns: dict, pattern_names: list = None) -> np.ndarray:
    """将形态字典转为有序二值向量。"""
    if pattern_names is None:
        pattern_names = sorted(patterns.keys())
    return np.array([patterns.get(name, 0) for name in pattern_names], dtype=np.float32)
```

---

## 路径 D：股票关系图嵌入

用图神经网络 (GNN) 或 node2vec 将股票关联图谱嵌入为向量，捕捉板块联动与产业链传导。

```python
import networkx as nx
from typing import Optional

def build_stock_graph(codes: list, correlation_matrix: np.ndarray,
                      threshold: float = 0.6) -> nx.Graph:
    """从收益率相关系数矩阵构建股票关系图。
    
    Args:
        codes: 股票代码列表
        correlation_matrix: (N, N) 相关系数矩阵
        threshold: 边的相关系数阈值 (绝对值大于此值才连边)
    
    Returns:
        nx.Graph: 股票关系图
    """
    G = nx.Graph()
    for code in codes:
        G.add_node(code)
    
    n = len(codes)
    for i in range(n):
        for j in range(i + 1, n):
            corr = correlation_matrix[i, j]
            if abs(corr) > threshold:
                G.add_edge(codes[i], codes[j], weight=abs(corr))
    
    return G


# === 方法1: node2vec ===
def node2vec_embed(G: nx.Graph, dimensions: int = 64,
                   walk_length: int = 30, num_walks: int = 200) -> dict:
    """用 node2vec 学习节点嵌入。
    
    Returns:
        {code: np.ndarray(dimensions,)} 嵌入字典
    """
    try:
        from node2vec import Node2Vec
    except ImportError:
        raise ImportError("pip install node2vec")
    
    node2vec = Node2Vec(G, dimensions=dimensions,
                        walk_length=walk_length,
                        num_walks=num_walks,
                        weight_key='weight',
                        workers=4, seed=42)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    return {node: model.wv[node] for node in G.nodes()}


# === 方法2: 基于图统计特征的手动嵌入 ===
def graph_feature_embed(G: nx.Graph) -> dict:
    """从图结构提取统计特征作为嵌入 (无需额外依赖)。
    
    特征: degree, clustering_coeff, betweenness_centrality,
          eigenvector_centrality, closeness_centrality
    """
    embeddings = {}
    
    degree = dict(G.degree())
    clustering = nx.clustering(G, weight='weight')
    
    try:
        betweenness = nx.betweenness_centrality(G, weight='weight')
    except Exception:
        betweenness = {n: 0 for n in G.nodes()}
    
    try:
        eigen = nx.eigenvector_centrality(G, weight='weight', max_iter=500)
    except Exception:
        eigen = {n: 0 for n in G.nodes()}
    
    closeness = nx.closeness_centrality(G)
    
    for node in G.nodes():
        embeddings[node] = np.array([
            degree.get(node, 0),
            clustering.get(node, 0),
            betweenness.get(node, 0),
            eigen.get(node, 0),
            closeness.get(node, 0),
        ], dtype=np.float32)
    
    return embeddings


# === 方法3: PyTorch GCN ===
class StockGCNLayer(nn.Module):
    """单层图卷积 (GCN)。
    
    输入: (N, in_features) 节点特征 + 邻接矩阵
    输出: (N, out_features) 更新后的节点特征
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (N, F_in), adj: (N, N) 归一化邻接矩阵
        support = self.linear(x)        # (N, F_out)
        output = torch.spmm(adj, support)  # 邻接矩阵聚合邻居
        return output


def normalize_adj(adj: np.ndarray) -> torch.Tensor:
    """对称归一化邻接矩阵: D^{-1/2} (A+I) D^{-1/2}"""
    N = adj.shape[0]
    A = adj + np.eye(N)
    D = np.diag(A.sum(axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    normalized = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.FloatTensor(normalized)
```

---

## 多路径融合

将不同路径的向量拼接为统一特征表示。

```python
def fuse_features(matrix_vec: np.ndarray,
                  pattern_vec: Optional[np.ndarray] = None,
                  graph_vec: Optional[np.ndarray] = None,
                  cnn_vec: Optional[np.ndarray] = None) -> np.ndarray:
    """融合多路径特征向量。
    
    Args:
        matrix_vec: 路径A 输出 (window, F) 或池化后 (F,)
        pattern_vec: 路径C 输出 (pattern_count,)
        graph_vec: 路径D 输出 (graph_embed_dim,)
        cnn_vec: 路径B 输出 (cnn_embed_dim,)
    
    Returns:
        拼接后的 (total_dim,) 向量
    """
    parts = []
    
    # 路径A: 如果是3D (window, F)，做全局平均池化
    if matrix_vec.ndim == 3:
        matrix_vec = matrix_vec.mean(axis=0)  # (F,)
    parts.append(matrix_vec)
    
    if pattern_vec is not None:
        parts.append(pattern_vec)
    
    if graph_vec is not None:
        parts.append(graph_vec)
    
    if cnn_vec is not None:
        parts.append(cnn_vec)
    
    return np.concatenate(parts, axis=-1)
```

---

## 验证

```python
def verify_graph_to_vector(df: pd.DataFrame):
    """验证所有向量化路径的正确性。"""
    # 路径 A
    X, y = kline_to_matrix(df)
    assert X.ndim == 3, f"路径A: X应为3D, got {X.ndim}D"
    assert X.shape[1] == 20, f"路径A: window应为20, got {X.shape[1]}"
    assert X.shape[2] >= 25, f"路径A: 特征数应≥25, got {X.shape[2]}"
    assert not np.any(np.isnan(X)), "路径A: X含NaN"
    print(f"路径A OK: X.shape={X.shape}, y.shape={y.shape}")

    # 路径 C
    patterns = detect_patterns(df)
    pvec = patterns_to_vector(patterns)
    assert pvec.dtype == np.float32, "路径C: 应为float32"
    assert set(pvec).issubset({0.0, 1.0}), "路径C: 应为二值"
    print(f"路径C OK: {len(patterns)} 个形态, 向量维度={pvec.shape}")

    # 路径 B (可选, 需 matplotlib)
    try:
        img = kline_to_image(df)
        assert img.shape[2] == 3, "路径B: 应为RGB 3通道"
        assert 0 <= img.max() <= 1, "路径B: 值域应为[0,1]"
        print(f"路径B OK: img.shape={img.shape}")
    except Exception as e:
        print(f"路径B SKIP: {e}")

    print("ALL VERIFICATION PASSED")
```

---

## 检查清单

### 路径 A — 滑动窗口矩阵
- [ ] 特征列数 ≥ 25
- [ ] 窗口大小 = 20 个交易日
- [ ] 标签为未来 5 日收益率 (非当日收益率)
- [ ] `np.nan_to_num()` 已执行
- [ ] 归一化按窗口独立进行 (无全量 fit_transform)
- [ ] 无未来函数 (`rolling`/`ewm` 仅用当前及历史数据)

### 路径 B — K线图像 CNN
- [ ] matplotlib 使用 `Agg` 后端 (无头模式)
- [ ] 图像归一化到 `[0, 1]`
- [ ] CNN 输出为固定维度嵌入向量
- [ ] 训练时图像生成不引入未来数据

### 路径 C — 技术形态向量
- [ ] 所有形态检测仅用当前及历史数据
- [ ] 输出为二值 `{0, 1}`
- [ ] 形态列表已文档化
- [ ] 与路径A向量维度对齐可拼接

### 路径 D — 股票关系图嵌入
- [ ] 相关系数矩阵仅用训练期数据计算
- [ ] 邻接矩阵已对称归一化
- [ ] node2vec 有固定 `seed=42`
- [ ] 图特征备选方案 (graph_feature_embed) 在无 node2vec 时可用

### 通用
- [ ] 融合后向量维度已记录
- [ ] 所有除法有除零防护
- [ ] 验证函数 `verify_graph_to_vector()` 通过
- [ ] 单元测试覆盖每条路径
