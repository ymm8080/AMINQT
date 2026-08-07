"""
替代数据源调研与可行性评估 (2026-07-27)
==========================================
目标: 对 10 类替代数据分别测试取数链路 → 评估覆盖度/数据质量/延迟 →
     独立训练 IC 贡献 → 决定是否纳入 PIPELINE1 特征引擎.

10 类数据:
  1. 北向资金        (Northbound Capital Flow)
  2. 融资融券        (Margin Trading & Short Selling)
  3. 基本面PIT       (Fundamental Point-in-Time)
  4. 主力资金        (Main Capital Flow)
  5. 机构龙虎榜      (Institutional Dragon-Tiger List)
  6. 主力资金持仓时序 (Main Capital Position Time Series)
  7. 游资席位净值     (Hot Money Seat Net Value)
  8. 股东增减持      (Shareholder Increase/Decrease Holdings)
  9. 股东户数变化     (Shareholder Count Changes)
 10. 股东平均股数变化 (Average Shares per Shareholder)

评估标准:
  - 数据可得性: 免费/低成本可获取 ✓
  - 覆盖度:    全A ≥80% 标的
  - 历史深度:   ≥3 年 (匹配当前训练窗口)
  - 更新频率:   日频 / 周频 / 季频
  - PIT 安全:   是否可严格做到 t-1 不前瞻
  - IC 贡献:    独立训练后 Rank IC ≥ 0.02 (弱有效)

数据源映射: Tushare (需积分≥2000) → AKShare (免费, 东财源) → Baostock (免费, 低频)

用法: python scripts/research_alternative_data.py [--fetch] [--train] [--all]
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── 数据源配置 ──────────────────────────────────────────────


@dataclass
class DataSourceSpec:
    """单个数据源的规格描述."""

    id: str  # 短标识符
    name: str  # 中文名称
    category: str  # 分类: 资金流/基本面/持仓/行为
    tushare_api: str | None  # Tushare 接口名 (None=无此接口)
    tushare_freq: str  # 更新频率
    tushare_points: int  # 最低积分要求
    akshare_func: str | None  # AKShare 函数名
    baostock_func: str | None  # Baostock 函数名
    pit_safe: bool  # 是否可严格 PIT
    snapshot_only: bool  # 是否仅有当日快照 (无历史)
    history_years: int  # 历史深度 (年)
    coverage_pct: float  # 估计全A覆盖率
    notes: str = ""


# ── 10 类数据源规格 ──────────────────────────────────────────

SOURCES: list[DataSourceSpec] = [
    DataSourceSpec(
        id="northbound",
        name="北向资金",
        category="资金流",
        tushare_api="moneyflow_hsgt",
        tushare_freq="日频 (交易日 19:00)",
        tushare_points=2000,
        akshare_func="stock_hsgt_north_net_flow_in_em / stock_hsgt_hist_em",
        baostock_func=None,
        pit_safe=True,
        snapshot_only=False,
        history_years=8,
        coverage_pct=0.15,  # 仅沪深港通标的(~1500只)
        notes="沪/深股通日度净买入额, 可区分沪市/深市. 关键特征: 连续净买入天数/累计净买入/占成交额比. Tushare 积分门槛2000.",
    ),
    DataSourceSpec(
        id="margin",
        name="融资融券",
        category="资金流",
        tushare_api="margin_detail",
        tushare_freq="日频 (次日 9:00)",
        tushare_points=2000,
        akshare_func="stock_margin_detail_sse / stock_margin_detail_szse",
        baostock_func="query_margin_detail",
        pit_safe=True,
        snapshot_only=False,
        history_years=14,
        coverage_pct=0.35,  # 两融标的(~1800只)
        notes="融资余额/融券余量日变化. 关键特征: 融资余额增速/融券占比/融资买入占成交比. Baostock 也有但仅限沪市.",
    ),
    DataSourceSpec(
        id="fundamental_pit",
        name="基本面PIT",
        category="基本面",
        tushare_api="fina_indicator",
        tushare_freq="随财报更新 (季频)",
        tushare_points=2000,
        akshare_func="stock_financial_analysis_indicator (东财财务指标)",
        baostock_func="query_growth_data (季频)",
        pit_safe=True,  # ann_date 可精确对齐
        snapshot_only=False,
        history_years=20,
        coverage_pct=0.98,
        notes="ROE/ROA/毛利率/净利率/营收增速/净利增速等, 含 ann_date (公告日) 可严格 PIT 对齐. 当前仅用 daily_basic 近似, 升级为 fina_indicator 可增加 20+ 因子. 季频更新, 日频填充需 merge_asof + ffill 但严格按 announce_date.",
    ),
    DataSourceSpec(
        id="main_money_flow",
        name="主力资金",
        category="资金流",
        tushare_api="moneyflow",
        tushare_freq="日频 (交易日 19:00)",
        tushare_points=2000,
        akshare_func="stock_individual_fund_flow_rank (已集成)",
        baostock_func=None,
        pit_safe=True,
        snapshot_only=False,
        history_years=5,
        coverage_pct=0.99,
        notes="已集成在 data_supply.fetch_money_flow(). 当前仅用「今日主力净流入-净额」和「超大单净流入-净额」. Tushare moneyflow 可补历史时序+小单/中单/大单细分. 升级路径: akshare 当日 → Tushare 补历史.",
    ),
    DataSourceSpec(
        id="lhb_inst",
        name="机构龙虎榜",
        category="行为",
        tushare_api="top_list + top_inst",
        tushare_freq="日频 (每日 20:00)",
        tushare_points=2000,
        akshare_func="stock_lhb_detail_em",
        baostock_func=None,
        pit_safe=True,
        snapshot_only=False,
        history_years=19,
        coverage_pct=0.05,  # 仅上榜股票, 日频稀疏
        notes="已部分集成 dim18_lhb. top_list=龙虎榜每日明细(上榜股票); top_inst=机构席位交易明细. 关键特征: 机构净买入额/机构上榜家数/机构买入占比. 数据稀疏(非全A日频), 需按前N日滚动聚合.",
    ),
    DataSourceSpec(
        id="inst_position_ts",
        name="主力资金持仓时序",
        category="持仓",
        tushare_api="fund_portfolio",
        tushare_freq="季频 (基金季报后)",
        tushare_points=2000,
        akshare_func="stock_fund_stock_holder (基金持股)",
        baostock_func=None,
        pit_safe=False,  # 季报延迟45天, 需严格按公告日对齐
        snapshot_only=False,
        history_years=10,
        coverage_pct=0.60,  # 有基金持仓的标的
        notes="公募基金重仓股季度持仓. 关键特征: 基金持仓家数变化/持仓市值变化/持仓占流通比. 季频数据, 需在公告日窗口内 ffill. PIT 风险: 季报截止日后45天才披露, 需严格控制公告日期.",
    ),
    DataSourceSpec(
        id="hot_money_seat",
        name="游资席位净值",
        category="行为",
        tushare_api="top_inst (衍生)",
        tushare_freq="日频",
        tushare_points=2000,
        akshare_func=None,  # 需从龙虎榜明细衍生
        baostock_func=None,
        pit_safe=True,
        snapshot_only=False,
        history_years=19,
        coverage_pct=0.02,  # 游资上榜标的极少
        notes="无直接的游资净值API. 需从 LHB 营业部席位数据中识别游资席位(非机构专用), 跟踪其买卖记录, 构建游资活跃度指标. 数据极稀疏, 作为条件特征(有LHB才激活).",
    ),
    DataSourceSpec(
        id="sh_holdings_change",
        name="股东增减持",
        category="行为",
        tushare_api="stk_holdertrade",
        tushare_freq="不定期 (公告后更新)",
        tushare_points=2000,
        akshare_func="stock_zh_a_gdhs_detail_em (十大股东/流通股东变动)",
        baostock_func="query_shareholder_change (仅限大股东)",
        pit_safe=False,  # 公告滞后, 需严格按公告日
        snapshot_only=False,
        history_years=10,
        coverage_pct=0.30,  # 有增减持公告的标的
        notes="大股东/高管增减持公告. 关键特征: 近N日净增持/减持金额/增持比例. 公告数据稀疏, 需按公告日 PIT 对齐. Baostock 也有但仅限前十大股东.",
    ),
    DataSourceSpec(
        id="sh_count_change",
        name="股东户数变化",
        category="行为",
        tushare_api="stk_holdernumber",
        tushare_freq="不定期 (季报/中报/年报后)",
        tushare_points=2000,
        akshare_func="stock_zh_a_gdhs_detail_em (股东户数详情)",
        baostock_func=None,
        pit_safe=False,  # 季报内披露, 需按公告日
        snapshot_only=False,
        history_years=10,
        coverage_pct=0.95,
        notes="股东户数(季频)+户均持股. 核心阿尔法信号: 股东户数减少=筹码集中=机构吸筹. 关键特征: 股东户数环比变化/同比变化/变化加速度. 已部分集成到 dim20_chip_proxy (OHLCV代理), 真实数据可显著提升准确率.",
    ),
    DataSourceSpec(
        id="avg_shares",
        name="股东平均股数变化",
        category="行为",
        tushare_api="衍生自 stk_holdernumber + 总股本",
        tushare_freq="不定期",
        tushare_points=2000,
        akshare_func="同上 + stock_info_a_code_name (总股本)",
        baostock_func=None,
        pit_safe=False,
        snapshot_only=False,
        history_years=10,
        coverage_pct=0.95,
        notes="户均持股 = 流通股本 / 股东户数. 比单纯股东户数更有意义(排除送转股扰动). 可直接从 stk_holdernumber 衍生, 无需额外API. 关键特征: 户均持股环比/户均持股集中度.",
    ),
]


# ── 数据拉取测试 ──────────────────────────────────────────────


def test_akshare_availability():
    """测试 akshare 各接口可用性 (不打全量数据, 仅验证接口可调通)."""
    results = {}
    try:
        import akshare as ak

        # 1. 北向资金
        try:
            df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
            results["northbound"] = f"✓ ({len(df)} 行, cols={list(df.columns)[:5]})"
        except Exception as e:
            results["northbound"] = f"✗ {e}"

        # 2. 融资融券 (沪市)
        try:
            df = ak.stock_margin_detail_sse(date="20260725")
            results["margin"] = f"✓ ({len(df)} 行, cols={list(df.columns)[:5]})"
        except Exception as e:
            results["margin"] = f"✗ {e}"

        # 3. 基本面PIT (东财财务指标)
        try:
            df = ak.stock_financial_analysis_indicator(
                symbol="000001", start_year="2024"
            )
            results["fundamental_pit"] = (
                f"✓ ({len(df)} 行, cols={list(df.columns)[:5]})"
            )
        except Exception as e:
            results["fundamental_pit"] = f"✗ {e}"

        # 4. 主力资金 (已集成)
        try:
            df = ak.stock_individual_fund_flow_rank(indicator="今日")
            results["main_money_flow"] = f"✓ ({len(df)} 行, 已集成)"
        except Exception as e:
            results["main_money_flow"] = f"✗ {e}"

        # 5. 龙虎榜
        try:
            df = ak.stock_lhb_detail_em(date="20260725")
            results["lhb_inst"] = f"✓ ({len(df)} 行, cols={list(df.columns)[:5]})"
        except Exception as e:
            results["lhb_inst"] = f"✗ {e}"

        # 6. 基金持仓
        try:
            df = ak.stock_fund_stock_holder(symbol="000001")
            results["inst_position_ts"] = (
                f"✓ ({len(df)} 行, cols={list(df.columns)[:5]})"
            )
        except Exception as e:
            results["inst_position_ts"] = f"✗ {e}"

        # 8. 股东增减持
        try:
            df = ak.stock_zh_a_gdhs_detail_em(symbol="000001")
            results["sh_holdings_change"] = (
                f"✓ ({len(df)} 行, cols={list(df.columns)[:5]})"
            )
        except Exception as e:
            results["sh_holdings_change"] = f"✗ {e}"

        # 9. 股东户数
        try:
            df = ak.stock_zh_a_gdhs_detail_em(symbol="000001")
            results["sh_count_change"] = (
                f"✓ ({len(df)} 行, cols={list(df.columns)[:5]})"
            )
        except Exception as e:
            results["sh_count_change"] = f"✗ {e}"

        # 7. 游资席位 (从 LHB 衍生)
        results["hot_money_seat"] = "需衍生: 从 LHB 明细识别游资席位"
        # 10. 股东平均股数 (衍生)
        results["avg_shares"] = "需衍生: 流通股本/股东户数"

    except ImportError:
        results["_error"] = "akshare 未安装"
    return results


def test_tushare_availability():
    """测试 Tushare 各接口可用性 (需 TUSHARE_TOKEN)."""
    token = settings.TUSHARE_TOKEN
    if not token:
        return {"_error": "TUSHARE_TOKEN 未配置"}
    results = {}
    try:
        import tushare as ts

        pro = ts.pro_api(token)

        # 测试各接口能否调通 (单只股票/单日)
        tests = {
            "moneyflow_hsgt": lambda: pro.moneyflow_hsgt(trade_date="20260725"),
            "margin_detail": lambda: pro.margin_detail(trade_date="20260725"),
            "fina_indicator": lambda: pro.fina_indicator(
                ts_code="000001.SZ", period="20251231"
            ),
            "moneyflow": lambda: pro.moneyflow(trade_date="20260725"),
            "top_list": lambda: pro.top_list(trade_date="20260725"),
            "top_inst": lambda: pro.top_inst(trade_date="20260725"),
            "fund_portfolio": lambda: pro.fund_portfolio(ts_code="000001.SZ"),
            "stk_holdertrade": lambda: pro.stk_holdertrade(ts_code="000001.SZ"),
            "stk_holdernumber": lambda: pro.stk_holdernumber(ts_code="000001.SZ"),
        }
        for name, fn in tests.items():
            try:
                df = fn()
                results[name] = (
                    f"✓ ({len(df)} 行)" if df is not None and len(df) > 0 else "✓ (空)"
                )
            except Exception as e:
                results[name] = f"✗ {e}"
    except ImportError:
        results["_error"] = "tushare 未安装"
    except Exception as e:
        results["_error"] = str(e)
    return results


# ── 决策矩阵 ─────────────────────────────────────────────────


def print_decision_matrix():
    """打印完整的纳入决策矩阵."""
    print("=" * 140)
    print("替代数据源 — 纳入 PIPELINE1 特征引擎决策矩阵")
    print("评估日期: 2026-07-27 | 当前特征维度: 21 (dim01-dim21)")
    print("=" * 140)

    header = (
        f"{'ID':<20s} {'名称':<14s} {'可得性':<10s} {'覆盖%':<7s} "
        f"{'历史':<6s} {'频率':<6s} {'PIT':<5s} {'优先级':<10s} {'决策':<10s} {'理由'}"
    )
    print(header)
    print("-" * 140)

    decisions = [
        (
            "fundamental_pit",
            "基本面PIT",
            "T+A",
            "98%",
            ">10y",
            "季频",
            "是",
            "P0-立即",
            "纳入",
            "daily_basic→fina_indicator升级,ann_date+20+财务因子",
        ),
        (
            "main_money_flow",
            "主力资金",
            "已集成",
            "99%",
            ">5y",
            "日频",
            "是",
            "P1-升级",
            "升级",
            "东财快照→Tushare moneyflow历史回填,四层资金滚动特征",
        ),
        (
            "sh_count_change",
            "股东户数变化",
            "T+A",
            "95%",
            ">10y",
            "季频",
            "否*",
            "P0-立即",
            "纳入",
            "最强筹码集中度信号,dim20仅OHLCV代理,真实数据IC预期+0.01-0.03",
        ),
        (
            "avg_shares",
            "股东平均股数",
            "衍生",
            "95%",
            ">10y",
            "季频",
            "否*",
            "P0-自动",
            "纳入",
            "从stk_holdernumber自动衍生,零额外API成本",
        ),
        (
            "northbound",
            "北向资金",
            "T+A",
            "15%",
            ">8y",
            "日频",
            "是",
            "P1-板块限定",
            "纳入",
            "仅沪深港通标的,独立训练或条件特征(board过滤)",
        ),
        (
            "margin",
            "融资融券",
            "T+A+B",
            "35%",
            ">14y",
            "日频",
            "是",
            "P1-板块限定",
            "纳入",
            "仅两融标的,融资余额增速=高质量情绪信号,独立训练",
        ),
        (
            "lhb_inst",
            "机构龙虎榜",
            "T+A",
            "5%",
            ">19y",
            "日频",
            "是",
            "P2-条件",
            "条件纳入",
            "dim18已占位,数据稀疏但信号质量高(机构专用席位买入)",
        ),
        (
            "sh_holdings_change",
            "股东增减持",
            "T+A+B",
            "30%",
            ">10y",
            "不定期",
            "否*",
            "P2-条件",
            "条件纳入",
            "内部人信号高质量,但稀疏+公告滞后,PIT对齐复杂",
        ),
        (
            "inst_position_ts",
            "机构持仓时序",
            "T+A",
            "60%",
            ">10y",
            "季频",
            "否*",
            "P2-条件",
            "待定",
            "季频+45天延迟实时价值有限,先观察IC",
        ),
        (
            "hot_money_seat",
            "游资席位净值",
            "需衍生",
            "2%",
            ">19y",
            "日频",
            "是",
            "P3-观察",
            "暂缓",
            "无现成API,需LHB营业部识别游资+净值时序,工程量大",
        ),
    ]

    for i, name, avail, cov, hist, freq, pit, pri, dec, reason in decisions:
        print(
            f"{i:<20s} {name:<12s} {avail:<10s} {cov:<7s} "
            f"{hist:<6s} {freq:<6s} {pit:<5s} {pri:<10s} {dec:<10s} {reason}"
        )

    print("-" * 140)
    print(
        "* PIT 标注: 季频/不定期数据公告滞后, 需严格按 announce_date/公告日 merge_asof, 不可直接用报告期."
    )
    print()
    print("优先级说明:")
    print("  P0-立即:  本周即纳入训练 → 新增 dim22-27")
    print("  P1-升级:  对已有数据源做历史补全 + 特征升级")
    print("  P2-条件:  独立训练验证 IC 后纳入 → 如 IC>0.02 则纳入")
    print("  P3-观察:  等数据源成熟/工程积累后再评估")


# ── 实施路线图 ─────────────────────────────────────────────────


def print_implementation_roadmap():
    """打印分步实施计划."""
    print()
    print("=" * 140)
    print("实施路线图 (4 阶段)")
    print("=" * 140)

    stages = [
        (
            "阶段0: 基础设施 (1-2天)",
            [
                "在 data_supply.py 新增 6 个 fetcher 方法: fetch_northbound / fetch_margin / fetch_fina_indicator / fetch_lhb / fetch_holdernumber / fetch_holdertrade",
                "每个 fetcher 遵循现有模式: Tushare 主源 + AKShare 降级 + 缓存 parquet",
                "在 panel_builder.py 注册可选 merge 步骤 (默认不阻断训练, 数据缺失填 NaN)",
            ],
        ),
        (
            "阶段1: P0 立即纳入 (2-3天)",
            [
                "dim22_fundamental_pit: fina_indicator PIT 对齐 (roe/roa/gross_margin/net_margin/eps_yoy/rev_yoy/profit_yoy/op_cashflow/debt_ratio/current_ratio...)",
                "dim23_shareholder_structure: 股东户数环比/同比/加速度 + 户均持股 + 户均持股集中度",
                "dim24_margin_trading: 融资余额增速/融券余额占比/融资买入占成交比/融资余额MA偏离",
                "训练: 全量面板 + IC 筛选裁决, 输出 factor_registry/factors_dual_2026W3x_alt_v1.json",
            ],
        ),
        (
            "阶段2: P1 升级+P2 验证 (3-5天)",
            [
                "P1-升级: main_money_flow Tushare 历史回填 (超大单/大单/中单/小单四层资金流滚动特征)",
                "P1-纳入: northbound 北向资金维度 (连续净买入天数/累计净买入/占成交额比/沪深分化)",
                "P2-验证: lhb_inst / sh_holdings_change 独立训练, 观察 IC (每个训练窗口自动裁决)",
                "P2-验证: inst_position_ts 机构持仓变化独立训练, IC>0.02 → 纳入",
            ],
        ),
        (
            "阶段3: P3 观察 (持续)",
            [
                "游资席位净值: 等待 community 数据源成熟或自行从 LHB 营业部数据构建",
                "AI 新闻情绪/产业链图谱/ESG 评级: 等 tushare 积分充足后评估",
            ],
        ),
    ]

    for title, items in stages:
        print(f"\n  {title}")
        for item in items:
            print(f"    - {item}")


# ── 特征设计预览 ───────────────────────────────────────────────


def print_feature_preview():
    """预览拟新增的维度及其特征列."""
    print()
    print("=" * 140)
    print("拟新增特征维度 — 列清单预览")
    print("=" * 140)

    dims = {
        "dim22_fundamental_pit (基本面PIT)": [
            "roe",
            "roa",
            "gross_margin",
            "net_margin",
            "eps_yoy",
            "rev_yoy",
            "profit_yoy",
            "op_cf_ratio",
            "debt_ratio",
            "current_ratio",
            "asset_turnover",
            "inventory_turnover",
            "pe_ttm_pit",
            "pb_lf_pit",
            "roe_5y_cv",
            "rev_3y_cagr",
            "accrual_ratio",
            "→ 17 因子, 季频 + merge_asof PIT 对齐到日频",
        ],
        "dim23_shareholder_structure (股东结构)": [
            "holder_count",
            "holder_count_qoq",
            "holder_count_yoy",
            "holder_count_qoq_accel",  # 二阶导: 减少速度加快=筹码加速集中
            "avg_shares_per_holder",
            "avg_shares_qoq",
            "avg_shares_yoy",
            "holder_concentration_zscore",  # 截面标准化后的集中度
            "→ 8 因子, 季频 → 日频 merge_asof",
        ],
        "dim24_margin_trading (融资融券)": [
            "margin_balance",
            "margin_balance_chg_1d",
            "margin_balance_chg_5d",
            "short_balance_ratio",  # 融券/融资比
            "margin_buy_ratio",  # 融资买入额/成交额
            "margin_balance_ma_deviation",  # 融资余额偏离20日均线
            "margin_balance_yoy",
            "→ 7 因子, 日频 (仅两融标的, 非两融填 NaN)",
        ],
        "dim25_northbound (北向资金)": [
            "north_net_buy",
            "north_net_buy_5d",
            "north_net_buy_20d",
            "north_consecutive_buy_days",  # 连续净买入天数
            "north_buy_ratio",  # 北向买入额/成交额
            "north_hold_pct",
            "north_hold_pct_chg_5d",  # 北向持股占比及变化
            "→ 7 因子, 日频 (仅沪深港通标的)",
        ],
        "dim26_lhb_enhanced (龙虎榜增强)": [
            "lhb_inst_net_buy_5d",
            "lhb_inst_net_buy_20d",
            "lhb_inst_count_5d",  # 近5日机构席位上榜次数
            "lhb_inst_buy_ratio",  # 机构买入占比
            "lhb_abnormal_days",  # 异常上榜天数 (偏离均值2σ)
            "→ 5 因子, 日频稀疏 → 前N日滚动聚合 (dim18 增强版)",
        ],
        "dim10_upgraded (资金流升级)": [
            "super_large_net_5d",
            "large_net_5d",
            "medium_net_5d",
            "small_net_5d",
            "super_large_ratio",  # 超大单/总成交
            "main_flow_diversion",  # 大单-小单背离 (机构vs散户)
            "main_flow_ma_deviation",  # 主力净流入偏离20日均值
            "→ 7 因子 (dim10 现有 2 因子升级为 7)",
        ],
    }

    for dim, cols in dims.items():
        print(f"\n  {dim}:")
        print(f"    {cols[0]}")
        if len(cols) > 1:
            print(f"    {cols[1]}")


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="替代数据源调研")
    p.add_argument("--fetch", action="store_true", help="测试数据源可用性")
    p.add_argument(
        "--train", action="store_true", help="独立训练各数据源评估 IC (需完整面板)"
    )
    p.add_argument("--all", action="store_true", help="执行全部步骤")
    args = p.parse_args()

    if not any([args.fetch, args.train, args.all]):
        # 默认仅打印报告
        print_decision_matrix()
        print_feature_preview()
        print_implementation_roadmap()
        print()
        print("用法:")
        print(
            "  python scripts/research_alternative_data.py              # 打印决策矩阵"
        )
        print(
            "  python scripts/research_alternative_data.py --fetch      # 测试数据源可用性"
        )
        print(
            "  python scripts/research_alternative_data.py --train      # 独立训练评估 IC"
        )
        print("  python scripts/research_alternative_data.py --all        # 全部步骤")
        sys.exit(0)

    if args.fetch or args.all:
        print("=" * 80)
        print("AKShare 接口可用性测试")
        print("=" * 80)
        ak_results = test_akshare_availability()
        for k, v in ak_results.items():
            print(f"  {k:<25s}: {v}")

        print()
        print("=" * 80)
        print("Tushare 接口可用性测试")
        print("=" * 80)
        ts_results = test_tushare_availability()
        for k, v in ts_results.items():
            print(f"  {k:<25s}: {v}")

    if args.train or args.all:
        print()
        print("=" * 80)
        print("独立训练评估: 需要完整面板数据 → 由 run_full_train.py 驱动")
        print("此功能需在 panel_full_enriched.parquet 就绪后运行")
        print("=" * 80)
        print("TODO: 各数据源独立训练 + IC 对比 → 见 scripts/run_alt_data_train.py")
