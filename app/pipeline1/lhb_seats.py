# -*- coding: utf-8 -*-
"""龙虎榜席位静态分类 (KIMI LHB v2.0 spec §2.2/§2.3/§2.4, 2025-08 清单).

top_inst 返回每只上榜股票每日的全部席位及买卖金额 (exalter 为营业部名).
本模块将 exalter 映射到资金类型, 供席位回填 / 每日 fetch 聚合使用.

静态分类为 spec §6.4 明确许可的初期方案 (席位溢价数据库稳定前).
类别优先级: inst > top > quant > retail > other.
"""

from __future__ import annotations

# 机构专用席位 (Tushare top_inst 中 exalter 的规范名)
INST_SEAT = "机构专用"

# 量化席位 (spec §2.3): 华鑫证券上海分公司 + 华宝证券上海东大名路
QUANT_PATTERNS: list[tuple[str, str]] = [
    ("华鑫证券", "上海分公司"),
    ("华宝证券", "东大名路"),
]

# 散户/混合席位 (spec §2.4): 东财拉萨系 (名含"拉萨")
RETAIL_KEYWORD = "拉萨"

# 顶级游资席位 (spec §2.2, 2025 更新清单) — 按营业部分支关键字匹配.
# 券商合并后名称变化 (国泰君安→国泰海通), 故按分支名匹配而非全名.
TOP_BRANCH_KEYWORDS: list[str] = [
    "江苏路",  # 章盟主-上海江苏路
    "彩虹北路",  # 章盟主-宁波彩虹北路
    "延安路",  # 章盟主-杭州延安路
    "六一中路",  # 六一中路-福州六一中路
    "二纬路",  # 六一中路-天津东丽开发区二纬路
    "黄河路",  # 陈小群-大连黄河路
    "留园路",  # 陈小群-苏州留园路
    "朱雀大街",  # 方新侠-西安朱雀大街
    "凯滨路",  # 呼家楼-上海凯滨路
    "桑田路",  # 宁波桑田路
    "太平南路",  # 作手新一/小鳄鱼-南京太平南路
    "大钟亭",  # 小鳄鱼-南京大钟亭
    "中信大厦",  # 呼家楼-中信建投北京中信大厦
]


def classify_seat(exalter: str | None) -> str:
    """返回席位的资金类型: inst / top / quant / retail / other."""
    if exalter is None or not isinstance(exalter, str):
        return "other"
    name = exalter.strip()
    if name == INST_SEAT:
        return "inst"
    if any(k in name for k in TOP_BRANCH_KEYWORDS):
        return "top"
    if any(a in name and b in name for a, b in QUANT_PATTERNS):
        return "quant"
    if RETAIL_KEYWORD in name:
        return "retail"
    return "other"
