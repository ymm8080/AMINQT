"""【已退役 RETIRED】极简日更追加脚本 — 不再使用.

原功能 (2026-08-03 前): 仅追加当日 OHLCV + LHB + CYQ, 其余面板列全部置 pd.NA
(不 ffill 慢列、不计算 rolling/派生特征), 是 07-27/28 缺 300 股与 07-30 荒行
数据残缺的根源之一. 已退役, 文件保留仅供审计参考 (完整历史见 git).

正式日更增广改用仓库根目录的全量脚本:
    python _daily_fetch.py [YYYYMMDD]

_daily_fetch.py 会拉取全部源 + 计算 rolling/派生特征 + 恒 ffill 慢列,
并原子替换当日行 (WORM 备份先行).
"""

import sys

MSG = """
============================================================
 append_today_to_panel.py 已退役 (RETIRED)
============================================================
此脚本只追加 OHLCV/LHB/CYQ, 其余面板列留空, 会造成日更数据残缺.

请改用全量日更脚本 (根目录):
    python _daily_fetch.py [YYYYMMDD]

_daily_fetch.py 行为:
  - 拉取 OHLCV / adj / daily_basic / stk_limit / suspend / cyq_perf /
    margin_detail / top_list / stock_basic 全部源 (带重试)
  - 计算 rolling 特征 (bias/ma/vol_surge/...) 与 CYQ 派生列
  - 恒前向填充 ffill 标记的慢列 (announce_date/财务/增减持/申万 等)
  - 原子替换当日行, 不产生重复
============================================================
"""

if __name__ == "__main__":
    print(MSG)
    sys.exit(1)
