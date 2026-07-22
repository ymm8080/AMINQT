/** 交易页: Pipeline-2 框架占位 (设计未定稿). */
export function TradingPage() {
  return (
    <>
      <h2>交易看板 · Pipeline 2</h2>
      <div className="panel">
        <p>
          <span className="badge">设计未定稿</span>
        </p>
        <p className="dim">
          Pipeline-2 (5分钟模型, 盘中 9:15~15:00 每 2 分钟) 设计尚未完成。
          本页预留: 左栏 行情/五档盘口 · 中栏 交易状态机 + 信号列表 · 右栏 持仓/委托/成交。
          规则引擎 v2 (L0-L4, 卖出状态机 P1-P12) 已就绪, 待 P2 定稿后接入实时信号。
        </p>
      </div>
    </>
  )
}
