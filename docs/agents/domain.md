# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`AGENTS.md`** at the repo root — primary context: iron rules, architecture, protocols, data format mapping.
- **`10_adr/`** — Architecture Decision Records. Read ADRs that touch the area you're about to work in.
  - `ADR-001-lstm-over-random-forest.md`
  - `ADR-002-akshare-over-tushare.md`
  - `ADR-003-memory-cache-for-stock-selection.md`
  - `ADR-004-sim-executor-for-broker-abstraction.md`
  - `ADR-TEMPLATE.md` (for new ADRs)
- **`rules/README.md`** — rule index; load relevant rules by task trigger.

If any file doesn't exist, **proceed silently**. Don't flag its absence.

## File structure

Single-context repo:

```
/
├── AGENTS.md              ← primary context (iron rules, architecture, protocols)
├── CLAUDE.md              ← Claude Code config (skills, tool chain)
├── rules/                 ← detailed rules (.md, IDE-agnostic)
├── 10_adr/                ← ADRs
│   ├── ADR-001-lstm-over-random-forest.md
│   ├── ADR-002-akshare-over-tushare.md
│   ├── ADR-003-memory-cache-for-stock-selection.md
│   └── ADR-004-sim-executor-for-broker-abstraction.md
├── app/
├── services/
├── data/
└── scripts/
```

## Use the glossary's vocabulary

No `CONTEXT.md` exists yet. Domain vocabulary is defined in `AGENTS.md` sections (e.g. "选股流程", "风控硬约束", "数据格式映射"). Use those terms as defined.

If a concept isn't documented, note it — either it's a gap or new language the project doesn't use.

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly:

> _Contradicts ADR-001 (LSTM over Random Forest) — but worth reopening because…_
