"""
ABC Test Orchestrator — launches all 12 agents and collects results.
"""
import subprocess, sys, json, os, time
from datetime import datetime

AGENTS = [
    # variant, solution, board
    ('A', 'dedup_l2', 'main'),
    ('A', 'dedup_l2', 'dual'),
    ('A', 'gate_d_v2', 'main'),
    ('A', 'gate_d_v2', 'dual'),
    ('B', 'dedup_l2', 'main'),
    ('B', 'dedup_l2', 'dual'),
    ('B', 'gate_d_v2', 'main'),
    ('B', 'gate_d_v2', 'dual'),
    ('C', 'dedup_l2', 'main'),
    ('C', 'dedup_l2', 'dual'),
    ('C', 'gate_d_v2', 'main'),
    ('C', 'gate_d_v2', 'dual'),
]

HARNESS = "_abc_test_harness.py"
OUT_DIR = "data/abc_test_results"

print("=" * 80)
print("ABC TEST ORCHESTRATOR — 12 Agents")
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Run max 4 concurrent to avoid thrashing
MAX_CONCURRENT = 4
results = {}
processes = {}

agent_idx = 0
while agent_idx < len(AGENTS) or processes:
    # Start new agents if slots available
    while len(processes) < MAX_CONCURRENT and agent_idx < len(AGENTS):
        variant, solution, board = AGENTS[agent_idx]
        label = f"{variant}_{solution}_{board}"
        logfile = os.path.join(OUT_DIR, f"{label}.log")

        prebuilt = os.path.join(OUT_DIR, f"prebuilt_{board}.parquet")
        cmd = [sys.executable, '-u', HARNESS,
               '--variant', variant,
               '--solution', solution,
               '--board', board,
               '--prebuilt', prebuilt]

        print(f"\n[{len(processes)+1}/{MAX_CONCURRENT} running] Launching: {label}")
        proc = subprocess.Popen(cmd, stdout=open(logfile, 'w'), stderr=subprocess.STDOUT)
        processes[label] = (proc, agent_idx, time.time())
        agent_idx += 1

    # Check for completed processes
    done = []
    for label, (proc, idx, start_t) in list(processes.items()):
        ret = proc.poll()
        if ret is not None:
            elapsed = time.time() - start_t
            done.append(label)

            # Parse result
            logfile = os.path.join(OUT_DIR, f"{label}.log")
            result = None
            try:
                with open(logfile) as f:
                    for line in f:
                        if 'ABC_RESULT_JSON:' in line:
                            result = json.loads(line.split('ABC_RESULT_JSON:', 1)[1].strip())
            except Exception as e:
                pass

            if result:
                results[label] = result
                status = f"IC={result['oos_ic']:+.5f} ICIR={result['oos_icir']:.4f} " \
                         f"Sharpe={result['top10_sharpe']:.2f} Composite={result['composite_score']:.4f}"
            else:
                results[label] = {'label': label, 'error': f'exit={ret}', 'elapsed_s': elapsed}
                status = f"FAILED (exit={ret})"

            print(f"  [{label}] DONE in {elapsed:.0f}s — {status}")
            del processes[label]

    if processes:
        time.sleep(2)

# ── Sort and display results ──
print("\n" + "=" * 80)
print("ABC TEST RESULTS — All 12 Agents")
print("=" * 80)

# Summary table sorted by composite score
print(f"\n{'Rank':<5} {'Agent':<28} {'Board':<6} {'IC':>8} {'ICIR':>8} {'Sharpe':>8} {'WinRate':>8} {'Composite':>10} {'Feats':>6} {'Time':>6}")
print("-" * 110)

ranked = sorted(
    [(k, v) for k, v in results.items() if 'composite_score' in v],
    key=lambda x: -x[1]['composite_score']
)

for rank, (label, r) in enumerate(ranked, 1):
    print(f"{rank:<5} {label:<28} {r['board']:<6} "
          f"{r['oos_ic']:>+8.5f} {r['oos_icir']:>8.4f} "
          f"{r['top10_sharpe']:>8.2f} {r['top10_win_rate']:>8.1%} "
          f"{r['composite_score']:>10.4f} {r['n_feat_used']:>6} {r['elapsed_s']:>5.0f}s")

# Also show any errors
errors = [(k, v) for k, v in results.items() if 'error' in v]
if errors:
    print(f"\nErrors ({len(errors)}):")
    for k, v in errors:
        print(f"  {k}: {v['error']}")

# ── Cross-tab analysis ──
print(f"\n{'='*80}")
print("CROSS-TAB: Composite Score by Variant × Solution × Board")
print(f"{'='*80}")

for board in ['main', 'dual']:
    print(f"\n  Board: {board.upper()}")
    print(f"  {'Solution':<15} {'A (baseline)':>15} {'B (before)':>15} {'C (after)':>15} {'C-B Delta':>15}")
    print(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    for sol in ['dedup_l2', 'gate_d_v2']:
        scores = {}
        for var in ['A', 'B', 'C']:
            label = f"{var}_{sol}_{board}"
            r = results.get(label, {})
            scores[var] = r.get('composite_score', None)

        a_s = f"{scores['A']:.4f}" if scores['A'] is not None else "N/A"
        b_s = f"{scores['B']:.4f}" if scores['B'] is not None else "N/A"
        c_s = f"{scores['C']:.4f}" if scores['C'] is not None else "N/A"
        if scores['B'] is not None and scores['C'] is not None:
            delta = scores['C'] - scores['B']
            delta_s = f"{delta:+.4f}"
        else:
            delta_s = "N/A"

        print(f"  {sol:<15} {a_s:>15} {b_s:>15} {c_s:>15} {delta_s:>15}")

# ── Save full results ──
out_json = os.path.join(OUT_DIR, f"abc_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f"\nFull results saved to: {out_json}")

# ── Best agent ──
if ranked:
    best = ranked[0]
    print(f"\n{'='*80}")
    print(f"BEST AGENT: {best[0]}")
    print(f"  Composite: {best[1]['composite_score']:.4f}")
    print(f"  OOS IC: {best[1]['oos_ic']:+.5f}  ICIR: {best[1]['oos_icir']:.4f}")
    print(f"  Top-10 Sharpe: {best[1]['top10_sharpe']:.2f}  Win Rate: {best[1]['top10_win_rate']:.1%}")
    print(f"  Features: {best[1]['n_feat_used']}  Source: {best[1]['feat_source']}")
    print(f"{'='*80}")
