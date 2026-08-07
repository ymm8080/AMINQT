import datetime
import os
import time
from collections import defaultdict

cache_root = "data/supply_cache"
cutoff = time.time() - 3 * 86400
agg = defaultdict(lambda: [0, 0])  # src -> [n_files, bytes]
samples = defaultdict(list)
mtime_range = defaultdict(lambda: [None, None])

for dirpath, _dirnames, filenames in os.walk(cache_root):
    for fn in filenames:
        fp = os.path.join(dirpath, fn)
        mtime = os.path.getmtime(fp)
        if mtime <= cutoff:
            continue
        rel = os.path.relpath(fp, cache_root).replace("\\", "/")
        parts = rel.split("/")
        if parts[0] == "alt_data" and len(parts) >= 2:
            key = "/".join(parts[:2])
            if len(parts) >= 3 and parts[2].startswith("backfill"):
                key = "/".join(parts[:3])
        else:
            key = parts[0]
        sz = os.path.getsize(fp)
        agg[key][0] += 1
        agg[key][1] += sz
        t = datetime.datetime.fromtimestamp(mtime)
        if mtime_range[key][0] is None or t < mtime_range[key][0]:
            mtime_range[key][0] = t
        if mtime_range[key][1] is None or t > mtime_range[key][1]:
            mtime_range[key][1] = t
        if len(samples[key]) < 3:
            samples[key].append(rel)

print(f"{'source':<48}{'files':>7}{'MB':>9}   mtime range")
for k, (n, b) in sorted(agg.items(), key=lambda x: -x[1][0]):
    r = mtime_range[k]
    print(f"{k:<48}{n:>7}{b / 1e6:>9.1f}   {r[0]:%m-%d %H:%M} ~ {r[1]:%m-%d %H:%M}")

print()
print("--- sample files ---")
for k in sorted(agg):
    print(k)
    for s in samples[k]:
        print("   ", s)
