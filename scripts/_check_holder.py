import pyarrow.parquet as pq
P = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
B = r"data\panel_full_enriched_v3.parquet"
s = pq.read_schema(P)
b = pq.read_schema(B)
print(f"Panel: {len(s.names)} cols, holder_count: {'holder_count' in s.names}")
print(f"Backup: {len(b.names)} cols, holder_count: {'holder_count' in b.names}")
missing = [c for c in b.names if c not in s.names and c not in ('turnover_rate','turnover_rate_f','turn')]
print(f"Missing from panel: {missing}")
