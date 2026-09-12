# -*- coding: utf-8 -*-
"""DNS对质测试: 本机DNS vs 阿里/腾讯DoH; 分流探测(国内回显); 新IP直连测试"""
import json
import socket
import ssl
import sys

sys.stdout.reconfigure(encoding="utf-8")
import requests

DOMS = ["www.iwencai.com", "www.10jqka.com.cn", "pass.10jqka.com.cn", "eq.10jqka.com.cn"]

print("== DNS对质")
local_ans = {}
for d in DOMS:
    try:
        local_ans[d] = socket.gethostbyname(d)
    except Exception:
        local_ans[d] = "DNS-FAIL"
    print(f"   local  {d:26s} {local_ans[d]}")

DOH = [("ali", "https://223.5.5.5/resolve"), ("dnspod", "https://1.12.12.12/resolve")]
new_ips = {}
for tag, url in DOH:
    for d in DOMS:
        try:
            r = requests.get(url, params={"name": d, "type": "A"}, timeout=8,
                             headers={"Accept": "application/dns-json"})
            data = r.json()
            ips = [a["data"] for a in data.get("Answer", []) if a.get("type") == 1]
            print(f"   {tag:7s} {d:26s} {ips}")
            if ips:
                new_ips.setdefault(d, []).extend(ips)
        except Exception as e:
            print(f"   {tag:7s} {d:26s} FAIL {type(e).__name__}: {str(e)[:60]}")

print("\n== 国内IP回显 (分流探测: 若显示家宽IP则THS流量未走VPN)")
for url in ["http://myip.ipip.net", "https://api.ipify.org"]:
    try:
        r = requests.get(url, timeout=10)
        print(f"   {url} -> {r.text.strip()[:100]}")
    except Exception as e:
        print(f"   {url} FAIL {type(e).__name__}: {str(e)[:80]}")

print("\n== 新IP TLS直连测试 (SNI=域名, 忽略证书校验)")
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

seen = set()
for d, ips in new_ips.items():
    for ip in ips:
        if ip in seen or ip == local_ans.get(d):
            continue
        seen.add(ip)
        try:
            s = socket.create_connection((ip, 443), timeout=10)
            w = ctx.wrap_socket(s, server_hostname=d)
            ver = w.version()
            w.close()
            print(f"   {d} via {ip}: TLS-OK {ver}")
        except Exception as e:
            print(f"   {d} via {ip}: FAIL {type(e).__name__}: {str(e)[:70]}")
