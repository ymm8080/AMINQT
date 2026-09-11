# -*- coding: utf-8 -*-
"""VPN开启后复测: 出口IP + 代理模式探测 + THS域TLS连通性"""
import os
import socket
import ssl
import sys

os.environ["NODE_OPTIONS"] = "--no-deprecation"
sys.stdout.reconfigure(encoding="utf-8")

print("== 环境代理变量")
print(f"   HTTP_PROXY={os.environ.get('HTTP_PROXY')}, HTTPS_PROXY={os.environ.get('HTTPS_PROXY')}")

print("\n== 本机出口IP (直连)")
import requests

try:
    ip = requests.get("https://api.ipify.org", timeout=10).text.strip()
    print(f"   egress IP = {ip}")
except Exception as e:
    print(f"   FAIL {type(e).__name__}: {str(e)[:100]}")

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def tls_probe(dom, timeout=10):
    try:
        ipd = socket.gethostbyname(dom)
    except Exception:
        return f"DNS-FAIL"
    try:
        s = socket.create_connection((ipd, 443), timeout=timeout)
    except Exception as e:
        return f"TCP-FAIL {ipd} ({type(e).__name__})"
    try:
        w = ctx.wrap_socket(s, server_hostname=dom)
        ver = w.version()
        cipher = w.cipher()[0]
        w.close()
        return f"TLS-OK {ipd} {ver} {cipher}"
    except Exception as e:
        s.close()
        return f"TLS-KILL {ipd} {type(e).__name__}: {str(e)[:50]}"


print("\n== THS域直连TLS复测 (VPN路由模式则经VPN)")
for dom in ["www.iwencai.com", "www.10jqka.com.cn", "pass.10jqka.com.cn", "eq.10jqka.com.cn"]:
    print(f"   {dom:26s} {tls_probe(dom)}")

print("\n== 常见VPN本地代理端口监听检测 (代理模式需显式走)")
for port in [1080, 1081, 7890, 7891, 7897, 10808, 10809, 8888, 8889, 9090, 2080]:
    try:
        s = socket.socket()
        s.settimeout(0.3)
        s.connect(("127.0.0.1", port))
        s.close()
        print(f"   port {port}: LISTENING")
    except Exception:
        pass
print("   (无输出=无本地代理端口)")
