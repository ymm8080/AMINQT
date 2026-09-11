# -*- coding: utf-8 -*-
"""探测THS各子域TLS封禁范围: DNS/TCP/TLS三段定位, 找没被封的子域"""
import socket
import ssl
import sys

sys.stdout.reconfigure(encoding="utf-8")

DOMAINS = [
    "www.iwencai.com",        # 对照: 已知被封
    "www.10jqka.com.cn",      # 对照: 已知被封
    "pass.10jqka.com.cn",     # 登录passport
    "login.10jqka.com.cn",
    "zxg.10jqka.com.cn",      # 自选股web
    "eq.10jqka.com.cn",       # 爱问诊股
    "basic.10jqka.com.cn",
    "d.10jqka.com.cn",
    "news.10jqka.com.cn",
    "stock.10jqka.com.cn",
    "comment.10jqka.com.cn",
    "data.10jqka.com.cn",
    "www.10jqka.com.cn.",     # 容错
]

ctx0 = ssl.create_default_context()
ctx0.check_hostname = False
ctx0.verify_mode = ssl.CERT_NONE

for dom in DOMAINS:
    dom = dom.rstrip(".")
    try:
        ip = socket.gethostbyname(dom)
    except Exception as e:
        print(f"{dom:28s} DNS-FAIL ({type(e).__name__})")
        continue
    try:
        s = socket.create_connection((ip, 443), timeout=8)
    except Exception as e:
        print(f"{dom:28s} TCP-FAIL ip={ip} ({type(e).__name__})")
        continue
    try:
        w = ctx0.wrap_socket(s, server_hostname=dom)
        cert_cn = "-"
        try:
            cert_cn = dict(x[0] for x in w.getpeercert()["subject"]).get("commonName", "-")
        except Exception:
            pass
        proto = w.version()
        w.close()
        print(f"{dom:28s} TLS-OK  ip={ip:15s} {proto} cn={cert_cn}")
    except Exception as e:
        s.close()
        print(f"{dom:28s} TLS-KILL ip={ip:15s} {type(e).__name__}: {str(e)[:60]}")
