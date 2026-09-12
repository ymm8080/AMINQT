# -*- coding: utf-8 -*-
"""SNI区分测试: 同一IP不同SNI的TLS握手 + HTTP80探测"""
import socket
import ssl
import sys

sys.stdout.reconfigure(encoding="utf-8")
IP = "43.248.130.133"

def tls_probe(sni):
    try:
        ctx = ssl.create_default_context()
        s = socket.create_connection((IP, 443), timeout=10)
        w = ctx.wrap_socket(s, server_hostname=sni if sni else None)
        print(f"SNI={sni or '(none)'}: TLS OK, cert={w.getpeercert()['subject']}")
        w.close()
    except Exception as e:
        print(f"SNI={sni or '(none)'}: FAIL {type(e).__name__}: {str(e)[:120]}")

tls_probe("www.iwencai.com")
tls_probe("www.10jqka.com.cn")
tls_probe("www.baidu.com")
tls_probe(None)

# HTTP 80
try:
    s = socket.create_connection((IP, 80), timeout=10)
    s.sendall(b"GET / HTTP/1.1\r\nHost: www.iwencai.com\r\nConnection: close\r\n\r\n")
    data = s.recv(300)
    print(f"HTTP80: {data[:200]!r}")
    s.close()
except Exception as e:
    print(f"HTTP80: FAIL {type(e).__name__}: {str(e)[:120]}")
