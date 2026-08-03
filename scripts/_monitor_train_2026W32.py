# -*- coding: utf-8 -*-
"""Tail UTF-16 log, emit matching lines as events (runs under Monitor's bash env)."""

import re
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

LOG = r"D:\AMINQT\AMINQT CODES\data\_train_2026W32.log"
NOISE = (
    "CategoryInfo",
    "FullyQualifiedErrorId",
    "RemoteException",
    "NativeCommandError",
    "At line:",
    "+ ~~~",
    "NotSpecified",
)
PAT = re.compile(
    r"(Traceback|Error|Exception|OOM|Killed|模型包|OOS weighted|OOS_IC|Results saved|"
    r"Registry prune|FeatureSelector 已初始化|\[训练\]|\[预测\]|\[质量\]|清洗后|保存)",
    re.IGNORECASE,
)
seen = set()
while True:
    try:
        raw = open(LOG, "rb").read()
    except FileNotFoundError:
        time.sleep(3)
        continue
    txt = raw.decode("utf-16-le", "ignore").replace("\x00", "").strip()
    if not txt.startswith("﻿"):
        txt = "﻿" + txt
    for ln in txt.splitlines():
        if ln in seen:
            continue
        seen.add(ln)
        if any(n in ln for n in NOISE):
            continue
        if PAT.search(ln):
            print(ln, flush=True)
    time.sleep(5)
