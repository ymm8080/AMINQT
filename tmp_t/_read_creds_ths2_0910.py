# -*- coding: utf-8 -*-
"""扫描两份凭证docx: 找THS条目 + 截断列出所有条目标签(防密码外泄)"""
import re
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")

DOCS = [
    r"D:\YMM\YMM ID & PW.docx",
    r"D:\YMM\YMM ID & PW (已自动恢复).docx",
]

KEY = re.compile(r"同花顺|问财|ths|10jqka|iwencai|wencai|股票|炒股|核新|hexin", re.I)

def paras_of(doc):
    with zipfile.ZipFile(doc) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    out = []
    for pm in re.finditer(r"<w:p[ >].*?</w:p>", xml, re.S):
        texts = re.findall(r"<w:t[^>]*>([^<]*)</w:t>", pm.group(0))
        line = "".join(texts).strip()
        if line:
            out.append(line)
    return out

for doc in DOCS:
    try:
        ps = paras_of(doc)
    except Exception as e:
        print(f"\n##### {doc}: FAIL {type(e).__name__}: {e}")
        continue
    print(f"\n##### {doc}  ({len(ps)} paras)")
    hits = [p for p in ps if KEY.search(p)]
    if hits:
        print("== FULL THS-RELATED LINES:")
        for p in hits:
            print("---")
            print(p)
    print("== ALL LINES TRUNCATED TO 40 CHARS:")
    for i, p in enumerate(ps):
        print(f"[{i:02d}] {p[:40]}")
