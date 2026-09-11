# -*- coding: utf-8 -*-
"""从 YMM ID & PW.docx 提取同花顺/问财相关条目 (只外显THS相关, 其余不入对话)"""
import re
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")

DOC = r"D:\YMM\YMM ID & PW.docx"

with zipfile.ZipFile(DOC) as z:
    xml = z.read("word/document.xml").decode("utf-8", errors="ignore")

# 按段落切分, 提取纯文本
paras = []
for pm in re.finditer(r"<w:p[ >].*?</w:p>", xml, re.S):
    texts = re.findall(r"<w:t[^>]*>([^<]*)</w:t>", pm.group(0))
    line = "".join(texts).strip()
    if line:
        paras.append(line)

print(f"total paragraphs: {len(paras)}")

KEY = re.compile(r"同花顺|问财|ths|THS|10jqka|iwencai|wencai", re.I)
hits = [p for p in paras if KEY.search(p)]
print(f"THS-related lines: {len(hits)}")
for p in hits:
    print("---")
    print(p)
