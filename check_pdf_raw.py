# -*- coding: utf-8 -*-
"""直接检查 PDF 原始内容"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pypdf import PdfReader

reader = PdfReader("demo.pdf")
print(f"[INFO] PDF 总页数: {len(reader.pages)}\n")

for i in range(min(3, len(reader.pages))):
    page = reader.pages[i]
    text = page.extract_text()
    print(f"=== Page {i+1} (前500字符) ===")
    print(text[:500] if text else "(无文本内容)")
    print()
