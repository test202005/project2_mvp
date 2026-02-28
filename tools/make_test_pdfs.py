# -*- coding: utf-8 -*-
"""生成测试用 PDF 文件"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# 确保输出目录存在
output_dir = os.path.join(os.path.dirname(__file__), "..", "pdfs")
output_dir = os.path.abspath(output_dir)
os.makedirs(output_dir, exist_ok=True)

# 基础内容（简化的业务描述）
BASE_CONTENT = """
RAG Copilot Project Overview

1. Business Background
This is a course design assistant product for XR editor.
Core users are course designers and content editors.
Their daily work includes:
- Arranging teaching workflows in the editor
- Uploading and managing course materials
- Generating course descriptions quickly

2. Problem Statement
As course volume increases, several issues emerge:
- Content scattered across multiple documents
- Repetitive work on similar courses
- New members difficulty understanding existing courses

3. Project Goals
Build a RAG-based assistant that:
- Answers course-related questions accurately
- Bases answers on existing documents
- Provides traceable references

4. Technical Architecture
The system uses minimal RAG architecture:
- User inputs question in editor
- System performs document retrieval
- Retrieved content injected as context
- Model generates answer from context

5. Data Sources
Course documentation in PDF/text format with:
- Unstructured content
- Business-oriented language
- Many similar expressions

6. RAG Boundary Definition
RAG responsibility: Find relevant content
Model responsibility: Generate from given content

7. Testing Focus
Key areas to test:
- Retrieval effectiveness
- Answer controllability
- Behavioral stability

8. Acceptance Criteria
- No obvious factual errors
- Answers traceable to source documents

9. Future Extensions
- More documents for robustness
- Structured chunking for better hits
- Test sets for regression validation
"""

# A 文档专属内容（放在开头确保不被截断）
A_UNIQUE = "XREDIT_A1_UNIQUE top_k=3"

# B 文档专属内容（放在开头确保不被截断）
B_UNIQUE = "XREDIT_B1_UNIQUE top_k=5"


def create_pdf(filename: str, unique_text: str):
    """创建包含英文和唯一标识的 PDF"""
    filepath = os.path.join(output_dir, filename)

    c = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4

    # 写入内容 - 唯一标识放在开头
    text_object = c.beginText(2*cm, height - 2*cm)
    text_object.setFont("Helvetica", 10)

    # 先写入唯一标识（确保不被截断）
    text_object.textLine(unique_text)
    text_object.textLine("")

    lines = BASE_CONTENT.strip().split('\n')
    for line in lines:
        # 每行约 80 字符换行
        if len(line) > 80:
            for i in range(0, len(line), 80):
                text_object.textLine(line[i:i+80])
        else:
            text_object.textLine(line)

    c.drawText(text_object)
    c.save()
    print(f"Generated: {filepath}")


if __name__ == "__main__":
    # 生成 A 和 B 测试 PDF
    create_pdf("course_A_test.pdf", A_UNIQUE)
    create_pdf("course_B_test.pdf", B_UNIQUE)

    print(f"\n测试 PDF 已生成到: {output_dir}")
    print("文件列表:")
    for f in os.listdir(output_dir):
        if f.endswith('.pdf'):
            print(f"  - {f}")
