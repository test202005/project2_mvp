# -*- coding: utf-8 -*-
"""调试脚本：检查 chunks 中是否包含关键词"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from retriever_keyword import build_chunks

# 直接使用 PDF 中实际提取的内容（手动转录）
pages = {
    "p1": """课程项目定义 为什么我们要做这个项目 1. 业务背景 在 XR 教育编辑器产品中 我们的用户是 课程设计人员 教学内容编辑人员 我们的日常难题 在编辑器里写教材效率太低 知识分散存在大量文档中 容易遗漏或重复投入 课程结构不好 知识点梳理混乱 新员工培训效果差 在信息过多时无法有效聚焦 2. 源于冲突 痛点不在于知识太多而在于 知识检索不可靠 3. 项目目标 最小可行目标 构建一个 MVP 能在课程文档上 稳定回答课程相关问题 RAG 能作为课程知识库的对话助手""",
    "p2": """最小可用产品 RAG Copilot 的工程边界 4. 功能范围 最小 RAG 能帮助我们回答 Agent 系统 用户在编辑器提问 系统能够检索相关文档 RAG 检索 回答 平台约束 5. 数据资源假设 教学资源 课程文档 PDF 文本 重点 结构统一 偏向领域知识 受内存限制 在 MVP 阶段我们关注 文档成本小 内容固定 知识不常变化 背景化处理 需要对内容进行组织 提示 好处是 内容人员不参与 实时准确 指规则""",
    "p3": """具体交付标准 评估 Copilot 的表现指标 第一页生成时 内容成本 第一页时 7. 评估重点关注 从测试角度看 项目重点关注的测试问题 7.1 检索是否有效 是否能正确找到相关文档 问答时回答是否可靠 8. 最小交付标准 MVP 项目最小交付标准 在固定课程文档集合 能够对课程问题 给出可追溯的回答 回答能够追溯到的文档 第一页 结果可验证 记录回归测试 复现性强 强调回归 这也意味着 MVP 通过前后对照 总体一句话 本项目的验证模型多轮强弱对比验证 在实际企业场景下 RAG 能否稳定可复现地解答问题"""
}

chunks = build_chunks(pages, max_len=220, overlap=40)
print(f"[INFO] 总共生成 {len(chunks)} 个 chunks\n")

# 调试：检查关键词是否在 chunks 中
keywords = ["工具调用", "函数调用", "tool", "Function Calling"]
stats = {k: [] for k in keywords}
for ch in chunks:
    for kw in keywords:
        if kw in ch.text:
            stats[kw].append(ch.chunk_id)

print("[DEBUG] 关键词在 chunks 中的分布:")
for kw, ids in stats.items():
    preview = ""
    if ids:
        # 显示第一个命中 chunk 的内容预览
        for ch in chunks:
            if ch.chunk_id == ids[0]:
                preview = f" | 内容预览: {ch.text[:60]}..."
                break
    print(f"  '{kw}': 命中 {len(ids)} 个, chunk_ids: {ids[:3]}{preview}")

# 如果都没命中，检查文档里到底有什么相关词
print("\n[DEBUG] 搜索包含'调用'的 chunks:")
for ch in chunks:
    if "调用" in ch.text:
        print(f"  {ch.chunk_id}: ...{ch.text[:80]}...")
