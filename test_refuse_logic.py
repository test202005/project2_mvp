# -*- coding: utf-8 -*-
"""测试拒答可解释化逻辑"""
from retriever_keyword import build_chunks, retrieve_topk

# 模拟一些 chunks
pages = {
    "p1": "函数调用是大模型与外部工具交互的重要方式。工具调用可以让模型获取实时信息。",
    "p2": "正确使用 Function Calling 需要定义好工具的 schema。",
    "p3": "RAG 系统包含检索和生成两个核心环节。"
}

chunks = build_chunks(pages, max_len=220, overlap=40)
print(f"[INFO] chunks built: {len(chunks)}")

# 测试1: 应该命中
print("\n=== 测试1: 正常命中查询 ===")
q1 = "什么是工具调用"
top1 = retrieve_topk(q1, chunks, top_k=3)
print(f"Query: {q1}")
top_chunk_ids = [c.chunk_id for c, _ in top1]
print(f"Top chunks IDs: {top_chunk_ids}")
if not top1:
    print(f"[DECISION] REFUSE reason=no_hit query='{q1}' top_chunks_ids={top_chunk_ids}")
    print("\n[ANSWER]")
    print("抱歉，文档中未提供相关信息。")
else:
    print("[DECISION] PROCEED - 走模型生成")

# 测试2: 应该 no-hit (触发拒答)
print("\n=== 测试2: No-hit 查询 (拒答) ===")
q2 = "什么是量子力学"
top2 = retrieve_topk(q2, chunks, top_k=3)
print(f"Query: {q2}")
top_chunk_ids = [c.chunk_id for c, _ in top2]
print(f"Top chunks IDs: {top_chunk_ids}")
if not top2:
    print(f"[DECISION] REFUSE reason=no_hit query='{q2}' top_chunks_ids={top_chunk_ids}")
    print("\n[ANSWER]")
    print("抱歉，文档中未提供相关信息。")
else:
    print("[DECISION] PROCEED - 走模型生成")

# 测试3: 另一个 no-hit 查询
print("\n=== 测试3: 另一个 No-hit 查询 ===")
q3 = "如何烹饪红烧肉"
top3 = retrieve_topk(q3, chunks, top_k=3)
print(f"Query: {q3}")
top_chunk_ids = [c.chunk_id for c, _ in top3]
print(f"Top chunks IDs: {top_chunk_ids}")
if not top3:
    print(f"[DECISION] REFUSE reason=no_hit query='{q3}' top_chunks_ids={top_chunk_ids}")
    print("\n[ANSWER]")
    print("抱歉，文档中未提供相关信息。")
else:
    print("[DECISION] PROCEED - 走模型生成")

print("\n=== 测试完成 ===")
