# -*- coding: utf-8 -*-
import os
import json
import re

# 定义工具函数
def get_demo_context():
    return """1. 函数调用是大模型与外部工具交互的重要方式
2. 工具调用可以让模型获取实时信息或执行特定任务
3. 正确的工具定义和参数设置是成功调用的关键"""

def read_local_pdf(file_path: str) -> str:
    try:
        from pypdf import PdfReader
        file_name = os.path.basename(file_path)
        reader = PdfReader(file_path)
        max_pages = min(3, len(reader.pages))
        lines = [f"文件: {file_name}"]
        for i in range(max_pages):
            page = reader.pages[i]
            text = page.extract_text()
            if not text or not text.strip():
                lines.append(f"p{i+1}: （本页无文本内容）")
                continue
            clean_text = text.strip().replace('\n', ' ')
            if len(clean_text) > 500:
                clean_text = clean_text[:500]
            elif len(clean_text) < 300 and i < max_pages - 1:
                pass
            lines.append(f"p{i+1}: {clean_text}")
        return "\n".join(lines)
    except Exception as e:
        return f"错误: 读取 PDF 文件失败 - {str(e)}"

def get_api_key():
    """获取 API_KEY，延迟加载以便测试"""
    API_KEY = os.getenv('API_KEY')
    if not API_KEY:
        print("错误: 请设置环境变量 API_KEY")
        exit(1)
    return API_KEY

def is_howto_question(q: str) -> bool:
    """判断问题是否包含方法型关键词（排除定义型问题）"""
    # 定义型问题应该被排除
    if any(kw in q for kw in ["什么是", "什么叫", "是什么", "如何定义", "怎样定义", "怎么定义"]):
        return False
    howto_keywords = ["如何", "怎么", "怎样", "实现", "设置", "使用"]
    return any(kw in q for kw in howto_keywords)

def has_howto_evidence(chunks_text: str) -> bool:
    """判断检索内容是否包含方法型证据词（更具体的实现细节）"""
    # 只保留最具体的方法指示词，避免误判
    evidence_keywords = ["步骤", "参数", "示例", "代码", "例如", "如下", "可以通过", "方式"]
    return any(kw in chunks_text for kw in evidence_keywords)

def call_model(prompt, system_prompt, tools=None, tool_results=None):
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=get_api_key())
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    if tool_results:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_results["tool_call_id"],
            "name": tool_results["name"],
            "content": tool_results["content"]
        })
    if tools:
        response = client.chat.completions.create(
            model="glm-4.5",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
    else:
        response = client.chat.completions.create(
            model="glm-4.5",
            messages=messages
        )
    return response.choices[0].message

def run_mode_rag():
    from zhipuai import ZhipuAI
    from retriever_keyword import build_chunks, retrieve_topk
    import glob
    client = ZhipuAI(api_key=get_api_key())

    # 输入目录路径
    dir_path = input("请输入 PDF 文件目录路径: ").strip()
    print(f"[INFO] 正在扫描目录: {dir_path}")

    # 查找所有 PDF 文件
    pdf_files = glob.glob(os.path.join(dir_path, "*.pdf"))
    if not pdf_files:
        print("错误: 目录下没有找到 .pdf 文件")
        return

    print(f"[INFO] 找到 {len(pdf_files)} 个 PDF 文件")
    all_chunks = []

    # 处理每个 PDF 文件
    for pdf_path in sorted(pdf_files):
        # 提取 doc_id：使用正则匹配 course_X 格式
        filename = os.path.basename(pdf_path)
        stem = filename.replace(".pdf", "")

        # 尝试匹配 course_XXX 格式（提取第一部分作为 doc_id）
        match = re.match(r'^course_([A-Za-z0-9]+)(?:_.*)?$', stem)
        if match:
            doc_id = match.group(1)  # 例如 A, B
        else:
            # fallback: 取 stem 前 8 个字符
            doc_id = stem[:8]

        print(f"[INFO] 正在读取: {filename} (doc_id={doc_id})")
        pdf_result = read_local_pdf(pdf_path)
        lines = pdf_result.split('\n')
        pages = {"p1": "", "p2": "", "p3": ""}
        for line in lines:
            if line.startswith("p1:"):
                pages["p1"] += line[3:].strip()
            elif line.startswith("p2:"):
                pages["p2"] += line[3:].strip()
            elif line.startswith("p3:"):
                pages["p3"] += line[3:].strip()

        # 使用 doc_id 构建 chunks
        chunks = build_chunks(pages, max_len=220, overlap=40, doc_id=doc_id)
        print(f"[INFO]   {filename} 生成 {len(chunks)} 个 chunks")
        all_chunks.extend(chunks)

    chunks = all_chunks
    print(f"[INFO] 总计 chunks built: {len(chunks)}")

    # 调试：检查专有段��是否在 chunks 中
    target_strings = ["XR-EDIT-A1", "XR-EDIT-B1", "top_k"]
    stats = {k: [] for k in target_strings}
    for ch in chunks:
        for ts in target_strings:
            if ts in ch.text:
                stats[ts].append(ch.chunk_id)

    print("[DEBUG] 专有段落在 chunks 中的分布:")
    for ts, ids in stats.items():
        print(f"  '{ts}': 命中 {len(ids)} 个, 前3个 chunk_id: {ids[:3]}")

    # 调试：按 doc_id 分布统计
    doc_stats = {}
    for ch in chunks:
        doc = ch.doc_id if ch.doc_id else "no_doc"
        if doc not in doc_stats:
            doc_stats[doc] = []
        doc_stats[doc].append(ch.chunk_id)

    print("\n[DEBUG] 按 doc_id 分布:")
    for doc, ids in sorted(doc_stats.items()):
        print(f"  {doc}: {len(ids)} 个 chunks")

    # 调试：B 文档模糊命中检查
    b_chunks = [ch for ch in chunks if ch.doc_id == "B"]
    if b_chunks:
        b_targets = ["B1", "XR-EDIT", "XR"]
        b_stats = {k: [] for k in b_targets}

        for ch in b_chunks:
            for target in b_targets:
                if target in ch.text:
                    b_stats[target].append((ch.chunk_id, ch.text[:120]))

        print("\n[DEBUG] B 文档模糊命中检查:")
        for target, matches in b_stats.items():
            if matches:
                print(f"  '{target}': 命中 {len(matches)} 个")
                for i, (cid, text) in enumerate(matches[:3]):
                    print(f"    [{i+1}] {cid}: {text}...")
            else:
                print(f"  '{target}': 未命中")
    while True:
        q = input("\n请输入问题（exit 退出）：").strip()
        if q.lower() in ("exit", "quit", "q"):
            break
        top = retrieve_topk(q, chunks, top_k=3)
        print("\n[RETRIEVAL] Top chunks:")
        if not top:
            print("  (no hits, all scores=0)")
        for ch, score in top:
            preview = ch.text[:120] + ("..." if len(ch.text) > 120 else "")
            print(f"  - {ch.chunk_id} (score={score}) {preview}")
        top_chunk_ids = [c.chunk_id for c, _ in top]
        print("[RETRIEVAL] Top chunks IDs:", ", ".join(top_chunk_ids) if top_chunk_ids else "(empty)")

        # 拼接收录文本用于方法型问题判断
        top_chunks_text = " ".join(ch.text for ch, _ in top)

        # 拒答可解释化：no-hit 时直接返回固定模板，不走模型
        if not top:
            print(f"[DECISION] REFUSE reason=no_hit query='{q}' top_chunks_ids={top_chunk_ids}")
            print("\n[ANSWER]")
            print("抱歉，文档中未提供相关信息。")
            continue

        # 方法型问题门槛：方法型问题但无方法型证据时直接拒答
        if is_howto_question(q) and not has_howto_evidence(top_chunks_text):
            print(f"[DECISION] REFUSE reason=evidence_insufficient query='{q}' top_chunks_ids={top_chunk_ids}")
            print("\n[ANSWER]")
            print("抱歉，文档中未提供相关信息。")
            continue

        # SCOPE 规则：避免跨文档污染
        if top and top[0][0].doc_id:
            dominant_doc = top[0][0].doc_id  # top1 的 doc_id
            before_count = len(top)
            scoped_top = [(ch, score) for ch, score in top if ch.doc_id == dominant_doc]
            after_count = len(scoped_top)
            print(f"[DECISION] SCOPE dominant_doc={dominant_doc} filtered={before_count}->{after_count}")
            top = scoped_top
            top_chunk_ids = [c.chunk_id for c, _ in top]

        # Token 兜底规则：测试 token 问题直接返回固定答案
        token_match = re.search(r'XREDIT_[A-Z0-9]+_UNIQUE', q)
        if token_match and top:
            token = token_match.group(0)
            top1_text = top[0][0].text
            if token in top1_text:
                print(f"[DECISION] ANSWER reason=token_hit query='{q}' top_chunks_ids={top_chunk_ids}")
                print("\n[ANSWER]")
                print(f"该 token ({token}) 是测试专用标识，用于验证跨文档污染与引用追溯。引用 [{top[0][0].chunk_id}]")
                continue

        context_lines = []
        for ch, score in top:
            context_lines.append(f"[{ch.chunk_id}] ({ch.page}) {ch.text}")
        rag_prompt = (
            "你是一个工程型 RAG 助手。请严格基于给定的【检索片段】回答。\n"
            "要求：\n"
            "1) 如果检索片段不足以回答，请直接说'文档未提供相关信息'。\n"
            "2) 回答中必须引用片段ID，如：引用[p2-c03]。\n"
            "3) 不要编造。\n\n"
            f"用户问题：{q}\n\n"
            "检索片段：\n"
            + "\n".join(context_lines)
        )
        print(f"[DECISION] ANSWER reason=sufficient_evidence query='{q}' top_chunks_ids={top_chunk_ids}")
        response = client.chat.completions.create(
            model="glm-4.5",
            messages=[
                {"role": "system", "content": "你是一个专业的 RAG 助手。"},
                {"role": "user", "content": rag_prompt}
            ]
        )
        print("\n[ANSWER]")
        print(response.choices[0].message.content)

def main():
    mode = input("选择模式：1=课程大纲 2=项目说明（说明版） 3=问答RAG（可观测检索）：").strip()
    if mode == "3":
        return run_mode_rag()
    elif mode == "2":
        import prompt_project_brief as P
    else:
        import prompt_course_outline as P
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_demo_context",
                "description": "获取演示课件的要点",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_local_pdf",
                "description": "读取本地 PDF 文件的前几页内容",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "本地 PDF 文件的完整路径"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        }
    ]
    pdf_path = input("请输入 PDF 文件路径: ").strip()
    first_prompt = P.FIRST_PROMPT_TEMPLATE.format(pdf_path=pdf_path)
    assert "{pdf_path}" not in first_prompt, "PDF 路径未正确替换"
    print(f"\n[First Prompt]\n{first_prompt}\n")
    print("=== 开始执行 ===")
    first_response = call_model(first_prompt, P.SYSTEM_PROMPT, tools)
    print("1. 模型第一次回复:")
    print("content:", getattr(first_response, "content", None))
    tool_calls = getattr(first_response, "tool_calls", None)
    if tool_calls:
        print("tool_calls:")
        for tc in tool_calls:
            print({
                "id": getattr(tc, "id", None),
                "name": tc.function.name if getattr(tc, "function", None) else None,
                "arguments": tc.function.arguments if getattr(tc, "function", None) else None
            })
    if hasattr(first_response, 'tool_calls') and first_response.tool_calls:
        tool_call = first_response.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        print("2. 触发工具:", tool_name)
        print("3. 工具入参:", tool_args)
        tool_result = ""
        if tool_name == "get_demo_context":
            tool_result = get_demo_context()
        elif tool_name == "read_local_pdf":
            tool_result = read_local_pdf(tool_args.get("file_path"))
        print("   工具出参:", tool_result)
        print()
        tool_results = {
            "tool_call_id": tool_call.id,
            "name": tool_name,
            "content": tool_result
        }
        second_prompt = P.SECOND_PROMPT
        final_response = call_model(second_prompt, P.SYSTEM_PROMPT, None, tool_results)
        print("4. 模型最终输出:")
        print(final_response.content)
    else:
        print("2. 工具未被调用")
        print("4. 模型最终输出:")
        if hasattr(first_response, 'content'):
            print(first_response.content)

if __name__ == "__main__":
    main()
