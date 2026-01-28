# -*- coding: utf-8 -*-
import os
import json

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

API_KEY = os.getenv('API_KEY')
if not API_KEY:
    print("错误: 请设置环境变量 API_KEY")
    exit(1)

def call_model(prompt, system_prompt, tools=None, tool_results=None):
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=API_KEY)
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
    client = ZhipuAI(api_key=API_KEY)
    pdf_path = input("请输入 PDF 文件路径: ").strip()
    print(f"[INFO] 正在读取 PDF: {pdf_path}")
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
    chunks = build_chunks(pages, max_len=220, overlap=40)
    print(f"[INFO] chunks built: {len(chunks)}")
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
        print("[RETRIEVAL] Top chunks IDs:", ", ".join([c.chunk_id for c, _ in top]))
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
