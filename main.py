import os
import json

# 定义工具函数
def get_demo_context():
    """返回固定的课件要点"""
    return """1. 函数调用是大模型与外部工具交互的重要方式
2. 工具调用可以让模型获取实时信息或执行特定任务
3. 正确的工具定义和参数设置是成功调用的关键"""

def read_local_pdf(file_path: str) -> str:
    """读取本地 PDF 文件的前几页内容"""
    try:
        from pypdf import PdfReader

        # 获取文件名
        file_name = os.path.basename(file_path)

        # 打开 PDF 文件
        reader = PdfReader(file_path)

        # 读取前 3 页
        max_pages = min(3, len(reader.pages))
        lines = [f"文件: {file_name}"]

        for i in range(max_pages):
            page = reader.pages[i]
            text = page.extract_text()

            # 处理空文本情况
            if not text or not text.strip():
                lines.append(f"p{i+1}: （本页无文本内容）")
                continue

            # 清理文本，取300-500字
            clean_text = text.strip().replace('\n', ' ')
            if len(clean_text) > 500:
                clean_text = clean_text[:500]
            elif len(clean_text) < 300 and i < max_pages - 1:
                # 如果不足300字且不是最后一页，可以继续读取
                pass
            lines.append(f"p{i+1}: {clean_text}")

        return "\n".join(lines)
    except Exception as e:
        return f"错误: 读取 PDF 文件失败 - {str(e)}"

# 从环境变量获取 API_KEY
API_KEY = os.getenv('API_KEY')
if not API_KEY:
    print("错误: 请设置环境变量 API_KEY")
    exit(1)

# 模型调用（使用智谱 ZhipuAI SDK）
def call_model(prompt, system_prompt, tools=None, tool_results=None):
    from zhipuai import ZhipuAI

    client = ZhipuAI(api_key=API_KEY)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # 如果有工具结果，添加到消息中
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

# 主函数
def main():
    # 模式选择
    mode = input("选择模式：1=课程大纲 2=项目说明（说明版）：").strip()
    if mode == "2":
        import prompt_project_brief as P
    else:
        import prompt_course_outline as P

    # 定义工具
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
    
    # 用户提示
    pdf_path = input("请输入 PDF 文件路径: ").strip()
    first_prompt = P.FIRST_PROMPT_TEMPLATE.format(pdf_path=pdf_path)

    # 防呆检查
    assert "{pdf_path}" not in first_prompt, "PDF 路径未正确替换"
    print(f"\n[First Prompt]\n{first_prompt}\n")

    # 第一次模型调用
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

    
    # 检查是否需要调用工具
    if hasattr(first_response, 'tool_calls') and first_response.tool_calls:
        tool_call = first_response.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        
        print("2. 触发工具:", tool_name)
        print("3. 工具入参:", tool_args)
        
        # 调用工具
        tool_result = ""
        if tool_name == "get_demo_context":
            tool_result = get_demo_context()
        elif tool_name == "read_local_pdf":
            tool_result = read_local_pdf(tool_args.get("file_path"))
        
        print("   工具出参:", tool_result)
        print()
        
        # 第二次模型调用，传入工具结果
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
