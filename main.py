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
        excerpts = []
        max_pages = min(3, len(reader.pages))
        
        for i in range(max_pages):
            page = reader.pages[i]
            text = page.extract_text()
            # 清理文本并限制长度
            clean_text = text.strip().replace('\n', ' ')
            if len(clean_text) > 200:
                clean_text = clean_text[:200] + "..."
            excerpts.append(f"摘录{i+1}: {clean_text}")
        
        # 构建返回内容
        result = f"文件: {file_name}\n"
        result += "\n".join(excerpts)
        
        return result
    except Exception as e:
        return f"错误: 读取 PDF 文件失败 - {str(e)}"

# 从环境变量获取 API_KEY
API_KEY = os.getenv('API_KEY')
if not API_KEY:
    print("错误: 请设置环境变量 API_KEY")
    exit(1)

# 模型调用（使用智谱 ZhipuAI SDK）
def call_model(prompt, tools=None, tool_results=None):
    from zhipuai import ZhipuAI
    
    client = ZhipuAI(api_key=API_KEY)
    
    messages = [
        {"role": "system", "content": "你是一个 helpful assistant，会根据需要调用工具来获取信息。当用户提供 PDF 文件路径时，你应该调用 read_local_pdf 工具来读取文件内容，然后基于内容生成课程大纲。"},
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
    prompt = "请分析以下 PDF 文件内容，并基于内容生成一个详细的课程大纲（使用 Markdown 格式）。每条大纲要点末尾请注明来源，格式为：（来源：文件名 摘录：...）。\n\nPDF 文件路径：{pdf_path}".format(
        pdf_path=input("请输入 PDF 文件路径: ")
    )
    
    # 第一次模型调用
    print("=== 开始执行 ===")
    first_response = call_model(prompt, tools)
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
        second_prompt = (
            "请仅基于上面 tool 返回的“文件名+摘录”生成课程大纲（Markdown）。\n"
            "要求：3~8章，每章3~7要点；每条要点末尾必须带（来源：文件名 摘录：...）。\n"
            "不要再次要求读取PDF，也不要再次调用任何工具。"
        )
        final_response = call_model(second_prompt, None, tool_results)

        print("4. 模型最终输出:")
        print(final_response.content)
    else:
        print("2. 工具未被调用")
        print("4. 模型最终输出:")
        if hasattr(first_response, 'content'):
            print(first_response.content)

if __name__ == "__main__":
    main()
