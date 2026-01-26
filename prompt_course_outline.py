SYSTEM_PROMPT = (
    "你是一个 helpful assistant，会根据需要调用工具来获取信息。"
    "当需要获取 PDF 内容时，你必须调用 read_local_pdf 工具读取文件内容。"
    "后续输出必须基于工具返回内容，不得虚构。"
)

FIRST_PROMPT_TEMPLATE = (
    "请读取以下 PDF 文件内容，然后基于内容生成一个详细的课程大纲（使用 Markdown）。\n"
    "要求：每条大纲要点末尾请注明来源，格式为：（来源：文件名 摘录：...）。\n\n"
    "PDF 文件路径：{pdf_path}"
)

SECOND_PROMPT = (
    "请仅基于上面 tool 返回的【文件名 + p1/p2/p3 摘录】生成课程大纲（Markdown）。\n"
    "要求：3~8章，每章3~7要点；每条要点末尾必须带（来源：文件名 pX 摘录：...）。\n"
    "不要再次要求读取PDF，也不要再次调用任何工具。"
)
