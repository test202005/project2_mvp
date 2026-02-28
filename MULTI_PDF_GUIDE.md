# 多 PDF 文档支持说明

## 修改内容

### 1. retriever_keyword.py
- `Chunk` 类新增 `doc_id` 字段
- `build_chunks()` 函数新增 `doc_id` 参数
- `chunk_id` 格式：`{doc_id}-p{page}-c{index}`

### 2. main.py - run_mode_rag()
- 输入改为：目录路径（自动扫描所有 .pdf 文件）
- `doc_id` 提取规则：
  - 去掉 `.pdf` 后缀
  - 如文件名包含 `_`，取最后一部分
  - 例：`course_A.pdf` → `A`，`course_B.pdf` → `B`

## 使用示例

### 准备测试文件
创建目录 `pdfs/`，放入：
- `course_A.pdf`
- `course_B.pdf`

### 运行程序
```bash
python main.py
# 选择模式：3

请输入 PDF 文件目录路径: pdfs/
[INFO] 正在扫描目录: pdfs/
[INFO] 找到 2 个 PDF 文件
[INFO] 正在读取: course_A.pdf (doc_id=A)
[INFO]   course_A.pdf 生成 8 个 chunks
[INFO] 正在读取: course_B.pdf (doc_id=B)
[INFO]   course_B.pdf 生成 6 个 chunks
[INFO] 总计 chunks built: 14
```

### 检索结果示例
```
[RETRIEVAL] Top chunks:
  - A-p2-c01 (score=3) RAG Copilot 的工程边界...
  - B-p1-c02 (score=2) 项目核心用户是...
  - A-p3-c00 (score=2) 最小验收标准...
[RETRIEVAL] Top chunks IDs: A-p2-c01, B-p1-c02, A-p3-c00
```

## chunk_id 格式说明

| 格式 | 示例 | 说明 |
|------|------|------|
| 旧格式 | `p2-c03` | 单文档模式 |
| 新格式 | `A-p2-c03` | 多文档模式，doc_id=A |
| 新格式 | `B-p1-c00` | 多文档模式，doc_id=B |

## 向后兼容性
- `doc_id` 参数可选（默认空字符串）
- 不传 `doc_id` 时保持原有格式 `p{page}-c{index}`
