# RAG Copilot - Project2 MVP

> 一个逐版本演进的 RAG 工程实验项目
> 聚焦多文档污染控制、规则稳定化与回归验证

---

## 项目定位

这是一个基于课程文档的最小化 RAG 问答系统。

目标不是"做一个万能 AI"，而是构建一个：

- 可控的 RAG 系统
- 可追溯引用来源
- 明确边界的问答机制
- 可回归验证的工程结构

应用场景：
XR 教学编辑器中的课程设计助手。

---

## 系统流程

```
用户问题
→ 文档检索（Retrieval）
→ SCOPE 过滤（防跨文档污染）
→ 决策门控（Gate）
→ 上下文注入
→ 模型生成
```

工程原则：

- **RAG 负责**"找内容"
- **模型负责**"组织表达"
- **规则负责**"控制边界"
- **回归负责**"保证稳定"

---

## 版本演进记录

### v0.3 - 拒答机制稳定化
- 建立最小回归集
- 引入 no_hit 拒答机制
- 方法型问题门槛（is_howto_question + has_howto_evidence）
- REFUSE 准确率：70% → 100%

### v0.4 - 多文档污染控制
- 支持多 PDF（doc_id）
- chunk_id 格式：{doc_id}-p{page}-c{index}
- 引入 SCOPE 过滤规则（防止跨文档污染）
- 新增 scope 回归断言

### v0.5 - 规则系统稳定化（当前）
- 修复 forced_doc 空格规范化问题（"B 版本"）
- 新增 "A 和 B" 跨文档对比模式识别
- 明确单文档回答边界
- 回归结果：
  - 总体准确率：100% (10/10)
  - ANSWER：100%
  - REFUSE：100%
  - Scope 污染：0

---

## 核心能力

### 1️⃣ 多文档支持
- 自动提取 doc_id
- 支持目录扫描加载多个 PDF
- chunk_id 标准化格式

### 2️⃣ SCOPE 污染防护
- 自动识别 dominant_doc
- 显式 forced_doc 规则（如 "B 版本"）
- 防止跨文档引用混杂

### 3️⃣ 决策门控（Gate）
- 方法型问题门槛
- 跨文档对比拒答
- 推荐类问题拒答
- Token 测试兜底规则

### 4️⃣ 回归验证体系
- 自动回归测试
- Scope 泄漏检测
- 决策日志可观测
- 版本可冻结（Tag + Release）

---

## 当前回归结果（v0.5）

| 指标 | 结果 |
|------|------|
| 总体准确率 | 100% (10/10) |
| ANSWER | 100% |
| REFUSE | 100% |
| Scope 污染 | 0 |

---

## 技术架��

核心组件：

- `retriever_keyword.py`
  基于关键词的检索器（2-gram tokenization）

- `main.py`
  RAG 流程与决策门控逻辑

- `tests/run_regression.py`
  回归测试套件（含 scope 断言）

---

## 快速开始

### 环境准备

```bash
pip install pymupdf python-dotenv
```

### 配置 API Key

```bash
export API_KEY=your_api_key_here
```

### 回归测试

```bash
python tests/run_regression.py
```

### 交互式问答

```bash
python main.py
# 选择模式 3（RAG 模式）
# 输入 PDF 目录路径，例如：pdfs/
```

### 测试文档准备

将 PDF 文件放入 `pdfs/` 目录，命名格式：

```
course_A.pdf → doc_id = "A"
course_B.pdf → doc_id = "B"
```

---

## 项目结构

```
project2_mvp/
├── main.py
├── retriever_keyword.py
├── tests/
│   ├── run_regression.py
│   ├── regression_cases_v04.jsonl
│   └── output/
├── tools/
│   └── make_test_pdfs.py
├── pdfs/
└── README.md
```

---

## 下一步计划

- [ ] 检索层优化（embedding / 向量化）
- [ ] 提升切分策略（结构化分块）
- [ ] 增加文档数量测试鲁棒性
- [ ] 自动化离线回归验证

---

本项目为工程演进型实验项目，
所有版本均通过 Tag 冻结，支持可追溯复现。
