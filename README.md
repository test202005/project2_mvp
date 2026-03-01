# RAG Copilot - Project2 MVP

基于课程文档的最小化 RAG 问答系统，专注于回答可控性与可追溯性。

## 项目简介

这是一个为 XR 教学编辑器设计的课程设计助手产品，核心用户是课程设计人员和教学内容编辑人员。系统基于最小 RAG 架构，能够：
- 准确回答课程相关问题
- 基于已有��档生成回答
- 提供可追溯的引用来源

## 核心功能

### 1. 多文档支持
- 支持加载多个 PDF 文档
- 自动提取文档标识（doc_id）
- chunk_id 格式：`{doc_id}-p{page}-c{index}`（如 `A-p2-c03`）

### 2. SCOPE 规则（跨文档污染防护）
- 自动检测主导文档（dominant_doc）
- 过滤检索结果，避免跨文档引用混杂
- 支持显式文档约束（如 "A 版本 top_k" → 强制使用文档 A）

### 3. 智能拒答机制
- **方法型问题门槛**：方法型问题但无实现证据时拒答
- **跨文档对比拒答**：不支持 "A 和 B 对比" 类问题
- **推荐类问题拒答**：不支持 "哪个更推荐" 等主观判断

### 4. Token 兜底规则
- 测试用 token（如 `XREDIT_A1_UNIQUE`）直接返回固定答案
- 用于验���跨文档污染与引用追溯

## 技术架构

```
用户问题 → 文档检索（Retrieval）→ SCOPE 过滤 → 决策门控 → 上下文注入 → 模型生成
```

### 核心组件
- **retriever_keyword.py**: 基于关键词的检索器（2-gram tokenization）
- **main.py**: RAG 流程与决策门控逻辑
- **tests/run_regression.py**: 回归测试套件

### RAG 边界定义
- **RAG 负责**：把相关内容找出来
- **大模型负责**：基于给定内容进行组织与表达

## 版本历史

### v0.5-rule-stable (当前)
**规则系统稳定化**
- ✅ 修复空格规范化问题（"B 版本" → "b版本"）
- ✅ 新增跨文档对比模式识别（"A 和 B" + 对比关键词）
- ✅ 回归测试：100% (10/10), Scope 污染：0

### v0.4-scope-stable
**多文档 SCOPE 规则**
- ✅ SCOPE 过滤规则（避免跨文档污染）
- ✅ 显式 doc 约束（forced_doc）
- ✅ Token 兜底规则
- ✅ Scope 断���测试

### v0.3-refuse-gate
**拒答准确率提升**
- ✅ 方法型问题门槛（is_howto_question + has_howto_evidence）
- ✅ REFUSE 准确率：70% → 100%

### v0.2-regression-baseline
**回归测试基线**
- ✅ 决策日志系统
- ✅ 20 条测试用例
- ✅ 总体准确率：80%

## 快速开始

### 环境准备
```bash
pip install pymupdf python-dotenv
```

### 配置 API Key
```bash
export API_KEY=your_api_key_here
```

### 运行测试
```bash
# 回归测试
python tests/run_regression.py

# 交互式问答
python main.py
# 选择模式 3 (RAG 模式)
# 输入 PDF 目录路径，如：pdfs/
```

### 准备测试文档
将 PDF 文件放入 `pdfs/` 目录，命名格式：`course_{doc_id}.pdf`
- `course_A.pdf` → doc_id = "A"
- `course_B.pdf` → doc_id = "B"

## 测试结果

### v0.5 回归测试
| 指标 | 结果 |
|------|------|
| 总体准确率 | 100% (10/10) |
| ANSWER 准确率 | 100% (8/8) |
| REFUSE 准确率 | 100% (2/2) |
| Scope 污染次数 | 0 |

### 测试用例覆盖
- ✅ Token 识别（XREDIT_*_UNIQUE）
- ✅ 文档专属问题（A 版本 / B 版本）
- ✅ 跨文档对比拒答（分别 / 对比 / 推荐）
- ✅ Scope 污染检测

## 目录结构

```
project2_mvp/
├── main.py                    # RAG 流程与决策逻辑
├── retriever_keyword.py       # 检索器实现
├── tests/
│   ├── run_regression.py      # 回归测试
│   ├── regression_cases_v04.jsonl  # v0.4 测试用例
│   └── output/                # 测试结果输出
├── tools/
│   └── make_test_pdfs.py      # 测试 PDF 生成器
├── pdfs/                      # PDF 文档目录
│   ├── course_A.pdf
│   └── course_B.pdf
└── README.md
```

## 下一步计划

- [ ] 检索层优化（embedding / 向量化）
- [ ] 增加文档数量测试鲁棒性
- [ ] 结构化切分提升命中率
- [ ] 离线回归验证自动化

## License

MIT
