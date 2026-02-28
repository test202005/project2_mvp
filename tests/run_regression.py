# -*- coding: utf-8 -*-
import os
import sys
import json
import re
import io
from contextlib import redirect_stdout
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import read_local_pdf, is_howto_question, has_howto_evidence
from retriever_keyword import build_chunks, retrieve_topk

# 回归测试不需要调用模型，API_KEY 检查跳过
# API_KEY = os.getenv('API_KEY')
# if not API_KEY:
#     print("错误: 请设置环境变量 API_KEY")
#     sys.exit(1)


def ask(pdf_dir: str, question: str) -> dict:
    """
    执行单次问答（支持多文档），捕获决策日志

    Args:
        pdf_dir: PDF 目录路径（支持多文档）或单 PDF 文件路径
        question: 用户问题

    Returns:
        dict: {
            "decision": "ANSWER" | "REFUSE",
            "top_chunks_ids": list,  # 检索后的 top chunk_ids
            "used_chunk_ids": list,  # SCOPE 过滤后的 chunk_ids
            "doc_scope": str,  # dominant_doc (如 "A", "B")
            "log": str
        }
    """
    import glob

    # 判断是目录还是单个文件
    if os.path.isdir(pdf_dir):
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    else:
        pdf_files = [pdf_dir]

    # 构建所有 chunks
    all_chunks = []
    for pdf_path in sorted(pdf_files):
        filename = os.path.basename(pdf_path)
        stem = filename.replace('.pdf', '')

        # 提取 doc_id
        import re
        match = re.match(r'^course_([A-Za-z0-9]+)(?:_.*)?$', stem)
        doc_id = match.group(1) if match else ''

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

        chunks = build_chunks(pages, max_len=220, overlap=40, doc_id=doc_id)
        all_chunks.extend(chunks)

    # 检索
    top = retrieve_topk(question, all_chunks, top_k=3)
    top_chunk_ids = [c.chunk_id for c, _ in top]

    # SCOPE 过滤
    used_chunks = top
    doc_scope = ""
    if top and top[0][0].doc_id:
        doc_scope = top[0][0].doc_id
        used_chunks = [(ch, score) for ch, score in top if ch.doc_id == doc_scope]

    used_chunk_ids = [c.chunk_id for c, _ in used_chunks]

    # 捕获输出日志
    log_buffer = io.StringIO()

    # 拼接收录文本用于方法型问题判断
    top_chunks_text = " ".join(ch.text for ch, _ in top)

    with redirect_stdout(log_buffer):
        if not top:
            print(f"[DECISION] REFUSE reason=no_hit query='{question}' top_chunks_ids={top_chunk_ids}")
            decision = "REFUSE"
        # 方法型问题门槛：方法型问题但无方法型证据时直接拒答
        elif is_howto_question(question) and not has_howto_evidence(top_chunks_text):
            print(f"[DECISION] REFUSE reason=evidence_insufficient query='{question}' top_chunks_ids={top_chunk_ids}")
            decision = "REFUSE"
        else:
            print(f"[DECISION] ANSWER reason=sufficient_evidence query='{question}' top_chunks_ids={top_chunk_ids}")
            decision = "ANSWER"

    log = log_buffer.getvalue()
    return {
        "decision": decision,
        "top_chunks_ids": top_chunk_ids,
        "used_chunk_ids": used_chunk_ids,
        "doc_scope": doc_scope,
        "log": log
    }


def run_regression():
    """运行回归测试"""
    # 支持 v0.4 测试用例
    cases_file_v04 = os.path.join(os.path.dirname(__file__), "regression_cases_v04.jsonl")
    if os.path.exists(cases_file_v04):
        cases_file = cases_file_v04
    else:
        cases_file = os.path.join(os.path.dirname(__file__), "regression_cases.jsonl")

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 读取测试用例
    cases = []
    with open(cases_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    results = []

    # 逐条执行
    print(f"=== 开始回归测试 ({len(cases)} 条用例) ===\n")

    for case in cases:
        print(f"[{case['id']}/{len(cases)}] Q: {case['question']}")

        # 兼容旧格式（pdf_path）和新格式（pdf_dir）
        pdf_input = case.get('pdf_dir', case.get('pdf_path', 'demo.pdf'))
        result = ask(pdf_input, case['question'])

        expected = case['expected']
        actual = result['decision']
        passed = (expected == actual)

        # Scope 断言（如果设置了 expected_scope）
        scope_passed = True
        scope_leak = False
        if 'expected_scope' in case and expected == 'ANSWER':
            expected_scope = case['expected_scope']
            used_chunk_ids = result['used_chunk_ids']
            # 检查所有 used_chunk_ids 是否以 expected_scope 开头
            for cid in used_chunk_ids:
                if not cid.startswith(f"{expected_scope}-"):
                    scope_passed = False
                    scope_leak = True
                    break

        result_item = {
            "id": case['id'],
            "question": case['question'],
            "expected": expected,
            "actual": actual,
            "passed": passed and scope_passed,
            "scope_passed": scope_passed,
            "scope_leak": scope_leak,
            "expected_scope": case.get('expected_scope', ''),
            "doc_scope": result['doc_scope'],
            "top_chunks_ids": result['top_chunks_ids'],
            "used_chunk_ids": result['used_chunk_ids']
        }
        results.append(result_item)

        status = "PASS" if (passed and scope_passed) else "FAIL"
        if scope_leak:
            print(f"  {status} Expected: {expected}, Actual: {actual}, Scope LEAK!")
        elif not passed:
            print(f"  {status} Expected: {expected}, Actual: {actual}")
        else:
            print(f"  {status} Expected: {expected}, Actual: {actual}")

    # 统计准确率
    total = len(cases)
    answer_cases = [r for r in results if r['expected'] == 'ANSWER']
    refuse_cases = [r for r in results if r['expected'] == 'REFUSE']

    overall_pass = sum(1 for r in results if r['passed'])
    answer_pass = sum(1 for r in answer_cases if r['passed'])
    refuse_pass = sum(1 for r in refuse_cases if r['passed'])

    overall_acc = overall_pass / total if total > 0 else 0
    answer_acc = answer_pass / len(answer_cases) if answer_cases else 0
    refuse_acc = refuse_pass / len(refuse_cases) if refuse_cases else 0

    # 统计 scope_leak
    scope_leak_count = sum(1 for r in results if r['scope_leak'])

    print("=" * 50)
    print("测试结果统计:")
    print(f"  总体准确率: {overall_acc:.1%} ({overall_pass}/{total})")
    print(f"  ANSWER准确率: {answer_acc:.1%} ({answer_pass}/{len(answer_cases)})")
    print(f"  REFUSE准确率: {refuse_acc:.1%} ({refuse_pass}/{len(refuse_cases)})")
    print(f"  Scope污染次数: {scope_leak_count}")
    print("=" * 50)

    # 保存结果
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": total,
            "overall_pass": overall_pass,
            "overall_acc": overall_acc,
            "answer_total": len(answer_cases),
            "answer_pass": answer_pass,
            "answer_acc": answer_acc,
            "refuse_total": len(refuse_cases),
            "refuse_pass": refuse_pass,
            "refuse_acc": refuse_acc,
            "scope_leak_count": scope_leak_count
        },
        "results": results
    }

    output_file = os.path.join(output_dir, "latest.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至: {output_file}")

    return overall_acc == 1.0


if __name__ == "__main__":
    success = run_regression()
    sys.exit(0 if success else 1)
