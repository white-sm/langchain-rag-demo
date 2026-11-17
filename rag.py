"""问答：检索 → 阈值过滤 →（有结果才生成，否则兜底）。

用法：
    python rag.py "希毅医学主要做什么"
    python rag.py "你们公司食堂几点开饭"
    python rag.py "什么是 MRCT" --trace
"""
from __future__ import annotations

import argparse
import sys

from langchain_core.documents import Document

from common import (
    FALLBACK_ANSWER,
    SCORE_THRESHOLD,
    TOP_K,
    build_llm,
    load_vectorstore,
)

SYSTEM_PROMPT = """你是希毅医学的内部知识助手。请严格依据下面检索到的资料回答问题，不要使用资料以外的知识。

【检索资料】
{context}

【回答要求】
1. 每个结论后面用 [编号] 标注它来自第几条资料
2. 资料里没有的信息，直接回答"资料中未提及"，不要推测或补充
3. 中文回答，控制在 150 字以内
4. 不要复述问题，直接给答案
5. 不要使用 Markdown 语法（如 ** 加粗），输出纯文本"""


def retrieve(question: str, k: int = TOP_K, threshold: float = SCORE_THRESHOLD):
    """返回 [(doc, score)]，只保留相似度达到阈值的。"""
    vs = load_vectorstore()
    hits = vs.similarity_search_with_relevance_scores(question, k=k)
    kept = [(doc, score) for doc, score in hits if score >= threshold]
    dropped = [(doc, score) for doc, score in hits if score < threshold]
    return kept, dropped


def build_context(hits: list[tuple[Document, float]]) -> str:
    """把检索结果编号拼成上下文，编号与答案里的 [1][2] 对应。"""
    parts = []
    for i, (doc, _score) in enumerate(hits, 1):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[{i}] 来源：{source}\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


def answer(question: str, trace: bool = False) -> dict:
    kept, dropped = retrieve(question)

    if trace:
        print(f"\n检索 Top-{TOP_K}（阈值 {SCORE_THRESHOLD}）:")
        for doc, score in kept:
            print(f"  [keep {score:.3f}] {doc.metadata.get('source')} :: {doc.page_content[:50]!r}")
        for doc, score in dropped:
            print(f"  [drop {score:.3f}] {doc.metadata.get('source')} :: {doc.page_content[:50]!r}")
        print()

    # ---- 兜底分支：对应 Dify 里的「条件分支 → ELSE」 ----
    if not kept:
        return {"answer": FALLBACK_ANSWER, "sources": [], "fallback": True}

    context = build_context(kept)
    llm = build_llm()
    messages = [
        ("system", SYSTEM_PROMPT.format(context=context)),
        ("user", question),
    ]
    resp = llm.invoke(messages)
    text = resp.content if isinstance(resp.content, str) else str(resp.content)

    sources = [
        {"index": i, "source": doc.metadata.get("source"), "score": round(score, 4)}
        for i, (doc, score) in enumerate(kept, 1)
    ]
    return {"answer": text.strip(), "sources": sources, "fallback": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--trace", action="store_true", help="打印检索命中的分块与分数")
    args = ap.parse_args()

    result = answer(args.question, trace=args.trace)

    print(result["answer"])
    if result["fallback"]:
        print("\n（走了兜底分支：检索结果全部低于阈值）")
    else:
        print("\n引用：")
        for s in result["sources"]:
            print(f"  [{s['index']}] {s['source']}  (score={s['score']})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
