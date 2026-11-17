"""验收测试：三条用例，跑完打印 PASS/FAIL。

用法：
    python test_rag.py

对应你在 Dify 里做的那三个测试：
    1. 知识库内的问题 → 要能答，且带 [编号] 引用
    2. 术语类问题     → 要能检索到另一个文档
    3. 知识库外的问题 → 必须走兜底，不能编
"""
from __future__ import annotations

import re
import sys

from rag import answer

CASES = [
    {
        "name": "知识库内问题",
        "q": "希毅医学主要做什么业务？",
        "check": lambda r: (not r["fallback"]) and ("[" in r["answer"]) and len(r["sources"]) > 0,
        "expect": "有答案 + 带 [编号] 引用 + 有来源",
    },
    {
        "name": "术语类问题（应命中另一个文档）",
        "q": "什么是 MRCT？",
        "check": lambda r: (not r["fallback"]) and len(r["sources"]) > 0,
        "expect": "有答案 + 有来源",
    },
    {
        "name": "知识库外问题（必须兜底）",
        "q": "你们公司食堂几点开饭？",
        "check": lambda r: r["fallback"] is True,
        "expect": "走兜底分支，不生成答案",
    },
]


def main() -> int:
    passed = 0
    print("=" * 70)
    for i, case in enumerate(CASES, 1):
        print(f"\n[{i}] {case['name']}")
        print(f"    问：{case['q']}")
        result = answer(case["q"])
        text = result["answer"].replace("\n", " ")
        print(f"    答：{text[:120]}{'...' if len(text) > 120 else ''}")

        ok = False
        try:
            ok = bool(case["check"](result))
        except Exception as e:  # noqa: BLE001
            print(f"    检查时异常：{e}")

        if ok:
            passed += 1
            print(f"    [PASS]  期望：{case['expect']}")
        else:
            print(f"    [FAIL]  期望：{case['expect']}")
            print(f"            实际 fallback={result['fallback']}, sources={len(result['sources'])}")

    print("\n" + "=" * 70)
    print(f"结果：{passed}/{len(CASES)} 通过")
    print("=" * 70)
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
