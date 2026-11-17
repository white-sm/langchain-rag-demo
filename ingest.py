"""建库：读取 docs/ 下的文档 → 切分 → 向量化 → 写入 Chroma。

用法：
    python ingest.py
    python ingest.py --trace     # 打印切出来的分块，检查切分是否合理
"""
from __future__ import annotations

import argparse
import shutil

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from common import CHROMA_DIR, COLLECTION, DOCS_DIR, build_embeddings

# 按 Markdown 标题切分：每个 ## 小节独立成块
MD_HEADERS = [("#", "h1"), ("##", "h2")]
_md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=MD_HEADERS)


def load_documents() -> list[Document]:
    if not DOCS_DIR.exists():
        raise SystemExit(f"找不到文档目录: {DOCS_DIR}")
    files = sorted(list(DOCS_DIR.glob("*.md")) + list(DOCS_DIR.glob("*.txt")))
    if not files:
        raise SystemExit(f"{DOCS_DIR} 里没有 .md / .txt 文件")

    docs: list[Document] = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": f.name, "path": str(f)}))
        print(f"  读取 {f.name}  ({len(text)} 字符)")
    return docs


def split_documents(docs: list[Document], chunk_size: int, overlap: int) -> list[Document]:
    """两级切分：

    第一级 —— 按 Markdown 标题切（## 一个术语分类一块）。
        为什么要这样：术语表是「分类 + 若干术语」的结构，如果按固定长度切，
        MRCT、EDC、SDV 这些不相关的术语会被挤进同一个块，向量互相稀释，
        导致搜「MRCT」搜不到。

    第二级 —— 某个小节如果还是超过 chunk_size，才按标点再细分。
        分隔符从粗到细，尽量避免把一句话拦腰截断。
    """
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "；", " ", ""],
        keep_separator=True,
    )

    chunks: list[Document] = []
    for doc in docs:
        sections = _md_splitter.split_text(doc.page_content)
        for sec in sections:
            # 保留原始文件信息
            sec.metadata["source"] = doc.metadata["source"]
            sec.metadata["path"] = doc.metadata["path"]
            if len(sec.page_content) <= chunk_size:
                chunks.append(sec)
            else:
                chunks.extend(char_splitter.split_documents([sec]))

    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--trace", action="store_true", help="打印每个分块的内容")
    args = ap.parse_args()

    print("[1/3] 读取文档")
    docs = load_documents()

    print(f"[2/3] 切分（chunk_size={args.chunk_size}, overlap={args.overlap}）")
    chunks = split_documents(docs, args.chunk_size, args.overlap)
    print(f"  得到 {len(chunks)} 个分块")
    if args.trace:
        for i, c in enumerate(chunks, 1):
            body = c.page_content.replace("\n", " ")[:80]
            print(f"  [{i}] {c.metadata['source']}  {len(c.page_content)}字  {body}...")

    print("[3/3] 向量化并写入 Chroma（会重新建库）")
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print("  已清空旧向量库")

    from langchain_chroma import Chroma

    emb = build_embeddings()

    # 先建集合、再加文档：比 from_documents(...) 稳，
    # 因为 from_documents 会把 kwargs 透传给不同版本的构造器，
    # 各版本接受的参数名不一致。
    try:
        store = Chroma(
            collection_name=COLLECTION,
            embedding_function=emb,
            persist_directory=str(CHROMA_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )
        print("  集合已创建（cosine 距离）")
    except Exception as exc:  # chromadb 版本差异时退回默认距离
        print(f"  [WARN] cosine 配置不被支持，退回默认: {type(exc).__name__}: {exc}")
        store = Chroma(
            collection_name=COLLECTION,
            embedding_function=emb,
            persist_directory=str(CHROMA_DIR),
        )

    print("  正在向量化，调用 embedding 接口…")
    store.add_documents(chunks)
    print(f"  完成，向量库位于 {CHROMA_DIR}")
    print(f"  集合内文档数: {store._collection.count()}")


if __name__ == "__main__":
    main()
