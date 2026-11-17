"""共享配置：embedding 模型、LLM、向量库连接。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / ".chroma"
COLLECTION = "cro_kb"

load_dotenv(BASE_DIR / ".env")


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _require_ascii_key(name: str, value: str) -> str:
    """API Key 必须是纯 ASCII。中文占位文本会让 http 头构造直接抛 UnicodeEncodeError，
    那个报错信息很难懂，所以在这里提前拦住，给一句人话。"""
    if not value:
        raise SystemExit(f"缺少 {name}，请在 .env 里填写真实值")
    if not value.isascii():
        raise SystemExit(
            f"{name} 里含有非 ASCII 字符（中文/全角空格等）。\n"
            f"请确认粘贴的是真实 Key，且没有带入多余字符。"
        )
    if value != value.replace(" ", ""):
        raise SystemExit(f"{name} 里含有空格，请去掉")
    return value


def build_embeddings() -> OpenAIEmbeddings:
    """硅基流动的 bge-m3，走 OpenAI 兼容协议。

    check_embedding_ctx_length=False 很关键：
    langchain 默认会用 tiktoken 按 OpenAI 的 tokenizer 预处理文本，
    但 bge-m3 不是 OpenAI 的模型，开着会报错或截断异常。
    """
    key = _require_ascii_key("SILICONFLOW_API_KEY", _env("SILICONFLOW_API_KEY"))
    return OpenAIEmbeddings(
        model=_env("EMBEDDING_MODEL", "BAAI/bge-m3"),
        base_url=_env("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        api_key=key,
        check_embedding_ctx_length=False,
    )


def build_llm() -> ChatOpenAI:
    key = _require_ascii_key("DEEPSEEK_API_KEY", _env("DEEPSEEK_API_KEY"))
    return ChatOpenAI(
        model=_env("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url=_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=key,
        temperature=0.3,
        max_tokens=800,
    )


def load_vectorstore() -> Chroma:
    if not CHROMA_DIR.exists():
        raise SystemExit("向量库不存在，先跑: python ingest.py")
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=build_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


TOP_K = int(_env("TOP_K", "3") or 3)
SCORE_THRESHOLD = float(_env("SCORE_THRESHOLD", "0.65") or 0.65)

FALLBACK_ANSWER = (
    "抱歉，知识库中没有检索到与这个问题相关的内容。\n\n"
    "你可以换个问法，或者联系对应业务同事确认。为免误导，我不会基于猜测作答。"
)
