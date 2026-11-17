# LangChain RAG 知识库问答（Dify 工作流的代码版）

同一套 RAG 链路，两个实现：

| 环节 | Dify（拖节点） | 本项目（写代码） |
|---|---|---|
| 文档切分 | 通用分段 500 / 50 | **按 Markdown 标题切**，超长再按标点细分 |
| 向量化 | `BAAI/bge-m3` | 同一个模型，走 OpenAI 兼容协议 |
| 向量库 | 内置（Weaviate） | Chroma，本地持久化（cosine） |
| 检索 | 知识检索节点 Top-3 | `similarity_search_with_relevance_scores` |
| 兜底分支 | 条件分支 → ELSE | `if not kept:` 直接返回兜底话术 |
| 带引用生成 | LLM 节点 + 提示词 | 同样的提示词，同样的 `[编号]` 规则 |

**为什么要做这件事**：Dify 证明了「我会用工具把链路搭起来」，这份代码证明「我知道那条链路每一步在做什么」。两个一起，RAG 这块就没有盲区了。

### 一个真实踩坑：切分方式决定召回效果

第一版我用固定长度切分（500/50），术语表 1097 字被切成 3 块、每块 300–460 字。结果搜「什么是 MRCT」时，**MRCT 那段根本没被检索到**：

```
[drop 0.432] cro-terms.md :: '## 数据与统计 - **EDC...'
[drop 0.361] cro-terms.md :: '## 运营相关 - **GCP...'
[drop 0.287] cro-terms.md :: '# 临床试验常用术语...'
```

原因：MRCT、EDC、SDV 这些不相关术语被挤进同一个块，向量被稀释了。

换成**按 Markdown 标题切分**（每个 `##` 小节独立成块），分块从 5 个变成 11 个，MRCT 独立成 125 字的块，分数直接变成：

```
[keep 0.574] cro-terms.md :: '- **MRCT（Multi-Regional Clinical Trial）**...'
```

**结论：结构化文档（带标题的术语表、手册、FAQ）优先按结构切，不要按字数切。** 固定长度切分只适合没有层级结构的连续文本。

### 阈值怎么定（不是拍脑袋）

改完切分后，实测两组分数：

| 类型 | 样本 | 分数 |
|---|---|---|
| 相关内容 | 希毅医学 / MRCT | 0.659 / 0.574 |
| 无关内容 | 食堂几点开饭 | 0.395 / 0.394 / 0.353 / 0.421 |

无关内容最高 0.421，相关内容最低 0.574，**取中点 0.50**，两边各留约 0.08 余量。

阈值调高（0.65）会漏掉 MRCT（0.574 被判无关）；调低（0.3）会让食堂问题也混进检索结果。**阈值只能靠实测分布定，不能靠感觉。**

---

## 一、跑起来

### 1. 装依赖

```powershell
cd C:\Users\20683\Documents\deepseek-harness\default-workspace\langchain-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 装 3–5 分钟。如果 `Activate.ps1` 提示禁止运行脚本，先执行一次：
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### 2. 配环境变量

```powershell
Copy-Item .env.example .env
notepad .env
```

填两个 Key（都能从你已有的地方拿到）：

| 变量 | 值 |
|---|---|
| `DEEPSEEK_API_KEY` | 你项目 `backend/.env` 里那个 |
| `SILICONFLOW_API_KEY` | 你在 Dify 里配的那个 `sk-...` |

### 3. 建库

```powershell
python ingest.py --trace
```

`--trace` 会打印每个分块，**先看一眼切分是否合理**（应该是完整段落，不是半句话）。

正常输出：

```
[1/3] 读取文档
  读取 cro-terms.md  (1097 字符)
  读取 xinyi-company.md  (650 字符)
[2/3] 切分（chunk_size=500, overlap=50）
  得到 8 个分块
[3/3] 向量化并写入 Chroma
  完成，向量库位于 ...\.chroma
```

### 4. 问一句

```powershell
python rag.py "希毅医学主要做什么"
python rag.py "什么是 MRCT" --trace
python rag.py "你们公司食堂几点开饭"
```

### 5. 跑验收

```powershell
python test_rag.py
```

期望 **3/3 通过**：两条正常回答、一条走兜底。

---

## 二、代码怎么读

**`common.py`** — 共享配置。三个要点：

1. `OpenAIEmbeddings(..., check_embedding_ctx_length=False)`
   langchain 默认会拿 tiktoken 按 OpenAI 的分词器预处理文本，bge-m3 不是 OpenAI 的模型，开着会报错。这一行是接第三方 embedding 的必备开关。

2. Chroma 建集合时用 `{"hnsw:space": "cosine"}`
   这样距离就是余弦距离，`相似度 = 1 - 距离` 落在 [0,1]，阈值 0.65 才有明确含义。换成 L2 距离，阈值的意义完全不同。

3. DeepSeek 走 `ChatOpenAI(base_url=...)`
   不需要 DeepSeek 专用 SDK，OpenAI 兼容协议就是标准入口。

**`ingest.py`** — 切分用 `RecursiveCharacterTextSplitter`，分隔符列表 `["\n\n", "\n", "。", "；", " ", ""]`：

> 它会**优先按段落切**，段落还太长才退到单换行、句号，最后才硬切字符。这和 Dify 里把「分段标识符」从 `\n` 改成 `\n\n` 是同一个道理——**别把一句话拦腰截断**。

**`rag.py`** — 核心就三行逻辑：

```python
kept = [(d, s) for d, s in hits if s >= THRESHOLD]   # 阈值过滤
if not kept:                                          # 兜底分支
    return FALLBACK_ANSWER
return llm.invoke(...)                                # 有料才生成
```

这就是 Dify 里那个条件分支的等价物。**区别在于：这里是 3 行代码，Dify 里是拖一个节点。本质一样。**

---
