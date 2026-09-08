# RAG 知识库助手使用说明

本文档面向第一次部署和使用本项目的开发者、内网管理员及业务用户，覆盖本地启动、模型配置、知识库管理、文档导入、问答、临时附件、评测和常见故障处理。

## 1. 项目简介

RAG 知识库助手是一个基于 Vue 3 和 FastAPI 的内网文档问答系统。用户先把资料导入知识库，系统解析文档、切分文本并生成向量；提问时先检索相关片段，再由聊天模型生成带引用的回答。

系统依赖以下组件：

- PostgreSQL：保存知识库、文档、文本片段、对话和消息。
- MinIO：保存用户上传的原始文件。
- Qdrant：保存文档片段和文件画像的向量索引。
- Ollama 或 OpenAI 兼容服务：提供聊天、意图判断和 embedding 模型。
- 可选 HTTP reranker：对召回片段重新排序。

当前版本是内网 MVP，未提供登录、用户身份认证和真正的多租户隔离。数据范围通过 `.env` 中的固定 `OWNER_ID` 过滤；不要把它直接暴露到不受信任的公网环境。

## 2. 环境要求

Windows 本地开发建议准备：

- Python 3.12 或更高版本
- Node.js 和 npm
- Docker Desktop，并确保 Docker Engine 正在运行
- 本地模型模式需要 Ollama

Linux 服务器脚本还要求：

- PostgreSQL 系统服务
- Qdrant 和 MinIO 可执行文件
- Conda 环境（默认名称为 `rag`）
- Bash、Python 和 npm

支持的知识库文档格式为：`.jsonl`、`.json`、`.pdf`、`.doc`、`.docx`、`.xlsx`、`.pptx`、`.txt`、`.md`、`.markdown`。

## 3. Windows 首次安装

在项目根目录执行：

```powershell
cd D:\startwell\RAG
Copy-Item .env.example .env
python -m pip install -r requirements.txt

Push-Location frontend
npm.cmd install
Pop-Location
```

`.env` 只在本机保存，不要提交到 Git。至少检查数据库、MinIO、Qdrant 和模型相关配置；没有特殊需求时可以先使用 `.env.example` 的本地默认值。

### 3.1 本地 Ollama 模式

启动 Ollama 后准备默认模型：

```powershell
ollama pull qwen3:4b
ollama pull qwen2.5:0.5b
ollama pull qwen3-embedding:0.6b
```

`.env` 中确认：

```dotenv
MODEL_MODE=local
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_CHAT_MODEL=qwen3:4b
RAG_DECISION_MODEL=qwen2.5:0.5b
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
QDRANT_COLLECTION=rag_chunks_qwen3_embedding
QDRANT_DOCUMENT_COLLECTION=rag_documents_qwen3_embedding
```

其中 `RAG_DECISION_MODEL` 是轻量的检索决策模型，不能与主聊天模型设置为同一个模型。

### 3.2 远程 OpenAI 兼容模式

远程模式需要聊天服务和支持 `/embeddings` 的 embedding 服务：

```dotenv
MODEL_MODE=remote

REMOTE_LLM_PROVIDER_NAME=DeepSeek
REMOTE_LLM_BASE_URL=https://api.deepseek.com
REMOTE_LLM_API_KEY=你的聊天_API_Key
REMOTE_LLM_MODELS=deepseek-chat,deepseek-reasoner
REMOTE_DEFAULT_CHAT_MODEL=deepseek-chat

REMOTE_EMBEDDING_PROVIDER_NAME=你的_embedding_服务
REMOTE_EMBEDDING_BASE_URL=https://你的_embedding_服务/v1
REMOTE_EMBEDDING_API_KEY=你的_embedding_API_Key
REMOTE_EMBEDDING_MODEL=你的_embedding_模型
QDRANT_COLLECTION=rag_chunks_remote_embedding
QDRANT_DOCUMENT_COLLECTION=rag_documents_remote_embedding
```

意图判断仍使用本地 Ollama 的 `RAG_DECISION_MODEL`。DeepSeek 官方 API 不提供 embedding，因此不能只配置 DeepSeek 聊天 API。切换 embedding 模型或维度后，应使用新的 Qdrant collection，并重新建立知识库和导入文档。

## 4. 启动和停止

### 4.1 一键启动（Windows）

确保 Docker Desktop、Ollama（本地模式）和前端依赖已准备好，然后执行：

```powershell
python run_all.py
```

脚本会启动 PostgreSQL、Qdrant、MinIO、FastAPI 和 Vite，并检查 8080、5173 端口是否可用。

- 前端：<http://127.0.0.1:5173>
- FastAPI 文档：<http://127.0.0.1:8080/docs>
- 健康检查：<http://127.0.0.1:8080/health>

按 `Ctrl+C` 会停止由脚本启动的后端和前端进程。停止基础设施容器：

```powershell
docker compose down
```

`docker compose down -v` 会删除 PostgreSQL、Qdrant 和 MinIO 数据卷，请仅在确认要清空本地数据时执行。

### 4.2 手动启动

终端一启动基础设施和后端：

```powershell
docker compose up -d postgres qdrant minio
python -m uvicorn rag_app.main:app --app-dir backend/src --host 127.0.0.1 --port 8080
```

终端二启动前端：

```powershell
cd frontend
npm.cmd run dev
```

也可以只启动 Docker 和后端：

```powershell
.\scripts\start-local.ps1
```

### 4.3 Linux 服务器

默认脚本路径为项目目录 `~/startwork/RAG`、基础服务目录 `/root/autodl-tmp/rag-services`、Conda 环境 `rag`：

```bash
bash scripts/start-server.sh
```

路径不同时覆盖环境变量：

```bash
RAG_PROJECT_DIR=/path/to/RAG \
RAG_SERVICES_DIR=/path/to/rag-services \
CONDA_ENV_NAME=rag \
bash scripts/start-server.sh
```

服务器前端默认使用 6008 端口，地址为 <http://127.0.0.1:6008>。脚本会复用已监听的 PostgreSQL、Qdrant 和 MinIO；按 `Ctrl+C` 只停止前后端。

## 5. 启动后检查

先打开健康检查接口：

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health | ConvertTo-Json
```

正常时 HTTP 状态为 `200` 且 JSON 中 `ok` 为 `true`。如果返回 `503`，查看启动终端日志和 `.env`，确认 PostgreSQL、MinIO、Qdrant 和模型服务都可用。

## 6. 网页端使用流程

### 6.1 创建和选择知识库

1. 打开前端地址。
2. 在左侧点击“新建知识库”。
3. 填写名称和可选描述后保存。
4. 在左侧列表选择知识库，进入“对话”或“文档”视图。

删除知识库会同时删除其中的原始文件、向量、文档片段和历史对话，且不可恢复。

### 6.2 导入文档

在“文档”视图的“批量导入资料”区域点击“选择文件”，可以一次选择多个文件。当前网页端文件选择器未设置目录选择属性；如果需要递归导入本机文件夹，请使用命令行导入。

1. 选择一个或多个支持的文件。
2. 检查待导入队列，移除不需要的文件。
3. 点击“开始导入”。
4. 等待每个文件变为“已完成”。
5. 确认文档状态为 `ready` 后再提问。

系统会自动跳过同一知识库中内容相同的文件。文档状态包括 `processing`、`ready` 和 `failed`。导入过程中不要刷新或关闭页面；浏览器刷新后无法恢复原始 `File` 对象，需要重新选择文件。后端上传接口先返回 `202`，后台在同一 FastAPI 进程的线程中完成后续处理。

### 6.3 命令行递归导入

先在网页或 API 中取得真实的 `knowledge_base_id`：

```powershell
python backend\src\rag_app\cli.py import-folder `
  --knowledge-base-id "实际的知识库 ID" `
  --folder "D:\data\documents"
```

命令会递归查找支持的文件并逐个导入。当前 CLI 不会把本地相对目录写入 `folder_path`，批量导入的文件会落在知识库根目录；需要保留目录树时，使用网页端选择文件并由浏览器提供相对路径，或调用上传 API 时传递 `folder_path`。

### 6.4 文档管理

文档页支持按标题、文件名和目录搜索，可以展开文件夹并删除单个文档或整个文件夹。删除会清理原始文件、PostgreSQL 记录和 Qdrant 向量。

### 6.5 发起问答

1. 切换到“对话”视图。
2. 点击“新建对话”，或直接在输入框输入问题。
3. 可在输入框下方选择回答模型。
4. 按 `Enter` 或点击发送按钮。
5. 展开“引用原文”查看文件名、页码和原文摘录。

回答会保存到当前知识库的历史对话中。历史对话支持切换、重命名和删除。系统会根据问题自动决定是否检索，并支持指代改写和复杂问题拆解；接口返回的 `query_plan`、`retrieval_trace` 和分阶段耗时可用于排查检索行为。

### 6.6 临时附件问答

点击回形针按钮可添加 1 到 10 个临时附件。单个附件最大 30 MB，送入当前回答上下文的临时文本最多 12,000 个字符。

- 附件解析完成后才能发送问题。
- 默认只用于当前问题，不写入 MinIO、PostgreSQL 或 Qdrant。
- 勾选“保存到当前知识库”后，系统会尝试把原文件导入知识库；保存失败不影响本次回答。

## 7. API 快速用法

完整接口参数以 <http://127.0.0.1:8080/docs> 为准。常用接口如下：

```text
GET    /health
GET    /api/v1/models
GET    /api/v1/knowledge-bases
POST   /api/v1/knowledge-bases
GET    /api/v1/knowledge-bases/{id}/documents
POST   /api/v1/knowledge-bases/{id}/documents
POST   /api/v1/knowledge-bases/{id}/chat
POST   /api/v1/knowledge-bases/{id}/chat-with-attachments
GET    /api/v1/knowledge-bases/{id}/conversations
POST   /api/v1/knowledge-bases/{id}/conversations
PATCH  /api/v1/conversations/{conversation_id}
DELETE /api/v1/conversations/{conversation_id}
```

创建知识库：

```powershell
$body = @{ name = "研发规范"; description = "研发流程和技术规范" } | ConvertTo-Json
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8080/api/v1/knowledge-bases `
  -ContentType "application/json" `
  -Body $body
```

上传文档：

```powershell
curl.exe -X POST `
  -F "file=@D:\data\documents\规范.pdf" `
  -F "folder_path=制度" `
  http://127.0.0.1:8080/api/v1/knowledge-bases/<knowledge_base_id>/documents
```

上传接口返回 `202` 和文档 ID 后，应轮询文档详情，直到 `status` 变成 `ready` 或 `failed`。

## 8. 评测和测试集工坊

顶部“测试”按钮可以选择测试集工坊或本地评测集（默认目录 `testsets`），运行全部 `Approved` 样本或指定题目。结果包括命中率、MRR、Recall@K、响应耗时、Token 用量，以及启用 Judge 后的正确性、完整性、忠实性和通过率。

命令行评测：

```powershell
python scripts\evaluate_rag.py `
  --knowledge-base-id "实际的知识库 ID" `
  --dataset ".\testsets\heishanliang_rag_eval_v2.jsonl" `
  --output ".\rag_eval_report.json"
```

指定题目或关闭 Judge：

```powershell
python scripts\evaluate_rag.py `
  --knowledge-base-id "实际的知识库 ID" `
  --dataset ".\testsets\heishanliang_rag_eval_v2.jsonl" `
  --question-id q0012 `
  --question-id q0011 `
  --no-judge
```

查看全部参数：

```powershell
python scripts\evaluate_rag.py --help
```

评测集中的 `source_document_ids` 和 `source_chunk_ids` 必须对应当前知识库。重新导入文件会产生新的 ID，旧评测集需要同步更新；默认只读取 `status=approved` 的样本。

同步测试集工坊需配置：

```dotenv
TESTSET_TOOL_BASE_URL=http://localhost:3000
TESTSET_TOOL_SYNC_TIMEOUT_SECONDS=60
```

文档处理完成后，可在网页顶部点击“同步工坊”，或调用：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8080/api/v1/knowledge-bases/<knowledge_base_id>/testset-sync"
```

## 9. 重要配置

完整列表见 `.env.example`。最常调整的配置如下：

| 配置 | 作用 | 默认值 |
| --- | --- | --- |
| `DATABASE_URL` | PostgreSQL 连接串 | `postgresql+psycopg://rag:rag@127.0.0.1:5432/rag` |
| `MINIO_ENDPOINT` / `MINIO_BUCKET` | 原始文件存储 | `127.0.0.1:9000` / `rag-documents` |
| `QDRANT_URL` / `QDRANT_COLLECTION` | 向量服务和 chunk collection | `http://127.0.0.1:6333` / 模板值 |
| `RAG_TOP_K` | 最终送入回答的片段数量 | `10` |
| `RAG_RETRIEVAL_CANDIDATE_K` | 初始召回候选数量 | `60` |
| `RAG_RERANK_ENABLED` | 是否启用 HTTP reranker | `true` |
| `RAG_MIN_RELEVANCE_SCORE` | reranker 相关性下限 | `0.1` |
| `RAG_ANSWER_MAX_OUTPUT_TOKENS` | 回答输出硬上限 | `1200` |
| `INGESTION_MAX_CONCURRENCY` | 入库并发 worker 数量 | `2` |
| `INGESTION_EMBEDDING_BATCH_SIZE` | embedding 批大小 | `32` |
| `MAX_DOCUMENT_BYTES` | 单个文档上限，0 表示不限制 | `0` |
| `RAG_JUDGE_ENABLED` | 评测是否启用答案 Judge | `true` |
| `EVALUATION_DATASET_DIR` | 本地 JSONL 评测集目录 | `testsets` |

本地模型内存不足时，把 `INGESTION_MAX_CONCURRENCY` 和 `INGESTION_EMBEDDING_MAX_CONCURRENCY` 调低，并且不要启动多个 Uvicorn worker。切换 embedding 模型后使用新的 collection，并重新导入文档。

已有文档需要补建文件画像索引时执行：

```powershell
python backend\src\rag_app\cli.py rebuild-document-index `
  --knowledge-base-id "实际的知识库 ID"
```

## 10. 常见故障

### 页面打不开

确认前端进程监听 5173（服务器模式为 6008），后端监听 8080；检查是否已有进程占用端口。前端 Vite 会把 `/api` 和 `/health` 代理到 8080。

### `/health` 返回 503

检查 `docker compose ps`、数据库和对象存储地址、Qdrant 地址、Ollama 模型，以及远程模式下的 API Key 和模型名称；查看后端启动日志中的具体初始化错误。

### 文档一直是 `processing`

处理队列在 FastAPI 进程内存中。后端重启会丢失未完成任务，数据库可能留下 `processing` 状态。重启后重新上传文件通常是最直接的处理方式；生产部署应增加持久化任务队列和恢复机制。

### 文档变为 `failed`

常见原因是文件没有可提取文本、格式损坏、Office 解析失败、embedding 服务不可用或 Qdrant 写入失败。扫描版 PDF 如果没有文本层，当前解析器不会自动 OCR。

### 上传重复文件

系统按内容 hash 跳过同一知识库中内容相同的文件；文件名不同也可能被识别为重复。若要导入新版本，请修改文件内容后重新上传，并确认旧版本是否需要删除。

### 问答没有引用或回答不相关

确认文档状态为 `ready`，问题使用了文档中的明确术语，并检查 embedding 模型。若更换过 embedding 模型，必须切换新的 collection 并重新导入文档；还可以暂时关闭或调整 reranker 后重新评测。

### 远程模式提示 embedding 错误

确认 `REMOTE_EMBEDDING_BASE_URL` 指向支持 OpenAI `/embeddings` 的服务，且 `REMOTE_EMBEDDING_MODEL` 与向量维度匹配。聊天 API 和 embedding API 可以是不同服务。

### 测试集读取失败

本地评测时确认 JSONL 文件位于 `EVALUATION_DATASET_DIR`，且文件名作为 `--dataset` 传入；不要把 `_chunk_candidates.jsonl` 当作正式评测集。工坊模式需要配置并启动 `TESTSET_TOOL_BASE_URL`。

## 11. 开发验证

提交代码前建议运行：

```powershell
python -m pytest backend/tests -q -p no:cacheprovider

Push-Location frontend
npm.cmd test
npm.cmd run build
Pop-Location
```

这些检查主要覆盖单元测试、mock 和前端工具测试，不等同于真实 PostgreSQL、MinIO、Qdrant、Ollama 和远程模型服务的集成冒烟测试。

## 12. 当前运行边界

- 没有登录认证、成员管理和文档级权限。
- 上传后的解析、embedding 和索引处理运行在后端进程内存中，没有 Redis/RabbitMQ 等持久化队列。
- 临时附件默认不持久化，保存附件是一次额外的知识库导入操作。
- 数据库、对象存储和向量库没有在应用层提供跨系统事务；删除或同步失败需要人工检查。
- 服务器脚本假设基础服务目录和运行用户环境已按项目约定准备好。
