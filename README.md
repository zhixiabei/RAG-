# RAG Knowledge Assistant

一个面向内网文档的检索增强问答（RAG）应用。系统使用 **PostgreSQL + MinIO + Qdrant** 保存业务数据、原始文件和向量索引，模型层支持本地 Ollama 或远程 OpenAI 兼容 API。

当前代码是一个可运行的 MVP/内网原型：可以创建知识库、导入文档、进行带引用的多轮问答，并运行离线评测。它还没有登录认证、持久化任务队列或生产级多租户能力。

## 能力概览

- 创建、选择和删除多个知识库；文档按文件夹展示，可单独或按文件夹删除。
- 通过网页选择文件或文件夹导入文档，显示解析、向量化和索引进度，并对重复文件跳过处理。
- 支持 `jsonl`、`json`、`pdf`、`docx`、`pptx`、`xlsx`、`md`、`markdown`、`txt`。
- 问答使用固定顺序的三个阶段：检索决策、Qdrant 向量检索、答案生成；回答返回文档级去重引用。
- 支持多轮对话、对话改名/删除、模型选择、Markdown/LaTeX 渲染，以及临时附件问答。
- 支持本地 JSONL 评测集，也可以把原始 RAG chunks 同步到测试集生成工具。

### 当前运行边界

文档上传接口在 MinIO 保存源文件并创建 PostgreSQL 记录后返回 `202`，解析、embedding 和 Qdrant 写入由同一后端进程内的后台线程处理，不是独立的 Redis/RabbitMQ 任务队列。后端重启会丢失内存中的待处理队列，数据库中可能留下 `processing` 状态的文档。

应用使用固定的 `OWNER_ID` 做数据范围过滤，但没有登录和身份认证。当前检索只有 dense cosine Top-K，没有 BM25、sparse 检索或 reranker。聊天附件只在当前请求中使用，不会作为知识库文档持久化。

## 架构和目录

```text
浏览器 Vue 3
    -> FastAPI API
        -> PostgreSQL（知识库、文档、chunk、会话和消息）
        -> MinIO（原始文件）
        -> Qdrant（embedding 向量）
        -> Ollama 或 OpenAI 兼容 API（聊天和 embedding）
```

```text
agent/
  retrieval_decision_agent.py  判断本轮是否需要检索
  knowledge_retrieval_agent.py 执行 Qdrant 向量检索
  answer_agent.py              生成最终回答和引用
backend/src/rag_app/
  api/              HTTP 路由和 Pydantic 请求模型
  application/      入库、删除和 RAG 流程编排
  domain/           领域模型和端口接口
  infrastructure/  PostgreSQL、MinIO、Qdrant、模型、解析器实现
  config.py         环境配置
  main.py           FastAPI 组装入口
frontend/src/
  components/       知识库、文档、导入和问答界面
  services/         Backend API 客户端
  stores/           Pinia 状态
docs/               架构、运行时分析和权限设计
scripts/            本地/服务器启动和 RAG 评测脚本
docker-compose.yml  本机 PostgreSQL、MinIO、Qdrant
.env.example        配置模板
```

前端使用 Vue 3 + JavaScript + Vite，不使用 TypeScript。

## 环境要求

- Python 3.12+
- Node.js 和 npm
- Docker Desktop 和 Docker Compose
- 本地模式：正在运行的 Ollama，以及 `qwen3:4b`、`qwen3-embedding:0.6b`
- 远程模式：聊天 API Key 和支持 OpenAI 兼容 `/embeddings` 的 embedding API Key

## Windows 快速启动

以下命令在项目根目录执行：

```powershell
cd D:\startwell\RAG
Copy-Item .env.example .env
python -m pip install -r requirements.txt

Push-Location frontend
npm.cmd install
Pop-Location
```

使用本地模型时，先启动 Ollama 并准备模型：

```powershell
ollama pull qwen3:4b
ollama pull qwen3-embedding:0.6b
```

然后一键启动 Docker 基础设施、FastAPI 和 Vite：

```powershell
python run_all.py
```

访问：

- 前端：<http://127.0.0.1:5173>
- API 文档：<http://127.0.0.1:8080/docs>
- 健康检查：<http://127.0.0.1:8080/health>

健康检查返回 `200` 且 JSON 中 `ok` 为 `true`，才表示 PostgreSQL、MinIO、Qdrant 和模型服务都已完成初始化。基础设施未就绪时后端仍可启动，但 `/health` 返回 `503`，业务接口不可用。

### 手动启动

需要分别打开终端时：

```powershell
docker compose up -d postgres qdrant minio
python -m uvicorn rag_app.main:app --app-dir backend/src --host 127.0.0.1 --port 8080
```

另开终端启动前端：

```powershell
cd frontend
npm.cmd run dev
```

只启动 Docker 和后端也可以运行：

```powershell
.\scripts\start-local.ps1
```

停止本机容器：

```powershell
docker compose down
```

`docker compose down -v` 会同时删除 PostgreSQL、Qdrant 和 MinIO 数据卷，仅在确认需要清空本地数据时使用。

## Linux 服务器启动

服务器脚本假设 PostgreSQL 使用系统服务，Qdrant 和 MinIO 使用本地二进制，Python/npm 位于 Conda 环境中。默认路径如下：

```text
项目目录       ~/startwork/RAG
基础服务目录   /root/autodl-tmp/rag-services
Conda 环境     rag
后端           http://127.0.0.1:8080
前端           http://127.0.0.1:6008
```

在项目根目录执行：

```bash
bash scripts/start-server.sh
```

路径或 Conda 环境不同时覆盖变量：

```bash
RAG_PROJECT_DIR=/path/to/RAG \
RAG_SERVICES_DIR=/path/to/rag-services \
CONDA_ENV_NAME=rag \
bash scripts/start-server.sh
```

脚本会复用已经监听的基础服务，统一看护 8080 后端和 6008 前端；按 `Ctrl+C` 只停止前后端，PostgreSQL、Qdrant 和 MinIO 继续运行。Qdrant/MinIO 日志写在基础服务目录下。

## 配置模型

复制 `.env.example` 后按实际环境修改。不要提交包含密钥的 `.env`。

### 本地 Ollama

```dotenv
MODEL_MODE=local
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_CHAT_MODEL=qwen3:4b
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
QDRANT_COLLECTION=rag_chunks_qwen3_embedding
```

### 远程 OpenAI 兼容 API

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
```

DeepSeek 官方 API 不提供 embedding，必须另配一个支持 `/embeddings` 的服务。切换 embedding 模型后，应使用新的 `QDRANT_COLLECTION`，新建知识库并重新导入文档；旧 collection 不能用不同维度或不同模型直接查询。

常用配置：

| 配置 | 说明 | 默认值 |
| --- | --- | --- |
| `DATABASE_URL` | PostgreSQL 连接串 | `postgresql+psycopg://rag:rag@127.0.0.1:5432/rag` |
| `MINIO_ENDPOINT` / `MINIO_BUCKET` | 对象存储地址和 bucket | `127.0.0.1:9000` / `rag-documents` |
| `QDRANT_URL` / `QDRANT_COLLECTION` | 向量库地址和 collection | `http://127.0.0.1:6333` / 见模板 |
| `RAG_TOP_K` | 每次向量召回数量 | `10` |
| `INGESTION_MAX_CONCURRENCY` | 入库 worker 数量 | `2` |
| `INGESTION_EMBEDDING_MAX_CONCURRENCY` | embedding 并发数 | `1` |
| `INGESTION_EMBEDDING_BATCH_SIZE` | 单批 embedding 数量 | `32` |
| `MAX_DOCUMENT_BYTES` | 单个知识库文档上限，`0` 表示不限制 | `0` |
| `TESTSET_TOOL_BASE_URL` | 测试集工具地址，留空可禁用同步 | 模板为 `http://localhost:3000` |

本地模型占用内存较高时，不要启动多个 Uvicorn worker；每个进程都会创建自己的入库 worker 和 embedding 并发限制。

## 导入和问答

### 网页导入

1. 在左侧创建知识库。
2. 在文档页选择文件或文件夹。
3. 等待文档状态变为 `ready` 后再提问。
4. 文档列表会显示 `processing`、`ready` 或 `failed`，失败原因由后端保存。

网页选择文件夹时会保留相对目录；同一知识库中相同目录和文件名的重复上传会被跳过。刷新页面不会保留尚未提交到后端的本地 `File` 对象。

### 命令行导入

先从 API 或网页取得真实的 `knowledge_base_id`，再在项目根目录执行：

```powershell
python backend\src\rag_app\cli.py import-folder `
  --knowledge-base-id "实际的知识库 ID" `
  --folder "D:\data\documents"
```

CLI 会递归查找支持的格式并逐个导入。当前 CLI 没有把相对目录传给入库服务，批量导入后的文件会落在知识库根目录；需要保留目录树时请使用网页文件夹导入。这是后续应修复的已知差异。

### 临时附件

问答输入框支持上传 1～10 个临时附件；直接 multipart 上传接口的附件总大小上限为 30 MB，预解析接口的单个附件上限为 30 MB，送入回答上下文的临时文本上限为 12,000 字符。附件只参与当前问题，不会写入 MinIO、PostgreSQL 或 Qdrant。

## RAG 评测

先启动后端，并确认评测集引用的文档已经在目标知识库中达到 `ready`：

```powershell
Invoke-RestMethod http://127.0.0.1:8080/api/v1/knowledge-bases |
  Format-Table id,name
```

### 本地 JSONL 评测集

```powershell
python scripts\evaluate_rag.py `
  --knowledge-base-id "实际的知识库 ID" `
  --dataset ".\heishanliang_rag_eval_v1.0.0.jsonl" `
  --output ".\rag_eval_report.json"
```

默认只评测 `status=approved` 的样本，并在每题结束后删除临时对话。报告统计文档与 chunk 的命中率、Precision/Recall/F1，基于 `expected_answer` 的归一化字符 F1，服务端响应时间，以及模型供应商实际返回的 Token Usage；未上报 usage 的模型调用不会使用估算值替代。

只评测指定题目时，重复传入 `--question-id`；本地 JSONL 和测试集工具模式都支持：

```powershell
python scripts\evaluate_rag.py `
  --knowledge-base-id "实际的知识库 ID" `
  --testset-tool-url "http://localhost:3000" `
  --question-id "q0012" `
  --question-id "q0011" `
  --output ".\rag_eval_selected_report.json"
```

指定题目后，RAG 会向测试集工具发送 `scope=selected` 和 `questionIds`，工具只返回这些题目；不传 `--question-id` 时仍导出全部 Approved 样本。

评测集中的 `source_document_ids` 和 `source_chunk_ids` 必须对应当前知识库中的 ID。重新导入文档会产生新 ID，需同步更新评测集，否则检索命中率会被计算为 0。没有配置对应 source ID 的样本不计入相应命中率分母。

运行以下命令查看全部参数：

```powershell
python scripts\evaluate_rag.py --help
```

### 同步到测试集生成工具

在 `.env` 中配置工具地址；不使用时将 `TESTSET_TOOL_BASE_URL` 清空：

```dotenv
TESTSET_TOOL_BASE_URL=http://localhost:3000
TESTSET_TOOL_SYNC_TIMEOUT_SECONDS=60
```

启动测试集工具后，新文档会在完成切分后同步同一批 chunk 和 ID：

```powershell
python scripts\evaluate_rag.py `
  --knowledge-base-id "实际的知识库 ID" `
  --testset-tool-url "http://localhost:3000" `
  --base-url "http://127.0.0.1:8080" `
  --output ".\rag_eval_report.json"
```

同步失败不会回滚 RAG 入库，文档的 `testset_sync_status` 会标记为 `failed`。工具恢复后可补同步整个知识库：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8080/api/v1/knowledge-bases/实际的知识库 ID/testset-sync"
```

## API 和验证

FastAPI 自动文档位于 `/docs`。主要接口包括：

```text
GET    /health
GET    /api/v1/models
GET    /api/v1/knowledge-bases
POST   /api/v1/knowledge-bases
GET    /api/v1/knowledge-bases/{id}/documents
POST   /api/v1/knowledge-bases/{id}/documents
POST   /api/v1/knowledge-bases/{id}/chat
POST   /api/v1/knowledge-bases/{id}/chat-with-attachments
GET/POST/PATCH/DELETE  /api/v1/.../conversations
```

提交代码前执行：

```powershell
python -m pytest backend/tests -q -p no:cacheprovider

Push-Location frontend
npm.cmd test
npm.cmd run build
Pop-Location
```

这些测试以单元测试、mock 和前端工具测试为主，不会替代真实 PostgreSQL、MinIO、Qdrant、Ollama 的集成冒烟测试。

## 文档索引

- [docs/project-runtime-code-analysis.md](docs/project-runtime-code-analysis.md)：当前代码调用链、文件职责和已知实现差异。
- [docs/rag-knowledge-base-design.md](docs/rag-knowledge-base-design.md)：目标架构和后续演进设计，部分内容尚未实现。
- [docs/knowledge-base-access-control-design.md](docs/knowledge-base-access-control-design.md)：权限和多租户设计草案。

## 后续优先级

1. **P0：真实基础设施冒烟测试。** 修正并验证全新 PostgreSQL volume 的 schema 初始化顺序，覆盖一次完整的上传、解析、Qdrant 写入、问答和删除链路。
2. **P1：可靠入库。** 将进程内线程队列替换为持久化任务队列，增加重试、恢复 `processing` 任务、幂等写入和跨 PostgreSQL/MinIO/Qdrant 的补偿机制。
3. **P1：访问控制。** 引入用户/租户身份、知识库成员关系和文档级权限，移除仅依赖固定 `OWNER_ID` 的数据隔离方式。
4. **P1：评测闭环。** 固定一套可复现的 embedding 和评测集，持续记录 Recall@K、引用准确率、答案质量、延迟和失败原因。
5. **P2：检索质量。** 在离线评测证明收益后再加入关键词或 sparse 混合检索、reranker、查询改写和上下文预算控制。
6. **P2：上线运维。** 补齐日志/指标、备份恢复演练、容量规划、反向代理和内网离线交付流程。
