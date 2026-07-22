# RAG Knowledge Assistant

这是一个基于 **PostgreSQL + MinIO + Qdrant** 的 RAG 项目，模型层可选择全本地 Ollama 或全远程 OpenAI 兼容 API。

本机开发不需要连接公司内网，但必须在本机启动真实基础设施；项目不使用 SQLite、本地文件存储或本地向量索引作为替代实现。

## 目录职责

```text
backend/src/rag_app/
  api/              HTTP 路由和请求模型
  application/      入库与问答用例编排
  domain/           领域模型和端口接口
  infrastructure/  PostgreSQL、MinIO、Qdrant、Ollama、解析器实现
  config.py         环境配置
  main.py           应用组装入口
frontend/src/
  components/       侧边栏、文档列表、导入面板、问答工作区
  services/         Backend API 客户端
  stores/           Pinia 状态
  App.vue           页面布局
docs/               架构设计文档
docker-compose.yml  本机 PostgreSQL、MinIO、Qdrant
requirements.txt    Python 依赖
.env.example        配置模板
```

前端使用 Vue 3 + JavaScript + Vite，不使用 TypeScript。

## 本机依赖

- Python 3.12+
- Docker Desktop + Docker Compose
- 本地模式：Ollama，以及 `qwen3:4b`、`qwen3-embedding:0.6b`
- 远程模式：聊天 API Key 和 embedding API Key

## 启动

```powershell
Copy-Item .env.example .env
pip install -r requirements.txt
docker compose up -d postgres qdrant minio
python -m uvicorn rag_app.main:app --app-dir backend/src --host 127.0.0.1 --port 8080
```

上面的命令必须在项目根目录 `D:\startwell\RAG` 执行。`python -m uvicorn --help` 只会显示帮助，不会启动后端。

如果当前终端位于 `frontend`，先回到项目根目录：

```powershell
cd D:\startwell\RAG
python backend\src\rag_app\main.py
```

也可以直接运行后端入口：

```powershell
python backend/src/rag_app/main.py
```

API 文档：`http://127.0.0.1:8080/docs`

也可以在 PyCharm 中直接右键运行 `backend/src/rag_app/main.py`，不需要手动配置 Uvicorn 模块参数。

后端进程可以在基础设施未启动时运行，但 `/health` 会返回 `503`，知识库接口也不可用。启动 Docker Desktop 和三个容器后，重启后端即可。

### 全远程 API 模式

远程模式下，聊天和 embedding 都通过 API，不会调用 Ollama。以 DeepSeek 负责聊天、另一个 OpenAI 兼容服务负责 embedding 为例：

```dotenv
MODEL_MODE=remote

REMOTE_LLM_PROVIDER_NAME=DeepSeek
REMOTE_LLM_BASE_URL=https://api.deepseek.com
REMOTE_LLM_API_KEY=你的_DeepSeek_API_Key
REMOTE_LLM_MODELS=deepseek-chat,deepseek-reasoner
REMOTE_DEFAULT_CHAT_MODEL=deepseek-chat

REMOTE_EMBEDDING_PROVIDER_NAME=你的_embedding_服务
REMOTE_EMBEDDING_BASE_URL=https://你的_embedding_服务/v1
REMOTE_EMBEDDING_API_KEY=你的_embedding_API_Key
REMOTE_EMBEDDING_MODEL=你的_embedding_模型

QDRANT_COLLECTION=rag_chunks_remote_embedding
```

DeepSeek 官方 API 不提供 embedding 接口，所以必须配置一个支持 OpenAI 兼容 `/embeddings` 的远程服务。若同一个平台同时提供聊天和 embedding，可将两组 Base URL 和 API Key 配成相同值。

切换 embedding 模型后需要新建知识库并重新导入文档，同时使用新的 `QDRANT_COLLECTION`；系统会阻止用不同 embedding 模型查询旧索引。问答输入框下方的模型菜单用于切换当前模式内配置的聊天模型。

## 一键启动前后端

第一次先安装前端依赖：

```powershell
cd frontend
npm.cmd install
cd ..
```

之后在 PyCharm 中右键运行根目录的 `run_all.py`，它会自动启动：

```text
Docker PostgreSQL / MinIO / Qdrant
FastAPI 后端 :8080
Vue 前端 :5173
```

也可以双击根目录的 `run_all.cmd`。退出一键进程时按 `Ctrl+C`，前后端子进程会一起停止。

## 导入本机文件夹

先创建知识库并取得实际的 `knowledge_base_id`，再执行：

```powershell
python backend\src\rag_app\cli.py import-folder `
  --knowledge-base-id "实际的知识库 ID" `
  --folder "D:\startwell\黑山梁资料\黑山梁资料"
```

不要输入尖括号。`<kb_id>` 只是文档中的占位符；PowerShell 会把尖括号解释为重定向符号。

正式架构见 [docs/rag-knowledge-base-design.md](docs/rag-knowledge-base-design.md)。
