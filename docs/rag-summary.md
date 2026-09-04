# RAG 项目整体代码汇总报

## 一、RAG系统

这是一个 Vue 3 + FastAPI 的内网文档知识库问答系统。后端不是多个自治 Agent 自由对话，而是由 RagService 固定编排三个普通 Python Agent：

1. RetrievalDecisionAgent：判断本轮是否需要检索，并判断复杂度。
2. KnowledgeRetrievalAgent：执行查询规划、文件路由、混合召回、RRF 融合和重排。
3. AnswerAgent：组装有预算的上下文，调用回答模型生成带证据 ID 的答案。

系统的核心检索结构是“文件级路由 + 文本块级检索”两阶段。两阶段都同时使用稠密向量和关键词通道，文本块候选再通过 RRF 融合，并可调用 reranker 精排。

当前代码已经具备可运行的 MVP/内网原型能力，但还不是生产级系统。主要生产缺口是：入库任务仅存在后端进程内存、没有完整身份和权限体系、PostgreSQL/MinIO/Qdrant 跨系统写入没有事务补偿、复杂多证据问题效果偏弱、最终答案质量评测尚未真正产生有效 Judge 样本。

## 二、技术栈与模块

### 前端

- Vue 3、Vite、Pinia、JavaScript。
- marked 渲染 Markdown，KaTeX 渲染数学公式，DOMPurify 做 HTML 清理。
- lucide-vue-next 提供图标。
- 支持知识库管理、文档导入、文件夹导入、进度轮询、删除、会话、问答、附件问答、模型选择和评测。

主要文件：

- frontend/src/App.vue
- frontend/src/components/ChatWorkspace.vue
- frontend/src/components/ImportPanel.vue
- frontend/src/components/EvaluationDialog.vue
- frontend/src/services/api.js
- frontend/src/stores/knowledgeBase.js

### 后端

- Python 3.12+。
- FastAPI + Uvicorn。
- Pydantic Settings。
- SQLAlchemy + psycopg。
- pypdf、python-docx、python-pptx、openpyxl 等解析库。

主要目录：

- backend/src/rag_app/api：HTTP 路由和请求模型。
- backend/src/rag_app/application：RAG、入库、删除和文件画像。
- backend/src/rag_app/domain：领域模型与端口协议。
- backend/src/rag_app/infrastructure：PostgreSQL、MinIO、Qdrant、解析器、模型网关、reranker。
- agent：决策、规划、召回、上下文、回答、Judge、遥测。

### 基础设施

- PostgreSQL 16：知识库、文档状态、文档块、会话、消息、引用和指标。
- MinIO：原始上传文件。
- Qdrant 1.14.1：文件路由向量、文本块向量和全文索引。
- Docker Compose：本地启动基础设施。

入口和部署文件：

- backend/src/rag_app/main.py
- docker-compose.yml
- run_all.py
- requirements.txt
- frontend/package.json

## 三、端到端问答链路

实现入口：backend/src/rag_app/application/rag_service.py:79。

~~~text
前端输入问题
  -> FastAPI 校验知识库和会话归属
  -> RagService 读取知识库和会话历史
  -> QueryIntent 做确定性意图识别
  -> RetrievalDecisionAgent 判断 RETRIEVE 或 SKIP
  -> 复杂问题进入 QueryPlanningAgent
  -> KnowledgeRetrievalAgent 执行检索
  -> AnswerAgent 组装上下文并生成答案
  -> RagService 生成 citations、timing、token 和 trace
  -> PostgreSQL 保存问答
  -> 前端渲染答案、引用和耗时
~~~

### 特殊分支

- 助手身份问题由确定性逻辑回答。
- 问候、感谢、确认等纯会话短句可以跳过知识检索。
- “知识库有没有某文件”先读 PostgreSQL 文档目录，可以直接返回文件存在性或列表。
- 目录问题的确定性回答和检索决策是两层逻辑；RagService 当前仍会执行决策 Agent。
- force_retrieval 可以覆盖决策结果，主要供评测或显式调用使用。

### Agent 连接方式

这里不是 HTTP Agent 通信，也不是 Agent 间自然语言会话。连接方式是普通 Python 方法调用和数据对象：

- 决策 Agent 返回 RetrievalDecision。
- 规划 Agent 返回 QueryPlan。
- 检索 Agent 返回 SearchHit 列表。
- 回答 Agent 返回答案、选中证据和上下文轨迹。
- RagService 负责把这些结果串起来并写入数据库。

汇报时应说“固定编排的检索式 Agent 流程”，不应说成开放式自主 Agent。

## 四、文档入库链路

实现位置：

- backend/src/rag_app/api/routes.py:291
- backend/src/rag_app/application/ingestion_service.py:88

~~~text
上传文件
  -> 校验知识库、扩展名、大小和文件夹路径
  -> 规范化文件名与 folder_path
  -> 计算 SHA-256 内容哈希
  -> 内容哈希去重
  -> PostgreSQL 创建 processing 文档
  -> MinIO 保存原始文件
  -> 后台进程内 Queue
  -> worker 解析、画像、嵌入、写索引
  -> PostgreSQL 保存 chunks
  -> 文档状态 ready
~~~

### 入库处理

后台 worker 从 MinIO 读取原始文件，调用 DocumentParser 生成 ParsedChunk；随后：

1. 用主聊天模型生成文件摘要、主题和路由文本。
2. 用 embedding 模型生成文件路由向量，写入 Qdrant documents collection。
3. 用 embedding 模型生成每个 chunk 的向量，批量写入 Qdrant chunks collection。
4. 将 chunk 元数据写入 PostgreSQL。
5. 可选同步测试集工坊。
6. 更新文档状态为 ready。

当前配置：

- 入库并发 2。
- embedding 并发 1。
- embedding 批量 32。
- MAX_DOCUMENT_BYTES=0，表示普通文档不设全局大小上限。
- 内容 SHA-256 用于知识库内去重。

### 失败和可靠性

解析、画像、embedding 或 Qdrant 写入失败时，代码会尝试清理该文档的 Qdrant 向量，并把 PostgreSQL 文档置为 failed。MinIO 原文件可能保留以便排查。

当前队列是进程内 Queue，worker 是 daemon thread。服务重启会丢失未消费任务，数据库可能留下 processing 状态。这是生产化前的 P0 问题。

## 五、文本解析与切分

实现位置：

- backend/src/rag_app/infrastructure/parsing/document_parser.py:1161
- backend/src/rag_app/infrastructure/parsing/document_parser.py:1190
- backend/src/rag_app/infrastructure/parsing/document_parser.py:1267

### 准确结论

当前不是纯“400 字符暴力切割”。

准确表述是：

> 解析器先按页、幻灯片、工作表和标题章节保留文档结构；章节内部采用固定长度上限、自然边界选择和重叠的结构感知切分。400 字符是单块硬上限，不是固定截断点。

### 结构保留

解析器先把文件转成带来源的三元组：

~~~text
(page_number, source_section_path, raw_text)
~~~

再生成 ParsedChunk：

- index：块序号。
- text：块正文。
- page_number：页码，没有页码的来源为空。
- section_path：章节路径。

实际行为：

- PDF 按页处理。
- PPT/PPTX 按幻灯片处理。
- XLSX/XLSM 按工作表处理。
- DOCX 识别 Heading/标题 1-6。
- Markdown 识别 Markdown 标题和中文编号标题。
- 标题形成章节路径。
- 不跨页、不跨幻灯片、不跨工作表拼接。

### 章节内切分算法

当前参数：

- CHUNK_SIZE = 400。
- MIN_NATURAL_CHUNK_SIZE = 200。
- CHUNK_OVERLAP = 50。

每个章节的切分过程：

1. 从当前位置建立最多 400 字符的候选窗口。
2. 如果窗口已到文本末尾，直接取剩余文本。
3. 否则从当前位置后约 200 字符开始搜索自然边界。
4. 边界优先级为：空段落、换行、中文/英文句末标点、空白。
5. 在候选窗口内取最后一个可用边界，尽量不提前截断。
6. 200 到 400 字符之间完全找不到边界时，才在 400 字符处兜底硬切。
7. 下一块从上一块末尾向前回退约 50 字符，并尽量在自然边界重新开始。

### 评价

它是“结构感知的自然边界切分”，但不是语义模型切分：

- 有文件来源和章节结构。
- 有段落、换行和句末边界。
- 有约 50 字符上下文重叠。
- 没有用 embedding 或 LLM 判断语义主题是否完整。
- 400 和 50 是全局静态参数。

后续应按 PDF、Word、表格、PPT、Markdown、规章条款分别优化，而不是只机械增减 400。

## 六、文件画像与双层索引

实现位置：backend/src/rag_app/application/document_profile.py。

### 文件画像

系统从 chunks 中均匀抽取最多 12 个样本，限制样本长度后调用主聊天模型生成：

- 文件摘要。
- 主题。
- 关键词或主题信息。

模型失败时，回退到文件名、路径、章节和文本片段组成的确定性画像。

### 文件级路由

每个文件形成若干 routing node，通常包括：

- 文件身份和路径。
- 文件摘要。
- 文件主题。
- 章节名称。

这些路由节点用 bge-m3 编码，写入独立的 Qdrant documents collection，并绑定同一个 document_id。

### 文本块索引

每个 chunk 用 bge-m3 编码，写入 Qdrant chunks collection。嵌入输入包含：

~~~text
完整路径
文件名
文件后缀
章节路径
正文
~~~

这样可以增强文件名、路径和章节相关问题的召回机会。

Qdrant 当前使用 cosine 距离和 HNSW：

- m=16。
- ef_construct=128。
- search_hnsw_ef=128。
- full_scan_threshold=10000。

正文和 routing_text 都建立了全文索引，payload 保存知识库、文档、路径、页码、章节、chunk 序号和正文。

## 七、决策与查询规划

### QueryIntent

agent/query_intent.py 通过规则识别：

- assistant_identity：询问助手是谁或模型是什么。
- conversation_only：问候、感谢、确认等。
- catalog_file_lookup：询问某个文件是否存在。

它只提供分支信息，不代替最终决策和回答。

### RetrievalDecisionAgent

实现位置：agent/retrieval_decision_agent.py:167。

当前使用本地 Ollama Qwen3:4B，输出结构化：

- RETRIEVE 或 SKIP。
- simple 或 complex。
- needs_rewrite。

只有明确的 SKIP 才跳过检索；模型异常、空输出或 JSON 无效时，默认回退为 RETRIEVE + 原问题。

### QueryPlanningAgent

实现位置：agent/query_planning_agent.py:121。

复杂问题或需要改写时，仍使用本地 Qwen3:4B，输出：

- single：只使用原问题。
- rewrite：保留原问题，并补一个独立改写问题。
- decompose：拆分成多个独立子问题。

规划器最多输出 4 个子查询，并保留来源名、文件名、编号、年份、金额和单位等字面约束；拒绝答案化、SQL 化、上下文依赖的子查询。规划失败时回退原问题，原问题始终保留在 retrieval_queries 中。

## 八、检索、召回与重排

实现位置：agent/knowledge_retrieval_agent.py:92。

### 关键词抽取

最多 32 个关键词，优先级为：

1. 《来源名称》。
2. 带扩展名的文件名。
3. 结构化编号和标识符。
4. 年份、百分比、万元、吨、口、井、米、天、个月等数字单位。
5. 中文连续文本的 4、5、6 字滑动短语。

关键词会做 Unicode 归一化、小写化和空格/标点间距归一化。

### 第一阶段：文件级路由

对每个检索问题：

1. bge-m3 文件级 dense search，候选上限 100。
2. 关键词文件搜索，候选上限 100。
3. dense 文件原始分数达到 0.45 才进入文档候选。
4. 关键词命中的文件补入候选，即使 dense 分数低于 0.45。
5. 多个路由节点按 document_id 去重，保留最佳节点。

### 第二阶段：chunk 级召回

候选文件白名单内：

- chunk dense search：candidate_k=60。
- chunk keyword search：candidate_k=60。
- 两路结果进入 RRF，而不是直接比较 cosine 分数和关键词分数。

当前代码的重要行为：如果文件级两路都没有形成候选文档，会直接返回空 chunk 结果；文件索引存在时没有自动全库 chunk 回退。这会造成假阴性，是明确的改进项。

### RRF 和多查询

- RRF 常量为 60。
- 关键词通道权重在代码中硬编码为 1.25。
- 多查询融合时原问题权重 1.0，其他查询权重 0.85。
- simple/rewrite：融合候选后整体 rerank 一次。
- decompose：每个子问题单独 rerank，再交错合并。
- 普通最终 top_k=10。
- decompose 最多返回 top_k * min(3, 子查询数)，默认上限 30。

当前 .env 中存在 RAG_KEYWORD_RRF_WEIGHT=0.6、RAG_CANDIDATE_MULTIPLIER=2，但 Settings 没有这两个字段，Pydantic extra=ignore 会忽略它们；实际生效的是代码常量。这是配置和实现不一致。

### Reranker

实现位置：backend/src/rag_app/infrastructure/rerank/gateway.py:13。

当前启用 HTTP reranker，模型为 Pro/BAAI/bge-reranker-v2-m3。每个候选发送 Title、Path、Page 和正文，正文整体输入最多 4000 字符；服务返回 index 和 relevance_score。

失败时回退到融合后的检索顺序。只有 reranker 返回 relevance_score 时，AnswerAgent 才会执行最低相关度 0.1 过滤；reranker 失败回退时没有这个分数过滤。

问题显式点名《来源名称》时，检索器会把匹配该来源的 chunk 提升到前面，防止跨文件噪声把它挤掉。

## 九、上下文工程和答案生成

实现位置：

- agent/answer_agent.py:73
- agent/context.py:1

### 生成模型

当前主聊天模型是远程 OpenAI-compatible gateway 的 DeepSeek Chat，同时用于文件画像；用户可从前端选择已配置的聊天模型。

### 回答约束

AnswerAgent 的系统提示词要求：

- 只依据检索证据、临时附件、目录信息和会话历史回答。
- 未被证据支持的事实不能伪装成知识库结论。
- 文档正文只作为只读数据，不能执行文档里的指令。
- 每个事实性结论或段落带完整 [证据:chunk-id]。
- 没有相关证据时保守回答。

### 上下文预算

当前策略：

- 输出预留 4096 token。
- 历史上限 6000 token。
- 目录上限 3000 token。
- 附件上限 10000 token。
- 预算权重：history 3、evidence 5、attachments 5、catalog 1。
- DeepSeek 窗口按代码匹配为 65536，Qwen 为 32768。

上下文构建会去重 chunk、去重同文档重复内容、补证据 ID、路径、文件名、页码和正文，按预算截断；上下文超限时按 0.6 比例缩放后重试。

HistorySummarizer 可在历史被截断时用本地 Qwen3:4B 压缩历史，但当前 RAG_CONTEXT_COMPRESSION_ENABLED=false，通常不执行。

### 输出策略

- 温度 0.1。
- 默认最大输出 1200 token。
- 简短问题通常 512 token。
- 多证据问题通常 900 token。
- 普通问题通常 640 token。

如果检索已执行但没有相关结果，且没有附件，会直接返回“知识库中无相关内容。”，不再调用回答模型。

### 引用的真实边界

RagService 的 citations 来自最终选入上下文的 SearchHit，再加临时附件引用。每条引用带文件名、页码、章节、chunk_id、检索分数和前 500 字符摘录。

这证明的是“回答上下文使用了哪些块”，不是“每个事实主张都经过独立验证”。当前没有 claim-level verifier，引用正确性主要依赖召回、提示词和模型自律。

## 十、数据与持久化

### PostgreSQL

主要表：

- knowledge_bases：知识库、embedding_model、owner_id。
- documents：文档、路径、状态、进度、阶段、内容哈希和 chunk_count。
- document_chunks：chunk 正文、页码、章节、文件夹路径和序号。
- conversations：会话。
- messages：问题、答案、引用 JSON 和 metrics JSON。

回答成功后保存答案、引用、responseTimeMs、tokenUsage 和 timing。

### MinIO

对象键包含知识库 ID、文档 ID、source 和安全文件名。正常问答阶段不重新读 MinIO，因为 Qdrant payload 已携带检索和回答所需的正文与来源元数据。

### 删除和一致性

删除服务会清理 PostgreSQL、MinIO 和 Qdrant，但这些是跨系统操作，不是单事务。中途失败时需要额外对账和补偿机制。

## 十一、前端工作流

### 文档导入

ImportPanel 支持文件和文件夹选择：

- 从 webkitRelativePath 推导 folder_path。
- 过滤 .DS_Store、._ 和 __MACOSX。
- 上传后轮询文档状态直到 ready 或 failed。
- 浏览器本地保存导入队列快照，刷新后提示恢复。

### 问答

ChatWorkspace 支持会话创建、切换、改名、删除、普通问答、临时附件问答、附件保存到知识库、Markdown/LaTeX 渲染、证据悬浮提示，以及请求断开后的服务端答案恢复轮询。

### 评测

EvaluationDialog 支持测试集工坊和本地 JSONL，支持全部 Approved 或指定题目，并展示命中率、Recall@K、MRR、耗时、token 和 Judge 结果。

## 十二、模型职责和当前实例配置

| 环节 | 当前模型 | 职责 |
| --- | --- | --- |
| 主聊天/文件画像 | DeepSeek Chat | 文件画像和最终答案 |
| 检索决策 | Ollama Qwen3:4B | RETRIEVE/SKIP、复杂度、是否改写 |
| 查询规划 | Ollama Qwen3:4B | single/rewrite/decompose |
| embedding | SiliconFlow Pro/BAAI/bge-m3 | 文件路由和 chunk 向量 |
| reranker | SiliconFlow Pro/BAAI/bge-reranker-v2-m3 | 候选 chunk 重排 |
| 上下文压缩 | Ollama Qwen3:4B | 可选历史压缩，当前关闭 |
| Judge | 未单独配置 | 若启用且未覆盖，复用默认回答 gateway；现有产物未产生 Judge 样本 |

当前主链路是本地决策/规划 + 远程回答/embedding/rerank，多模型调用叠加会增加复杂问题时延和运行依赖。

## 十三、评测方法

实现位置：backend/src/rag_app/evaluation.py:979。

评测通过 HTTP 调真实 chat 接口：

1. 读取 JSONL 测试集，默认只取 Approved。
2. 为每题创建会话。
3. 非拒答题 force_retrieval=true。
4. 记录答案、retrieved_chunks、耗时、token 和 trace。
5. 计算文档命中、precision、recall、MRR 和 chunk 指标。
6. 有 evidence facts 时，用 embedding 比较 fact 与单 chunk 或相邻 2/3 chunk 窗口，默认阈值 0.72。
7. 配置 Judge 时，按 correctness 0.4、completeness 0.3、faithfulness 0.3 计算总分，0.7 以上通过。


## 十四、本次实际验证

本次实际执行：

- 后端：python -m pytest backend/tests -q，205 passed，4 warnings。
- 前端：npm test，13 passed。
- 前端：npm run build，Vite 生产构建成功。

这些结果证明单元测试覆盖的行为通过，不等同于真实业务数据上的答案正确率，也不能替代线上容量、权限、容错和质量评测。

## 十五、关键代码索引

- 主问答编排：backend/src/rag_app/application/rag_service.py:79
- 入库服务：backend/src/rag_app/application/ingestion_service.py:88
- 文档切分：backend/src/rag_app/infrastructure/parsing/document_parser.py:1161
- 自然边界切分：backend/src/rag_app/infrastructure/parsing/document_parser.py:1267
- 文件画像：backend/src/rag_app/application/document_profile.py:1
- 检索 Agent：agent/knowledge_retrieval_agent.py:92
- 决策 Agent：agent/retrieval_decision_agent.py:167
- 查询规划：agent/query_planning_agent.py:121
- 答案 Agent：agent/answer_agent.py:73
- 上下文工程：agent/context.py:1
- 模型工厂：backend/src/rag_app/model_gateway_factory.py:9
- Qdrant：backend/src/rag_app/infrastructure/qdrant/vector_store.py:12
- Reranker：backend/src/rag_app/infrastructure/rerank/gateway.py:13
- 评测：backend/src/rag_app/evaluation.py:979
- API 路由：backend/src/rag_app/api/routes.py:291
- 数据库 schema：backend/src/rag_app/infrastructure/postgres/schema.py:3
