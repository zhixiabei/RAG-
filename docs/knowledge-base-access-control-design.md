# 知识库权限与删除功能设计

## 1. 文档状态

| 项目 | 内容 |
| --- | --- |
| 状态 | 待实施 |
| 适用范围 | 知识库、文档、对话和问答接口 |
| 目标 | 隔离个人知识库，提供只读总知识库，并支持安全删除 |
| 当前实现 | 已有知识库、文档和历史对话删除能力；尚未包含用户认证和资源归属 |

本文档描述后续功能的目标方案，不代表这些能力已经上线。

## 2. 目标与边界

### 2.1 目标

1. 每个登录用户只能看到和使用自己的个人知识库，以及系统总知识库。
2. 个人知识库的所有者可以创建和删除知识库，也可以上传和删除其中的文档。
3. 普通用户可以查看、检索和问答总知识库，但不能向总知识库上传或删除内容。
4. 管理员负责维护总知识库，可以上传和删除其中的文档。
5. 后端对每个资源接口执行权限校验，不能依赖前端隐藏按钮保证安全。
6. 删除时同步清理 PostgreSQL、MinIO 和 Qdrant 中的关联数据，并支持失败重试。

### 2.2 暂不包含

- 知识库共享给指定用户或部门。
- 文档级单独授权。
- 多组织、多租户计费与配额。
- 用户自行注册和找回密码。
- 复杂角色策略编辑器。

## 3. 核心概念

### 3.1 用户角色

| 角色 | 说明 |
| --- | --- |
| `user` | 普通用户，管理自己的个人知识库，只读使用总知识库 |
| `admin` | 管理员，可维护总知识库，并可处理异常个人知识库 |

若未来需要限制总知识库本身的删除，可以再增加 `super_admin`；第一阶段也可以直接规定总知识库不可通过 API 删除。

### 3.2 知识库类型

| 类型 | `scope` | 所有者 | 说明 |
| --- | --- | --- | --- |
| 个人知识库 | `personal` | 必须有 `owner_id` | 仅所有者和管理员可访问 |
| 总知识库 | `global` | 可为空或指向系统账号 | 所有登录用户可读，仅管理员可写 |

系统只保留一个总知识库。总知识库应通过初始化脚本或管理员操作创建，普通创建接口只创建个人知识库。

## 4. 权限矩阵

| 操作 | 个人知识库所有者 | 其他普通用户 | 管理员 | 总知识库普通用户 |
| --- | --- | --- | --- | --- |
| 查看知识库 | 允许 | 拒绝 | 允许 | 允许 |
| 查看文档列表 | 允许 | 拒绝 | 允许 | 允许 |
| 检索和问答 | 允许 | 拒绝 | 允许 | 允许 |
| 查看自己的对话 | 允许 | 拒绝 | 允许 | 允许 |
| 上传文档 | 允许 | 拒绝 | 允许 | 拒绝 |
| 删除文档 | 允许 | 拒绝 | 允许 | 拒绝 |
| 删除知识库 | 允许 | 拒绝 | 允许 | 拒绝 |

补充规则：

1. 普通用户不能读取其他用户的个人知识库，即使知道知识库、文档或对话 ID。
2. 总知识库普通用户只能读取、检索和问答，不能创建、上传、修改或删除其中的内容。
3. 对话属于创建该对话的用户。总知识库是共享资料库，但用户的对话记录不是共享数据。
4. 管理员是否可以查看个人对话内容需要单独审计；默认仅用于故障处理，不在普通管理页面展示。

## 5. 身份认证

优先接入公司现有的 OIDC、OAuth2 或统一身份系统。后端验证访问令牌后生成当前用户上下文：

```text
CurrentUser
- id
- external_subject
- display_name
- role
```

浏览器请求使用：

```http
Authorization: Bearer <access-token>
```

安全要求：

1. 不接受前端提交的 `owner_id` 作为资源所有者。
2. 创建个人知识库时，后端必须使用 `current_user.id` 写入 `owner_id`。
3. 不使用可伪造的 `X-User-Id` 请求头作为正式认证方式。
4. 如果项目自行管理密码，密码只能保存 Argon2 或 bcrypt 哈希，不能明文存储。
5. 所有业务接口默认要求登录，只有健康检查和登录回调可以匿名访问。

## 6. 数据模型调整

### 6.1 用户表

```
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    external_subject TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_users_role CHECK (role IN ('user', 'admin'))
);
```

### 6.2 知识库表

在 `knowledge_bases` 增加：

```
ALTER TABLE knowledge_bases
    ADD COLUMN owner_id TEXT REFERENCES users(id),
    ADD COLUMN scope TEXT NOT NULL DEFAULT 'personal',
    ADD COLUMN status TEXT NOT NULL DEFAULT 'active';

ALTER TABLE knowledge_bases ADD CONSTRAINT ck_knowledge_base_scope
    CHECK (scope IN ('personal', 'global'));

ALTER TABLE knowledge_bases ADD CONSTRAINT ck_knowledge_base_owner
    CHECK (
        (scope = 'personal' AND owner_id IS NOT NULL)
        OR scope = 'global'
    );

CREATE INDEX idx_knowledge_bases_owner ON knowledge_bases(owner_id);
CREATE UNIQUE INDEX uq_single_global_knowledge_base
    ON knowledge_bases(scope) WHERE scope = 'global';
```

`status` 预留 `active`、`deleting` 和 `delete_failed`，用于跨存储删除任务。

### 6.3 对话表

在 `conversations` 增加 `owner_id`：

```
ALTER TABLE conversations
    ADD COLUMN owner_id TEXT REFERENCES users(id);

CREATE INDEX idx_conversations_owner
    ON conversations(owner_id, updated_at DESC);
```

文档不重复保存所有者。文档权限通过所属知识库的 `scope` 和 `owner_id` 判断。

### 6.4 外键清理规则

PostgreSQL 内部关系建议设置级联删除：

```text
knowledge_bases
  -> documents
    -> document_chunks
  -> conversations
    -> messages
```

数据库级联只负责 PostgreSQL，MinIO 对象和 Qdrant 向量仍需由删除服务显式清理。

## 7. 后端权限设计

在 API 层统一获取 `current_user`，在应用层集中实现权限判断：

```
python
require_knowledge_base_read(user, knowledge_base)
require_knowledge_base_write(user, knowledge_base)
require_knowledge_base_delete(user, knowledge_base)
require_conversation_access(user, conversation)
```

判断原则：

```
text
个人知识库读取/写入：owner_id == current_user.id，或当前用户是管理员
总知识库读取：任意已登录用户
总知识库写入：仅管理员
总知识库删除：默认禁止
```

权限检查必须覆盖：

- 知识库获取、列表和删除。
- 文档列表、上传和删除。
- 对话创建、列表和消息读取。
- 问答和向量检索。
- 管理员维护接口。

资源不存在和无权访问个人资源时，建议统一返回 `404`，避免通过 `403` 枚举其他用户的资源。对已知总知识库的只读限制可以返回 `403` 并给出明确说明。

## 8. API 调整

### 8.1 知识库

```http
GET    /api/v1/knowledge-bases
POST   /api/v1/knowledge-bases
GET    /api/v1/knowledge-bases/{knowledge_base_id}
DELETE /api/v1/knowledge-bases/{knowledge_base_id}
```

- `GET` 只返回当前用户自己的个人知识库和总知识库。
- `POST` 只创建个人知识库，所有者取自当前登录用户。
- `DELETE` 仅允许个人库所有者或管理员调用，总知识库默认不可删除。

### 8.2 文档

```http
GET    /api/v1/knowledge-bases/{knowledge_base_id}/documents
POST   /api/v1/knowledge-bases/{knowledge_base_id}/documents
DELETE /api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}
```

- 个人库所有者可以上传和删除文档。
- 普通用户只能查看总知识库文档。
- 管理员可以上传和删除总知识库文档。
- 删除文档时必须同时校验文档确实属于路径中的知识库。

### 8.3 对话和问答

现有对话和问答接口增加当前用户校验：

```http
POST /api/v1/knowledge-bases/{knowledge_base_id}/chat
GET  /api/v1/knowledge-bases/{knowledge_base_id}/conversations
POST /api/v1/knowledge-bases/{knowledge_base_id}/conversations
GET  /api/v1/conversations/{conversation_id}/messages
```

总知识库可以被所有用户问答，但对话列表只能返回当前用户自己的对话。

## 9. 删除流程

### 9.1 删除文档

文档数据同时存在于三个存储中：

1. PostgreSQL：文档和 chunk 元数据。
2. MinIO：原始文件。
3. Qdrant：文档 chunk 向量。

建议流程：

```text
鉴权
  -> 将文档状态标记为 deleting
  -> 按 document_id 删除 Qdrant points
  -> 按 source_object_key 删除 MinIO object
  -> 删除 PostgreSQL 文档记录及关联 chunks
  -> 返回成功
```

删除方法必须幂等：目标已经不存在时也应视为清理完成。发生部分失败时，记录失败原因并允许后台任务重试。

### 9.2 删除个人知识库

```text
鉴权并阻止删除总知识库
  -> 将知识库标记为 deleting
  -> 禁止新的上传、问答和对话创建
  -> 清理该知识库全部 Qdrant points
  -> 清理该知识库全部 MinIO objects
  -> 删除 PostgreSQL 知识库，级联删除文档、chunks、对话和消息
```

数据量较小时可以同步执行。正式环境建议使用删除任务或 outbox：API 返回 `202 Accepted`，后台任务执行并重试，防止跨存储操作中断后产生残留。

### 9.3 存储接口补充

对象存储端口增加：

```
python
delete_object(object_key: str) -> None
delete_prefix(prefix: str) -> None
```

向量存储端口增加：

```
python
delete_document(document_id: str) -> None
delete_knowledge_base(knowledge_base_id: str) -> None
```

Qdrant 删除必须使用 payload filter，分别按 `document_id` 或 `knowledge_base_id` 清理。

## 10. 前端设计

### 10.1 导航和展示

侧栏分组展示：

```text
总知识库
  公司总知识库

我的知识库
  用户创建的个人知识库
```

总知识库显示只读标识。个人库显示所有者可用的管理操作。

### 10.2 操作控制

- 个人知识库：显示上传、删除文档、删除知识库按钮。
- 总知识库普通用户：隐藏上传和删除按钮，保留文档查看与问答。
- 总知识库管理员：显示上传和删除文档按钮。
- 删除操作使用确认对话框，并显示将被删除的知识库或文档名称。
- 删除中禁用重复操作，失败后显示后端返回的可理解错误信息。
- 前端按钮控制只改善体验，后端仍必须执行完整鉴权。

## 11. 现有数据迁移

建议按以下顺序迁移：

1. 创建系统管理员用户。
2. 给表增加允许为空的 `owner_id`、`scope` 和 `status` 字段。
3. 明确现有知识库归属，将其分配给管理员或指定用户。
4. 从现有会话来源补齐 `conversations.owner_id`；无法确认的历史会话归档给管理员。
5. 创建唯一的总知识库，或将指定的现有知识库转换为总知识库。
6. 补齐所有数据后增加检查约束和索引。
7. 上线鉴权接口后再开放删除按钮。

迁移必须先备份 PostgreSQL、MinIO 和 Qdrant。不要在没有所有者映射的情况下直接把所有历史知识库暴露给所有用户。

## 12. 审计与安全

建议记录以下审计事件：

- 用户创建或删除知识库。
- 用户上传或删除文档。
- 管理员修改总知识库。
- 被拒绝的越权写入和删除请求。

审计字段至少包含：用户 ID、操作、资源类型、资源 ID、时间、结果和请求追踪 ID。审计日志不应保存访问令牌或完整文档内容。

## 13. 测试与验收

### 13.1 后端权限测试

1. 用户 A 可以创建个人知识库并上传、删除其中的文档。
2. 用户 A 不能查看、问答、上传或删除用户 B 的个人知识库。
3. 用户 A 即使直接构造知识库、文档或对话 ID，也不能越权访问。
4. 普通用户可以查看和问答总知识库，但上传与删除返回 `403`。
5. 管理员可以上传和删除总知识库文档。
6. 总知识库不能通过普通知识库删除接口删除。
7. 总知识库中的用户 A 对话不会出现在用户 B 的对话列表中。

### 13.2 删除一致性测试

1. 删除文档后，PostgreSQL、MinIO 和 Qdrant 均无残留。
2. 删除个人知识库后，其文档、chunks、向量、对象、对话和消息均被清理。
3. 重复执行同一删除任务不会报错或误删其他知识库数据。
4. MinIO 或 Qdrant 临时失败后，删除任务能够重试并最终完成。
5. 删除进行中不能继续上传文档或发起问答。

### 13.3 前端验收

1. 普通用户只看到总知识库和自己的个人知识库。
2. 个人库正常显示上传和删除操作。
3. 总知识库对普通用户不显示任何写入或删除操作。
4. 删除前显示明确确认信息，删除后列表和当前选中项正确更新。
5. 接口返回 `401` 时进入登录流程，返回 `403` 时显示权限提示。

## 14. 推荐实施顺序

1. **身份认证**：接入登录系统，建立 `CurrentUser`。
2. **数据归属**：增加用户、知识库所有者、知识库类型和对话所有者字段。
3. **读取隔离**：先保护知识库、文档、对话和问答查询，关闭越权读取。
4. **写入鉴权**：限制个人库和总知识库的创建、上传操作。
5. **文档删除**：补齐 PostgreSQL、MinIO、Qdrant 的单文档清理。
6. **知识库删除**：增加状态机、后台任务和跨存储重试。
7. **前端权限界面**：增加分组、只读标识、删除确认和错误反馈。
8. **审计与验收**：补齐权限回归测试、删除一致性测试和审计日志。

读取隔离应早于删除功能上线，否则系统即使有删除按钮，仍存在通过直接调用 API 越权访问数据的风险。
