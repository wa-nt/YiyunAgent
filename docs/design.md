# 第二大脑 Agent —— 设计方案

> 目标：面向大模型应用开发实习的简历项目。载体是「个人知识管理 Agent」，
> 核心卖点是四件套：**分层记忆 × 上下文治理 × Skill 化 × 评测与可观测**。
> 设计原则：每个数字有口径、每个模块有消融、功能克制不铺张。

## 1. 项目定位

个人知识库（笔记 / 面经 / 收藏文章 / 课程资料）的智能问答与主动回顾助手。
解决的真实问题：**知识库越用越笨** —— 记忆会冲突、会过期、会相互污染，
上下文会爆，检索会漏。本项目把这些"治理问题"作为一等公民来解决并用实验验证。

面试叙事线：
> "我不光建了知识库，还发现它用得越久回答越差。我把问题拆成记忆污染、
> 上下文膨胀、检索漏召回三类，分别设计了治理机制，并用自建评测集和
> 代表性消融实验验证了每一项的收益。"

## 2. 总体架构

```
┌────────────────────────────────────────────────────┐
│  接入层  FastAPI (SSE 流式) / CLI                    │
├────────────────────────────────────────────────────┤
│  Agent 运行时 (LangGraph 状态图)                     │
│  ├─ 主循环: 路由 → 检索/工具 → 生成 → 引用校验        │
│  ├─ Skill 注册表 (渐进式披露加载)                     │
│  └─ Harness: 步数/token 预算、超时、重试、checkpoint │
├────────────────────────────────────────────────────┤
│  治理层                                              │
│  ├─ Context Governor: compaction / 工具结果清理      │
│  │   / token 预算分配                                │
│  └─ Memory Governor: 写入(重要性/去重/冲突检测)      │
│      召回(语义+BM25+实体 三路融合) / 过期与纠错       │
├────────────────────────────────────────────────────┤
│  LLM 接入层 (双协议适配)                              │
│  ├─ OpenAI Chat Completions 适配器 (OpenAI/DeepSeek/ │
│  │   通义等)                                         │
│  └─ Anthropic Messages 适配器 (Claude)               │
├────────────────────────────────────────────────────┤
│  存储层                                              │
│  ├─ PostgreSQL + PGVector: 文档块 / 长期记忆 / 向量   │
│  ├─ BM25 索引 (rank_bm25 起步, 需要时换 ES)           │
│  └─ Redis: 会话状态 / checkpoint / 缓存               │
├────────────────────────────────────────────────────┤
│  观测层  OpenTelemetry trace + token 成本看板         │
└────────────────────────────────────────────────────┘
```

设计借鉴边界：记忆版本链参考 Mem0 的追加式写入和 Letta 的显式 Agent 状态；Skill
目录契约参考 Agent Skills 的 `SKILL.md` 规范；追踪与实验数据使用
OpenTelemetry/OpenInference 语义，观测后端首选本地 Phoenix。只借鉴边界和数据契约，
不引入完整平台或多用户能力。

## 3. 核心功能边界（只做这些）

| 功能 | 说明 |
|---|---|
| 采集入库 | Markdown / PDF / 网页剪藏 → 解析 → 语义切分 → 向量化入库 |
| 知识问答 | 三路混合检索 + Rerank + 生成 + 引用溯源（可跳回原文块） |
| 记忆学习 | 对话中自动抽取用户偏好与事实，写入分层记忆 |
| 主动回顾 | 基于记忆 + 间隔重复生成「本周该复习什么」 |

明确不做：多用户/权限体系、前端美化（能演示即可）、移动端、文件同步。

## 4. LLM 双协议接入层（W1 内置）

### 4.1 统一内部接口

```python
class LLMClient(Protocol):
    async def chat(
        self,
        messages: list[Message],        # 内部统一消息模型
        tools: list[ToolDef] | None,    # 内部统一工具定义
        stream: bool,
    ) -> ChatResult | AsyncIterator[ChatChunk]: ...
```

配置驱动选择后端：`llm.provider: openai_compat | anthropic`。

### 4.2 两个薄适配器

- **OpenAIChatAdapter**：基于官方 `openai` SDK，`base_url` 可配即可覆盖
  OpenAI / DeepSeek / 通义 / Moonshot 等所有 OpenAI 兼容端点。
- **AnthropicAdapter**：基于官方 `anthropic` SDK。

### 4.3 协议差异点（实现重点，也是面试素材）

| 差异 | OpenAI | Anthropic | 适配方式 |
|---|---|---|---|
| 工具定义 | `tools[].function` | `tools[].input_schema` | 统一为内部 ToolDef，双向转换 |
| 工具调用 | `message.tool_calls[]` | `content` 中的 `tool_use` 块 | 统一抽取为 ToolCall 列表 |
| 工具结果 | `role=tool` 消息 | `user` 消息内 `tool_result` 块 | 双向映射 |
| system | 消息列表内 | 独立 `system` 参数 | 抽取首条 system |
| 流式 | delta 增量 | 事件序列 (content_block_delta 等) | 统一为内部 chunk 流 |

实现纪律：**各写一个薄适配器 + 分支即可，不搞三层抽象继承。**

## 5. 分层记忆系统（Memory Governor）

### 5.1 三层结构

| 层 | 内容 | 存储 | 生命周期 |
|---|---|---|---|
| 短期记忆 | 当前会话的工具结果、中间状态 | Redis | 会话级 |
| 工作记忆 | 结构化任务状态：目标/约束/已完成/待办/错误 | Redis + checkpoint 落盘 | 任务级 |
| 长期记忆 | 用户偏好、稳定事实（带时间戳、置信度、来源引用） | PostgreSQL + 向量 | 持久 |

### 5.2 写入侧治理（大多数人漏掉的半边）

抽取管道：对话 → LLM 抽取候选记忆 → **重要性评分**（低于阈值丢弃）→
**去重**（与现有记忆向量相似度）→ **冲突检测**：

- 与旧记忆语义冲突时，不覆盖，标记冲突态：保留新旧两条 + 版本号，
  召回时优先"更近且证据更强"的一条，必要时向用户确认。
- 敏感信息（密码、密钥等）过滤不落盘。

长期记忆采用**追加式版本链**，不直接更新历史记录。最小字段为：

```text
memory_id / content / memory_type(preference|fact)
status(candidate|accepted|conflicted|superseded)
version / supersedes_id / importance / confidence
source_refs / created_at / valid_from / valid_to
```

冲突解决流程为 `candidate → accepted|conflicted → superseded`。查询读取当前有效视图，
审计和用户确认读取完整版本链；用户确认通过显式写操作完成，不使用 GET 修改状态。

### 5.3 召回侧

三路并行召回后归一化融合：

1. 向量语义相似（PGVector）
2. BM25 关键词（解决专有名词/精确匹配）
3. 实体匹配（人名、课程名、公司名）

融合后按 query 类型加权（temporal 类查询提升时间新鲜度权重）。

文档和记忆都保存来源时间字段；temporal 查询先识别时间约束，再参与召回排序。
实体检索第一版使用结构化实体字段和简单规范化，不额外引入知识图谱服务。

### 5.4 关键设计表态（面试答法）

- "为什么不用 Mem0/Zep 现成框架？" → 先跑通 Mem0 作为 baseline 对照组，
  再讲教育/求职场景下偏好 vs 事实 vs 临时状态的边界不同所以自研分层，
  **并给出两者在自建评测集上的对比数字**。
- "记忆过期怎么办？" → 时间衰减 + 置信度，高相关旧记忆的 staleness
  作为已知 limitation 写入报告。

## 6. 上下文治理（Context Governor）

三种策略，各自独立可开关（消融需要）：

1. **Compaction**：历史对话摘要压缩，保留"决策与结论"丢"过程废话"
2. **工具结果清理**：大结果（长文档/检索全文）落盘，上下文只留
   "调用发生过 + 摘要 + 引用指针"
3. **Token 预算分配**：按信息类型分区给预算（system/目标/已确认事实/
   检索证据/历史对话），超预算时按优先级裁剪

配置固定为：

```yaml
context:
  compaction: true
  tool_result_cleanup: true
  token_budget: true
```

每种策略都记录压缩前后 token、关键信息保留率、延迟和答案指标。原始对话、工作记忆
和给模型看的 prompt view 分开保存，避免治理过程覆盖可审计数据。

设计表态（对齐字节高频题"压缩放链路哪一步"）：压缩发生在**组装提示词时**
而非写入存储时 —— 存储层保留完整历史，压缩只是"给模型看的视图"。

## 7. Skill 系统

- 3-5 个真实流程封装为 `SKILL.md`（YAML frontmatter + 正文 + 可选脚本）：
  剪藏入库 / 周复习计划生成 / 面经模拟提问 / 知识盲区分析
- **渐进式披露三层加载**：name+description 常驻 → 触发时加载正文 →
  引用资源按需读取
- 触发准确率单独建测试集：应触发/不应触发样本各半，报 Precision/Recall
- 量化 Skill 化收益：同一任务 Skill 化前后 system prompt token 数对比

目录契约固定为：

```text
skills/<skill-name>/
  SKILL.md          # name、description、触发条件、输入输出、核心流程
  scripts/          # 按需执行的脚本
  resources/        # 模板、示例和参考资料
```

正文只在触发后加载，脚本和资源必须由注册表允许后才能执行或读取。

## 8. Agent 运行时（Harness）

LangGraph 状态图：路由 → 检索/工具 → 生成 → 引用校验。

运行时兜底能力：

- 步数上限 + 总 token/费用预算 + 单步超时 + 全局 deadline
- 工具错误分类重试（可重试/不可重试/降级跳过并标记）
- checkpoint 落盘，长任务中断可恢复
- 死循环检测：连续 N 步状态无进展则终止并报告

## 9. 评测与可观测（项目护身符）

### 9.1 自建评测集（150-300 条）

用自己的真实知识库（笔记 + 面经 + 收藏）构造，按 LoCoMo 分类切片：

| 类别 | 示例 | 占比 |
|---|---|---|
| single-hop | "KV Cache 是什么" | 30% |
| multi-hop | "对比我收藏的两篇讲量化的文章的结论" | 25% |
| temporal | "我上个月收藏的那篇讲 RAG 的文章" | 20% |
| 跨会话偏好 | "按我喜欢的格式总结这篇" | 15% |
| 对抗/无答案 | 知识库不存在的内容，应拒答 | 10% |

评测集以版本化 JSONL 保存。每条样本包含 `id`、`category`、`query`、
`expected_answer`、`evidence_ids`、`must_refuse`，必要时包含期望记忆。真实知识库在
开源前使用脱敏或合成替代数据。

### 9.2 指标与口径

- 检索：Recall@5、MRR（分子分母写进报告）
- 记忆：Top-K 命中率（按类别分报）、记忆污染率（错误记忆被引用的比例）
- 生成：答案准确率（LLM-as-Judge + rubric，人工抽检校准）、引用忠实度
- 效率：每 query 平均 token、P95 延迟、单 query 成本（元）

### 9.3 消融矩阵（核心交付物）

主矩阵覆盖记忆、上下文治理和检索的代表性组合：

- 记忆：关闭 / 开启；
- 上下文治理：无治理 / compaction / 工具结果清理 / token 预算 / 三策略全开；
- 检索：纯向量 / 混合检索。

每种上下文策略至少做一次单因素对照，完整组合再作为最终配置；每格使用同一数据集、
模型和预算，报告准确率、引用忠实度、Recall@5、MRR、token 成本、P95 延迟和记忆污染率。

### 9.4 可观测

OpenTelemetry span 级埋点：每次 LLM 调用 / 工具调用 / 重试 / 降级，
归因 token 与成本到模块维度，导出到 Phoenix（基于 OpenTelemetry/OpenInference）。
项目只实现必要的成本汇总页，不自建完整观测平台。

## 10. API 概要

| 接口 | 说明 |
|---|---|
| `POST /ingest` | 导入文档（md/pdf/url） |
| `POST /chat` | SSE 流式问答 |
| `GET /memories` | 查看冲突记忆和版本链 |
| `POST /memories/{id}/resolve` | 接受、保留或 supersede 冲突记忆 |
| `POST /review/plan` | 生成周复习计划 |
| `GET /eval/report` | 评测结果与消融矩阵 |

## 11. 技术栈

Python 3.12 / FastAPI / LangGraph / openai + anthropic SDK /
PostgreSQL + PGVector / rank_bm25 / Redis / OpenTelemetry + Phoenix / Docker。
LLM：DeepSeek 或通义 API 为主（成本低），Claude 走 Anthropic 适配器做对照。

## 12. 10 周排期

| 周 | 交付 | 验收 |
|---|---|---|
| W1 | 项目骨架、配置、数据库迁移、统一消息模型、双协议接入层 | 服务可启动，两种协议均可完成最小调用 |
| W2 | Markdown/PDF 入库、切分、向量索引、引用定位 | 文档可导入并返回证据块 |
| W3 | 向量 + BM25 + 基础实体检索 | 基础混合检索 baseline 可复现 |
| W4 | 三层记忆模型、短期/工作记忆和 checkpoint | 三层记忆均可读写 |
| W5 | 重要性、去重、冲突检测和版本链 | 冲突不覆盖，旧新版本均可查询 |
| W6 | compaction、工具结果清理、token 预算分配 | 三策略独立开关并记录指标 |
| W7 | 3-5 个渐进式披露 Skill | 剪藏、周复习、面经流程可运行 |
| W8 | Skill 触发评测、150-300 条评测集整理 | Precision/Recall 和数据版本可生成 |
| W9 | OpenTelemetry、Phoenix、成本看板和失败记录 | trace 可关联模型、工具、检索和成本 |
| W10 | 消融矩阵、报告、演示、复现脚本 | `make eval` 生成完整结果和原始产物 |

每周保留一次回归和数据校验；模型供应商故障、评测失败和数据清洗使用额外缓冲处理，
不通过删除评测样本或放宽评分规则制造通过结果。

可选增强（有余力再做）：服务层暴露 `/v1/chat/completions` +
`/v1/messages` 双协议入口，项目升级为可被 Claude Code / Cursor
直接接入的 Agent 后端（Infra 向加分 bullet）。

## 13. 简历 bullet（做完填真实数字）

- 设计三层记忆系统（短期/工作/长期），写入侧实现重要性判断、去重与
  冲突检测，temporal 类查询召回率提升 __%，记忆污染率从 __% 降至 __%
- 实现上下文三策略治理（compaction / 工具结果清理 / token 预算分区），
  单会话 token 成本下降 __%，P95 延迟 __s
- 封装 LLM 双协议接入层（OpenAI Chat Completions / Anthropic Messages），
  统一流式与工具调用差异，支持多模型配置切换与 A/B 对比
- 自建 __ 条评测集（参考 LoCoMo 分类），代表性消融实验验证各模块贡献；
  OpenTelemetry 全链路 trace，token 成本按模块归因

## 14. 已知 Limitation（写进报告，面试主动讲）

- 单用户场景，未做多租户与权限隔离
- 记忆 staleness 只做时间衰减，高相关旧事实的"自信地错误"是开放问题
- 评测集规模有限（__ 条），结论按置信区间解读
- LLM-as-Judge 存在长度/位置偏差，已用人工抽检 __ 条校准
- 真实个人知识库不直接随代码开源，公开复现使用脱敏或合成评测数据
- Phoenix 等观测系统记录的提示词和文档内容需要按部署环境配置脱敏策略
