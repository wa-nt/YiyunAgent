# SecondBrainAgent 产品审视报告（两轮合并 · 逐条源码验证版）

- 日期：2026-10-04
- 来源：两轮独立审视（第一轮：UX/功能/流程/性能/可用性全量走查；第二轮：外部深度报告），本轮对两轮全部可验证断言逐条对照当前工作区源码复核后合并。
- 验证方式：grep 计数、vendor marked.min.js 实机重放（mermaid 管线）、pytest 全量、数据库行数统计、逐行读 `app/` 与 `web/index.html`。
- 图例：✅ 已验证存在 ｜ ⚠️ 结构成立、实测数字无法在当前工作区复现（本地库近乎为空：1 chunk / 1 会话）｜ ❌ 不成立或需更正

**TL;DR**：上一轮路线图（P0-P2）已全部落地且 566 个测试全绿。当前真正拦住重度用户的是：输入法选词即发送（中文用户每天踩）、消息/代码零复制按钮、数据只能出不能进（导出格式无导入方）、流式渲染 O(n²) + 强制滚动、LLM 调用零超时（一次卡住的 provider 晾用户 30 分钟）、mermaid 是一条看得见的坏图路径且占 vendor 85%、长会话每轮重复付一次摘要 LLM 调用。多数修复是小 diff。

---

## 一、交互流程

### 1.1 输入法选词按 Enter 直接把消息发出去 ✅
- 证据：`web/index.html:1736` 仅判 `e.key === "Enter" && !e.shiftKey`；全文 `isComposing|keyCode` **0 命中**（已 grep 验证）。
- 影响：UI 是 `lang="zh-CN"`，Chrome 确认拼音时仍触发 keydown Enter——用户按 Enter 选词，半截拼音直接发出。这是中文用户最高频的劝退点。
- 建议：守卫加 `if (e.isComposing || e.keyCode === 229) return;`，一行。

### 1.2 没有任何复制按钮 ✅（两轮均发现）
- 证据：全文 `copy` 0 命中（大小写不敏感）；唯一 `clipboard` 引用在 `index.html:1150`（读粘贴的文件）。`msg-actions` 只有 编辑/重新生成（`index.html:924-931`）；hljs 已接上（`:1015`）但代码块无复制。
- 影响：抄答案/抄代码是重度用户点击频率最高的动作，竞品全是标配。
- 建议：消息 actions 加「复制」；`pre` 右上角加代码复制钮。

### 1.3 编辑必连带重新生成，没有「只保存」路径 ✅
- 证据：`editMessage()` 保存成功后无条件 `respond()`（`index.html:1644-1645`）。
- 影响：只想改个错字也要付一次完整生成（token + 等待 + 分支膨胀）。
- 建议：编辑行加「仅保存」，或保存后询问是否重答；后端 `/api/sessions/{id}/respond` 本就可独立调用，纯前端拆分。

### 1.4 ＋ 按钮名不副实 ✅
- 证据：`attach-btn` 的 onclick 是 `focusIngest()`（`index.html:1294`、`:1287-1293`），打开的是侧栏知识库导入框，不是给消息加附件；title 写了「导入文档」但视觉直觉是「附件」。
- 建议：要么改图标/文案为「入库」，要么真做成消息附件（后者涉及多模态，本期建议维持 YAGNI、改文案）。

### 1.5 / 命令面板只有技能、没有内置命令 ✅
- 证据：面板数据源仅 `/api/skills`（`index.html:1746`），选中 = 把首个触发词填进输入框（`:1773-1780`）；无 `/new`、`/export`、`/model` 等本地命令。
- 建议：保持现状可接受；若做内置命令，client-side 拦截即可，不必动后端。

### 1.6 会话列表没有空态 ✅
- 证据：其余 5 个列表（文档/记忆/漏洞/任务/通知）都有空态文案；唯独 `loadSessions()` 在 `index.html:1161` 直接 `innerHTML = ""`，搜不到时一片沉默空白。
- 建议：补一行「没有匹配的对话」。

### 1.7 生成期间整个应用被锁死 ✅（第一轮发现）
- 证据：全局 `busy` 挡住切会话（`index.html:1219`）、新对话（`:1376`）、切模式（`:881`）、编辑/重生成（`:1622`、`:1651`）。
- 影响：长回答跑几十秒，用户连看别的会话都不行。ChatGPT/Claude 均可边流式边导航。
- 建议：busy 改 per-request（每流一个 controller、渲染到各自容器）；最低限度放行「切会话/新对话」（后台流照常落库，回来 `refreshMessages()` 恢复）。

### 1.8 强制滚动到底，和用户抢滚动条 ✅（第一轮发现）
- 证据：每个 text_delta 调 `scrollBottom()`（`index.html:1703`、`:909-911`）。
- 建议：距底 >80px 时暂停自动滚动 + 「回到底部」浮钮。

### 1.9 原生 `prompt()`/`confirm()` + ESC 关不掉弹层 ✅（两轮均发现）
- 证据：重命名/记忆编辑用 `window.prompt`（`index.html:1173`、`:1408`）；Escape 只关 / 面板（`:1734`），引用气泡（`:1037`）、设置/任务面板（`:1858`、`:1581`）均无 keydown 处理。
- 建议：行内编辑替代 prompt；两个 mask 监听 ESC。

## 二、渲染

### 2.1 Mermaid 是死代码，且是用户看得见的坏图 ✅（本轮实机复现）
- 复现结果（用 vendored marked.min.js + 文件原样的 `esc()`/`html()` 渲染器）：
  ```
  marked output: "<p>前文一段。</p>\n&lt;pre class=&quot;mermaid-slot&quot;&gt;&lt;/pre&gt;\n\n<p>后文一段。</p>\n"
  querySelectorAll can find: false
  ```
- 链路：`:988-990` 把 ```mermaid 块换成原始 HTML 占位符 → `:993` 交给 marked → `:889` 自定义 `html(token){return esc(token.text)}` 把它转义成**可见字面量** → `:1029` `querySelectorAll("pre.mermaid-slot")` 拿到空集合，mermaid.run 永不执行。
- 代价：`web/vendor/mermaid.min.js` 实测 **2,571,900 B**，vendor 主体（不含 katex 字体）约 3.04 MB，占 **约 85%**（外部报告称 76%/3.3MB，数字略偏，结论不变）。
- 建议：删掉 mermaid（回收 2.5MB + 消灭唯一可见坏图），或改用 data 属性传递源码而不是原始 HTML 占位符。

### 2.2 表格与图片样式缺失——知识库核心内容类型渲染坏了 ✅
- 证据：`.msg-body table|th|td|img` **0 命中**（已 grep 验证）。
- 影响：GFM 表格无边框无内边距散排；图片无 max-width，会撑破 760px 的 `.chat-inner`（`:281`）横向破坏布局。
- 建议：十几行 CSS。表格建议加 `border-collapse` + 发丝线边框；img 加 `max-width:100%`。

## 三、功能完整性

### 3.1 数据只能出不能进（本轮最严重共识项）✅（两轮均发现）
- 证据：`/api/export` 产出 `version`/`exported_at`（`main.py:484-485`）但**无任何消费方**（全库各出现恰 1 次，已验证）；唯一导入口 `/api/import/chatgpt`（`main.py:572`）只嗅探 ChatGPT `mapping` 与 Claude `chat_messages`，与自家格式不匹配。
- 影响：能备份、永远不能还原；导出格式也不是任何第三方标准，双向单向门。
- 建议：`/api/import` 按 `version` 识别自家格式原样回灌；顺手在 UI 导入按钮旁说明支持的三种格式。

### 3.2 全硬删除，confirm() 是唯一防线 ✅
- 证据：`deleted_at` 全库 0 命中；`main.py:305/360/462/737/824` 均立即 DELETE。
- 建议：消息/会话先做软删除（`deleted_at` 列 + 列表过滤），30 天后物理清理。

### 3.3 无自动备份、迁移不可检测 ✅
- 证据：`PRAGMA user_version` 全库 0 命中；迁移是裸 ALTER TABLE 堆（`db.py:53-124`），中途失败不可回滚也不可检测。
- 建议：至少记 `user_version`；「下载数据库」旁加一键定时备份（复制 app.db 到 backups/，保留 N 份）。

### 3.4 会话管理只有扁平列表 ✅（两轮均发现，角度互补）
- 证据：无文件夹/标签/置顶/归档；搜索是 LIKE（`main.py:241` 注释自认「量大了再考虑 FTS」）；后端已返回 `mode`/`source`/`created_at`（`main.py:244-256`）但前端只渲染标题（`index.html:1162-1194`）——定时任务自动会话与手动会话分不清，无日期分组。
- 建议：先做零成本的（列表项渲染 mode 徽章 + 今天/昨天分组），FTS 等有量级证据再说。

### 3.5 技能只能祈祷措辞命中，无手动运行入口 ✅
- 证据：仅 `GET /api/skills`；触发靠关键词 + 0.7 占优阈值（`app/skills/trigger.py:38-77`、`config.py:81`）。
- 建议：`POST /api/skills/{name}/run` 一个端点即可让用户显式触发。

### 3.6 付费的「思考强度」看不到任何产出 ✅
- 证据：UI 四档（`index.html:732-737`）映射到 `reasoning_effort`/`thinking.budget_tokens`（`openai_compat.py:109-113`、`anthropic.py:114-121`），但 SSE 只发 5 种事件（`runtime.py:129` 注释与实现一致），推理过程从不展示。
- 影响：选 Max、付了钱、只见最终答案——比不摆这个开关更糟。
- 建议：要么把 reasoning/thinking 流出来（Anthropic 的 thinking delta、OpenAI 的 reasoning summary 均可透传），要么先藏掉开关。

### 3.7 code 模式是诚实禁用桩 ✅
- 证据：按钮灰着写「即将推出」（`index.html:716`），后端显式 400（`main.py:96-97`）。
- 结论：诚实，无需改动；对写代码主力用户是评估终点，属已知边界。

### 3.8 文档导入后不可预览 ✅（第一轮发现）
- 证据：文档列表只有标题/chunk 数/删除（`index.html:1082-1095`），点击无反应；「为什么没检索到」只能靠猜。
- 建议：加按 doc 列 chunk 的接口 + 点击展开预览（单条 `/api/chunks/{id}` 已有）。

### 3.9 定时任务要手写 cron、无下次运行预览 ✅（第一轮发现）
- 证据：表单仅一个 cron 输入框（`index.html:810-812`）；croniter 已是依赖。
- 建议：常用预设 + 输入时显示「下次运行时间」。

### 3.10 首次启动无引导 ✅（第一轮发现）
- 证据：未配 key 时 hero 照常欢迎，首条消息只换来一条 6 秒后消失的报错。
- 建议：页面加载时探测 `/api/settings`，未配置显示引导卡直通设置面板。

## 四、性能

### 4.1 LLM 调用完全没有超时 ✅（最高优先级后端项）
- 证据：`timeout|max_retries` 在 `app/llm/` **0 命中**（已 grep 验证）；已安装 openai **3.19.2**（`.venv` 实测），默认 read=600s、max_retries=2。
- 影响：chat_stream 首字节前无任何产出——provider 卡一次，空白气泡干等最长 30 分钟，无进度无报错。
- 建议：三个 client 构造时传 `timeout=httpx.Timeout(connect=10, read=120)` + `max_retries=1`，几行。

### 4.2 每个 token 全量重解析整个回答 ✅（两轮均发现）
- 证据：每个 text_delta 调 `renderCitations`（`index.html:1700-1703`）：marked 全文 parse + innerHTML 重写 + hljs 全量 + KaTeX + mermaid.run + 强制 `scrollBottom()`；全文 `requestAnimationFrame` 0 命中。
- 影响：结构性 O(n²)，2000 token 回答 ≈ 2000 次完整解析，越到后面越卡。
- 建议：rAF/150ms 节流重渲染；hljs/mermaid 推迟到 done 后跑一次。

### 4.3 检索冷加载与串行 LIKE ⚠️（结构全部核实，数字未复现）
- 证据（结构已核实）：
  - `pipeline.py:52/80` 每次 ingest/删文档无条件 `invalidate()`，下次搜索重付全额冷加载；
  - `entity_search.py:39-46` 对每个 token 串行 `await` 一次 `LIKE '%x%'` 全表扫；
  - embedding 在 `hybrid.py:28` 先于三路 gather 串行执行。
- 外部报告实测（20,000 chunks：BM25 冷加载 2577ms、entity_search 1012ms）**无法在本工作区复现**（本地库 1 chunk），但代码结构支撑量级结论。
- 建议：`chunks.content` 建 FTS5 索引替掉逐 token LIKE；embedding 与三路并行。

### 4.4 BM25 在事件循环上做同步计算 ✅（外部报告的「亮点」描述需更正 ❌）
- 证据：`BM25Index.search` 是同步 `def`（`bm25_search.py:64`，numpy + 纯 Python sorted 遍历全部 chunk），`bm25_search()` 直接调用（`:133`），阻塞事件循环。
- **更正**：外部报告称「唯一做对的是 load() 走了 asyncio.to_thread」——不成立。全库 `to_thread` 仅 1 处且在 `app/ingest/pipeline.py:21`（文件读取），BM25 的 load/search 都在事件循环上。问题比报告说的更重，不存在该亮点。
- 建议：search 包 `asyncio.to_thread`；量大后 load 同理。

### 4.5 busy_timeout 只在托盘线程有 ✅
- 证据：全库唯一一处 `desktop.py:135`（只读连接）；`db.py:41` 只设 WAL + foreign_keys。
- 影响：HTTP 写入路径 busy_timeout=0，写者相撞立即 `database is locked` 而不是等待。
- 建议：`connect()` 里加 `PRAGMA busy_timeout = 5000`，一行。

### 4.6 messages.parent_id 无索引 ✅
- 证据：`schema.sql:98` 仅 `idx_messages_session_id`；`parent_id` 却是递归 CTE 的 join key（`runtime.py:283-293`、`main.py:433`）。
- 建议：加 `CREATE INDEX idx_messages_parent_id`，一行（外部报告实测长会话 20.2ms 的数字未复现，结构成立）。

### 4.7 前端按钮永久锁死风险 ✅
- 证据：`index.html:1103/1123/1590/1953/1997` 五处 try 内禁用、finally 恢复，但 fetch 无超时无 AbortSignal——请求挂起则「上传中…」卡到刷新。
- 建议：统一 fetch 包装加 AbortSignal.timeout（如 60s）；大文件导入另做后台任务 + 进度（ingest 目前同步阻塞 HTTP 请求，50MB PDF 期间无进度无取消）。

### 4.8 长会话每轮重复付一次摘要 LLM 调用，且阻塞首 token ✅（第一轮发现）
- 证据：`HISTORY_LIMIT=20`、压缩阈值 10（`runtime.py:37`、`config.py:54`）；治理产物不落库（`context.py:7-9`）——会话超 10 条消息后，**每轮** `/api/chat` 在流式开始前同步调一次 LLM 摘要（`runtime.py:691`、`context.py:107-131`）。
- 影响：首 token 多等几秒，token 成本每轮重复。
- 建议：摘要随 session 持久化（存摘要文本 + 压缩位置），下轮增量压缩。

### 4.9 每轮结束 4 个刷新请求 + 会话接口无 LIMIT ✅（第一轮发现）
- 证据：流结束 finally 连发 4 个请求（`index.html:1716-1724`）；`loadSessionMode`/`loadSessionModel` 各拉**全量** `/api/sessions` 再 find（`:1245-1247`、`:1269-1271`）；`/api/sessions` 无 LIMIT（`main.py:236`）。
- 建议：加 `GET /api/sessions/{id}` 单会话接口；列表加 limit。
- 量级参考：外部报告实测 300 会话/12000 消息下 `/api/sessions` 1.05ms、recall 13ms——当前规模无感，属「量大了再付」项。

### 4.10 token 估算偏乐观 ✅（第一轮发现）
- 证据：CJK 1:1、其余 4 字符/token（`context.py:33` 注释自认低估中文）；8k 预算对中文可能实际超限。traces 已存真实 usage，可校准。

## 五、可用性

### 5.1 刷新后几乎全丢 ✅
- 证据：localStorage 仅 `brain.theme` 与 sessionId 两键（`index.html:842/851`，已验证无其他 setItem）；0 处 IndexedDB。草稿输入、当前模式、侧栏折叠全丢。
- 建议：输入草稿防抖写 localStorage，一行级改动，收益大。

### 5.2 错误 6 秒自动消失、无重试 ✅（两轮均发现）
- 证据：`showError` setTimeout 6s（`index.html:902`）；SSE 出错时半截回答已落库（`runtime.py:741` degraded 路径），toast 消失后无「重试」入口。
- 建议：错误条常驻至手动关闭 + 重试按钮（后端 `/respond` 已支持）。

### 5.3 无障碍基本为零 ✅（两轮均发现）
- 证据：`aria-*` 全文 1 处（人格 textarea 的 aria-label）、`role=`/`tabindex` 0 处、图标按钮只有 title；流式正文无 `aria-live`（读屏完全感知不到生成）。
- 建议：按底线补：stream 容器 `aria-live="polite"`、图标按钮 aria-label、弹层 ESC + 焦点管理。属不可 YAGNI 项。

### 5.4 中英混排 + 硬编码中文 ✅（两轮均发现）
- 证据：思考强度 Off/Low/High/Max（`index.html:733-736`）混在中文 UI 里；整体无 i18n。i18n 维持 YAGNI 砍掉，但四档文案应本地化。

### 5.5 离线/断连无处理 ✅
- 证据：`navigator.onLine`、`offline` 事件、503、timeout 全部 0 命中。配合 4.1/4.7：后端不可达时整个 app 无声卡住。
- 建议：全局 fetch 包装 + 顶部连接状态条。

### 5.6 单 `$` 被当数学定界符 ✅（第一轮发现）
- 证据：`renderMathInElement` 开单 `$`（`index.html:1021-1022`），「花了 $5 和 $10」会被渲染成公式。
- 建议：关单 `$` 或加负向断言。

### 5.7 细节缺口（第一轮发现，均小改动）✅
- 模型 chip 每次展开重拉 `/api/models`，失败静默降级（`index.html:1306-1323`）→ 内存缓存，设置保存后失效；
- 每轮 4 个刷新请求（见 4.9）；
- 消息不显示时间（`created_at` 落库但 UI 不渲染）；
- `/api/traces` 后端齐全（`main.py:830-867`）但无任何 UI，成本归因无从下手。

## 六、工作区未提交改动（392+/113-，11 文件）注意事项 ✅

已验证整体是改进而非半成品：被删 DOM 无悬空引用、新 id 的 handler 都在、**566 个测试全绿（本轮实测复现，41.78s）**。三处需注意：

1. **new-session 漏重置新增全局变量** ✅：`resetStaleSession()` 有 `sessModel = ""; sessEffort = "";`（`index.html:1257-1259`），但 `new-session`（`:1375-1385`）只清 sessionId/sessionMode；`loadSessionModel` 因 sessionId 为 null 提前 return（`:1267`）——开过设模型的会话再点「新对话」，chip 一直挂着旧模型标签直到发第一条消息。一行修复。
2. **Enter 发送提示被删、行为还在** ✅：占位符已是「有什么可以帮你的吗？」（`:709`），Enter 逻辑在 `:1736`。修 1.1 时顺手在 hint 区补回「Enter 发送 · Shift+Enter 换行」。
3. **effort 集成链路无测试** ✅：新增测试只覆盖 `_effort_kwarg` 映射（`test_llm.py:152-187`）；DB 列 → `session_effort()` → `get_llm(effort=)` → 请求 kwargs 的缝无覆盖（`test_agent.py` 无 effort 引用）。

## 七、无法本地验证、值得手测的风险 ⚠️

- **DeepSeek + reasoning_effort**：`_effort_kwarg` 对所有 openai_compat 端点一律发 `reasoning_effort`（`openai_compat.py:109-113`），`detect_provider` 只用于定价。若 DeepSeek 拒收该参数，选了 Low/High/Max 的每一轮都 400。手测一次即知；稳妥做法是按 base_url 白名单发送。
- **marked v15 的 href trim 依赖**：XSS 白名单实测 15 向量全不执行（外部报告实测），但安全性依赖 marked 先 trim `token.href`——建议在 `index.html:886` 注释写明，防止未来升级踩坑。

## 八、验证过的亮点（两轮共识，照实说）

- **分支式消息模型**：编辑=不可变兄弟分支、删除=递归子树+active_leaf 回退修复、重生成不毁旧答案（`main.py:325-443`）——比多数竞品讲究；
- **引用链路端到端真通**：`[chunk N]` → `<cite>` → `/api/chunks/{id}` 原文+父文档标题，URL 来源给回链；
- **调度器是生产级**：创建时冻结时区 + 变更显式警告、TICK_INTERVAL 30s 幂等 claim、失败转通知不重试；
- **导入双格式正确**：ChatGPT 树形（沿 children[0] 主分支拉直）与 Claude 线性都处理了；
- **会话失效竞态守卫**（`index.html:1241-1263`）与 skill loader 对用户可写目录的失败处理（`runtime.py:547-585`）是范本级；
- 成本核算带三档定价 + 分组看板；代码对已知局限直接写注释不藏——工程素养在同类项目里属高。

## 九、动手顺序（合并两轮，按「会不会让迁移用户掉头」排序）

**今天就能改（全部验证过，都是小 diff）：**
1. IME Enter 守卫（1.1，一行）
2. LLM 客户端 `timeout` + `max_retries=1`（4.1）
3. new-session 补 `sessModel/sessEffort` 重置（六.1，一行）
4. 流式渲染 rAF 节流 + 滚动让权（4.2 + 1.8，同一段代码顺手做）
5. table/img CSS（2.2）
6. 消息/代码块复制按钮（1.2）
7. 错误条常驻 + 重试（5.2）
8. busy_timeout（4.5）与 parent_id 索引（4.6）——各一行

**本周：**
9. `/api/import` 支持自家导出格式（3.1，最担心的单向门）
10. 删 mermaid 或改 data-* 方案（2.1）
11. 解除 busy 全局锁：放行切会话/新对话（1.7）
12. effort：透传推理过程或先藏开关；同时验证 DeepSeek 兼容（3.6 + 七）
13. 历史压缩摘要持久化（4.8，后端省钱省首 token）
14. 会话列表空态 + mode 徽章（1.6 + 3.4）；编辑「仅保存」路径（1.3）

**之后：** 软删除/回收站（3.2）、FTS5（4.3）、单会话接口+limit（4.9）、文档 chunk 预览（3.8）、cron 预设+下次运行预览（3.9）、首启引导（3.10）、traces 页（5.7）、草稿持久化（5.1）、a11y 基线（5.3）、技能手动运行（3.5）、bm25 to_thread（4.4）、自动备份+user_version（3.3）。

**维持砍掉（YAGNI，两轮共识）：** 语音、画布、图片生成、移动端、i18n、MCP、多模态。仅两处文案收尾：＋ 按钮改叫「入库」（1.4），粘贴图片的 422 提示说清楚「暂不支持图片」。
