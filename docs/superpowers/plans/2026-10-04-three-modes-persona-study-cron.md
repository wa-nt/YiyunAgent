# 三模式 + 人格 + 漏洞记忆 + 定时任务 + 启动优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 YiyunAgent 从知识库问答扩展为 chat/work 两种可用模式：可配置去-AI-味人格、复习漏洞自动记录（简化 SM-2）、用户自设定时任务（cron + 托盘通知 + 自动开会话），并优化冷启动与可选开机自启。Code 入口本期只展示为禁用状态，不实现运行时。

**Architecture:**

- 模式绑定在 `sessions.mode` 上。当前只实现 `chat`、`work`；`code` 仅作为禁用的产品入口，不接受 API 创建，不进入运行时工具白名单。
- 已有会话永远以数据库中的 mode 为准。请求中的 `mode` 只用于创建新会话；已有会话若显式传入不同 mode，返回 `409`。
- 漏洞记忆使用新表 `knowledge_gaps`，由 work 模式工具 `record_knowledge_gap` / `review_knowledge_gap` 操作。漏洞是可重复复习卡片；`passed=True` 只延长间隔，不自动 resolved。
- 定时任务使用新表 `scheduled_tasks` 和 `notifications`。lifespan 启动一个单进程调度循环，每次触发先事务性占位 `last_fired_at`，再在事务外运行 Agent；失败写通知但不重试同一 occurrence，也不能杀死调度循环。
- 通知渠道本期二选一，选择托盘通知：托盘通知线程直接只读 SQLite，不从后台线程直调 ASGI；Web 通知铃铛及通知中心本期不实现。逐条已读 API、任务 PATCH 也本期推迟。
- 定时任务创建的会话带 `source=scheduled`、`scheduled_task_id` 和 `scheduled_occurrence_at`，与用户手动会话可区分；任务删除不删除历史会话和通知。
- 人格是用户数据，单独存 SQLite，支持多行原文；未设置时加载仓库内默认人格文件，空字符串表示明确禁用人格。

**Tech Stack:** Python 3.12 / FastAPI / aiosqlite / pywebview / pystray / 原生 JS 前端。新增依赖：`croniter`（纯 Python cron 解析，唯一新依赖）。

**Spec:** 本文件即 spec（grilling 会话结论，无独立设计文档）。关键决策：隐私不分级全云端；chat/work 本期可用；Code 模式运行时、Pi sidecar、多 Agent/角色卡、Web 通知中心、逐条已读、任务编辑均推迟；通知渠道在 Web 铃铛与托盘通知中选择托盘通知；定时任务只由用户显式创建。聊天默认人格借鉴 `liliMozi/openhanako` 的公开 Hanako persona 模板所体现的原则，但重新编写为本项目的原创中文规则，不复制原文。

## Scope / Deferred

本期必须交付：

- chat/work 模式和会话级模式绑定；
- work 模式漏洞记录、复习和列表 API/UI；
- 多行人格设置及默认人格；
- cron 任务创建、启停、删除、触发、失败通知、自动会话；
- 托盘通知；
- 冷启动的定性懒加载改造；
- Windows 打包版开机自启开关。

明确推迟：

- Code 模式运行时、Code 工具、Pi sidecar；
- Web 通知铃铛和通知中心（通知渠道本期已选择托盘通知，不并行实现 Web 通知）；
- `POST /api/notifications/read {ids}` 逐条已读 API；
- `PATCH /api/tasks/{id}` 任务编辑 API；
- 精确的冷启动毫秒门槛或 CI 性能测试；
- 多用户通知隔离、任务时区选择器、语义去重、复杂任务队列。

## Global Constraints

- 测试沿用 `tests/` 现有风格：`db(tmp_path, monkeypatch)` fixture（`settings.db_path` + `init_db(..., embed_dim=8)`）+ `httpx.ASGITransport(app=app)`。需要验证 lifespan 的测试另加 lifespan 专用 fixture，不能假设现有 API fixture 会启动后台任务。
- 新表一律加进 `app/schema.sql`（`CREATE TABLE IF NOT EXISTS`）；给已有表加列用 `app/db.py:init_db` 的 ad-hoc ALTER 模式。
- 工具定义用 `app.llm.types.ToolDef(name, description, parameters)`；handler 签名为 `async def fn(args: dict, db_path: str | None) -> tuple[str, str]`。保留名 `search_knowledge` 不可用。
- SSE 事件类型只用现有的：`session / text_delta / tool_start / tool_end / error / done`。`session` 事件允许增加字段，但不新增事件类型。
- 不引入前端框架；`web/index.html` 单文件内增量修改。
- 所有持久化时间统一使用 UTC ISO 8601；cron 表达式按运行机器当前本地时间解释，并在任务 UI 中明确这一点。
- cron 计算使用任务创建时记录的本机时区标识；本期不提供时区选择器。机器时区改变后不自动迁移既有任务，UI 必须显示当前解释时区。夏令时重复时间只执行一次，缺失时间跳过。
- Windows 开机自启使用 `winreg` 写 `HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run`，默认关闭；源码环境拒绝启用并返回可读错误。
- 调度器只支持单进程、单 worker。桌面入口必须使用 Windows 单实例互斥体；服务入口必须明确以一个 worker 启动，开发 reload 不启动 scheduler。数据库事务占位只防止同一进程内重复触发，本期不提供跨进程调度锁。
- `init_db` 必须先用 `PRAGMA table_info` 检测旧列，再在事务中幂等迁移；迁移后归一化非法存量 mode，并保留旧库备份/失败可恢复路径。
- 资源文件统一经 `resource_path()` 解析源码和 PyInstaller 路径；打包 smoke test 必须验证 schema、默认 persona 和 skills 均可读取。

## Review Focus / Required Invariants

1. 休眠/关机期间错过的 cron occurrence 不补跑；恢复后只处理启动时刻前一个 scheduler grace window 内的最近 occurrence，超过窗口全部跳过。grace window 等于一个 tick 周期，且必须在测试中固定。
2. `last_fired_at` 在执行 Agent 前写入；LLM/API 失败也不清除，不重复重试同一 occurrence。
3. `fire_task` 失败必须写 error notification，并吞掉异常返回调度层；坏 cron 只暂停/记录该任务，不得杀死调度循环。
4. `review_gap(passed=False)` 必须把间隔重置为 1 天；`passed=True` 翻倍，设置最大间隔 180 天。
5. `run_agent(mode=None)` 未显式提供模式时从 session 读取；已有 session 的 `/api/chat` 与 `respond` 不能退回 chat。
6. 工具白名单在“发送给 LLM”和“实际 dispatch”两层检查；不可用工具返回明确的 mode 错误。
7. 多行 persona 必须原样 round-trip；默认人格文件在源码和 PyInstaller 打包版都可读取。
8. 调度器关闭顺序：停止接收新任务 → 取消/等待 scheduler → 处理在途 fire task → drain memory/traces。
9. fire task 使用全局并发上限和单任务超时；shutdown 有最大等待时间，超时取消任务并写入可读错误通知，不阻塞进程退出。

---

### Task 1: 模式框架（sessions.mode + prompt/工具白名单）

**Files:**

- Modify: `app/schema.sql`（不直接改已有 sessions 结构；新安装也可不写 mode，统一由迁移保证）
- Modify: `app/db.py`（`init_db` 增加 `sessions.mode`、`source`、`scheduled_task_id`、`scheduled_occurrence_at` 的幂等迁移）
- Modify: `app/agent/runtime.py`（chat/work prompt、模式读取、工具白名单、dispatch 防护）
- Modify: `app/main.py`（`ChatRequest.mode`、新会话建模、已有会话冲突校验、respond 使用 session mode、session SSE 增加 mode）
- Test: `tests/test_modes.py`

**Interfaces / invariants:**

- `SUPPORTED_MODES = ("chat", "work")` 是唯一模式校验来源；`code` 不属于本期可接受 API mode。创建和 API 输入非法 mode 返回 400；迁移时将 NULL/未知存量 mode 归一化为 `chat` 并记录 warning，运行时读到仍未知的 mode 返回可识别错误。
- `MODE_PROMPTS` 只包含 chat/work；`MODE_TOOLS = {"chat": ["search_knowledge"], "work": ["search_knowledge", "record_knowledge_gap", "review_knowledge_gap"]}`。
- chat 默认人格原则：有温度但不谄媚；保持边界和自主判断；优先从事实、原理和证据分析；用类比或具体例子解释抽象概念；不编造不确定信息，明确说需要确认；少用破折号和模板化收尾。该原则参考 OpenHanako 的 Hanako 模板（Apache-2.0），采用本项目原创表述，不保留其用户/访客身份设定。
- `get_session_mode(session_id, db_path=None) -> str`：旧数据经迁移后默认为 chat；未知 mode 不静默进入未定义运行时，返回可识别错误。
- `ensure_session(session_id, db_path=None, *, mode="chat")`：只在真正创建时使用 mode；已存在 session 不覆盖 mode。
- 自动会话写入 `source=scheduled`、任务 ID 和 occurrence；普通会话写入 `source=manual`。会话列表 API 返回 source，前端可区分自动会话。
- `assemble_messages(..., mode: str, persona: str | None = None)`：system prompt = `SYSTEM_PROMPT + MODE_PROMPTS[mode]`，persona 非空时前置。
- `run_agent(..., mode: str | None = None)`：`None` 时从 session 读取；只在新会话创建路径显式传 mode。
- `/api/chat`：无 session_id 时 `mode` 缺省为 chat，非空值必须是 chat/work；有 session_id 时先确认 session 存在并从数据库读 mode，若请求 mode 非空且不同则 `409`，session 不存在返回 `404`。
- `/api/sessions/{id}/respond`：始终从 session 读取 mode。
- 新建成功的 `session` SSE 数据为 `{type: "session", session_id, mode}`。
- `code` API 请求返回 400/409，并说明本期未开放；前端 disabled 只是 UX，不是安全边界。

- [ ] **Step 1: 写失败测试**：新建 work 会话持久化 mode；已有 work 会话不带 mode 仍用 work；已有 work 会话带 chat 返回 409；respond 读取 work；非法 mode/code 被拒绝；chat 工具列表不含 gap 工具；旧 sessions 迁移默认为 chat。
- [ ] **Step 2:** 跑 `pytest tests/test_modes.py -v`，确认 FAIL。
- [ ] **Step 3:** 实现迁移、`ensure_session`、模式读取、路由校验、runtime 工具过滤和 dispatch 二次检查。
- [ ] **Step 4:** 跑 `pytest tests/test_modes.py tests/test_agent.py tests/test_api.py -v`，全 PASS。
- [ ] **Step 5:** Commit：`feat: bind sessions to chat and work modes`

### Task 2: 可配置人格（支持多行）

**Files:**

- Create: `app/agent/prompts/persona_default.md`（仓库内定稿，不依赖运行时访问外部 GitHub；内容按下方原则原创编写）
- Modify: `app/schema.sql`（新表 `app_settings(key TEXT PRIMARY KEY, value TEXT NOT NULL)`）
- Create or modify: `app/settings_store.py`（最小 SQLite `get_setting/set_setting`，只服务 persona）
- Modify: `app/config.py`（不要把多行 persona 写入 `.env`；保留运行时默认字段或移除 persona 配置字段）
- Modify: `app/main.py`（SettingsUpdate、settings GET/POST 从 SQLite 读写 persona）
- Modify: `app/agent/runtime.py`（`load_persona(db_path=None) -> str`）
- Modify: `web/index.html`（设置面板加多行文本框，沿现有 load/save 模式）
- Test: `tests/test_persona.py`
- Modify: PyInstaller spec/build 配置（把 `persona_default.md` 加入打包资源，并统一资源路径解析）

**Storage contract:**

- persona 原文存 `app_settings`，支持换行、引号和中文；不进入 `.env` 和 `EDITABLE_FIELDS`。存储层必须区分“无记录/NULL”和空字符串：前者加载默认人格，后者表示用户明确禁用人格并按空字符串 round-trip，不回退默认文件。
- 默认人格文件缺失时抛出明确启动/请求错误，不静默生成空人格。
- `/api/settings` 返回 `persona`；其它设置管道保持现状。

- [ ] **Step 1:** 写失败测试：多行 persona round-trip；无记录加载默认人格；空字符串保持为空；settings API 返回值；默认资源路径在源码环境可读。
- [ ] **Step 2:** 跑 `pytest tests/test_persona.py -v`，确认 FAIL。
- [ ] **Step 3:** 实现 SQLite 存储、runtime 加载、设置 API/UI 和打包资源配置。默认人格直接写入本仓库，采用“温暖但有边界、独立判断、原理和证据优先、类比落地、坦诚不确定、避免模板化收尾”等原创规则；禁“作为 AI”，允许说不知道。不得复制 OpenHanako 模板的句子、身份设定或占位符。
- [ ] **Step 4:** 跑 `pytest tests/test_persona.py tests/test_settings.py -v`，PASS。
- [ ] **Step 5:** Commit：`feat: store configurable persona as multiline user data`

### Task 3: 知识漏洞表 + 复习工具

**Files:**

- Modify: `app/schema.sql`（`knowledge_gaps` + 索引）
- Create: `app/study/gaps.py`（`record_gap / review_gap / list_gaps / delete_gap / due_gaps`）
- Create: `app/study/tools.py`（两个 ToolDef 及 handler 适配器）
- Modify: `app/agent/runtime.py`（work 模式注册并 dispatch 两个内置工具）
- Modify: `app/main.py`（`GET /api/gaps?all=false`、`DELETE /api/gaps/{gap_id}`）
- Test: `tests/test_gaps.py`

**Data contract:**

```sql
knowledge_gaps(
  id INTEGER PRIMARY KEY,
  topic TEXT NOT NULL,
  detail TEXT NOT NULL,
  interval_days INTEGER NOT NULL DEFAULT 1,
  next_review_at TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
)
```

- 本期不实现归档/恢复，因此不创建 `resolved` 字段；未来实现归档时另行增加迁移和 API，避免保留一个没有写入路径的伪功能。
- `record_gap`：next review = now + 1 day。
- `review_gap(gap_id, passed)`：通过则 `min(interval_days * 2, 180)`；失败则 `1`；next review = now + interval days；不存在 ID 返回可识别错误。
- `due_gaps`：`next_review_at <= now`；时间统一 UTC。
- 本期允许同主题重复记录，不做语义去重。

- [ ] **Step 1:** 写失败测试：记录后未到期不出现在 due；改时间后出现；通过翻倍；失败重置；180 天上限；GET/DELETE；不存在 ID；非法参数。
- [ ] **Step 2:** 跑确认 FAIL。
- [ ] **Step 3:** 实现纯 SQL 领域函数、工具 handler 适配器、runtime 注册和 API。
- [ ] **Step 4:** 跑 `pytest tests/test_gaps.py -v`，PASS。
- [ ] **Step 5:** Commit：`feat: add knowledge gap review cards`

### Task 4: 漏洞列表 UI + work 模式复习行为

**Files:**

- Modify: `web/index.html`（独立“漏洞”面板入口、列表和删除按钮；模式切换器放新建会话/顶栏区域，不放 `#model-row`）
- Modify: `app/agent/runtime.py`（work prompt：发现未掌握时调 `record_knowledge_gap`，答完追问确认）
- Test: `tests/test_modes.py`（work prompt 断言）

**UI contract:**

- `currentMode` 默认 chat；模式切换只影响新会话。
- 新会话请求带 `mode`；旧会话打开后以 SSE/session 或 sessions API 返回的 mode 同步 UI。
- chat/work button 可用；code button disabled 且标注“即将推出”。
- 不新增 SSE 事件类型。

- [ ] **Step 1:** 追加 `record_knowledge_gap` 和复习教练行为断言。
- [ ] **Step 2:** 跑 `pytest tests/test_modes.py -v` 确认 FAIL。
- [ ] **Step 3:** 实现 prompt、面板、模式切换和旧会话同步。
- [ ] **Step 4:** 跑相关 pytest；启动页面验证切 work → 新会话 → 发送学习困难描述 → 列表显示记录。使用浏览器预览流程而不是把手工结果只写进提交信息。
- [ ] **Step 5:** Commit：`feat: add study gaps panel and work mode UI`

### Task 5: 定时任务核心（cron 表 + 调度器 + API）

**Files:**

- Modify: `pyproject.toml`（加 `croniter`）
- Modify: `app/schema.sql`（`scheduled_tasks`、`notifications` 两张新表及索引）
- Create: `app/scheduler.py`（`TaskRow`、CRUD、occurrence 计算、`run_scheduler_tick`、`scheduler_loop`、`fire_task`）
- Modify: `app/main.py`（lifespan 启动/停止 scheduler；任务 CRUD API；最近通知只读 API）
- Test: `tests/test_scheduler.py`

**Tables:**

```sql
scheduled_tasks(
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  cron TEXT NOT NULL,
  prompt TEXT NOT NULL,
  mode TEXT NOT NULL DEFAULT 'work',
  timezone TEXT NOT NULL,
  enabled INTEGER NOT NULL DEFAULT 1,
  last_fired_at TEXT,
  created_at TEXT NOT NULL
)

notifications(
  id INTEGER PRIMARY KEY,
  task_id INTEGER,
  session_id TEXT,
  kind TEXT NOT NULL,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  created_at TEXT NOT NULL
)
```

- 单用户桌面应用，本期不设 `read` 字段和逐条已读 API；只返回最近 N 条通知，托盘从应用启动时刻开始查询并用 `last_notified_id`（进程内）去重，避免重启重复弹出旧通知。保留期限先按最近 100 条查询；后台清理推迟，但必须记录数据库长期增长限制。
- 创建任务只接受 chat/work；默认 work。创建时保存当前本机时区标识，cron 按该时区解释；数据库时间和 occurrence 记录存 UTC。机器时区改变后，UI 显示实际解释时区并提示任务仍使用创建时区。
- API：`GET/POST /api/tasks`、`POST /api/tasks/{id}/enable`、`POST /api/tasks/{id}/disable`、`DELETE /api/tasks/{id}`、`GET /api/notifications`。不做 PATCH。
- `POST /api/tasks` 校验 cron、mode、非空且有长度上限的 name/prompt；非法 cron 返回 400。已存坏 cron 在 tick 时捕获、禁用并只写一条错误通知，继续处理其他任务。
- `run_scheduler_tick(now, started_at, grace_window)`：使用任务保存的 timezone 计算 `prev_occurrence(now)`；仅当 occurrence 在 `[started_at - grace_window, now]` 内才允许触发，启动前更早 occurrence 全部跳过。事务内以任务 ID + occurrence 条件更新 `last_fired_at`，提交后再创建 fire task。
- `fire_task`：创建带 source/task/occurrence 元数据的指定 mode 新会话，消费 `run_agent` 完整事件流；仅有完整 done 且无 error 才写 success notification，异常或超时写经截断和敏感信息清洗的 error notification，并吞掉业务异常返回调度层。
- 调度器串行扫描，fire task 使用全局 semaphore 和单任务 timeout；同一任务已有 active fire task 时跳过重入。删除/禁用只影响后续 occurrence，在途任务按 shutdown 策略继续或取消。
- scheduler shutdown：先设置 accepting=false，API 不再创建新 fire；取消 loop；在最大等待时间内等待 active fire tasks，超时取消并写通知；最后执行现有 memory/tracing drain，使用 `try/finally` 保证 drain。

- [ ] **Step 1:** 写失败测试：合法/非法 cron；时区/DST边界；同一 occurrence 只 fire 一次；并发 tick 不重复；失败/超时写 error notification 且不抛；坏 cron 不杀循环；错过 occurrence 不补跑；任务 mode 和来源元数据传给新 session；禁用/删除 CRUD；重复 tick 不重入。
- [ ] **Step 2:** 跑确认 FAIL。
- [ ] **Step 3:** 实现 occurrence 计算、事务性 last_fired_at 占位、异常隔离、active task 管理、CRUD/API 和 lifespan。
- [ ] **Step 4:** 对 API 使用现有 fixture；增加 lifespan、并发 tick、shutdown 中途 fire、单 worker 约束测试。跑 `pytest tests/test_scheduler.py -v`。
- [ ] **Step 5:** Commit：`feat: add idempotent cron tasks and session notifications`

### Task 6: 任务 UI + 托盘通知（通知渠道选择）

**Files:**

- Modify: `web/index.html`（定时任务管理页：列表、新建表单、启停、删除；显示最近任务结果，不做 Web 通知铃铛）
- Modify: `app/desktop.py`（daemon 轮询线程直接用同步 `sqlite3` 查询 notifications；`icon.notify` 去重；停止事件）
- Test: `tests/test_scheduler.py`（任务 API 和 notification 查询断言）

**Interfaces:**

- 前端只消费 Task 5 的任务 API 和最近通知 API；点击任务结果不承担通知中心职责。UI 显示任务解释时区、启用状态、最近结果和自动会话入口。
- desktop 线程不创建新的 ASGI client、不调用 FastAPI route、不跨事件循环复用 aiosqlite 连接。
- 每条 notification 只通知一次；轮询只查询应用启动后的记录，轮询线程退出时由 `threading.Event` 控制；SQLite 读连接设置 busy timeout，图标实例生命周期由 desktop 主线程持有。pystray 不可用时记录日志并保持 Web 任务管理可用。

- [ ] **Step 1:** 追加任务启停和最近通知查询测试。
- [ ] **Step 2:** 实现任务 UI、同步 SQLite 轮询和 pystray 通知。
- [ ] **Step 3:** pytest PASS；用浏览器预览验证创建任务、启停、删除；打包/桌面环境手工验证托盘通知一次性弹出。通知渠道已选择托盘，Web 端不实现铃铛或通知中心。
- [ ] **Step 4:** Commit：`feat: add task management UI and tray notifications`

### Task 7: 启动优化 + 开机自启开关

**Files:**

- Modify: `app/main.py`（只对已确认的重型导入做函数内延迟 import，重点检查 `app.ingest.pipeline`、`app.tracing`，不要盲目延迟核心 runtime/config）
- Create: `app/autostart.py`（`enable/disable/is_enabled`；支持判断可内联或保持最小 helper）
- Modify: `app/main.py`（`autostart` 设置特判；不进入 `.env`）
- Modify: 前端设置面板（显示 `autostart_supported`；不支持时禁用开关）
- Test: `tests/test_autostart.py`
- Modify: PyInstaller/build 配置（与 Task 2 共同确认资源打包）

**Interfaces:**

- `is_supported = sys.platform == "win32" and bool(getattr(sys, "frozen", False))`。
- `enable()` 使用 `sys.executable` 和固定启动参数写入完整、正确引用的注册表命令；重复 enable 幂等。
- `disable()` 删除本应用键，不存在时也成功；注册表访问失败返回明确 500/400。
- GET settings 返回 `autostart` 和 `autostart_supported`；源码环境设置 `true` 返回 400 并提示仅打包版可用，设置 `false` 可成功。
- 不增加精确冷启动性能阈值；以当前基线记录 `-X importtime -c "import app.main"`，只把“明确列出的重型模块未被加载”作为自动化门槛，观测值用于说明，不作为性能通过条件。

- [ ] **Step 1:** 写失败测试：源码环境拒绝 true；false 成功；支持状态字段；注册表 enable/disable/is_enabled 使用 monkeypatch；惰性导入断言只覆盖明确重型模块。
- [ ] **Step 2:** 跑确认 FAIL。
- [ ] **Step 3:** 实现懒加载、自启模块、设置 API/UI 和打包资源联动。
- [ ] **Step 4:** 跑 `pytest tests/test_autostart.py tests/test_settings.py -v`；源码环境确认无回归。打包环境的注册表和冷启动只做手工验收，不设毫秒门槛。
- [ ] **Step 5:** Commit：`perf: lazy-load heavy imports; feat: opt-in Windows autostart`

---

## Verification Matrix

- [ ] 全量 `pytest -q`。
- [ ] 模式：新建 work、旧 work 继续、respond、非法 code/mode、工具越权防护。
- [ ] 人格：多行保存、未设置回退默认、空字符串保持为空、源码资源、PyInstaller 资源；默认人格与 OpenHanako 的借鉴关系和 Apache-2.0 归因记录在项目文档中。
- [ ] 漏洞：记录、到期、通过翻倍、失败重置、上限、删除、列表、空白/超长输入和并发复习边界。
- [ ] 调度：非法 cron 隔离、时区/DST、occurrence 幂等、并发 tick、失败/超时通知、休眠不补跑、单实例、shutdown 清理。
- [ ] UI：模式切换、漏洞列表、任务创建/启停/删除、code disabled。
- [ ] 桌面：托盘通知只提示启动后新增记录，单次去重，线程退出不阻塞窗口，pystray 不可用时降级。
- [ ] 自启：源码拒绝启用，打包版注册表 enable/disable 幂等。
- [ ] 迁移/数据：旧 schema 升级幂等；导出和整库备份覆盖 persona、knowledge gaps、scheduled tasks、notifications；删除策略明确。
- [ ] 打包：PyInstaller 产物可读取 schema、默认 persona、skills，并完成一次启动 smoke test。

## Self-Review 记录

- **Spec 覆盖**：chat/work 模式、人格、多行存储、漏洞+简化 SM-2、cron+失败通知、托盘通知、懒加载、自启均有任务和测试；通知渠道明确选择托盘通知；Code 运行时、Web 铃铛、逐条已读、PATCH、精确性能门槛明确推迟。
- **模式一致性**：`/api/chat`、`respond`、定时任务和 runtime 均以 session mode 为准；新会话才接受 mode。
- **存储一致性**：persona 不再走单行 `.env`，NULL 与空字符串语义分离；notifications 不再声明未实现的 read API；新表均列入 schema 和导出/备份清单。
- **调度一致性**：先占位再执行，按保存的时区和 grace window 处理 occurrence，坏 cron/LLM/超时失败隔离，active task、单实例和 shutdown 顺序有明确契约。
- **数据边界**：通知正文截断并清洗敏感信息；自动会话带来源元数据；删除任务不误删历史会话，长期通知增长作为已知限制记录。
- **桌面一致性**：托盘线程直读 SQLite，避免同进程跨事件循环直调 ASGI。
- **资源一致性**：默认人格文件和 PyInstaller 数据资源在同一实施链路中处理。
