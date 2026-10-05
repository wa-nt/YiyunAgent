CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    source TEXT,
    title TEXT,
    ingested_at TEXT,
    meta TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER REFERENCES documents(id),
    idx INTEGER,
    content TEXT,
    token_count INTEGER,
    embedding BLOB
);

-- chunk_vectors 是 sqlite-vec 的 vec0 虚拟表（chunk_id INTEGER PRIMARY KEY,
-- embedding FLOAT[N]）。维度 N 来自配置 embed_dim，由 app/db.py 在初始化时
-- 用 Python 格式化生成建表语句，因此不写在本文件里。
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    kind TEXT,
    content TEXT,
    confidence REAL,
    source TEXT,
    created_at TEXT,
    updated_at TEXT,
    status TEXT DEFAULT 'active',
    supersedes INTEGER
);

-- 模式框架（T1）用到的 mode / source / scheduled_task_id / scheduled_occurrence_at
-- 刻意不写在这张表里：新老安装都只走 db.init_db 里那段幂等 ALTER TABLE + 归一化，
-- 避免「新库由 schema.sql 建列、旧库由迁移补列」两条路径各自漂移。
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT,
    -- 用户改名 / LLM 自动标题 / ChatGPT 导入写入；NULL = 列表回退到首条消息前 30 字。
    -- 旧库没有这一列，由 db.init_db 的 ALTER TABLE 兜底补列
    title TEXT,
    -- 分支模型：当前生效分支的叶子消息；NULL（无消息/旧数据）时回退 max(id)
    active_leaf INTEGER,
    -- per-session 供应商覆盖；NULL = 跟随全局 llm_provider
    provider TEXT,
    -- per-session 模型覆盖；NULL = 跟随全局该供应商默认模型
    model TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    role TEXT,
    content TEXT,
    created_at TEXT,
    -- 分支模型：指向前驱消息；同 parent 的多条消息互为分支。旧数据由迁移回填成线性链
    parent_id INTEGER
);

CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY,
    ts TEXT,
    kind TEXT,
    name TEXT,
    detail TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost REAL
);

-- 用户在界面上编辑的多行文本数据（当前只有 persona）。.env 是单行 KEY=VALUE 口径，
-- 多行原文会被写坏，所以这类用户数据单独存表。无记录 = 没设置过（加载默认人格文件），
-- 空字符串 = 用户明确不要人格，两者语义不同，不能互相顶替。
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- 知识漏洞：work 模式的间隔重复卡片（record/review 两个内置工具与 /api/gaps 共用）。
-- interval_days 只由 app/study/gaps.py 的复习逻辑维护；next_review_at 与两种时间戳都是
-- UTC ISO 8601（带 +00:00 偏移），定长格式让「到期」可以直接做字符串比较。
-- 本期不做归档，因此没有 resolved 一类的字段（见 app/study/gaps.py 的模块说明）。
CREATE TABLE IF NOT EXISTS knowledge_gaps (
    id INTEGER PRIMARY KEY,
    topic TEXT NOT NULL,
    detail TEXT NOT NULL,
    interval_days INTEGER NOT NULL DEFAULT 1,
    next_review_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- 三张热表的过滤列都缺索引，实际查询退化成全表扫描：按 session_id 取会话历史与
-- 消息列表、按 status 筛有效记忆、看板按 (kind, name, ts) 过滤 trace。
-- 全部 IF NOT EXISTS，init_db 每次启动重放本文件时给存量库补上，无需迁移脚本。
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
-- parent_id 是分支树递归 CTE 与编辑截断的 join key，无索引则长会话全表扫
CREATE INDEX IF NOT EXISTS idx_messages_parent_id ON messages(parent_id);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_traces_kind_name_ts ON traces(kind, name, ts);
-- /api/gaps?all=false 与定时复习任务（T5）只查「已到期」，列表按到期时间排序
CREATE INDEX IF NOT EXISTS idx_knowledge_gaps_next_review_at ON knowledge_gaps(next_review_at);

-- 定时任务（T5）。name/cron/prompt 是用户输入（创建时校验非空与长度，cron 还要能被
-- croniter 解析）；mode 只允许 chat/work，默认 work；timezone 是**创建时**记录的本机
-- IANA 时区标识（如 Asia/Shanghai），cron 永远按它解释——机器时区后来变了也不迁移既有
-- 任务，UI 只提示「仍在用创建时区」。除 created_at/last_fired_at 外的所有时间都是
-- UTC ISO 8601（带 +00:00 偏移），定长格式让比较可以直接走字符串序。
-- last_fired_at 是最近一次已触发的 occurrence：它在跑 Agent **之前**用
-- 「任务 ID + occurrence」条件事务占位，因此既是幂等键（同一 occurrence 只 fire 一次），
-- 也是「不重试失败轮次」的依据（失败只写通知，不回滚占位）。enabled=0 的任务不再被扫描。
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    cron TEXT NOT NULL,
    prompt TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'work',
    timezone TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_fired_at TEXT,
    created_at TEXT NOT NULL
);

-- 任务结果通知（T5 写，T6 的托盘线程只读）。单用户桌面应用，本期不做 read 字段与逐条
-- 已读 API：托盘从应用启动时刻开始查询，用进程内的 last_notified_id 去重，重启不会把旧
-- 通知再弹一遍。这张表只增不删——是已知的长期增长点，后台清理推迟到有量级证据之后；
-- 查询侧一律只取最近 N 条（默认 100，见 app/scheduler.py 的 NOTIFICATION_LIMIT）。
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY,
    task_id INTEGER,
    session_id TEXT,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- 调度扫描每轮都要按 enabled 过滤；通知列表按时间倒序取最近 N 条
CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_enabled ON scheduled_tasks(enabled);
CREATE INDEX IF NOT EXISTS idx_notifications_created_at ON notifications(created_at);