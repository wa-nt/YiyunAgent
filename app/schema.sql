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

-- 三张热表的过滤列都缺索引，实际查询退化成全表扫描：按 session_id 取会话历史与
-- 消息列表、按 status 筛有效记忆、看板按 (kind, name, ts) 过滤 trace。
-- 全部 IF NOT EXISTS，init_db 每次启动重放本文件时给存量库补上，无需迁移脚本。
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_traces_kind_name_ts ON traces(kind, name, ts);