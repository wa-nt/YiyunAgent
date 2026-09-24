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

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    role TEXT,
    content TEXT,
    created_at TEXT
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