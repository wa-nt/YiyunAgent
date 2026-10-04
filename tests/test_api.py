"""前端新增接口的用例：chunk 引用回跳、会话列表、文件上传、记忆列表。

隔离方式跟 tests/test_agent.py 的 API 段一致：db 夹具把 settings.db_path 指到 tmp 并
init_db（ASGITransport 不跑 lifespan，建表的责任在夹具），所以完全不碰 data/app.db。
"""

from pathlib import Path

import httpx
import pytest

from app.config import settings
from app.db import get_db, init_db
from app.main import app

DIM = 8


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "app.db"))
    await init_db(tmp_path / "app.db", DIM)
    yield tmp_path / "app.db"


@pytest.fixture
async def client(db):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def seed_chunk(
    db, content="RAG 结合检索与生成", title="检索笔记", source="notes/rag.md"
) -> int:
    async with get_db(db) as conn:
        cursor = await conn.execute(
            "INSERT INTO documents (source, title, ingested_at) VALUES (?, ?, ?)",
            (source, title, "2026-09-28T10:00:00"),
        )
        doc_id = cursor.lastrowid
        cursor = await conn.execute(
            "INSERT INTO chunks (doc_id, idx, content, token_count) VALUES (?, 0, ?, ?)",
            (doc_id, content, len(content)),
        )
        await conn.commit()
        return cursor.lastrowid


async def test_chunk_detail_joins_its_document(client, db):
    chunk_id = await seed_chunk(db)

    resp = await client.get(f"/api/chunks/{chunk_id}")

    assert resp.status_code == 200
    assert resp.json() == {
        "id": chunk_id,
        "content": "RAG 结合检索与生成",
        "title": "检索笔记",
        "source": "notes/rag.md",
    }


async def test_chunk_detail_missing_is_404(client, db):
    resp = await client.get("/api/chunks/999")
    assert resp.status_code == 404


async def test_sessions_list_orders_by_last_message(client, db):
    async with get_db(db) as conn:
        await conn.executemany(
            "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
            [
                ("s1", "2026-09-28T09:00:00"),
                ("s2", "2026-09-28T10:00:00"),
                ("s3", "2026-09-28T11:00:00"),
            ],
        )
        await conn.executemany(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            [
                ("s1", "user", "一" * 60, "2026-09-28T09:00:00"),
                ("s1", "assistant", "回答", "2026-09-28T09:05:00"),
                ("s2", "user", "RAG 是什么？", "2026-09-28T10:00:00"),
            ],
        )
        await conn.commit()

    rows = (await client.get("/api/sessions")).json()

    # s2 最近发言排最前；s3 一条消息都没有，只能排最后且标题为空
    assert [r["id"] for r in rows] == ["s2", "s1", "s3"]
    assert rows[0] == {
        "id": "s2",
        "created_at": "2026-09-28T10:00:00",
        "provider": None,
        "model": None,
        "title": "RAG 是什么？",
        "message_count": 1,
    }
    assert rows[1]["title"] == "一" * 30 and rows[1]["message_count"] == 2
    assert rows[2]["title"] == "" and rows[2]["message_count"] == 0


async def test_sessions_list_breaks_ties_by_message_id(client, db):
    """created_at 只有秒级精度，同一秒内发言的两条会话靠自增 id 定序。"""
    async with get_db(db) as conn:
        await conn.executemany(
            "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
            [("s1", "2026-09-28T09:00:00"), ("s2", "2026-09-28T09:00:00")],
        )
        await conn.executemany(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            [
                ("s1", "user", "先发言", "2026-09-28T09:00:00"),
                ("s2", "user", "后发言", "2026-09-28T09:00:00"),
            ],
        )
        await conn.commit()

    rows = (await client.get("/api/sessions")).json()

    assert [r["id"] for r in rows] == ["s2", "s1"]


def _spy_ingest(monkeypatch, seen: list[str]) -> None:
    async def fake(source: str) -> int:
        seen.append(source)
        return 2

    monkeypatch.setattr("app.main.ingest", fake)


async def test_ingest_accepts_data_dir_files_and_urls(client, db, tmp_path, monkeypatch):
    """非 URL 的 source 只认数据目录内的文件；URL 不走路径检查（由 loaders 守公网）。"""
    seen: list[str] = []
    _spy_ingest(monkeypatch, seen)
    note = tmp_path / "uploads" / "笔记.md"
    note.parent.mkdir()
    note.write_text("# 标题\n", encoding="utf-8")

    # 绝对路径、相对路径（按数据目录解析）、URL 三种都放行
    for source in (str(note), "uploads/笔记.md", "https://example.com/a.md"):
        resp = await client.post("/api/ingest", json={"source": source})
        assert resp.status_code == 200
        assert resp.json() == {"chunks": 2}

    # 本地路径（绝对/相对）经白名单校验后统一以解析后的绝对路径进管线，URL 原样透传
    assert seen == [note.resolve(), note.resolve(), "https://example.com/a.md"]


async def test_ingest_rejects_file_outside_data_dir(client, db, tmp_path, monkeypatch):
    """修复前 source 原样进管线 = 任意本地文件读取（.md/.txt/.pdf 都能被读走）。"""
    seen: list[str] = []
    _spy_ingest(monkeypatch, seen)
    outside = tmp_path.parent / "机密.md"
    outside.write_text("# 不该被读到\n", encoding="utf-8")

    resp = await client.post("/api/ingest", json={"source": str(outside)})

    assert resp.status_code == 400
    assert "数据目录" in resp.json()["detail"]
    assert seen == []


async def test_ingest_rejects_relative_traversal(client, db, tmp_path, monkeypatch):
    seen: list[str] = []
    _spy_ingest(monkeypatch, seen)
    (tmp_path.parent / "机密.md").write_text("# 不该被读到\n", encoding="utf-8")

    resp = await client.post("/api/ingest", json={"source": "../机密.md"})

    assert resp.status_code == 400
    assert seen == []


async def test_ingest_relative_source_loads_from_data_dir(client, db, tmp_path, monkeypatch):
    """校验口径必须等于加载口径：相对路径读到的必须是数据目录里的那份，而不是 CWD 下的同名文件。"""
    monkeypatch.setattr(settings, "embed_dim", DIM)

    async def fake_embed(texts):
        return [[1.0] + [0.0] * (DIM - 1) for _ in texts]

    monkeypatch.setattr("app.ingest.pipeline.embed_texts", fake_embed)
    note = tmp_path / "notes" / "a.md"
    note.parent.mkdir()
    note.write_text("# 数据目录里的笔记\n内容", encoding="utf-8")

    resp = await client.post("/api/ingest", json={"source": "notes/a.md"})

    assert resp.status_code == 200
    chunk = (await client.get("/api/chunks/1")).json()
    assert chunk["title"] == "数据目录里的笔记"
    assert Path(chunk["source"]) == note


async def test_upload_rejects_unsupported_suffix(client, tmp_path):
    resp = await client.post(
        "/api/ingest/upload", files={"file": ("evil.sh", b"rm -rf /", "text/plain")}
    )

    assert resp.status_code == 422
    assert not (tmp_path / "uploads").exists()


async def test_upload_rejects_oversized_file(client, tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.UPLOAD_MAX_BYTES", 4)

    resp = await client.post(
        "/api/ingest/upload", files={"file": ("笔记.md", b"12345", "text/markdown")}
    )

    assert resp.status_code == 422
    assert not (tmp_path / "uploads").exists()


async def test_upload_markdown_lands_in_uploads_and_ingests(client, db, tmp_path, monkeypatch):
    seen = []

    async def fake_ingest(source):
        seen.append(source)
        return 2

    monkeypatch.setattr("app.main.ingest", fake_ingest)

    resp = await client.post(
        # 带 ../ 的文件名不得写到 uploads/ 之外
        "/api/ingest/upload",
        files={"file": ("../笔记.md", "# 标题\n正文".encode(), "text/markdown")},
    )

    assert resp.status_code == 200
    assert resp.json() == {"chunks": 2}
    saved = tmp_path / "uploads" / "笔记.md"
    assert saved.read_bytes() == "# 标题\n正文".encode()
    assert seen == [saved]


async def test_upload_markdown_ingests_end_to_end(client, db, tmp_path, monkeypatch):
    """走真 ingest（只把 embedding 换成桩）：落盘 → 分块入库 → 引用回跳取得到原文。"""
    monkeypatch.setattr(settings, "embed_dim", DIM)

    async def fake_embed(texts):
        return [[1.0] + [0.0] * (DIM - 1) for _ in texts]

    monkeypatch.setattr("app.ingest.pipeline.embed_texts", fake_embed)

    resp = await client.post(
        "/api/ingest/upload",
        files={"file": ("笔记.md", "# KV Cache\n注意力缓存".encode(), "text/markdown")},
    )

    assert resp.status_code == 200
    assert resp.json() == {"chunks": 1}

    chunk = (await client.get("/api/chunks/1")).json()
    assert chunk["title"] == "KV Cache"
    assert "注意力缓存" in chunk["content"]
    assert Path(chunk["source"]) == tmp_path / "uploads" / "笔记.md"


async def test_memories_list_newest_first(client, db):
    async with get_db(db) as conn:
        await conn.executemany(
            "INSERT INTO memories (kind, content, confidence, created_at, status) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                ("fact", "已作废的旧事实", 0.5, "2026-09-28T08:00:00", "superseded"),
                ("fact", "用户在北京上学", 0.9, "2026-09-28T09:00:00", "active"),
                ("preference", "喜欢 Python", 0.8, "2026-09-28T10:00:00", "active"),
            ],
        )
        await conn.commit()

    rows = (await client.get("/api/memories")).json()

    # id 倒序（最新的在后插入）；作废的记忆也在列表里，由状态字段交代
    assert [r["content"] for r in rows] == ["喜欢 Python", "用户在北京上学", "已作废的旧事实"]
    assert rows[0] == {
        "id": 3,
        "kind": "preference",
        "content": "喜欢 Python",
        "status": "active",
        "confidence": 0.8,
        "created_at": "2026-09-28T10:00:00",
    }