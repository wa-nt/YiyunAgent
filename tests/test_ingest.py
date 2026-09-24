import pymupdf
import pytest

from app.config import settings
from app.db import get_db, init_db
from app.ingest import loaders, pipeline
from app.ingest.chunker import chunk_text
from app.ingest.loaders import load

DIM = 8
SIZE = 300
OVERLAP = 60


def _common_overlap(a: str, b: str) -> int:
    for n in range(min(len(a), len(b)), 0, -1):
        if a.endswith(b[:n]):
            return n
    return 0


def _paragraphs(word_count: int = 400, per_paragraph: int = 20) -> list[str]:
    words = [f"w{i:04d}" for i in range(word_count)]
    return [
        " ".join(words[i : i + per_paragraph])
        for i in range(0, word_count, per_paragraph)
    ]


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("  \n\n  \t ") == []


def test_chunk_text_respects_size_and_overlap():
    text = "\n\n".join(_paragraphs())
    chunks = chunk_text(text, SIZE, OVERLAP)

    assert len(chunks) > 3
    assert all(len(c) <= SIZE for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert _common_overlap(a, b) >= OVERLAP - 2

    covered = "\n".join(chunks)
    assert "w0000" in covered and "w0399" in covered


def test_chunk_text_prefers_paragraph_boundary():
    paragraphs = _paragraphs(80, 19)
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, SIZE, OVERLAP)

    expected = "\n\n".join(paragraphs[:2])
    assert len(expected) <= SIZE
    # 第一块在段落边界收尾，而不是塞满 size 个字符
    assert chunks[0] == expected


def test_chunk_text_hard_splits_long_paragraph():
    text = "".join(f"s{i:04d}" for i in range(200))
    chunks = chunk_text(text, SIZE, OVERLAP)

    assert len(chunks) >= 3
    assert all(len(c) <= SIZE for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert _common_overlap(a, b) == OVERLAP


def test_chunk_text_clamps_runaway_overlap():
    # overlap ≥ size/2 的退化配置：不钳制时窗口每次只前进 1 字符，块数会逼近文本长度
    text = "".join(f"s{i:04d}" for i in range(500))
    size = 200
    chunks = chunk_text(text, size, size)

    assert len(chunks) < len(text) // 50
    assert all(len(c) <= size for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert size // 2 - 1 >= _common_overlap(a, b) >= size // 2 - 1 - 2


def test_chunk_text_keeps_indentation():
    code = "    def f():\n        return 1"
    chunks = chunk_text(f"# 笔记\n\n{code}\n\n结尾", SIZE, OVERLAP)

    assert len(chunks) == 1
    assert code in chunks[0]


def test_load_markdown_and_dispatch(tmp_path):
    note = tmp_path / "note.md"
    note.write_text("# 标题\n\n第一段\n\n第二段\n", encoding="utf-8")

    title, text = load(str(note))
    assert title == "标题"
    assert "第二段" in text

    other = tmp_path / "note.txt"
    other.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError):
        load(str(other))


def test_load_pdf(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "PDF body text")
    doc.set_metadata({"title": "PDF 标题"})
    doc.save(pdf_path)
    doc.close()

    title, text = loaders.load_pdf(pdf_path)
    assert title == "PDF 标题"
    assert "PDF body text" in text


def test_load_url_strips_tags_without_network(monkeypatch):
    html = (
        "<html><head><title> 网页标题 </title><style>b{}</style>"
        "<script>var x=1;</script></head>"
        "<body><h1>头部</h1><p>第一段</p><p>第二段</p></body></html>"
    )

    class _Response:
        text = html

        def raise_for_status(self):
            pass

    monkeypatch.setattr(loaders.httpx, "get", lambda *a, **k: _Response())

    title, text = loaders.load_url("https://example.com/article")
    assert title == "网页标题"
    assert "第一段" in text and "第二段" in text
    assert "var x=1" not in text and "b{}" not in text and "<p>" not in text


def test_extract_html_without_head_close_tag():
    # 不少页面缺 </head>；head 一旦计入跳过，正文会被整体丢弃
    html = "<html><head><title>标题</title><body><p>正文第一段</p><p>正文第二段</p>"
    title, text = loaders.extract_html(html)

    assert title == "标题"
    assert "正文第一段" in text and "正文第二段" in text


def test_extract_html_keeps_indentation():
    html = "<p>说明</p><pre>    code line\n        nested\n\n\n\n尾部</pre>"

    _, text = loaders.extract_html(html)

    assert "    code line\n        nested" in text
    assert "\n\n\n" not in text


@pytest.mark.asyncio
async def test_ingest_list_and_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "embed_dim", DIM)
    batches: list[list[str]] = []

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        batches.append(list(texts))
        return [[float(i % 7) for i in range(DIM)] for _ in texts]

    monkeypatch.setattr(pipeline, "embed_texts", fake_embed_texts)

    note = tmp_path / "note.md"
    note.write_text(
        "# KV Cache 笔记\n\n" + "\n\n".join(_paragraphs(200, 20)), encoding="utf-8"
    )
    db_path = tmp_path / "app.db"

    count = await pipeline.ingest(str(note), db_path)

    assert count > 1
    assert len(batches) == 1
    assert len(batches[0]) == count

    async with get_db(db_path) as conn:
        docs = await conn.execute_fetchall("SELECT id, source, title FROM documents")
        chunks = await conn.execute_fetchall(
            "SELECT id, doc_id, idx, content, token_count, embedding FROM chunks"
        )
        vectors = await conn.execute_fetchall("SELECT chunk_id FROM chunk_vectors")

    assert len(docs) == 1
    assert docs[0]["title"] == "KV Cache 笔记"
    assert docs[0]["source"] == str(note)

    assert len(chunks) == count
    assert [c["idx"] for c in chunks] == list(range(count))
    assert all(c["doc_id"] == docs[0]["id"] for c in chunks)
    assert all(c["token_count"] == len(c["content"]) for c in chunks)
    assert all(len(c["embedding"]) == DIM * 4 for c in chunks)  # float32 blob

    assert len(vectors) == count
    assert {v["chunk_id"] for v in vectors} == {c["id"] for c in chunks}

    listed = await pipeline.list_documents(db_path)
    assert len(listed) == 1
    assert listed[0]["id"] == docs[0]["id"]
    assert listed[0]["chunk_count"] == count

    await pipeline.delete_document(docs[0]["id"], db_path)

    assert await pipeline.list_documents(db_path) == []
    async with get_db(db_path) as conn:
        assert await conn.execute_fetchall("SELECT id FROM documents") == []
        assert await conn.execute_fetchall("SELECT id FROM chunks") == []
        assert await conn.execute_fetchall("SELECT chunk_id FROM chunk_vectors") == []


@pytest.mark.asyncio
async def test_ingest_embedding_failure_leaves_no_document(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "embed_dim", DIM)

    async def broken_embed_texts(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(pipeline, "embed_texts", broken_embed_texts)

    note = tmp_path / "note.md"
    note.write_text("# 标题\n\n" + "内容 " * 100, encoding="utf-8")
    db_path = tmp_path / "app.db"

    with pytest.raises(RuntimeError):
        await pipeline.ingest(str(note), db_path)

    assert await pipeline.list_documents(db_path) == []
    async with get_db(db_path) as conn:
        assert await conn.execute_fetchall("SELECT id FROM chunks") == []
        assert await conn.execute_fetchall("SELECT chunk_id FROM chunk_vectors") == []


@pytest.mark.asyncio
async def test_ingest_empty_document_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "embed_dim", DIM)

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        raise AssertionError("空文档不应触发 embedding")

    monkeypatch.setattr(pipeline, "embed_texts", fake_embed_texts)

    note = tmp_path / "empty.md"
    note.write_text("\n\n   \n\n", encoding="utf-8")

    assert await pipeline.ingest(str(note), tmp_path / "app.db") == 0


@pytest.mark.asyncio
async def test_ingest_embeds_before_opening_write_transaction(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "embed_dim", DIM)
    db_path = tmp_path / "app.db"
    await init_db(db_path)  # 先建表，便于在 embedding 期间直接查库

    seen: dict[str, int] = {}

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        async with get_db(db_path) as conn:
            seen["documents"] = len(
                await conn.execute_fetchall("SELECT id FROM documents")
            )
        return [[1.0] * DIM for _ in texts]

    monkeypatch.setattr(pipeline, "embed_texts", fake_embed_texts)

    note = tmp_path / "note.md"
    note.write_text("# 标题\n\n" + "正文 " * 100, encoding="utf-8")

    count = await pipeline.ingest(str(note), db_path)

    # embedding 期间还没有任何写入，写事务不跨网络 I/O（也无需读连接阻塞）
    assert seen["documents"] == 0
    assert count > 0
    async with get_db(db_path) as conn:
        assert len(await conn.execute_fetchall("SELECT id FROM documents")) == 1
        assert len(await conn.execute_fetchall("SELECT id FROM chunks")) == count