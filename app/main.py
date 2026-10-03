import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent.runtime import (
    drain_memory_writes,
    ensure_session,
    list_messages,
    run_agent,
)
from app.config import settings
from app.db import get_db, init_db
from app.ingest.pipeline import delete_document, ingest, list_documents
from app.tracing import (
    InvalidTimestamp,
    drain_traces,
    list_traces,
    summarize_traces,
)

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

UPLOAD_SUFFIXES = {".md", ".markdown", ".pdf", ".txt"}
UPLOAD_MAX_BYTES = 50 * 1024 * 1024  # 固定上限，需要时再做配置


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    # 记忆写入与 trace 写入都是 fire-and-forget（进程内后台任务），退出前给它们一个收尾
    # 窗口，否则最后几轮对话的记忆与埋点会随进程一起消失。drain 的超时是软上限
    # （每轮 DRAIN_TIMEOUT，最坏还要加上 sqlite busy timeout），不会无限卡住关闭。
    # 顺序不能反：记忆抽取自己也会调 LLM（因此产生 llm trace），先收记忆再收 trace
    await drain_memory_writes()
    await drain_traces()


app = FastAPI(title="第二大脑 Agent", lifespan=lifespan)


class IngestRequest(BaseModel):
    source: str


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


async def _sse(req: ChatRequest):
    """SSE 流：每个 AgentEvent 一行 `data: {json}`。session_id 为空的请求
    先把新建的 id 作为首个事件发出，前端据此续聊。

    整个流包在 try/except 里：run_agent 抛出、或写库 / json.dumps 失败时，
    HTTP 状态码已经发出去了，只能尽量补一个 error 事件再结束——否则前端
    收到的是 200 加静默截断，会一直等 done。
    """
    try:
        if not req.session_id:
            req.session_id = await ensure_session(None)
            yield _sse_line("session", {"session_id": req.session_id})
        async for event in run_agent(req.session_id, req.message):
            yield _sse_line(event.type, event.data)
    except Exception:
        # 异常文本对用户没有意义（还可能带出内部路径），只记日志，界面给一句可重试的提示
        logger.exception("SSE 流中断")
        yield _sse_line("error", {"message": "生成出错，请重试"})


def _sse_line(event_type: str, data: dict) -> str:
    """SSE 一行。事件数据都由 runtime 用 JSON 可序列化的值构造，无需再做兜底。"""
    return f"data: {json.dumps({'type': event_type, 'data': data}, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _sse(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/sessions/{session_id}/messages")
async def session_messages(session_id: str) -> list[dict]:
    return await list_messages(session_id)


@app.get("/api/sessions")
async def api_sessions() -> list[dict]:
    """会话列表，最近活跃在前。title 取首条 user 消息前 30 字，还没发过言的会话为空串。"""
    async with get_db() as conn:
        rows = await conn.execute_fetchall(
            "SELECT s.id, s.created_at, "
            "COALESCE((SELECT substr(content, 1, 30) FROM messages WHERE session_id = s.id "
            "AND role = 'user' ORDER BY id LIMIT 1), '') AS title, "
            "(SELECT COUNT(*) FROM messages WHERE session_id = s.id) AS message_count "
            "FROM sessions s "
            "ORDER BY (SELECT MAX(id) FROM messages WHERE session_id = s.id) DESC"
        )
    return [dict(row) for row in rows]


@app.get("/api/chunks/{chunk_id}")
async def api_chunk(chunk_id: int) -> dict:
    """单条 chunk 及所属文档的标题/来源，供前端从引用角标跳到原文。"""
    async with get_db() as conn:
        rows = await conn.execute_fetchall(
            "SELECT c.id, c.content, d.title, d.source FROM chunks c "
            "JOIN documents d ON d.id = c.doc_id WHERE c.id = ?",
            (chunk_id,),
        )
    if not rows:
        raise HTTPException(status_code=404, detail="chunk 不存在")
    return dict(rows[0])


@app.post("/api/ingest")
async def api_ingest(req: IngestRequest) -> dict:
    """按 source 导入。非 URL 的来源只能是数据目录内的文件。

    校验放在 HTTP 边界（不动 pipeline.ingest，内部调用方如 eval 仍可读任意路径）：
    否则这个接口等于任意本地文件读取——source 传个 .env 之外的 .md/.txt/.pdf 就能被读走。
    URL 不过路径检查，由 loaders 的公网校验负责。
    """
    source: str | Path = req.source
    if not req.source.startswith(("http://", "https://")):
        root = Path(settings.db_path).resolve().parent
        target = Path(req.source)
        # 相对路径按数据目录解析：上传接口落盘的 uploads/xxx 与 data/ 下的笔记都靠它
        if not target.is_absolute():
            target = root / target
        # 必须 resolve 后再比（`..` 与符号链接都能逃出目录）；resolve 会归一 Windows 路径大小写
        target = target.resolve()
        if not target.is_relative_to(root):
            raise HTTPException(status_code=400, detail="只允许导入数据目录下的文件，其他文件请用上传接口")
        # 查的和读的必须是同一个文件：把解析后的绝对路径交给管线，而不是按 CWD 再解析一次的原串
        source = target
    try:
        chunks = await ingest(source)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"chunks": chunks}


@app.post("/api/ingest/upload")
async def api_upload(file: UploadFile) -> dict:
    """上传本地文件，落盘到 db_path 同级的 uploads/ 后走与 /api/ingest 相同的管线。

    用文件名（不带客户端给的目录）做目标名，避免 ../ 写出目录。
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        raise HTTPException(status_code=422, detail=f"不支持的文件类型：{suffix or file.filename}")
    data = await file.read()
    if len(data) > UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=422, detail="文件超过 50MB 上限")
    dest = Path(settings.db_path).parent / "uploads" / Path(file.filename).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    try:
        chunks = await ingest(dest)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"chunks": chunks}


@app.get("/api/memories")
async def api_memories() -> list[dict]:
    """记忆列表，最新在前限 200 条。"""
    async with get_db() as conn:
        rows = await conn.execute_fetchall(
            "SELECT id, kind, content, status, confidence, created_at FROM memories "
            "ORDER BY id DESC LIMIT 200"
        )
    return [dict(row) for row in rows]


@app.get("/api/documents")
async def api_documents() -> list[dict]:
    return await list_documents()


@app.delete("/api/documents/{doc_id}")
async def api_delete_document(doc_id: int) -> dict:
    await delete_document(doc_id)
    return {"deleted": doc_id}


@app.get("/api/traces")
async def api_traces(
    kind: str | None = None,
    name: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    """trace 列表（最新在前），可按 kind / name / 时间范围过滤并分页。

    name 精确匹配；start / end 必须是带时区的 ISO 8601 时间戳（任意偏移或 Z 均可，
    如 2026-09-25T10:00:00Z、2026-09-25T18:00:00+08:00），闭区间 [start, end]，
    不规范的值回 422。
    """
    try:
        return await list_traces(
            kind=kind, name=name, start=start, end=end, limit=limit, offset=offset
        )
    except InvalidTimestamp as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/traces/summary")
async def api_traces_summary(
    kind: str | None = None,
    name: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """成本看板：总调用次数 / 总 tokens / 总成本，外加按 kind、按 name 分组。

    过滤器与 /api/traces 同义（含时间戳格式要求），用于按模块或时间窗口归因成本。
    """
    try:
        return await summarize_traces(kind=kind, name=name, start=start, end=end)
    except InvalidTimestamp as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# T6 的前端目录；不存在时跳过挂载（挂到 / 会吞掉未匹配的 API 路径，所以放最后）
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")