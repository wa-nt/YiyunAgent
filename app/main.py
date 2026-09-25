import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent.runtime import (
    drain_memory_writes,
    ensure_session,
    list_messages,
    run_agent,
)
from app.db import init_db
from app.ingest.pipeline import delete_document, ingest, list_documents
from app.tracing import drain_traces, list_traces, summarize_traces

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    # 记忆写入与 trace 写入都是 fire-and-forget（进程内后台任务），退出前给它们一个收尾
    # 窗口，否则最后几轮对话的记忆与埋点会随进程一起消失。drain 自带超时，不会卡住关闭。
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
    except Exception as exc:
        yield _sse_line("error", {"message": f"{type(exc).__name__}: {exc}"})


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


@app.post("/api/ingest")
async def api_ingest(req: IngestRequest) -> dict:
    try:
        chunks = await ingest(req.source)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc
    return {"chunks": chunks}


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

    name 精确匹配；start / end 与 ts 做字典序比较，要传带偏移量的完整时间戳
    （例：2026-09-25T10:00:00.000+00:00），只传日期时 end 会漏掉当天。
    """
    return await list_traces(
        kind=kind, name=name, start=start, end=end, limit=limit, offset=offset
    )


@app.get("/api/traces/summary")
async def api_traces_summary(
    kind: str | None = None,
    name: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """成本看板：总调用次数 / 总 tokens / 总成本，外加按 kind、按 name 分组。

    过滤器与 /api/traces 同义，用于按模块或时间窗口归因成本。
    """
    return await summarize_traces(kind=kind, name=name, start=start, end=end)


# T6 的前端目录；不存在时跳过挂载（挂到 / 会吞掉未匹配的 API 路径，所以放最后）
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")