import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent.runtime import ensure_session, list_messages, run_agent
from app.db import init_db
from app.ingest.pipeline import delete_document, ingest, list_documents

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


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
    """异常兜底也要保证可序列化：data 里可能带着无法 json 化的对象。"""
    try:
        payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
    except (TypeError, ValueError):
        payload = json.dumps(
            {"type": event_type, "data": {"message": "服务器内部错误"}}, ensure_ascii=False
        )
    return f"data: {payload}\n\n"


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


# T6 的前端目录；不存在时跳过挂载（挂到 / 会吞掉未匹配的 API 路径，所以放最后）
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")