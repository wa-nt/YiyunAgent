import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError, AsyncOpenAI, AuthenticationError, OpenAIError
from pydantic import BaseModel, Field

from app.agent.runtime import (
    DEFAULT_MODE,
    SUPPORTED_MODES,
    UnknownModeError,
    drain_memory_writes,
    ensure_session,
    get_session_mode,
    list_messages,
    run_agent,
)
from app.config import EDITABLE_FIELDS, env_path, mask_secret, settings, update_env_file
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
    # 模式：None = 未指定（新会话按 chat，已有会话沿用库里的值）。取值不在 pydantic 里
    # 收窄成 Literal：非法值要统一回 400 并说明原因，Literal 会先把它变成 422 的通用
    # 校验错误；校验唯一来源是 runtime.SUPPORTED_MODES（见 _requested_mode）。
    mode: str | None = None


def _requested_mode(mode: str | None) -> str | None:
    """校验请求里的 mode，返回 None（未指定）或合法模式名。

    code 本期只有前端占位，后端照样按非法拒绝并说明原因：前端的 disabled 是 UX，
    不是安全边界。
    """
    if mode is None:
        return None
    if mode == "code":
        raise HTTPException(status_code=400, detail="code 模式本期未开放，请使用 chat 或 work")
    if mode not in SUPPORTED_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的模式：{mode}（可选：{'、'.join(SUPPORTED_MODES)}）",
        )
    return mode


async def _existing_session_mode(session_id: str) -> str | None:
    """已有会话的模式；会话不存在时返回 None。

    库里的 mode 是未知值时由 get_session_mode 抛 UnknownModeError，交给调用方收口。
    """
    async with get_db() as conn:
        rows = await conn.execute_fetchall("SELECT 1 FROM sessions WHERE id = ?", (session_id,))
    if not rows:
        return None
    return await get_session_mode(session_id)


async def _sse_events(events) -> AsyncIterator[str]:
    """把 AgentEvent 流编成 SSE 行。整个流包在 try/except 里：run_agent 抛出、或写库 /
    json.dumps 失败时，HTTP 状态码已经发出去了，只能尽量补一个 error 事件再结束——
    否则前端收到的是 200 加静默截断，会一直等 done。"""
    try:
        async for event in events:
            yield _sse_line(event.type, event.data)
    except Exception:
        # 异常文本对用户没有意义（还可能带出内部路径），只记日志，界面给一句可重试的提示
        logger.exception("SSE 流中断")
        yield _sse_line("error", {"message": "生成出错，请重试"})


async def _sse(req: ChatRequest, new_session_mode: str | None = None):
    """SSE 流：每个 AgentEvent 一行 `data: {json}`。session_id 为空的请求
    先把新建的 id 连同模式作为首个事件发出，前端据此续聊。

    new_session_mode 只在「请求没带 session_id、由本函数新建会话」时用得上；已有会话
    与补建路径都不给 run_agent 传 mode（模式以库里的为准，见 run_agent）。
    """
    created_mode: str | None = None
    try:
        if not req.session_id:
            created_mode = new_session_mode or DEFAULT_MODE
            req.session_id = await ensure_session(None, mode=created_mode)
            yield _sse_line("session", {"session_id": req.session_id, "mode": created_mode})
    except Exception:
        # 建会话失败同样只能补 error 事件：HTTP 状态码已发出，不能让前端干等
        logger.exception("创建会话失败")
        yield _sse_line("error", {"message": "生成出错，请重试"})
        return
    # 只有刚新建的会话显式传 mode：其余路径交给 run_agent 从会话读，避免两个来源打架
    extra = {"mode": created_mode} if created_mode else {}
    async for line in _sse_events(run_agent(req.session_id, req.message, **extra)):
        yield line


def _sse_line(event_type: str, data: dict) -> str:
    """SSE 一行。事件数据都由 runtime 用 JSON 可序列化的值构造，无需再做兜底。"""
    return f"data: {json.dumps({'type': event_type, 'data': data}, ensure_ascii=False)}\n\n"


def _sse_response(lines: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        lines,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """聊天入口。新会话（不带 session_id）用请求里的 mode（缺省 chat）建立会话，
    session 事件把它回给前端；已有会话一律按库里的 mode 跑，请求带的 mode 只用来
    校验一致性——不一致说明前端拿着另一个模式的会话在聊，回 409 而不是偷偷改模式。"""
    mode = _requested_mode(req.mode)
    if req.session_id is not None:
        try:
            session_mode = await _existing_session_mode(req.session_id)
        except UnknownModeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if session_mode is None:
            # 未知 id：老调用方（含测试）一直靠 run_agent 里的 ensure_session 补建，
            # 只有显式声明了 mode 的请求才按「指错了会话」处理
            if mode is not None:
                raise HTTPException(status_code=404, detail="会话不存在")
        elif mode is not None and mode != session_mode:
            raise HTTPException(
                status_code=409,
                detail=f"会话模式为 {session_mode}，不能按 {mode} 继续（模式在创建会话时固定）",
            )
    return _sse_response(_sse(req, new_session_mode=mode or DEFAULT_MODE))


@app.post("/api/sessions/{session_id}/respond")
async def respond(session_id: str, mid: int | None = None) -> StreamingResponse:
    """回答会话里已落库的用户提问（编辑后重答 / 重新生成的共用入口）。

    mid 给出时先把 active_leaf 切到那条提问上（重新生成 = 回到提问再答一次，
    旧回答留在自己的分支上不动）；不给则用当前叶子。叶子必须是 user 消息。
    与 /api/chat 的差别只在不再插入用户消息——提问已经由调用方写好了；模式同样不传，
    由 run_agent 从会话读。
    """
    async with get_db() as conn:
        if mid is not None:
            rows = await conn.execute_fetchall(
                "SELECT role FROM messages WHERE id = ? AND session_id = ?",
                (mid, session_id),
            )
            if not rows:
                raise HTTPException(status_code=404, detail="消息不存在")
            if rows[0]["role"] != "user":
                raise HTTPException(status_code=422, detail="只能对用户提问重新生成")
            await conn.execute(
                "UPDATE sessions SET active_leaf = ? WHERE id = ?", (mid, session_id)
            )
            await conn.commit()
        rows = await conn.execute_fetchall(
            "SELECT role, content FROM messages WHERE session_id = ?1 AND id = "
            "COALESCE((SELECT active_leaf FROM sessions WHERE id = ?1), "
            "(SELECT MAX(id) FROM messages WHERE session_id = ?1))",
            (session_id,),
        )
    if not rows or rows[0]["role"] != "user":
        raise HTTPException(status_code=409, detail="会话末尾没有待回答的提问")
    message = rows[0]["content"] or ""
    return _sse_response(_sse_events(run_agent(session_id, message, user_saved=True)))


@app.get("/api/sessions/{session_id}/messages")
async def session_messages(session_id: str) -> list[dict]:
    return await list_messages(session_id)


def _like_escape(q: str) -> str:
    """LIKE 的 %/_ 转义（配合 ESCAPE '\\'），否则用户搜个 100% 就变成全匹配。"""
    return q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@app.get("/api/sessions")
async def api_sessions(q: str = "") -> list[dict]:
    """会话列表，最近活跃在前。title 优先用户改名 / 自动标题，回退首条 user 消息前 30 字；
    带上 mode 与 source，前端据此显示模式、区分定时任务创建的自动会话。NULL 按迁移口径
    归一（chat / manual），列表读到的 mode、source 因此永远是合法值。
    q 命中标题或任意一条消息内容的会话才返回（LIKE 子串匹配，量大了再考虑 FTS）。"""
    like = f"%{_like_escape(q)}%" if q else ""
    async with get_db() as conn:
        rows = await conn.execute_fetchall(
            "SELECT s.id, s.created_at, s.provider, s.model, "
            "COALESCE(s.mode, 'chat') AS mode, COALESCE(s.source, 'manual') AS source, "
            "COALESCE(s.title, (SELECT substr(content, 1, 30) FROM messages "
            "WHERE session_id = s.id AND role = 'user' ORDER BY id LIMIT 1), '') AS title, "
            "(SELECT COUNT(*) FROM messages WHERE session_id = s.id) AS message_count "
            "FROM sessions s "
            "WHERE (? = '' OR COALESCE(s.title, '') LIKE ? ESCAPE '\\' "
            "OR EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.id "
            "AND m.content LIKE ? ESCAPE '\\')) "
            "ORDER BY (SELECT MAX(id) FROM messages WHERE session_id = s.id) DESC",
            (q, like, like),
        )
    return [dict(row) for row in rows]


class SessionUpdate(BaseModel):
    """字段缺省 = 不动；provider/model 传空串 = 清掉覆盖（跟随全局）。"""

    title: str | None = None
    provider: Literal["openai_compat", "anthropic", ""] | None = None
    model: str | None = None


@app.patch("/api/sessions/{session_id}")
async def api_session_update(session_id: str, req: SessionUpdate) -> dict:
    if req.title is None and req.provider is None and req.model is None:
        raise HTTPException(status_code=422, detail="没有要更新的字段")
    async with get_db() as conn:
        rows = await conn.execute_fetchall(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        )
        if not rows:
            raise HTTPException(status_code=404, detail="会话不存在")
        if req.title is not None:
            title = req.title.strip()
            if not title:
                raise HTTPException(status_code=422, detail="标题不能为空")
            await conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)
            )
        if req.provider is not None:
            await conn.execute(
                "UPDATE sessions SET provider = ? WHERE id = ?",
                (req.provider or None, session_id),
            )
        if req.model is not None:
            await conn.execute(
                "UPDATE sessions SET model = ? WHERE id = ?",
                (req.model.strip() or None, session_id),
            )
        await conn.commit()
    return {"updated": session_id}


@app.delete("/api/sessions/{session_id}")
async def api_session_delete(session_id: str) -> dict:
    """删会话连带消息：messages 引用 sessions 且没有 ON DELETE CASCADE，
    必须先删子表再删父表，否则外键约束直接拒。"""
    async with get_db() as conn:
        rows = await conn.execute_fetchall(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        )
        if not rows:
            raise HTTPException(status_code=404, detail="会话不存在")
        await conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await conn.commit()
    return {"deleted": session_id}


class MessageUpdate(BaseModel):
    content: str


@app.put("/api/messages/{message_id}")
async def api_message_edit(message_id: int, req: MessageUpdate) -> dict:
    """编辑用户提问 = 开分支：原消息原样保留，在同 parent 下新建一条兄弟消息
    并把 active_leaf 切过去，前端随后调 /respond 在新分支上生成回答。
    只允许编辑 user 消息——改 assistant 的回答等于伪造历史。"""
    content = req.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="内容不能为空")
    async with get_db() as conn:
        rows = await conn.execute_fetchall(
            "SELECT session_id, role, parent_id FROM messages WHERE id = ?", (message_id,)
        )
        if not rows:
            raise HTTPException(status_code=404, detail="消息不存在")
        if rows[0]["role"] != "user":
            raise HTTPException(status_code=422, detail="只能编辑自己的提问")
        cursor = await conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at, parent_id) "
            "VALUES (?, 'user', ?, ?, ?)",
            (
                rows[0]["session_id"],
                content,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                rows[0]["parent_id"],
            ),
        )
        new_id = cursor.lastrowid
        await conn.execute(
            "UPDATE sessions SET active_leaf = ? WHERE id = ?",
            (new_id, rows[0]["session_id"]),
        )
        await conn.commit()
    return {"updated": message_id, "new_id": new_id}


@app.delete("/api/messages/{message_id}")
async def api_message_delete(message_id: int) -> dict:
    """删除一条消息及其整个子分支；当前分支被删断时 active_leaf 回退到被删节点的 parent
    （None 也行——读取侧会回退到 max(id)）。"""
    async with get_db() as conn:
        rows = await conn.execute_fetchall(
            "SELECT session_id, parent_id FROM messages WHERE id = ?", (message_id,)
        )
        if not rows:
            raise HTTPException(status_code=404, detail="消息不存在")
        session_id, parent_id = rows[0]["session_id"], rows[0]["parent_id"]
        # aiosqlite 对 CTE+DELETE 的 rowcount 不可靠（恒为 -1），用 changes() 取真实删除数
        await conn.execute(
            "WITH RECURSIVE sub AS ("
            "  SELECT id FROM messages WHERE id = ?"
            "  UNION ALL"
            "  SELECT m.id FROM messages m JOIN sub s ON m.parent_id = s.id"
            ") DELETE FROM messages WHERE id IN (SELECT id FROM sub)",
            (message_id,),
        )
        deleted_n = (await conn.execute_fetchall("SELECT changes() AS n"))[0]["n"]
        leaf = await conn.execute_fetchall(
            "SELECT active_leaf FROM sessions WHERE id = ?", (session_id,)
        )
        if leaf and leaf[0]["active_leaf"] is not None:
            alive = await conn.execute_fetchall(
                "SELECT 1 FROM messages WHERE id = ?", (leaf[0]["active_leaf"],)
            )
            if not alive:
                await conn.execute(
                    "UPDATE sessions SET active_leaf = ? WHERE id = ?",
                    (parent_id, session_id),
                )
        await conn.commit()
    return {"deleted": deleted_n}


class BranchSwitch(BaseModel):
    direction: Literal[-1, 1]


@app.post("/api/messages/{message_id}/branch")
async def api_message_branch(message_id: int, req: BranchSwitch) -> dict:
    """切到相邻分支：找到同 parent 的上一个/下一个兄弟，把 active_leaf 落到它
    子树里最新的一支（分支切换后看到的是该分支最近一次对话的结尾）。"""
    async with get_db() as conn:
        rows = await conn.execute_fetchall(
            "SELECT session_id, parent_id FROM messages WHERE id = ?", (message_id,)
        )
        if not rows:
            raise HTTPException(status_code=404, detail="消息不存在")
        session_id, parent_id = rows[0]["session_id"], rows[0]["parent_id"]
        if parent_id is None:
            sibs = await conn.execute_fetchall(
                "SELECT id FROM messages WHERE session_id = ? AND parent_id IS NULL "
                "ORDER BY id",
                (session_id,),
            )
        else:
            sibs = await conn.execute_fetchall(
                "SELECT id FROM messages WHERE session_id = ? AND parent_id = ? "
                "ORDER BY id",
                (session_id, parent_id),
            )
        ids = [s["id"] for s in sibs]
        target = ids.index(message_id) + req.direction
        if not 0 <= target < len(ids):
            raise HTTPException(status_code=404, detail="那个方向没有更多分支")
        # 沿最新子节点下探到叶子
        leaf = ids[target]
        while True:
            child = (
                await conn.execute_fetchall(
                    "SELECT MAX(id) AS c FROM messages WHERE parent_id = ?", (leaf,)
                )
            )[0]["c"]
            if child is None:
                break
            leaf = child
        await conn.execute(
            "UPDATE sessions SET active_leaf = ? WHERE id = ?", (leaf, session_id)
        )
        await conn.commit()
    return {"leaf": leaf}


@app.patch("/api/memories/{memory_id}")
async def api_memory_edit(memory_id: int, req: MessageUpdate) -> dict:
    content = req.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="内容不能为空")
    async with get_db() as conn:
        cursor = await conn.execute(
            "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
            (content, datetime.now(timezone.utc).isoformat(timespec="seconds"), memory_id),
        )
        await conn.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return {"updated": memory_id}


@app.delete("/api/memories/{memory_id}")
async def api_memory_delete(memory_id: int) -> dict:
    async with get_db() as conn:
        cursor = await conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        await conn.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return {"deleted": memory_id}


@app.get("/api/export")
async def api_export() -> JSONResponse:
    """用户数据导出（会话/消息/记忆/文档清单）。chunks 与向量是可从文档重建的
    派生数据，不进导出文件；要完整备份请用 /api/export/db。"""
    async with get_db() as conn:
        data: dict = {
            "version": 1,
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        for table in ("sessions", "messages", "memories", "documents"):
            rows = await conn.execute_fetchall(f"SELECT * FROM {table}")
            data[table] = [dict(row) for row in rows]
    return JSONResponse(
        data,
        headers={"Content-Disposition": 'attachment; filename="second-brain-export.json"'},
    )


@app.get("/api/export/db")
async def api_export_db() -> FileResponse:
    """整库下载。WAL 模式下最新写入可能还在 -wal 文件里，先 checkpoint 合并进主库文件再发。"""
    async with get_db() as conn:
        await conn.execute("PRAGMA wal_checkpoint(FULL)")
    return FileResponse(settings.db_path, filename="app.db")


def _iso_from_ts(ts) -> str:
    """ChatGPT 导出里的 unix 秒时间戳转 ISO；缺失/异常时回退当前时间。"""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError, OSError, OverflowError):
        return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _linearize_chatgpt(convo: dict) -> list[tuple[str, str, str]]:
    """把 ChatGPT conversations.json 的树形 mapping 拉直成 [(role, content, created_at)]。

    从根节点（parent 为 None）沿 children[0] 走主分支：导出文件里一条会话可能有多个
    分支，只取当前生效的那条。system/tool 角色、非文本 parts（图片等）跳过。
    """
    mapping = convo.get("mapping") or {}
    node = next((n for n in mapping.values() if n.get("parent") is None), None)
    out: list[tuple[str, str, str]] = []
    while node:
        msg = node.get("message")
        role = ((msg or {}).get("author") or {}).get("role")
        if role in ("user", "assistant"):
            parts = ((msg.get("content") or {}).get("parts")) or []
            text = "\n".join(p for p in parts if isinstance(p, str)).strip()
            if text:
                out.append((role, text, _iso_from_ts(msg.get("create_time"))))
        children = node.get("children") or []
        node = mapping.get(children[0]) if children else None
    return out


def _linearize_claude(convo: dict) -> list[tuple[str, str, str]]:
    """Claude 导出（conversations.json）是线性结构：chat_messages 已按序排列，
    sender 用 human/assistant，正文在 text 或 content[].text。"""
    out: list[tuple[str, str, str]] = []
    for m in convo.get("chat_messages") or []:
        role = {"human": "user", "assistant": "assistant"}.get(m.get("sender"))
        if not role:
            continue
        text = (m.get("text") or "").strip()
        if not text:
            parts = [
                c.get("text", "")
                for c in (m.get("content") or [])
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            text = "\n".join(p for p in parts if p).strip()
        if text:
            # Claude 的 created_at 本身已是 ISO 字符串，原样保留
            out.append(
                (
                    role,
                    text,
                    m.get("created_at")
                    or datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
            )
    return out


@app.post("/api/import/chatgpt")
async def api_import_chatgpt(file: UploadFile) -> dict:
    """导入 ChatGPT / Claude 的 conversations.json：逐会话嗅探格式（mapping 树 =
    ChatGPT，chat_messages 列表 = Claude），每个会话建一条 session（保留原标题），
    消息按序落库并串成 parent 链。重复导入会重复建会话，由用户自行删除。"""
    data = await file.read()
    if len(data) > UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=422, detail="文件超过 50MB 上限")
    try:
        convos = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(status_code=422, detail="不是有效的 JSON 文件")
    if not isinstance(convos, list):
        raise HTTPException(status_code=422, detail="不是 ChatGPT/Claude 导出格式（应为会话数组）")

    sessions_n = messages_n = 0
    async with get_db() as conn:
        for convo in convos:
            if not isinstance(convo, dict):
                continue
            if "mapping" in convo:
                msgs = _linearize_chatgpt(convo)
            elif "chat_messages" in convo:
                msgs = _linearize_claude(convo)
            else:
                continue
            if not msgs:
                continue
            session_id = uuid.uuid4().hex
            title = (convo.get("title") or convo.get("name") or "").strip() or None
            await conn.execute(
                "INSERT INTO sessions (id, created_at, title) VALUES (?, ?, ?)",
                (session_id, _iso_from_ts(convo.get("create_time") or convo.get("created_at")), title),
            )
            prev_id = None
            for role, content, ts in msgs:
                cursor = await conn.execute(
                    "INSERT INTO messages (session_id, role, content, created_at, parent_id) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, role, content, ts, prev_id),
                )
                prev_id = cursor.lastrowid
            await conn.execute(
                "UPDATE sessions SET active_leaf = ? WHERE id = ?",
                (prev_id, session_id),
            )
            sessions_n += 1
            messages_n += len(msgs)
        await conn.commit()
    return {"sessions": sessions_n, "messages": messages_n}


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


class SettingsUpdate(BaseModel):
    """设置面板的可编辑字段（全集见 config.EDITABLE_FIELDS）。api_key 留空 = 保持现状。"""

    llm_provider: Literal["openai_compat", "anthropic"] | None = None
    openai_base_url: str | None = None
    openai_api_key: str | None = None
    openai_model: str | None = None
    anthropic_api_key: str | None = None
    anthropic_model: str | None = None
    embed_base_url: str | None = None
    embed_api_key: str | None = None
    embed_model: str | None = None
    embed_dim: int | None = Field(default=None, gt=0)


# 密钥字段不接受空串覆盖：清空密钥属于破坏性操作，让它只能去改 .env 完成
_SECRET_FIELDS = {"openai_api_key", "anthropic_api_key", "embed_api_key"}


@app.get("/api/settings")
async def api_settings_get() -> dict:
    """当前模型 / embedding 配置，密钥脱敏回显（只露末 4 位）。"""
    return {
        "llm_provider": settings.llm_provider,
        "openai_base_url": settings.openai_base_url or "",
        "openai_api_key": mask_secret(settings.openai_api_key),
        "openai_model": settings.openai_model or "",
        "anthropic_api_key": mask_secret(settings.anthropic_api_key),
        "anthropic_model": settings.anthropic_model,
        "embed_base_url": settings.embed_base_url or "",
        "embed_api_key": mask_secret(settings.embed_api_key),
        "embed_model": settings.embed_model or "",
        "embed_dim": settings.embed_dim,
    }


@app.post("/api/settings")
async def api_settings_update(req: SettingsUpdate) -> dict:
    """保存设置：写 .env（重启不丢）并同步 settings 单例。

    LLM / embedding 客户端都是每次调用时新建（llm/__init__.py、llm/embed.py），
    所以保存立即生效，无需重启。注意 embed_dim 只影响之后写入的向量：与现有
    vec0 表维度不一致时检索会报错，前端已提示需重新导入文档。
    """
    updates: dict = {}
    for name in EDITABLE_FIELDS:
        value = getattr(req, name)
        if value is None or (name in _SECRET_FIELDS and value == ""):
            continue
        updates[name] = value

    # 校验「保存后的完整配置」而不是这次增量：切了供应商但没配齐密钥/模型就拒绝，
    # 否则写出来的是一用就 401 的半残配置。openai_base_url 允许为空（SDK 默认官方端点）
    merged = {name: getattr(settings, name) for name in EDITABLE_FIELDS} | updates
    if merged["llm_provider"] == "anthropic":
        missing = [n for n in ("anthropic_api_key", "anthropic_model") if not merged[n]]
    else:
        missing = [n for n in ("openai_api_key", "openai_model") if not merged[n]]
    if missing:
        raise HTTPException(status_code=422, detail=f"当前供应商缺少必填项：{', '.join(missing)}")

    if updates:
        update_env_file(env_path(), {n: str(v) for n, v in updates.items()})
        for name, value in updates.items():
            setattr(settings, name, value)
    return {"updated": sorted(updates)}


class ModelListRequest(BaseModel):
    """拉取模型列表。base_url / api_key 用面板里**当前填的**值，留空则回落已保存配置
    （密钥输入框平时是空的，只有用户新填时才带上）。"""

    kind: Literal["llm", "embed"]
    base_url: str | None = None
    api_key: str | None = None


def _model_context_length(model) -> int | None:
    """供应商在 /models 里附带的上下文窗口长度。字段名各家不一，OpenAI 官方压根不给。"""
    for field in ("context_length", "context_window", "max_context_length"):
        value = getattr(model, field, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


@app.post("/api/models")
async def api_models(req: ModelListRequest) -> dict:
    """转发 GET {base_url}/models，把供应商的模型清单给前端做输入建议。

    必须走后端：浏览器直连供应商会跨域失败，而且密钥只该存在服务端。只读、不落库，
    拉取失败就是一次普通的 HTTP 错误，不影响任何已保存配置。
    """
    if req.kind == "llm":
        if settings.llm_provider == "anthropic":
            raise HTTPException(status_code=400, detail="Anthropic 暂不支持拉取，请手动填写模型名")
        base_url = req.base_url or settings.openai_base_url
        api_key = req.api_key or settings.openai_api_key
    else:
        base_url = req.base_url or settings.embed_base_url
        api_key = req.api_key or settings.embed_api_key

    try:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url or None, timeout=20.0)
    except OpenAIError as exc:  # 没有 key 时构造函数就抛
        raise HTTPException(status_code=422, detail=f"缺少 API Key：{exc}") from exc

    try:
        page = await client.models.list()
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail="API Key 无效或无权访问该端点") from exc
    except APIConnectionError as exc:
        raise HTTPException(status_code=502, detail=f"连不上 {base_url or 'OpenAI 官方端点'}") from exc
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"拉取失败：{exc}") from exc
    finally:
        await client.close()

    # 去重后按 id 排序。context 缺失就是 null，前端只把它当提示，不强制
    seen: dict[str, int | None] = {}
    for m in page.data:
        seen.setdefault(m.id, _model_context_length(m))
    return {"models": [{"id": mid, "context": seen[mid]} for mid in sorted(seen)]}


@app.get("/api/skills")
async def api_skills() -> list[dict]:
    """已加载的 skill 清单，给前端 / 命令面板用。load_skills 每次现扫目录，
    用户改完 skills/ 目录刷新即生效，无需重启。"""
    from app.skills.loader import load_skills

    if not settings.skills_enabled:
        return []
    return [
        {"name": s.name, "description": s.description, "triggers": list(s.triggers)}
        for s in load_skills(settings.skills_dir).values()
    ]


# T6 的前端目录；不存在时跳过挂载（挂到 / 会吞掉未匹配的 API 路径，所以放最后）
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")