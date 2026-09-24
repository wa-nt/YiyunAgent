import re

from app.config import settings


def _normalize(text: str) -> str:
    """统一换行与特殊空白，但保留行首缩进（代码笔记的缩进是内容）。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+(?=\n)", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")


def chunk_text(
    text: str, size: int | None = None, overlap: int | None = None
) -> list[str]:
    """按段落优先切成 ≤ size 的块，相邻块共享 overlap 个字符。"""
    size = settings.chunk_size if size is None else size
    overlap = settings.chunk_overlap if overlap is None else overlap
    if not text or not text.strip():
        return []

    size = max(1, size)
    # overlap 超过半个块会让窗口每次只前进 1 字符（块数爆炸），因此钳到 size//2 - 1
    overlap = max(0, min(overlap, max(size // 2 - 1, 0)))
    normalized = _normalize(text)

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind("\n\n", start + size // 2, end)
            if boundary > start:
                end = boundary
        piece = normalized[start:end].strip("\n")
        if piece.strip():
            chunks.append(piece)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks