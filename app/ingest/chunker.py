import re

from app.config import settings


def chunk_text(
    text: str, size: int | None = None, overlap: int | None = None
) -> list[str]:
    """按段落优先切成 ≤ size 的块，相邻块共享 overlap 个字符。"""
    size = settings.chunk_size if size is None else size
    overlap = settings.chunk_overlap if overlap is None else overlap
    if not text or not text.strip():
        return []

    size = max(1, size)
    overlap = max(0, min(overlap, size - 1))
    normalized = re.sub(r"[ \t\r\f\v\xa0]+", " ", text).strip()

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind("\n\n", start + size // 2, end)
            if boundary > start:
                end = boundary
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks