from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """一路检索的命中结果。score 的含义由所在检索路决定：

    向量路是 L2 distance（越小越近），BM25 路是 BM25 分，
    实体路是命中 token 数，融合后是 RRF 分（越大越靠前）。
    """

    chunk_id: int
    doc_id: int
    content: str
    title: str | None
    score: float