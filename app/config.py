from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: Literal["openai_compat", "anthropic"] = "openai_compat"

    openai_base_url: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None

    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-sonnet-4-5"
    anthropic_max_tokens: int = 4096

    embed_base_url: Optional[str] = None
    embed_api_key: Optional[str] = None
    embed_model: Optional[str] = None
    embed_dim: int = 1024

    db_path: str = "data/app.db"
    chunk_size: int = 500
    chunk_overlap: int = 80

    memory_enabled: bool = True
    memory_recall_top_k: int = 5
    # 去重阈值：归一化（去称呼前缀/标点/空白）后整句的序列相似度。0.92 是刻意的保守值——
    # 一词之差改变事实的句子（「用户在北京上学」vs「用户在北京上班」）相似度约 0.86，
    # 插入否定词的说法（「喜欢」vs「不喜欢」，0.966）另由 same_claim 的否定词守卫兜住，
    # 两者都必须落在阈值下方才能并存/判冲突，而不是被当成重复丢掉
    memory_dedup_ratio: float = 0.92
    memory_decay: float = 0.9

    # 上下文治理（T8）。三个策略各自独立可开关：T10 的消融实验靠单因素对照
    # （只关一个、其余不动）验证各自贡献，所以不要把它们合并成一个总开关
    context_compaction_enabled: bool = True
    context_tool_clean_enabled: bool = True
    context_token_budget_enabled: bool = True
    # prompt view 的总预算（token，按 1 token ≈ 4 字符估算）
    context_max_tokens: int = 8000
    # 历史超过 N 条时触发压缩；保留最近 N/2 条完整对话，更早的由 LLM 生成摘要
    context_compaction_threshold: int = 10


settings = Settings()