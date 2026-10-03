from pathlib import Path
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

    # 检索模式：hybrid（三路 RRF 融合）/ vector / bm25。T10 消融实验的检索对照组
    # （D/E/F/G/H）靠这个开关切单路检索，runtime 的工具执行每次从 settings 读
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = "hybrid"

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

    # 可观测（T9）。总开关：关掉后 LLM / 工具调用都不记 trace（T10 的观测开销对照用）。
    # traces 表本身不动——开关只管写，历史数据仍在
    tracing_enabled: bool = True

    # LLM 定价表：每 1M tokens 的单价（美元），traces 的成本按 provider 选档估算。
    # 数字就是各家公开的按百万 tokens 牌价（gpt-4o-mini、claude-sonnet、deepseek-chat），
    # 直接抄下来，不换算成别的单位，避免与官方报价对不上。
    # provider 由客户端类型与 base_url 判定（见 app/llm/openai_compat.py 的
    # detect_provider），表里没有的（如通义）回落到 openai 档。
    # 只用于看板上的量级归因，不追求与账单一致
    price_openai_input: float = 0.15  # gpt-4o-mini input，美元 / 1M tokens
    price_openai_output: float = 0.60  # gpt-4o-mini output，美元 / 1M tokens
    price_anthropic_input: float = 3.0  # claude-sonnet input，美元 / 1M tokens
    price_anthropic_output: float = 15.0  # claude-sonnet output，美元 / 1M tokens
    price_deepseek_input: float = 0.14  # deepseek-chat input，美元 / 1M tokens
    price_deepseek_output: float = 0.28  # deepseek-chat output，美元 / 1M tokens

    # Skill 系统（T11）。总开关关掉后完全不触发：不扫 skills 目录、不注入正文、
    # 不注册专用工具（「这个部署不带 skill」与将来的消融对照都用它）
    skills_enabled: bool = True
    # skill 根目录，相对路径按当前工作目录解析（与 db_path 同一口径）
    skills_dir: str = "skills"
    # 触发阈值：命中多个 skill 时，赢家的命中数占全部命中的比例低于此值就视为
    # 意图不明，本轮不加载 skill。0.7 让「帮我优化简历」（1/1）触发、
    # 「简历和面试分别要准备什么」（1/2）不触发
    skills_trigger_threshold: float = 0.7


settings = Settings()


# ---- 设置面板（/api/settings）的运行时读写 ----
# 只有这些字段能在界面上改；记忆/上下文/定价等其余配置仍只走 .env 与环境变量
EDITABLE_FIELDS = (
    "llm_provider",
    "openai_base_url",
    "openai_api_key",
    "openai_model",
    "anthropic_api_key",
    "anthropic_model",
    "embed_base_url",
    "embed_api_key",
    "embed_model",
    "embed_dim",
)


def mask_secret(value: Optional[str]) -> str:
    """密钥脱敏回显：只露末 4 位，全量密钥永远不出进程。"""
    if not value:
        return ""
    return f"…{value[-4:]}" if len(value) > 4 else "…"


def env_path() -> Path:
    """.env 的位置：与 pydantic-settings 的读取口径一致（相对 CWD）。单独成函数
    是让测试能把它 monkeypatch 到临时目录，不动真实 .env。"""
    return Path(".env")


def update_env_file(path: Path, updates: dict[str, str]) -> None:
    """就地更新 .env：已存在的键替换原行，新键追加到末尾；注释与无关行原样保留。
    键名按 .env 惯例写大写（pydantic-settings 读取时大小写不敏感）。"""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    pending = dict(updates)
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in line:
            key = line.split("=", 1)[0].strip().upper()
            hit = next((k for k in pending if k.upper() == key), None)
            if hit is not None:
                out.append(f"{key}={pending.pop(hit)}")
                continue
        out.append(line)
    out.extend(f"{k.upper()}={v}" for k, v in pending.items())
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
