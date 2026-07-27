"""
agent_factory.py — Agent 工厂 + LLM Provider 抽象层
支持 DeepSeek / OpenAI / Anthropic，通过环境变量配置切换。
"""
import logging
import os

from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit

from core.config import LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_API_BASE
from core.db import create_db
from agent.prompts import build_rich_prefix

_log = logging.getLogger(__name__)

# ── LLM Provider 工厂 ──────────────────────────────────────

_PROVIDER_REGISTRY = {}


def _provider(name: str):
    """装饰器：注册 LLM provider 工厂函数"""
    def decorator(fn):
        _PROVIDER_REGISTRY[name] = fn
        return fn
    return decorator


def _resolve_api_key(provider: str) -> str:
    """按优先级解析 API Key：LLM_API_KEY > 各 provider 专属环境变量"""
    if LLM_API_KEY:
        return LLM_API_KEY
    env_map = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    key = os.getenv(env_map.get(provider, ""), "")
    if not key:
        raise RuntimeError(
            f"LLM_PROVIDER={provider} 但未找到 API Key。"
            f"请设置 LLM_API_KEY 或 {env_map.get(provider, '')} 环境变量。"
        )
    return key


@_provider("deepseek")
def _create_deepseek():
    from langchain_deepseek import ChatDeepSeek
    kwargs = {"model": LLM_MODEL}
    api_key = _resolve_api_key("deepseek")
    kwargs["api_key"] = api_key
    if LLM_API_BASE:
        kwargs["api_base"] = LLM_API_BASE
    return ChatDeepSeek(**kwargs)


@_provider("openai")
def _create_openai():
    from langchain_openai import ChatOpenAI
    kwargs = {"model": LLM_MODEL}
    api_key = _resolve_api_key("openai")
    kwargs["api_key"] = api_key
    if LLM_API_BASE:
        kwargs["base_url"] = LLM_API_BASE
    return ChatOpenAI(**kwargs)


@_provider("anthropic")
def _create_anthropic():
    from langchain_anthropic import ChatAnthropic
    kwargs = {"model": LLM_MODEL}
    api_key = _resolve_api_key("anthropic")
    kwargs["api_key"] = api_key
    if LLM_API_BASE:
        kwargs["base_url"] = LLM_API_BASE
    return ChatAnthropic(**kwargs)


def create_llm():
    """根据 LLM_PROVIDER 环境变量创建对应的 LLM 实例。
    支持: deepseek | openai | anthropic
    """
    factory = _PROVIDER_REGISTRY.get(LLM_PROVIDER)
    if factory is None:
        available = ", ".join(sorted(_PROVIDER_REGISTRY))
        raise RuntimeError(
            f"不支持的 LLM_PROVIDER='{LLM_PROVIDER}'。"
            f"可用: {available}。请检查 .env 中的 LLM_PROVIDER 配置。"
        )
    try:
        return factory()
    except ImportError as e:
        pkg_map = {
            "deepseek": "langchain-deepseek",
            "openai": "langchain-openai",
            "anthropic": "langchain-anthropic",
        }
        pkg = pkg_map.get(LLM_PROVIDER, LLM_PROVIDER)
        raise RuntimeError(
            f"缺少 {LLM_PROVIDER} 的依赖包。请运行: pip install {pkg}"
        ) from e


# ── Agent 工厂 ─────────────────────────────────────────────

def create_agent(db=None, extra_tools=None, skill_name=None):
    """
    创建 SQL Agent。统一入口。

    Args:
        db: SQLDatabase 实例，不传则自动创建
        extra_tools: 额外工具列表
        skill_name: 可选，激活的 skill 名称

    Returns: AgentExecutor
    """
    if db is None:
        db = create_db()

    llm = create_llm()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    prefix = build_rich_prefix(db, skill_name=skill_name)

    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="tool-calling",
        verbose=False,
        max_iterations=int(os.getenv("MAX_ITERATIONS", "100")),
        max_execution_time=int(os.getenv("MAX_EXECUTION_TIME", "120")),
        extra_tools=extra_tools,
        prefix=prefix,
    )

    _log.info("Agent 创建: provider=%s, model=%s, tools=%d, skill=%s",
              LLM_PROVIDER, LLM_MODEL,
              len(extra_tools) if extra_tools else 0,
              skill_name or "-")
    return agent
