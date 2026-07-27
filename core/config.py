"""
config.py — 统一配置 & 环境变量校验
导入安全：仅解析环境变量常量，不创建目录、不初始化日志。
调用 init_config() 显式初始化运行时资源。
"""
import logging
import logging.handlers
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── 环境变量 ──────────────────────────────────────────────

load_dotenv(override=True)

# ── 数据库 ────────────────────────────────────────────────
DB_URI = os.getenv("DB_URI")
if not DB_URI:
    raise RuntimeError("环境变量 DB_URI 未设置，请在 .env 文件或系统环境变量中配置")

# 从 URI 提取数据库名: mysql+pymysql://user:pass@host:port/dbname → dbname
_DB_NAME_MATCH = __import__("re").search(r"/([^/?]+)(?:\?|$)", DB_URI.rstrip("/"))
DB_NAME = _DB_NAME_MATCH.group(1) if _DB_NAME_MATCH else "database"

# ── LLM ───────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")  # deepseek | openai | anthropic
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_BASE = os.getenv("LLM_API_BASE", "")

# ── 文件处理 ──────────────────────────────────────────────
LARGE_FILE_MB = 100
CHUNKSIZE = 50_000
MAX_OUTPUT_ROWS = 50
_CSV_ENCODINGS = [
    "utf-8", "gbk", "gb18030", "gb2312", "utf-16", "utf-16-le", "latin-1",
]

# ── API 服务 ──────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", "30"))

# ── 路径（仅作为 Path 对象，不创建目录） ──────────────────
_PROJ_ROOT = Path(__file__).resolve().parent.parent
CHART_DIR = Path(os.getenv("CHART_DIR", str(_PROJ_ROOT / "charts")))
BACKUP_DIR = Path(os.getenv("BACKUP_DIR", str(_PROJ_ROOT / "backups")))
RAW_DIR = Path(os.getenv("RAW_DIR", str(_PROJ_ROOT / "raw")))
WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", str(_PROJ_ROOT / "workspace")))
SKILLS_DIR = Path(os.getenv("SKILLS_DIR", str(_PROJ_ROOT / "skills")))
LOG_DIR = Path(os.getenv("LOG_DIR", str(_PROJ_ROOT / "logs")))

# ── 图表 ──────────────────────────────────────────────────
CHART_TTL_DAYS = int(os.getenv("CHART_TTL_DAYS", "7"))

# ── 备份 ──────────────────────────────────────────────────
MAX_BACKUPS_PER_TABLE = int(os.getenv("MAX_BACKUPS_PER_TABLE", "10"))

# ── Redis 会话持久化（可选） ─────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "")

# ── 日志级别 ──────────────────────────────────────────────
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ── 初始化 ────────────────────────────────────────────────

_initialized = False


def init_config():
    """显式初始化：创建运行时目录 + 配置日志。幂等，多次调用无副作用。"""
    global _initialized
    if _initialized:
        return

    # 创建目录
    for d in (CHART_DIR, BACKUP_DIR, RAW_DIR, WORKSPACE_DIR, SKILLS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 配置日志
    _setup_logging()

    _initialized = True


def _setup_logging():
    """配置控制台 + 文件日志（按天轮转，保留 7 天，UTF-8 编码）"""
    _log = logging.getLogger("core")
    _log.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))

    if _log.handlers:
        return

    # 控制台 handler（Windows 强制 utf-8）
    try:
        import io
        utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        _ch = logging.StreamHandler(utf8_stdout)
    except Exception:
        _ch = logging.StreamHandler(sys.stdout)
    _ch.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    _ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S",
    ))
    _log.addHandler(_ch)

    # 文件 handler（按天轮转，保留 7 天，utf-8 编码）
    _fh = logging.handlers.TimedRotatingFileHandler(
        LOG_DIR / "agent.log", when="midnight", interval=1, backupCount=7,
        encoding="utf-8",
    )
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _log.addHandler(_fh)

    _log.info("配置已加载: LLM=%s, LARGE_FILE_MB=%d, CHUNKSIZE=%d", LLM_MODEL, LARGE_FILE_MB, CHUNKSIZE)
