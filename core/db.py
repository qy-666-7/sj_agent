"""
db.py — 数据库连接（自动创建默认库）
"""
import logging
from urllib.parse import urlparse, urlunparse

from langchain_community.utilities import SQLDatabase
import sqlalchemy

from core.config import DB_URI

_log = logging.getLogger(__name__)


def _ensure_database_exists(uri: str) -> None:
    """如果 URI 指定的数据库不存在，自动创建。连接 MySQL 不指定库来执行 CREATE。"""
    parsed = urlparse(uri)
    db_name = parsed.path.lstrip("/")
    if not db_name:
        return

    # 构造一个不带数据库名的 URI（连到服务器本身）
    server_uri = urlunparse(parsed._replace(path=""))
    try:
        engine = sqlalchemy.create_engine(server_uri)
        with engine.connect() as conn:
            conn.exec_driver_sql(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        engine.dispose()
    except Exception as e:
        _log.warning("无法自动创建数据库 '%s': %s", db_name, e)


def create_db() -> SQLDatabase:
    """创建 SQLDatabase 连接（自动创建默认库如果不存在）"""
    _ensure_database_exists(DB_URI)
    _log.info("创建数据库连接: %s", DB_URI[:DB_URI.index("@")] + "@***")
    return SQLDatabase.from_uri(DB_URI)


def refresh_tables(db: SQLDatabase) -> list[str]:
    """重新读取数据库表名，将新表加入 SQLDatabase 内部缓存。

    SQLDatabase 在 __init__ 时通过 information_schema 一次性缓存
    _all_tables / _usable_tables，之后永不刷新。调用本函数可在
    CREATE TABLE 之后强制重新扫描，使新表对新会话可见。

    Returns:
        新发现的表名列表（已排序）
    """
    inspector = db._inspector
    # 清除 SQLAlchemy Inspector 缓存，确保读到最新表结构
    if hasattr(inspector, 'info_cache'):
        inspector.info_cache.clear()
    current = set(inspector.get_table_names())
    new_tables = current - db._all_tables

    if not new_tables:
        return []

    db._all_tables = current
    if db._include_tables:
        db._usable_tables = set(db._include_tables)
    else:
        db._usable_tables = current - db._ignore_tables

    # 为新表反射 SQLAlchemy MetaData（供 get_table_info 使用）
    for t in new_tables:
        try:
            db._metadata.reflect(bind=db._engine, only=[t])
        except Exception:
            _log.warning("无法反射新表元数据: %s", t)

    _log.info("表缓存已刷新，发现 %d 张新表: %s",
              len(new_tables), sorted(new_tables))
    return sorted(new_tables)
