"""
backup.py — MySQL 备份 / 恢复引擎
通过 subprocess 调用 mysqldump / mysql，从 DB_URI 自动解析连接参数
"""
import logging
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from langchain_community.utilities import SQLDatabase

from core.config import DB_URI, BACKUP_DIR, MAX_BACKUPS_PER_TABLE

_log = logging.getLogger(__name__)


def _parse_db_uri(uri: str = DB_URI) -> dict:
    """从 SQLAlchemy URI 提取 mysql CLI 所需的连接参数"""
    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = str(parsed.port or 3306)
    user = parsed.username or "root"
    password = parsed.password or ""
    database = parsed.path.lstrip("/")
    return {"host": host, "port": port, "user": user, "password": password, "database": database}


def _find_mysqldump() -> str:
    """查找 mysqldump 可执行文件；找不到则抛异常"""
    # 常见路径
    candidates = [
        "mysqldump",
        r"C:\Program Files\MySQL\MySQL Server 8.0\bin\mysqldump.exe",
        r"C:\Program Files\MySQL\MySQL Server 8.4\bin\mysqldump.exe",
        r"C:\Program Files\MySQL\MySQL Server 9.0\bin\mysqldump.exe",
        "/usr/bin/mysqldump",
        "/usr/local/bin/mysqldump",
    ]
    for p in candidates:
        try:
            subprocess.run([p, "--version"], capture_output=True, timeout=5)
            return p
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    raise FileNotFoundError("找不到 mysqldump。请确认 MySQL 已安装并将其 bin 目录加入 PATH。")


def _find_mysql_cli() -> str:
    """查找 mysql CLI；找不到则退回到 Python sqlalchemy 方式恢复"""
    candidates = [
        "mysql",
        r"C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe",
        r"C:\Program Files\MySQL\MySQL Server 8.4\bin\mysql.exe",
        "/usr/bin/mysql",
        "/usr/local/bin/mysql",
    ]
    for p in candidates:
        try:
            subprocess.run([p, "--version"], capture_output=True, timeout=5)
            return p
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return ""  # 回退到 Python 方式


def _rotating_cleanup(table_name: str):
    """保留最近 MAX_BACKUPS_PER_TABLE 个备份，删除旧文件"""
    if MAX_BACKUPS_PER_TABLE <= 0:
        return
    # 剥离跨库前缀，与 backup_table 的文件命名保持一致
    bare = table_name.split(".", 1)[-1] if "." in table_name else table_name
    prefix = f"{bare}_" if bare else "full_"
    files = sorted(
        BACKUP_DIR.glob(f"{prefix}*.sql"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in files[MAX_BACKUPS_PER_TABLE:]:
        old.unlink(missing_ok=True)
        _log.debug("清理旧备份: %s", old.name)


def backup_table(table_name: str, output_path: str = "") -> str:
    """
    备份单张表到 .sql 文件。

    Args:
        table_name: 表名（如 "users" 或 "lx.users"）
        output_path: 可选输出路径；不传则自动生成 backups/{table}_{timestamp}.sql

    Returns:
        生成的 .sql 文件路径

    Raises:
        FileNotFoundError: mysqldump 未找到
        subprocess.CalledProcessError: 备份执行失败
    """
    conn = _parse_db_uri()
    mysqldump = _find_mysqldump()

    # 解析 库名.表名 格式（支持跨库备份）
    db_name = conn["database"]
    bare_table = table_name
    if "." in table_name:
        parts = table_name.split(".", 1)
        db_name = parts[0].strip()
        bare_table = parts[1].strip()

    if not output_path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = str(BACKUP_DIR / f"{bare_table}_{ts}.sql")

    cmd = [
        mysqldump,
        f"--host={conn['host']}",
        f"--port={conn['port']}",
        f"--user={conn['user']}",
        f"--password={conn['password']}",
        "--single-transaction",
        "--routines",
        "--triggers",
        "--add-drop-table",
        db_name,
        bare_table,
    ]

    _log.info("备份开始: %s → %s", table_name, output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, check=True, timeout=120)

    size_kb = Path(output_path).stat().st_size / 1024
    _log.info("备份完成: %s, %.1f KB", table_name, size_kb)

    # 轮转清理旧备份
    _rotating_cleanup(table_name)

    return output_path


def backup_all(output_path: str = "") -> str:
    """
    全库备份。

    Args:
        output_path: 可选输出路径；不传则自动生成 backups/full_{timestamp}.sql

    Returns:
        生成的 .sql 文件路径
    """
    conn = _parse_db_uri()
    mysqldump = _find_mysqldump()

    if not output_path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = str(BACKUP_DIR / f"full_{ts}.sql")

    cmd = [
        mysqldump,
        f"--host={conn['host']}",
        f"--port={conn['port']}",
        f"--user={conn['user']}",
        f"--password={conn['password']}",
        "--single-transaction",
        "--routines",
        "--triggers",
        "--add-drop-table",
        conn["database"],
    ]

    _log.info("全库备份开始 → %s", output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, check=True, timeout=300)

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    _log.info("全库备份完成, %.1f MB", size_mb)

    _rotating_cleanup("")

    return output_path


def restore_backup(db: SQLDatabase, sql_path: str) -> str:
    """
    从 .sql 文件恢复数据到 MySQL。

    优先使用 mysql CLI；不可用时回退到 db.run() 逐行执行。

    Args:
        db: SQLDatabase 连接（用于回退方式）
        sql_path: .sql 备份文件路径

    Returns:
        结果描述
    """
    path = Path(sql_path)
    if not path.exists():
        return f"错误: 备份文件不存在 — {sql_path}"

    size_mb = path.stat().st_size / (1024 * 1024)
    _log.info("恢复开始: %s, %.1f MB", sql_path, size_mb)

    # 尝试 mysql CLI
    mysql_cli = _find_mysql_cli()
    if mysql_cli:
        conn = _parse_db_uri()
        cmd = [
            mysql_cli,
            f"--host={conn['host']}",
            f"--port={conn['port']}",
            f"--user={conn['user']}",
            f"--password={conn['password']}",
            conn["database"],
        ]
        try:
            with open(sql_path, "r", encoding="utf-8") as f:
                subprocess.run(cmd, stdin=f, capture_output=True, check=True, timeout=300)
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode("utf-8", errors="replace")[:500]
            _log.error("mysql CLI 恢复失败: %s", err)
            return f"错误: 恢复失败 — {err}"

        _log.info("恢复完成: %s", sql_path)
        return f"✅ 恢复成功\n📄 文件: {sql_path}\n📦 大小: {size_mb:.1f} MB"

    # 回退：通过 db.run() 执行每条 SQL
    _log.warning("mysql CLI 不可用，使用 db.run() 逐 SQL 恢复")
    sql_content = path.read_text(encoding="utf-8", errors="replace")
    statements = _split_sql_statements(sql_content)

    executed = 0
    errors = []
    for stmt in statements:
        stmt = stmt.strip()
        if not stmt:
            continue
        try:
            db.run(stmt)
            executed += 1
        except Exception as e:
            errors.append(str(e)[:100])

    if errors:
        _log.warning("部分 SQL 执行失败: %d/%d 条", len(errors), len(statements))
        return (
            f"⚠️ 恢复部分完成\n"
            f"📄 文件: {sql_path}\n"
            f"✅ 成功: {executed}/{len(statements)} 条\n"
            f"❌ 失败: {len(errors)} 条\n"
            f"首个错误: {errors[0] if errors else '无'}"
        )

    _log.info("逐 SQL 恢复完成: %d 条", executed)
    return f"✅ 恢复成功\n📄 文件: {sql_path}\n📦 大小: {size_mb:.1f} MB\n📏 执行: {executed} 条 SQL"


def _split_sql_statements(content: str) -> list[str]:
    """简单拆分 SQL 文件为语句列表（按 ; 分隔，跳过注释行）"""
    lines = []
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("--") or stripped.startswith("/*") or not stripped:
            continue
        lines.append(line)
    return "\n".join(lines).split(";")


def list_backups(table_name: str = "") -> list[dict]:
    """
    列出备份文件。

    Args:
        table_name: 可选过滤表名

    Returns:
        [{"name": ..., "size_kb": ..., "time": ...}, ...]
    """
    pattern = f"{table_name}_*.sql" if table_name else "*.sql"
    files = sorted(
        BACKUP_DIR.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    result = []
    for p in files:
        stat = p.stat()
        result.append({
            "name": p.name,
            "size_kb": round(stat.st_size / 1024, 1),
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        })
    return result
