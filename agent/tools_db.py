"""
tools_db.py — 数据库相关工具
sql_to_dataframe / import_data_to_db / backup_database / restore_database / copy_to_workspace
"""
import logging
import shutil
from pathlib import Path

import pandas as pd

from langchain.tools import tool as lc_tool

from core import config
from core.context import DataFrameContext

_log = logging.getLogger(__name__)


def create_db_tools(ctx: DataFrameContext, db):
    """创建数据库相关工具。sql_to_dataframe 始终返回；其余仅在 db 可用时返回。"""
    # Note: copy_to_workspace has been moved to tools_core.py (no DB dependency)

    # ── sql_to_dataframe（始终可用，使用 config.DB_URI） ──

    @lc_tool
    def sql_to_dataframe(sql: str) -> str:
        """
        执行 SELECT 查询，结果加载为 DataFrame，后续可查询/清洗/作图。
        【优先用此工具而非 sql_db_query——一次查出全部所需数据再分析】

        Parameters:
        sql: SELECT 语句
        """
        if not sql.strip().upper().startswith("SELECT"):
            # 去掉前置 -- 注释后再检查（Agent 常在 SQL 前加注释说明）
            import re
            stripped = re.sub(r'^\s*--[^\n]*\n', '', sql.strip(), flags=re.MULTILINE).strip()
            if not stripped.upper().startswith("SELECT"):
                return "错误: 仅支持 SELECT。"

        sql_upper = sql.upper()
        if "LIMIT" not in sql_upper:
            sql = sql.rstrip(";") + " LIMIT 10000"
            _log.info("自动添加 LIMIT 10000")

        if ctx.loaded_df is not None:
            ctx.snapshot()

        try:
            import sqlalchemy
            engine = sqlalchemy.create_engine(config.DB_URI)
            # DATE_FORMAT / strftime 的 %Y/%m 会被 SQLAlchemy pyformat 误当参数占位符。
            # %% 是 pyformat 的 % 转义写法，传给 DB 时自动还原为单个 %。
            safe_sql = sql.replace('%', '%%')
            result_df = pd.read_sql(safe_sql, engine)
            engine.dispose()
        except Exception as e:
            return f"错误: SQL 执行失败 — {e}"

        if len(result_df) == 0:
            return "⚠️ 查询结果为空。"

        limit_msg = " (已自动限制 10000 行)" if "LIMIT 10000" in sql.upper() else ""
        ctx.loaded_df = result_df
        ctx.loaded_filename = "SQL查询"
        ctx.large_file_path = ""
        ctx.large_file_encoding = ""
        ctx.large_file_suffix = ""
        ctx.log_action("sql_to_df", sql[:80], 0, len(result_df))

        _log.info("SQL→DF: %d 行", len(result_df))

        return (
            f"✅ SQL 结果已加载{limit_msg}\n"
            f"📏 {len(result_df):,} 行 × {len(result_df.columns)} 列\n"
            f"📋 {', '.join(result_df.columns[:12])}"
            f"{'…' if len(result_df.columns) > 12 else ''}\n"
            f"## 前5行\n" + result_df.head(5).to_string(index=False)
        )

    tools = [sql_to_dataframe]

    if db is None:
        return tools

    # ── 以下工具仅在 db 可用时创建 ────────────────────────

    # ── import_data_to_db ─────────────────────────────────

    @lc_tool
    def import_data_to_db(table_name: str, create_table: bool = False) -> str:
        """
        导入已加载数据到 MySQL。表存在→INSERT(列匹配)，不存在+create_table=True→自动建表。

        Parameters:
        table_name: 目标表名
        create_table: 表不存在时是否自动建表
        """
        from core.importer import import_to_db
        return import_to_db(ctx, db, table_name, create_table=create_table)

    # ── move_table_to_db ───────────────────────────────────

    @lc_tool
    def move_table_to_db(source_table: str, target_table: str,
                         create_target: bool = False, drop_source: bool = False) -> str:
        """
        跨库/同库迁移表数据（纯 SQL，不依赖 DataFrame 加载）。
        用于将默认库中的临时表迁移到目标库，或任意库间数据迁移。

        Parameters:
        source_table: 源表名（支持 库名.表名）
        target_table: 目标表名（支持 库名.表名）
        create_target: 是否先 CREATE TABLE target LIKE source（表不存在时用）
        drop_source: 迁移成功后是否删除源表
        """
        from core.importer import _parse_table_name
        src_db, src_bare = _parse_table_name(source_table)
        tgt_db, tgt_bare = _parse_table_name(target_table)

        def _fq(t: str) -> str:
            """backtick-quoted fully-qualified table name"""
            db_n, bare = _parse_table_name(t)
            if db_n:
                return f"`{db_n}`.`{bare}`"
            return f"`{bare}`"

        try:
            if create_target:
                create_sql = f"CREATE TABLE {_fq(target_table)} LIKE {_fq(source_table)}"
                db.run(create_sql)
                from core.db import refresh_tables
                refresh_tables(db)

            insert_sql = f"INSERT INTO {_fq(target_table)} SELECT * FROM {_fq(source_table)}"
            db.run(insert_sql)

            # 验证行数
            count_src = db.run(f"SELECT COUNT(*) FROM {_fq(source_table)}")
            count_tgt = db.run(f"SELECT COUNT(*) FROM {_fq(target_table)}")
            try:
                import ast
                n_src = int(ast.literal_eval(count_src)[0][0])
                n_tgt = int(ast.literal_eval(count_tgt)[0][0])
            except Exception:
                n_src = n_tgt = "?"

            if drop_source:
                db.run(f"DROP TABLE {_fq(source_table)}")
                return (
                    f"✅ 迁移完成（源表已删除）\n"
                    f"📋 {source_table} → {target_table}\n"
                    f"📏 迁移行数: {n_tgt}"
                )

            return (
                f"✅ 迁移完成\n"
                f"📋 {source_table} → {target_table}\n"
                f"📏 源表行数: {n_src}\n"
                f"📏 目标表行数: {n_tgt}"
            )
        except Exception as e:
            return f"错误: 迁移失败 — {e}"

    # ── backup_database ───────────────────────────────────

    @lc_tool
    def backup_database(table_name: str = "") -> str:
        """
        备份 MySQL 表/全库到 backups/ 目录（.sql 文件）。
        执行 DELETE/UPDATE/DROP 前强烈建议先备份！

        Parameters:
        table_name: 表名，空字符串 = 全库备份
        """
        from core.backup import backup_table, backup_all, list_backups
        try:
            if table_name.strip():
                path = backup_table(table_name.strip())
                size_kb = round(Path(path).stat().st_size / 1024, 1)
                return (
                    f"✅ 备份成功\n"
                    f"📋 表: {table_name.strip()}\n"
                    f"📄 文件: {path}\n"
                    f"📦 大小: {size_kb:.1f} KB"
                )
            else:
                path = backup_all()
                size_mb = round(Path(path).stat().st_size / (1024 * 1024), 1)
                existing = list_backups()
                return (
                    f"✅ 全库备份成功\n"
                    f"📄 文件: {path}\n"
                    f"📦 大小: {size_mb:.1f} MB\n"
                    f"📚 现有备份: {len(existing)} 个文件"
                )
        except FileNotFoundError as e:
            return f"错误: {e}"
        except Exception as e:
            return f"错误: 备份失败 — {e}"

    # ── restore_database ──────────────────────────────────

    @lc_tool
    def restore_database(sql_file: str) -> str:
        """
        从 backups/ 目录中的 .sql 备份文件恢复数据。
        ⚠️ 会覆盖当前表内容，使用前务必让用户确认！

        Parameters:
        sql_file: 备份文件名或完整路径，如 "users_20260717_200530.sql"
        """
        from core.backup import restore_backup, BACKUP_DIR
        path = Path(sql_file)
        if not path.is_absolute():
            path = BACKUP_DIR / sql_file
        return restore_backup(db, str(path))

    tools.extend([
        import_data_to_db, move_table_to_db, backup_database, restore_database,
    ])
    return tools
