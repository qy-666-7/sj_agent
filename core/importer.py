"""
importer.py — 数据导入引擎
DataFrame → MySQL 表：列匹配 / CREATE TABLE / 分块 INSERT
"""
import ast
import logging
import re

import numpy as np
import pandas as pd

from langchain_community.utilities import SQLDatabase

from core.config import CHUNKSIZE
from core.context import DataFrameContext
from core.db import refresh_tables
from core.file_io import read_csv_full

_log = logging.getLogger(__name__)

# ── 类型映射 ─────────────────────────────────────────────


def _dtype_to_mysql(series: pd.Series) -> str:
    """pandas dtype → MySQL 列类型"""
    dtype = series.dtype

    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE"
    if pd.api.types.is_bool_dtype(dtype):
        return "TINYINT(1)"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "DATETIME"

    # 字符串：估算最大长度
    str_series = series.astype(str)
    max_len = int(str_series.str.len().max()) if len(series) > 0 else 0
    if pd.isna(max_len) or max_len <= 0:
        max_len = 255
    max_len = min(max_len * 2, 65535)
    if max_len <= 255:
        return f"VARCHAR({max(max_len, 1)})"
    return "TEXT"


def _quote(name: str) -> str:
    """用反引号包裹 MySQL 标识符"""
    return f"`{name}`"


def _safe_value(val) -> str:
    """将 Python 值转为 MySQL 安全的字面量"""
    if val is None:
        return "NULL"
    # pd.NaT / np.nan / pd.NA → NULL
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return "NULL"
    elif pd.isna(val):
        return "NULL"
    # bool 必须在 int 之前检查（Python 中 bool 是 int 的子类）
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, (int, np.integer)):
        return str(val)
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return "NULL"
        return repr(val)
    if isinstance(val, pd.Timestamp):
        return f"'{val.strftime('%Y-%m-%d %H:%M:%S')}'"
    # 字符串：转义单引号
    s = str(val)
    return f"'{s.replace(chr(39), chr(39)+chr(39))}'"


# ── 表名解析 ─────────────────────────────────────────────


def _parse_table_name(table_name: str) -> tuple[str, str]:
    """解析 库名.表名 → (库名, 表名)；纯表名 → ('', 表名)"""
    if "." in table_name:
        parts = table_name.split(".", 1)
        return parts[0].strip(), parts[1].strip()
    return "", table_name.strip()


def _fq_table(table_name: str) -> str:
    """返回完全限定表名：库名.表名 或 纯表名"""
    db_name, bare = _parse_table_name(table_name)
    if db_name:
        return f"`{db_name}`.`{bare}`"
    return f"`{bare}`"


# ── 表信息查询 ───────────────────────────────────────────


def _get_table_columns(db: SQLDatabase, table_name: str,
                       db_name: str = "") -> list[str] | None:
    """查询表的列名列表；表不存在返回 None"""
    schema = f"'{db_name}'" if db_name else "DATABASE()"
    _, bare = _parse_table_name(table_name)
    try:
        raw = db.run(
            "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
            f"WHERE TABLE_SCHEMA = {schema} AND TABLE_NAME = '{bare}' "
            "ORDER BY ORDINAL_POSITION"
        )
        rows = ast.literal_eval(raw)
        return [r[0] for r in rows]
    except Exception:
        return None


# ── 列比较 ───────────────────────────────────────────────


def _compare_columns(table_cols: list[str], df_cols: list[str]) -> dict:
    """比较表列与 DataFrame 列，返回匹配/多余/缺失"""
    table_set = set(table_cols)
    df_set = set(df_cols)
    return {
        "matched": sorted(table_set & df_set),
        "only_in_table": sorted(table_set - df_set),
        "only_in_df": sorted(df_set - table_set),
    }


# ── CREATE TABLE ──────────────────────────────────────────


def _build_create_table_sql(table_name: str, df: pd.DataFrame) -> str:
    """根据 DataFrame 生成 CREATE TABLE 语句（支持 库名.表名）"""
    col_defs = []
    for col in df.columns:
        mysql_type = _dtype_to_mysql(df[col])
        col_defs.append(f"  {_quote(col)} {mysql_type}")
    cols_sql = ",\n".join(col_defs)
    return f"CREATE TABLE {_fq_table(table_name)} (\n{cols_sql}\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"


# ── INSERT ────────────────────────────────────────────────


def _insert_chunked(db: SQLDatabase, table_name: str, df: pd.DataFrame) -> int:
    """分块 INSERT DataFrame 到表（支持 库名.表名），返回写入行数"""
    cols = list(df.columns)
    quoted_cols = [_quote(c) for c in cols]
    cols_clause = ", ".join(quoted_cols)

    total_inserted = 0
    for start in range(0, len(df), CHUNKSIZE):
        chunk = df.iloc[start:start + CHUNKSIZE]
        rows_sql_parts = []
        for _, row in chunk.iterrows():
            vals = ", ".join(_safe_value(row[c]) for c in cols)
            rows_sql_parts.append(f"  ({vals})")

        sql = (
            f"INSERT INTO {_fq_table(table_name)} ({cols_clause}) VALUES\n"
            + ",\n".join(rows_sql_parts)
        )
        db.run(sql)
        total_inserted += len(chunk)

    return total_inserted


# ── 主入口 ────────────────────────────────────────────────


def import_to_db(
    ctx: DataFrameContext,
    db: SQLDatabase,
    table_name: str,
    create_table: bool = False,
) -> str:
    """
    将当前 loaded_df（或大文件路径）导入 MySQL 表。

    Args:
        ctx: DataFrameContext
        db: SQLDatabase 连接
        table_name: 目标表名
        create_table: 表不存在时是否自动建表

    Returns: 结果摘要文本
    """
    # ── 1. 获取 DataFrame ──
    if ctx.loaded_df is not None:
        df = ctx.loaded_df
        source = ctx.loaded_filename or "内存数据"
    elif ctx.source_file_path:
        # 采样模式 → 触发延迟全量加载（与 _resolve_df 逻辑一致）
        suffix = ctx.source_file_suffix
        _log.info("导入前触发全量加载: %s", ctx.source_file_path)
        try:
            if suffix == ".csv":
                from core.file_io import read_csv_full
                df_full, _enc = read_csv_full(ctx.source_file_path)
            elif suffix in (".xlsx", ".xls"):
                import pandas as pd
                df_full = pd.read_excel(ctx.source_file_path)
                df_full.columns = [str(c) for c in df_full.columns]
            else:
                return f"错误: 不支持的采样文件格式 {suffix}"
            ctx.loaded_df = df_full
            ctx.source_file_path = ""
            ctx.source_file_suffix = ""
            df = ctx.loaded_df
            source = ctx.loaded_filename or "采样文件"
        except Exception as e:
            return f"错误: 全量加载失败 — {e}"
    elif ctx.large_file_path:
        try:
            df, _ = read_csv_full(ctx.large_file_path)
        except Exception as e:
            return f"错误: 读取大文件失败 — {e}"
        source = ctx.loaded_filename or "大文件"
    else:
        return "错误: 没有已加载的数据。请先用 read_data_file 读取一个文件。"

    if len(df) == 0:
        return "错误: 数据为空，无法导入。"

    db_name, bare_name = _parse_table_name(table_name)
    display_name = table_name
    _log.info("导入开始: %s → %s (%d 行)", source, display_name, len(df))

    # ── 2. 目标表是否存在 ──
    table_cols = _get_table_columns(db, table_name, db_name)

    if table_cols is None:
        # ── 表不存在 ──
        if not create_table:
            return (
                f"⚠️ 目标表 `{display_name}` 不存在。\n\n"
                f"如要自动建表，请使用 create_table=True 参数重新调用。\n"
                f"DataFrame 包含 {len(df.columns)} 列: "
                f"{', '.join(df.columns[:10])}"
                f"{'...' if len(df.columns) > 10 else ''}"
            )

        # 自动建表（已支持 库名.表名）
        create_sql = _build_create_table_sql(table_name, df)
        try:
            db.run(create_sql)
            _log.info("自动建表: %s", display_name)
            refresh_tables(db)          # 强制刷新缓存
        except Exception as e:
            return f"错误: 建表失败 — {e}\n\nSQL:\n{create_sql}"

        # 插入数据
        try:
            inserted = _insert_chunked(db, table_name, df)
        except Exception as e:
            return (
                f"⚠️ 表已创建，但数据插入失败 — {e}\n\n"
                f"表结构:\n{create_sql}"
            )

        return (
            f"✅ 导入成功\n"
            f"📋 新建表: `{display_name}`\n"
            f"📐 列数: {len(df.columns)}\n"
            f"📏 导入行数: {inserted:,}\n"
            f"📄 来源: {source}"
        )

    # ── 3. 表存在 — 比较列 ──
    df_cols = list(df.columns)
    comparison = _compare_columns(table_cols, df_cols)

    lines = [
        f"🔍 列比较: 表 `{display_name}` ({len(table_cols)} 列) ↔ DataFrame ({len(df_cols)} 列)",
        f"   匹配: {len(comparison['matched'])} 列",
    ]

    if comparison["only_in_table"]:
        lines.append(
            f"   ⚠️ 仅表中存在: {', '.join(comparison['only_in_table'])}"
            f"（导入时这些列将使用默认值或 NULL）"
        )

    if comparison["only_in_df"]:
        lines.append(
            f"   ❌ 仅 DataFrame 中存在: {', '.join(comparison['only_in_df'])}"
        )
        lines.append(
            "\n⚠️ 导入中止: DataFrame 中包含表中不存在的列。"
            "\n请先用 clean_data 删除多余列，或先在数据库中 ALTER TABLE 添加这些列。"
        )
        return "\n".join(lines)

    # ── 4. 列匹配 → 导入 ──
    if not comparison["matched"]:
        return "错误: 没有匹配的列，无法导入。"

    # 只导入匹配的列，DataFrame 列顺序对齐到表结构
    insert_df = df[comparison["matched"]].copy()
    # 填充缺失列为 NULL（仅表中存在但 DataFrame 中不存在的列，但我们已经
    # 检查了 only_in_df，且只选 matched 列，所以不存在这种情况）

    try:
        inserted = _insert_chunked(db, table_name, insert_df)
    except Exception as e:
        return f"错误: 数据插入失败 — {e}"

    _log.info("导入完成: %d 行 → `%s`", inserted, display_name)

    lines.append(f"\n✅ 导入成功")
    lines.append(f"📋 目标表: `{display_name}`")
    lines.append(f"📐 匹配列: {', '.join(comparison['matched'][:8])}")
    if len(comparison["matched"]) > 8:
        lines.append(f"    ... 共 {len(comparison['matched'])} 列")
    lines.append(f"📏 导入行数: {inserted:,}")
    lines.append(f"📄 来源: {source}")

    return "\n".join(lines)
