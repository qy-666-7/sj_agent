"""
cleaning.py — 数据清洗与查询引擎
分块聚合 / 全量加载查询 / 大文件清洗 / 结果截断
"""
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import CHUNKSIZE, MAX_OUTPUT_ROWS
from core.file_io import get_file_size_mb, read_csv_full

_log = logging.getLogger(__name__)

_EVAL_NS = {"df": None, "pd": pd, "np": np, "str": str, "int": int, "float": float, "len": len, "bool": bool, "__builtins__": {}}


def _safe_eval(expression: str, df: pd.DataFrame):
    """安全 eval：仅允许 df/pd/np/len/bool，禁止 builtins。"""
    ns = dict(_EVAL_NS, df=df)
    stripped = expression.strip()

    # 单行无赋值 → eval
    if "\n" not in stripped and ";" not in stripped:
        try:
            return eval(stripped, ns)
        except SyntaxError:
            pass  # 可能是赋值语句，走 exec

    # 多行/多语句：先尝试整体 eval（跨行表达式），失败则整体 exec（多语句）
    try:
        return eval(stripped, ns)
    except SyntaxError:
        exec(stripped, ns)
    return ns["df"]


# ── 结果截断 ─────────────────────────────────────────────

def _truncate(result) -> str:
    """将 DataFrame/Series 截断到 MAX_OUTPUT_ROWS，超出部分标注"""
    if isinstance(result, pd.DataFrame):
        total = len(result)
        if total > MAX_OUTPUT_ROWS:
            truncated = result.head(MAX_OUTPUT_ROWS)
            return (
                f"结果 (共 {total:,} 行，仅展示前 {MAX_OUTPUT_ROWS}):\n"
                + truncated.to_string(index=False)
                + f"\n… 省略 {total - MAX_OUTPUT_ROWS:,} 行"
            )
        return f"结果 ({total:,} 行):\n" + result.to_string(index=False)
    if isinstance(result, pd.Series):
        total = len(result)
        if total > MAX_OUTPUT_ROWS:
            truncated = result.head(MAX_OUTPUT_ROWS)
            return (
                f"结果 (共 {total:,} 行，仅展示前 {MAX_OUTPUT_ROWS}):\n"
                + truncated.to_string()
                + f"\n… 省略 {total - MAX_OUTPUT_ROWS:,} 行"
            )
        return f"结果:\n" + result.to_string()
    return f"结果: {result}"


# ── 友好错误 ──────────────────────────────────────────────

def _friendly_error(e: Exception, df: pd.DataFrame) -> str:
    """将 pandas 异常转为用户可理解的提示"""
    msg = str(e)
    # KeyError: 列名不存在
    if isinstance(e, KeyError) or "not in index" in msg.lower():
        col_name = str(e).strip("'\"[] ")
        available = ", ".join(df.columns[:15])
        return (
            f"错误: 列名不存在 — {msg}\n"
            f"可用列 ({len(df.columns)} 个): {available}"
            f"{'…' if len(df.columns) > 15 else ''}\n"
            f"提示: 请检查列名拼写，或先用 read_data_file 查看字段列表"
        )
    # AttributeError: 方法不存在
    if isinstance(e, AttributeError):
        return (
            f"错误: 方法或属性不存在 — {msg}\n"
            f"提示: 可用方法如 .sum(), .mean(), .groupby(), .dropna(), "
            f".drop_duplicates(), .fillna(), .rename(), .astype()"
        )
    # SyntaxError / 表达式格式错误
    if isinstance(e, SyntaxError):
        return (
            f"错误: 表达式格式错误 — {msg}\n"
            f"提示: 表达式应如 \"df['列名'].sum()\" 或 \"df.groupby('列名')['列名'].sum()\""
        )
    # TypeError
    if isinstance(e, TypeError):
        return (
            f"错误: 类型不匹配 — {msg}\n"
            f"提示: 检查是否对文本列使用了数值运算，或参数类型错误"
        )
    return f"错误: 表达式执行失败 — {msg}"


# ── 聚合判断 ─────────────────────────────────────────────

_AGG_KEYWORDS = [
    ".sum(", ".mean(", ".min(", ".max(", ".count(", ".nunique(",
    ".sum()", ".mean()", ".min()", ".max()", ".count()", ".nunique()",
]


def is_aggregation(expression: str) -> bool:
    if any(kw in expression for kw in _AGG_KEYWORDS):
        return True
    if "groupby" in expression.lower():
        return True
    if "value_counts" in expression:
        return True
    return False


# ── 分块聚合 ─────────────────────────────────────────────

def eval_chunked(expression: str, file_path: str, encoding: str) -> str:
    chunk_results: list[tuple[int, object]] = []
    total_rows = 0

    _log.info("分块查询开始: %s", expression[:80])

    try:
        reader = pd.read_csv(file_path, encoding=encoding, chunksize=CHUNKSIZE)
    except Exception as e:
        return f"错误: 无法读取文件 — {e}"

    first_chunk_cols = None
    for chunk in reader:
        chunk.columns = [str(c).lstrip("﻿") for c in chunk.columns]
        if first_chunk_cols is None:
            first_chunk_cols = list(chunk.columns)
        total_rows += len(chunk)
        try:
            result = _safe_eval(expression, chunk)
            chunk_results.append((len(chunk), result))
        except Exception as e:
            return _friendly_error(e, chunk)

    if not chunk_results:
        return "错误: 文件中无数据"

    chunk_sizes = [r[0] for r in chunk_results]
    results = [r[1] for r in chunk_results]
    first = results[0]

    if all(isinstance(r, (int, float, np.integer, np.floating, np.bool_))
           for r in results):
        return _merge_scalar(expression, results, chunk_sizes, total_rows,
                             file_path, encoding)

    if all(isinstance(r, pd.Series) for r in results):
        combined = pd.concat(results).groupby(level=0).sum()
        if ".mean(" in expression:
            return eval_full_reload(expression, file_path, encoding)
        return _truncate(combined)

    if all(isinstance(r, pd.DataFrame) for r in results):
        combined = pd.concat(results, ignore_index=True)
        return _truncate(combined)

    return f"结果: {first}"


def _merge_scalar(expression: str, results: list, chunk_sizes: list[int],
                  total_rows: int, file_path: str, encoding: str) -> str:
    if ".sum(" in expression or ".count(" in expression:
        combined = sum(results)
    elif ".nunique(" in expression:
        return eval_full_reload(expression, file_path, encoding)
    elif ".mean(" in expression:
        total_weighted = sum(r * sz for r, sz in zip(results, chunk_sizes))
        combined = total_weighted / total_rows if total_rows > 0 else 0
    elif ".min(" in expression:
        combined = min(results)
    elif ".max(" in expression:
        combined = max(results)
    else:
        combined = sum(results)
    return f"结果: {combined}"


# ── 全量加载查询 ─────────────────────────────────────────

def eval_full_reload(expression: str, file_path: str, encoding: str) -> str:
    _log.info("全量查询: %s", expression[:80])
    try:
        df, _ = read_csv_full(file_path)
    except Exception as e:
        return f"错误: 读取文件失败 — {e}"

    file_mb = get_file_size_mb(file_path)
    try:
        result = _safe_eval(expression, df)
    except Exception as e:
        return _friendly_error(e, df)
    finally:
        del df

    prefix = f"⚠️ 全量加载 {file_mb:.0f} MB 文件，建议优先使用 MySQL 查询\n\n"
    return prefix + _truncate(result)


# ── 小文件清洗 ───────────────────────────────────────────

def clean_in_memory(df: pd.DataFrame, expression: str) -> tuple[pd.DataFrame, str]:
    # 拒绝无意义的 df.copy() 调用
    stripped = expression.strip()
    if stripped in ("df.copy()", "df"):
        raise ValueError(
            "表达式不会改变数据，浪费一轮操作。请直接进行有实际效果的清洗，"
            "如 df.ffill() / df.dropna() / df.assign(...) 等。"
        )

    before_rows = len(df)
    before_cols = len(df.columns)
    before_nulls = int(df.isna().sum().sum())

    try:
        result = _safe_eval(expression, df)
    except Exception as e:
        raise ValueError(_friendly_error(e, df))

    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            f"清洗表达式必须返回 DataFrame，当前返回了 {type(result).__name__}。\n"
            f"提示: 试试 df.dropna() 而非 df.dropna(inplace=True)"
        )

    after_rows = len(result)
    after_nulls = int(result.isna().sum().sum())

    summary_lines = [
        f"✅ 清洗完成",
        f"📏 行数: {before_rows:,} → {after_rows:,} "
        f"({'−' if before_rows >= after_rows else '+'}{abs(before_rows - after_rows):,})",
        f"📐 列数: {before_cols} → {len(result.columns)}"
        f"{' (列已变更)' if before_cols != len(result.columns) else ''}",
        f"🗑️ 空值: {before_nulls:,} → {after_nulls:,} "
        f"({'−' if before_nulls >= after_nulls else '+'}{abs(before_nulls - after_nulls):,})",
        "",
        "## 清洗后前5行",
        result.head(5).to_string(index=False),
    ]

    _log.info("清洗完成: %d→%d 行", before_rows, after_rows)
    return result, "\n".join(summary_lines)


# ── 大文件清洗 ───────────────────────────────────────────

def squash_large(file_path: str, encoding: str, expression: str,
                 temp_files: list[str]) -> tuple[str, str]:
    if not Path(file_path).exists():
        return f"错误: 文件不存在 — {file_path}", ""

    file_mb = get_file_size_mb(file_path)
    _log.info("大文件清洗开始: %s (%.0f MB)", expression[:80], file_mb)

    try:
        df, _ = read_csv_full(file_path)
    except Exception as e:
        return f"错误: 读取文件失败 — {e}", ""

    before_rows = len(df)
    before_cols = len(df.columns)
    before_nulls = int(df.isna().sum().sum())

    try:
        result = _safe_eval(expression, df)
    except Exception as e:
        err_msg = _friendly_error(e, df)
        del df
        return err_msg, ""

    if not isinstance(result, pd.DataFrame):
        err_msg = (
            f"错误: 清洗表达式必须返回 DataFrame，当前返回了 {type(result).__name__}。\n"
            f"示例: df.dropna() 或 df[df['price'] > 0]"
        )
        del df
        return err_msg, ""

    cleaned = result
    del df

    tmp = tempfile.NamedTemporaryFile(
        suffix=".csv", prefix="cleaned_", delete=False,
    )
    new_path = tmp.name
    temp_files.append(new_path)
    cleaned.to_csv(new_path, index=False, encoding="utf-8")

    after_rows = len(cleaned)
    after_cols = len(cleaned.columns)
    after_nulls = int(cleaned.isna().sum().sum())
    del cleaned

    new_size_mb = get_file_size_mb(new_path)

    summary = "\n".join([
        "✅ 清洗完成（大文件模式）",
        f"📏 行数: {before_rows:,} → {after_rows:,} "
        f"({'−' if before_rows >= after_rows else '+'}{abs(before_rows - after_rows):,})",
        f"📐 列数: {before_cols} → {after_cols}"
        f"{' (列已变更)' if before_cols != after_cols else ''}",
        f"🗑️ 空值: {before_nulls:,} → {after_nulls:,} "
        f"({'−' if before_nulls >= after_nulls else '+'}{abs(before_nulls - after_nulls):,})",
        f"📦 清洗后文件大小: {new_size_mb:.1f} MB",
        f"⚠️ 全量加载 {file_mb:.0f} MB 文件完成清洗，已缓存为临时 CSV",
    ])

    _log.info("大文件清洗完成: %d→%d 行, 新文件=%.1f MB",
              before_rows, after_rows, new_size_mb)
    return summary, new_path
