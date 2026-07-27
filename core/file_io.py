"""
file_io.py — 文件 I/O 纯函数
编码检测 / CSV 读取 / 分块扫描 / 摘要格式化
所有函数无副作用，不持有全局状态
"""
import logging
from pathlib import Path
from typing import Any

import chardet
import numpy as np
import pandas as pd

from core.config import CHUNKSIZE, _CSV_ENCODINGS

_log = logging.getLogger(__name__)

# ── 编码检测 ─────────────────────────────────────────────

# 编码别名 → 标准化名
_ENCODING_MAP = {
    "gb2312": "gbk", "gbk": "gbk", "gb18030": "gb18030",
    "utf-8": "utf-8", "utf-16": "utf-16",
    "ascii": "utf-8", "iso-8859-1": "latin-1",
}


def detect_encoding(file_path: str) -> str:
    """用 chardet 检测文件编码，返回规范化的编码名"""
    # ★ 先试 UTF-8（最常见，跳过 chardet 开销）
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            f.read(2000)
        return "utf-8"
    except (UnicodeDecodeError, UnicodeError):
        pass
    with open(file_path, "rb") as f:
        raw = f.read(20_000)  # 100KB→20KB: chardet 够用了
    result = chardet.detect(raw)
    detected = (result.get("encoding") or "").lower()
    return _ENCODING_MAP.get(detected, detected)


def probe_csv_encoding(file_path: str) -> str:
    """只读第一块验证编码（不加载全量），返回第一个可用的编码。"""
    candidates = [detect_encoding(file_path)] + [
        e for e in _CSV_ENCODINGS if e != detect_encoding(file_path)
    ]
    for enc in candidates:
        try:
            next(pd.read_csv(file_path, encoding=enc, chunksize=1))
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            continue
    return "utf-8"  # fallback


# ── CSV 采样读取 ─────────────────────────────────────────

def read_csv_sample(file_path: str, nrows: int = 100) -> tuple[pd.DataFrame, str]:
    """采样读取 CSV 前 N 行，返回 (DataFrame, 编码)。不加载全量。"""
    enc = probe_csv_encoding(file_path)
    df = pd.read_csv(file_path, encoding=enc, nrows=nrows)
    df.columns = [str(c).lstrip("﻿") for c in df.columns]
    _log.debug("CSV 采样: %s, 编码=%s, 采样行数=%d", file_path, enc, len(df))
    return df, enc


# ── CSV 全量读取 ─────────────────────────────────────────

def read_csv_full(file_path: str) -> tuple[pd.DataFrame, str]:
    """智能读取 CSV：先 chardet 探测编码，失败则逐一尝试（全量加载）。"""
    candidates = [detect_encoding(file_path)] + [
        e for e in _CSV_ENCODINGS if e != detect_encoding(file_path)
    ]
    for enc in candidates:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            df.columns = [str(c).lstrip("﻿") for c in df.columns]
            _log.debug("CSV 读取成功: %s, 编码=%s, 行数=%d", file_path, enc, len(df))
            return df, enc
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            continue

    # 保底方案
    _log.warning("所有编码尝试失败，使用 utf-8 + replace 读取: %s", file_path)
    df = pd.read_csv(file_path, encoding="utf-8", encoding_errors="replace")
    df.columns = [str(c).lstrip("﻿") for c in df.columns]
    return df, "utf-8(replace)"


# ── 文件大小 ─────────────────────────────────────────────

def get_file_size_mb(file_path: str) -> float:
    """返回文件大小（MB）"""
    return Path(file_path).stat().st_size / (1024 * 1024)


# ── 分块扫描 ─────────────────────────────────────────────

def read_csv_chunked(file_path: str, encoding: str,
                     chunksize: int = CHUNKSIZE) -> dict[str, Any]:
    """分块读取 CSV，增量计算元数据摘要（含 Welford 方差），不保留 DataFrame。"""
    col_names: list[str] | None = None
    col_dtypes: dict[str, Any] = {}
    total_rows = 0
    null_counts: dict[str, int] = {}
    nunique_sets: dict[str, set] = {}           # 分类列唯一值采样
    numeric_stats: dict[str, dict] = {}
    head_sample = None
    tail_sample = None

    reader = pd.read_csv(file_path, encoding=encoding, chunksize=chunksize)
    for chunk in reader:
        chunk.columns = [str(c).lstrip("﻿") for c in chunk.columns]

        if col_names is None:
            col_names = list(chunk.columns)
            col_dtypes = {c: chunk[c].dtype for c in col_names}
            null_counts = {c: 0 for c in col_names}
            nunique_sets = {
                c: set() for c in col_names
                if not pd.api.types.is_numeric_dtype(chunk[c])
                and not pd.api.types.is_datetime64_any_dtype(chunk[c])
            }
            head_sample = chunk.head(5)

        total_rows += len(chunk)
        tail_sample = chunk.tail(5)

        for c in col_names:
            null_counts[c] += int(chunk[c].isna().sum())

            # ── 分类列：采样唯一值（最多存 1000 个） ──
            if c in nunique_sets and len(nunique_sets[c]) < 1000:
                try:
                    nunique_sets[c].update(chunk[c].dropna().unique()[:500])
                except Exception:
                    pass

            # ── 数值列：Welford 算法增量计算方差 ──
            if not pd.api.types.is_numeric_dtype(chunk[c]):
                continue
            col_data = chunk[c].dropna()
            if len(col_data) == 0:
                continue
            if c not in numeric_stats:
                numeric_stats[c] = {
                    "min": float("inf"),
                    "max": float("-inf"),
                    "sum": 0.0,
                    "count": 0,
                    # Welford 状态
                    "M2": 0.0,         # 平方差累计（方差 * n）
                    "mean": 0.0,        # 当前均值
                }
            s = numeric_stats[c]
            s["min"] = min(s["min"], float(col_data.min()))
            s["max"] = max(s["max"], float(col_data.max()))
            s["sum"] += float(col_data.sum())

            # Welford 增量更新
            for x in col_data:
                x = float(x)
                s["count"] += 1
                delta = x - s["mean"]
                s["mean"] += delta / s["count"]
                delta2 = x - s["mean"]
                s["M2"] += delta * delta2

    _log.info("分块扫描完成: %s, 行数=%d, 列数=%d", file_path, total_rows, len(col_names or []))

    # 计算最终 std
    for s in numeric_stats.values():
        if s["count"] > 1:
            s["std"] = (s["M2"] / (s["count"] - 1)) ** 0.5
        else:
            s["std"] = 0.0
        s.pop("M2", None)
        s.pop("mean", None)

    # nunique 估计
    estimated_nunique = {c: len(v) for c, v in nunique_sets.items()}

    return {
        "columns": col_names or [],
        "dtypes": col_dtypes,
        "row_count": total_rows,
        "null_counts": null_counts,
        "nunique_estimated": estimated_nunique,
        "numeric_stats": numeric_stats,
        "head": head_sample,
        "tail": tail_sample,
    }


# ── DataFrame 元数据 ─────────────────────────────────────

def df_meta(df: pd.DataFrame) -> dict[str, Any]:
    """从 DataFrame 快速构建元数据 dict（含 describe 完整统计）。"""
    null_counts = {c: int(df[c].isna().sum()) for c in df.columns}
    nunique_counts = {
        c: int(df[c].nunique())
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    }
    numeric_stats = {}
    for c in df.select_dtypes(include=["number"]).columns:
        col_data = df[c].dropna()
        if len(col_data) > 0:
            numeric_stats[c] = {
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "sum": float(col_data.sum()),
                "count": len(col_data),
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "p25": float(col_data.quantile(0.25)),
                "p50": float(col_data.quantile(0.50)),
                "p75": float(col_data.quantile(0.75)),
            }
    return {
        "columns": list(df.columns),
        "dtypes": {c: df[c].dtype for c in df.columns},
        "row_count": len(df),
        "null_counts": null_counts,
        "nunique": nunique_counts,
        "numeric_stats": numeric_stats,
        "head": df.head(5),
    }


# ── 摘要格式化 ───────────────────────────────────────────

def build_summary(meta: dict, filename: str, encoding: str,
                  is_chunked: bool, converted_from: str = "",
                  label: str = "已加载") -> str:
    """将元数据格式化为可读摘要报告。"""
    lines: list[str] = []

    if is_chunked:
        lines.append(f"✅ {label}（大文件模式 — 仅缓存摘要）: {filename}")
        lines.append(f"📏 行数: {meta['row_count']:,} | 列数: {len(meta['columns'])}")
        lines.append(f"🔤 编码: {encoding}")
        if converted_from:
            lines.append(f"📄 原始格式: {converted_from}")
        lines.append("⚠️ 数据未全量驻留内存，后续查询时将重新读取文件")
    else:
        lines.append(f"✅ {label}: {filename}")
        lines.append(f"📏 行数: {meta['row_count']:,} | 列数: {len(meta['columns'])}")
        lines.append(f"🔤 编码: {encoding}")
        if converted_from:
            lines.append(f"📄 原始格式: {converted_from}")

    # 字段列表
    lines.append("")
    lines.append("## 字段列表")
    nunique = meta.get("nunique_estimated") or meta.get("nunique") or {}
    for i, col in enumerate(meta["columns"]):
        dtype = meta["dtypes"].get(col, "unknown")
        null_count = meta["null_counts"].get(col, 0)
        n_unq = nunique.get(col, 0)
        parts = []
        if null_count > 0:
            parts.append(f"空值={null_count}")
        if n_unq > 0:
            parts.append(f"唯一值≈{n_unq}")
        extra = f", {', '.join(parts)}" if parts else ""
        lines.append(f"  {i + 1}. {col} ({dtype}{extra})")

    if meta.get("head") is not None:
        lines.append("")
        lines.append("## 前5行数据")
        lines.append(meta["head"].to_string(index=False))

    if meta.get("tail") is not None:
        lines.append("")
        lines.append("## 最后5行数据")
        lines.append(meta["tail"].to_string(index=False))

    if meta.get("numeric_stats"):
        lines.append("")
        lines.append("## 数值列统计")
        for col, stats in meta["numeric_stats"].items():
            mean_val = stats.get("mean") or (stats["sum"] / stats["count"] if stats["count"] > 0 else 0)
            row = (
                f"  {col}: count={stats['count']:,}, "
                f"min={stats['min']:.4f}, max={stats['max']:.4f}, "
                f"mean={mean_val:.4f}"
            )
            if stats.get("std") and stats["std"] > 0:
                row += f", std={stats['std']:.4f}"
            if stats.get("p25"):
                row += f", [p25={stats['p25']:.4f} p50={stats['p50']:.4f} p75={stats['p75']:.4f}]"
            lines.append(row)

    return "\n".join(lines)
