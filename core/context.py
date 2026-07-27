"""
context.py — DataFrameContext
每个会话独立一个实例，替代模块级 global 变量。
v4: 多槽位 — 支持同时持有多个命名 DataFrame，切换不丢数据。
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

_log = logging.getLogger(__name__)


@dataclass
class _Slot:
    """单个数据槽位：一个 DataFrame + 关联状态"""
    df: pd.DataFrame | None = None
    filename: str = ""
    large_file_path: str = ""
    large_file_encoding: str = ""
    large_file_suffix: str = ""
    source_file_path: str = ""
    source_file_suffix: str = ""
    _snapshot: pd.DataFrame | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DataFrameContext:
    """
    持有一次数据加载会话的所有状态。

    v4 多槽位: 支持多个命名 DataFrame 并存，通过 switch() 切换。
    - 默认槽位名: "default"
    - read_data_file / sql_to_dataframe 自动使用当前活跃槽位
    - switch("名称") 切换槽位（不存在则自动创建）
    - 旧代码无需修改：loaded_df / snapshot / undo 等自动代理到活跃槽位
    """

    # ── 多槽位存储 ──────────────────────────────────────────
    _slots: dict[str, _Slot] = field(default_factory=dict)
    _active: str = "default"

    # ── 会话工作区 ──────────────────────────────────────────
    session_ws: str = ""  # 会话专属 workspace 子目录，未设置时回退到全局 WORKSPACE_DIR

    # ── 全局共享状态 ────────────────────────────────────────
    temp_files: list[str] = field(default_factory=list)
    # merge_file 关联缓存（内部追踪，供外部可选读取）
    side_files: dict[str, pd.DataFrame] = field(default_factory=dict)
    chart_files: list[str] = field(default_factory=list)

    def __post_init__(self):
        if "default" not in self._slots:
            self._slots["default"] = _Slot()

    # ── 工作区路径 ──────────────────────────────────────────

    @property
    def workspace_dir(self) -> str:
        """有效的 workspace 目录：会话专属 > 全局 WORKSPACE_DIR"""
        if self.session_ws:
            return self.session_ws
        from core.config import WORKSPACE_DIR
        return str(WORKSPACE_DIR)

    # ── 槽位访问 ────────────────────────────────────────────

    @property
    def _slot(self) -> _Slot:
        """当前活跃槽位"""
        if self._active not in self._slots:
            self._slots[self._active] = _Slot()
        return self._slots[self._active]

    @property
    def active_slot(self) -> str:
        """当前活跃槽位名称"""
        return self._active

    def switch(self, name: str) -> str:
        """切换到命名槽位（不存在则自动创建）。最多保留 5 个槽位。返回切换信息。"""
        name = name.strip()
        if not name:
            return "错误: 槽位名称不能为空"
        old = self._active
        if name not in self._slots:
            # 槽位上限：最多 5 个，超出时删除最旧的（非活跃槽位）
            MAX_SLOTS = 5
            if len(self._slots) >= MAX_SLOTS:
                # 找最久未使用的非活跃槽位
                candidates = [n for n in self._slots if n != old]
                if candidates:
                    oldest = candidates[0]
                    dropped_slot = self._slots.pop(oldest)
                    rows = len(dropped_slot.df) if dropped_slot.df is not None else 0
                    _log.warning("槽位超限(%d)，自动删除: '%s' (%d 行)", MAX_SLOTS, oldest, rows)
            self._slots[name] = _Slot()
            _log.info("槽位创建: '%s'", name)
        self._active = name
        slot = self._slot
        info = f"已切换到 '{name}'"
        if slot.df is not None:
            info += f" ({len(slot.df):,} 行 × {len(slot.df.columns)} 列)"
        elif slot.source_file_path:
            info += f" (采样文件: {Path(slot.source_file_path).name})"
        _log.info("槽位切换: '%s' → '%s'", old, name)
        return info

    def list_slots(self) -> str:
        """列出所有槽位及其状态摘要"""
        if not self._slots:
            return "📋 无数据槽位"
        lines = [f"📋 数据槽位 ({len(self._slots)} 个):"]
        for name, s in self._slots.items():
            marker = " ◀ 当前" if name == self._active else ""
            if s.df is not None:
                lines.append(
                    f"  [{name}]{marker}: {len(s.df):,} 行 × {len(s.df.columns)} 列 "
                    f"({s.filename or '内存数据'})"
                )
            elif s.source_file_path:
                lines.append(
                    f"  [{name}]{marker}: 采样模式 — "
                    f"{Path(s.source_file_path).name}"
                )
            elif s.large_file_path:
                lines.append(
                    f"  [{name}]{marker}: 大文件模式 — "
                    f"{Path(s.large_file_path).name}"
                )
            else:
                lines.append(f"  [{name}]{marker}: (空)")
        return "\n".join(lines)

    def drop_slot(self, name: str) -> str:
        """删除命名槽位（不能删除活跃槽位）"""
        name = name.strip()
        if name == self._active:
            return f"错误: 不能删除当前活跃槽位 '{name}'。请先 switch 到其他槽位。"
        if name not in self._slots:
            return f"错误: 槽位 '{name}' 不存在"
        slot = self._slots.pop(name)
        rows = len(slot.df) if slot.df is not None else 0
        _log.info("槽位删除: '%s' (%d 行)", name, rows)
        return f"✅ 已删除槽位 '{name}' ({rows:,} 行)"

    # ── 属性代理（读取/写入活跃槽位，保持向后兼容）───────────

    @property
    def loaded_df(self) -> pd.DataFrame | None:
        return self._slot.df

    @loaded_df.setter
    def loaded_df(self, value: pd.DataFrame | None):
        self._slot.df = value

    @property
    def loaded_filename(self) -> str:
        return self._slot.filename

    @loaded_filename.setter
    def loaded_filename(self, value: str):
        self._slot.filename = value

    @property
    def large_file_path(self) -> str:
        return self._slot.large_file_path

    @large_file_path.setter
    def large_file_path(self, value: str):
        self._slot.large_file_path = value

    @property
    def large_file_encoding(self) -> str:
        return self._slot.large_file_encoding

    @large_file_encoding.setter
    def large_file_encoding(self, value: str):
        self._slot.large_file_encoding = value

    @property
    def large_file_suffix(self) -> str:
        return self._slot.large_file_suffix

    @large_file_suffix.setter
    def large_file_suffix(self, value: str):
        self._slot.large_file_suffix = value

    @property
    def source_file_path(self) -> str:
        return self._slot.source_file_path

    @source_file_path.setter
    def source_file_path(self, value: str):
        self._slot.source_file_path = value

    @property
    def source_file_suffix(self) -> str:
        return self._slot.source_file_suffix

    @source_file_suffix.setter
    def source_file_suffix(self, value: str):
        self._slot.source_file_suffix = value

    @property
    def history(self) -> list[dict[str, Any]]:
        return self._slot.history

    @history.setter
    def history(self, value: list[dict[str, Any]]):
        self._slot.history = value

    @property
    def _snapshot(self) -> pd.DataFrame | None:
        return self._slot._snapshot

    @_snapshot.setter
    def _snapshot(self, value: pd.DataFrame | None):
        self._slot._snapshot = value

    # ── 判断（代理到活跃槽位）────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return (self._slot.df is not None
                or bool(self._slot.large_file_path)
                or bool(self._slot.source_file_path))

    @property
    def is_large(self) -> bool:
        return self._slot.df is None and bool(self._slot.large_file_path)

    @property
    def is_sampled(self) -> bool:
        return self._slot.df is None and bool(self._slot.source_file_path)

    # ── 临时文件 ──────────────────────────────────────────

    def register_temp(self, path: str):
        if path and path not in self.temp_files:
            self.temp_files.append(path)

    # ── 加载 DataFrame（来自 SQL 等） ─────────────────────

    def load_df(self, df: pd.DataFrame, name: str = "查询结果"):
        """直接加载 DataFrame（如 SQL 查询结果），替代 read_data_file"""
        self.snapshot()           # 拍快照（如果已有数据）
        self._slot.df = df
        self._slot.filename = name
        self._slot.large_file_path = ""
        self._slot.large_file_encoding = ""
        self._slot.large_file_suffix = ""
        _log.info("DataFrame 加载到 '%s': %s (%d 行)", self._active, name, len(df))

    # ── 快照 & 通用撤销 ───────────────────────────────────

    def snapshot(self):
        """保存当前活跃槽位的 DataFrame 快照，用于 undo"""
        if self._slot.df is not None:
            self._slot._snapshot = self._slot.df.copy()
            _log.debug("快照已保存: '%s' %d 行", self._active, len(self._slot._snapshot))

    def undo(self) -> str:
        """撤销最近一次数据变更（clean / merge / sql_to_df）"""
        if self._slot._snapshot is not None and self._slot.df is not None:
            before = len(self._slot.df)
            self._slot.df = self._slot._snapshot
            self._slot._snapshot = None
            if self._slot.history:
                self._slot.history.pop()
            _log.info("撤销: '%s' %d → %d 行", self._active, before, len(self._slot.df))
            return (
                f"🔄 已撤销最近一次操作\n"
                f"📏 恢复至 {len(self._slot.df):,} 行 × {len(self._slot.df.columns)} 列"
            )
        return "错误: 没有可撤销的操作（快照不存在或已撤销过）"

    def log_action(self, action: str, expression: str, before_rows: int,
                   after_rows: int):
        """记录操作到活跃槽位的历史"""
        self._slot.history.append({
            "action": action,
            "expression": expression,
            "before_rows": before_rows,
            "after_rows": after_rows,
        })

    # ── 副文件 ────────────────────────────────────────────

    def add_side_file(self, name: str, df: pd.DataFrame):
        self.side_files[name] = df
        _log.info("关联文件添加: %s (%d 行)", name, len(df))

    # ── 图表追踪 ──────────────────────────────────────────

    def track_chart(self, path: str):
        if path not in self.chart_files:
            self.chart_files.append(path)

    # ── 清理 ──────────────────────────────────────────────

    def cleanup(self):
        for p in self.temp_files:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass
        self.temp_files.clear()
        for p in self.chart_files:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass
        self.chart_files.clear()
        self._slots.clear()
        self._active = "default"
        self._slots["default"] = _Slot()
        self.side_files.clear()
        _log.debug("DataFrameContext 已清理")

    def reset(self):
        self.cleanup()
