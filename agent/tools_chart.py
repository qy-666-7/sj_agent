"""
tools_chart.py — 图表生成工具（基于 pyecharts / ECharts）
输出交互式 HTML 文件（支持缩放、悬停读数、日期轴自动格式化）。
"""
import time

import numpy as np
import pandas as pd

from langchain.tools import tool as lc_tool
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Pie, Scatter, HeatMap

from core import config
from core.context import DataFrameContext
from agent.tools_core import _resolve_df

# ═══════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════

def _is_datetime_col(df: pd.DataFrame, col: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return True
    try:
        s = pd.to_datetime(df[col], errors='coerce')
        return s.notna().sum() > len(df) * 0.8
    except Exception:
        return False


def _parse_dates(series: pd.Series):
    """尝试解析为日期，成功返回 datetime Series，失败返回 None。"""
    try:
        dt = pd.to_datetime(series, errors='coerce')
        valid = dt.dropna()
        if len(valid) < len(series) * 0.5:
            return None
        return dt
    except Exception:
        return None


def _round_float(v: float) -> float:
    """消除浮点累积误差（如 1.230000000001 → 1.23），保留最高 8 位有效小数。"""
    if v == 0.0 or abs(v) >= 1e15:
        return v
    return round(v, 8)


def _safe_scalar(v) -> float | int | None:
    """单个标量值 → JSON 安全的 Python 数字。"""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        fv = float(v)
        if np.isnan(fv) or np.isinf(fv):
            return 0.0
        return _round_float(fv)
    if isinstance(v, pd.Timestamp):
        return v.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _to_json_safe(obj):
    """将 numpy/pandas 类型转为 JSON 安全的 Python 类型。"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        fv = float(obj)
        return 0.0 if np.isnan(fv) or np.isinf(fv) else _round_float(fv)
    if isinstance(obj, np.ndarray):
        return [_safe_scalar(x) for x in obj.flat]
    if isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(obj, pd.Series):
        return [_safe_scalar(v) for v in obj]
    return obj


def _make_fname(ctx: DataFrameContext, suffix: str) -> str:
    # 纳秒精度时间戳，避免同一秒内生成多张图时文件名冲突覆盖
    ts = str(time.time_ns())[-12:]
    safe_name = (ctx.loaded_filename or "chart").rsplit(".", 1)[0]
    return f"{safe_name}_{suffix}_{ts}.html"


def _build_global_opts(title: str, n_labels: int = 0) -> opts.InitOpts:
    """构建通用全局配置：响应式容器 + 动态高度（标签多时自动增大）。"""
    if n_labels > 30:
        height = "650px"
    elif n_labels > 15:
        height = "580px"
    else:
        height = "500px"
    return opts.InitOpts(
        width="100%",
        height=height,
        bg_color="transparent",
        animation_opts=opts.AnimationOpts(animation=True, animation_threshold=2000),
    )


def _axis_label_str(series, freq: str = 'auto') -> list[str]:
    """将日期或数值序列转为可读的轴标签字符串列表。freq 用于选择格式。"""
    s = pd.Series(series)
    dt = _parse_dates(s)
    if dt is not None:
        # 按月/季度聚合 → 年-月 格式
        if freq.startswith('M') or freq.startswith('Q'):
            return [d.strftime('%Y-%m') for d in dt]
        if freq.startswith('Y') or freq.startswith('A'):
            return [d.strftime('%Y') for d in dt]
        # 按日期自身特征判断
        rng = (dt.max() - dt.min()).days
        if rng >= 365:
            return [d.strftime('%Y-%m') for d in dt]
        if rng >= 2:
            return [d.strftime('%m-%d') for d in dt]
        if (dt.dt.hour != 0).any():
            return [d.strftime('%m-%d %H:%M') for d in dt]
        return [d.strftime('%m-%d') for d in dt]
    return [str(v) for v in series]


def _gen_label_opts(label: str, n_labels: int = 0) -> opts.AxisOpts:
    """X 轴配置：分类轴 + 旋转标签 + 自动间隔。"""
    rotate = 45 if n_labels > 10 else 30
    interval = max(0, n_labels // 30 - 1) if n_labels > 30 else 0
    return opts.AxisOpts(
        name=label,
        type_="category",
        axislabel_opts=opts.LabelOpts(rotate=rotate, interval=interval),
    )


def _common_opts(n_labels: int) -> dict:
    """bar/line 图表通用配置：网格自适应 + 提示 + 工具栏 + 数据缩放。"""
    extra = {}
    if n_labels > 10:
        pct = 100 if n_labels <= 20 else max(40, 2000 // n_labels)
        extra["datazoom_opts"] = [opts.DataZoomOpts(range_start=0, range_end=pct)]
    return {
        "tooltip_opts": opts.TooltipOpts(trigger="axis"),
        "toolbox_opts": opts.ToolboxOpts(
            feature=opts.ToolBoxFeatureOpts(
                save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(title="保存为图片"),
                data_zoom=opts.ToolBoxFeatureDataZoomOpts(),
                restore=opts.ToolBoxFeatureRestoreOpts(),
            ),
        ),
        **extra,
    }


def _y_axis_opts(label: str = "") -> opts.AxisOpts:
    """Y 轴配置（自动单位格式化）。"""
    return opts.AxisOpts(
        name=label,
        type_="value",
        axislabel_opts=opts.LabelOpts(formatter="{value}"),
        splitline_opts=opts.SplitLineOpts(is_show=True),
    )


# ═══════════════════════════════════════════════════════════
# 图表生成
# ═══════════════════════════════════════════════════════════

def _get_date_agg(df: pd.DataFrame, dt_series: pd.Series) -> tuple[str, str]:
    """根据数据密度自动选择日期聚合粒度，返回 (pandas freq, 中文标签后缀)。
    'auto' 表示不聚合，保持原始粒度。"""
    n = len(df)
    if n <= 1:
        return 'auto', ''
    rng = dt_series.max() - dt_series.min()
    rng_days = rng.days

    if rng_days <= 2:
        return 'auto', ''
    # 按数据密度判断（日均点数），避免密集日数据被误聚合为月
    density = n / max(rng_days, 1)
    if rng_days >= 90 and density >= 1.5:
        return 'ME', '（按月）'
    if rng_days >= 30 and density >= 1.5:
        return 'W-MON', '（按周）'
    if rng_days >= 7:
        return 'D', '（按日）'
    return 'auto', ''


def _auto_pivot(df: pd.DataFrame, x: str, y_cols: list[str],
                group_col: str = "") -> tuple[pd.DataFrame, str, list[str]]:
    """检测长格式数据并自动透视，返回 (new_df, new_x, new_y_cols)。

    Case 1: y_cols 不在 df 列名中但匹配某分类列的取值
            (日期, 产品类型, 销售额) + y='产品A,产品B' → 宽格式
    Case 2: y_cols 在 df 中但旁边有维度列（如平台/城市），
            自动 pivot 拆分为多系列 — 解决「折线图只出一条线」问题
            (平台, 月份, 实收) + y='实收' → (月份, 饿了么, 美团)
    """
    # ── 辅助：找分类列 ──
    def _cat_cols(df, exclude: set) -> list:
        result = []
        for c in df.columns:
            if c in exclude:
                continue
            col = df[c].dropna()
            if len(col) == 0:
                continue
            n_unique = col.nunique()
            if 1 < n_unique <= 12 and col.apply(type).eq(str).all():
                result.append(c)
        return result

    # ── Case 1: y 值藏在分类列中（现有逻辑） ──
    if not all(yc in df.columns for yc in y_cols):
        for cat_col in _cat_cols(df, {x}):
            cats = set(df[cat_col].dropna().unique())
            match_count = sum(1 for yc in y_cols if yc in cats)
            if match_count >= len(y_cols) * 0.5:
                val_cols = [c for c in df.columns if c != x and c != cat_col]
                if not val_cols:
                    continue
                val_col = val_cols[0]
                try:
                    pivoted = df.pivot_table(
                        index=x, columns=cat_col, values=val_col,
                        aggfunc='sum', fill_value=0,
                    ).reset_index()
                    pivoted.columns = [str(c) for c in pivoted.columns]
                    return pivoted, x, y_cols
                except Exception:
                    continue
        return df, x, y_cols

    # ── Case 2: y 列都在 df 中，但存在分类维度列需要 pivot 成多系列 ──
    dim_cols = _cat_cols(df, {x} | set(y_cols))
    if not dim_cols:
        return df, x, y_cols

    cat_col = dim_cols[0]  # 取第一个分类列作为系列拆分键
    try:
        pivoted = df.pivot_table(
            index=x, columns=cat_col, values=y_cols,
            aggfunc='sum', fill_value=0,
        ).reset_index()

        # 扁平化分层列名
        new_cols = []
        for c in pivoted.columns:
            if isinstance(c, tuple):
                if c[1]:  # ('实收', '饿了么') → '饿了么' or '饿了么_实收'
                    new_cols.append(c[1] if len(y_cols) == 1 else f'{c[1]}_{c[0]}')
                else:     # ('月份', '') → '月份' (reset_index 列在 MultiIndex 下的表示)
                    new_cols.append(str(c[0]))
            else:
                new_cols.append(str(c))
        pivoted.columns = new_cols
        new_y_cols = [c for c in new_cols if c != str(x)]
        return pivoted, x, new_y_cols
    except Exception:
        return df, x, y_cols


def _prepare_xy(df, x: str, y_cols: list[str]):
    """准备 X/Y 数据：日期列自动解析并按密度聚合，分类列截断。
    返回 (x_labels, y_series, x_label, is_date, freq) — freq 用于标签格式化。"""
    dt_series = _parse_dates(df[x])
    x_label = x
    is_date = dt_series is not None
    freq = 'auto'

    if is_date and y_cols:
        df_plot = df.copy()
        df_plot['_dt'] = dt_series
        freq, freq_label = _get_date_agg(df, dt_series)

        if freq == 'auto':
            x_data = dt_series.apply(_to_json_safe).tolist()
        else:
            grouper = pd.Grouper(key='_dt', freq=freq)
            agg_dict = {yc: 'sum' for yc in y_cols}
            grouped = df_plot.groupby(grouper).agg(agg_dict).reset_index().sort_values('_dt')
            x_data = grouped['_dt'].apply(_to_json_safe).tolist()
            df_plot = grouped
            x_label = x + freq_label

        y_series = {yc: df_plot[yc].apply(_to_json_safe).tolist() for yc in y_cols}
    elif is_date:
        x_data = dt_series.apply(_to_json_safe).tolist()
        y_series = {}
    else:
        plot_df = df.head(60) if len(df) > 60 else df
        x_data = plot_df[x].astype(str).tolist()
        y_series = {yc: plot_df[yc].apply(_to_json_safe).tolist() for yc in y_cols} if y_cols else {}

    return x_data, y_series, x_label, is_date, freq


def _chart_title(ctx: DataFrameContext, chart_type: str, title: str) -> str:
    if title:
        return title
    safe_name = (ctx.loaded_filename or "data").rsplit(".", 1)[0]
    return f"{safe_name}: {chart_type}"


def build_bar(ctx, df, x: str, y: str, title: str, y_cols: list[str]):
    x_data, y_series, x_label, is_date, freq = _prepare_xy(df, x, y_cols)
    x_labels = _axis_label_str(x_data, freq) if is_date else x_data
    bar = Bar(init_opts=_build_global_opts(title, len(x_labels)))

    bar.add_xaxis(x_labels)
    bar.set_global_opts(
        xaxis_opts=_gen_label_opts(x_label, len(x_labels)),
        yaxis_opts=_y_axis_opts("" if not y_cols else y),
        **_common_opts(len(x_labels)),
    )

    for yc, values in y_series.items():
        bar.add_yaxis(yc, values, label_opts=opts.LabelOpts(is_show=False))

    if not y_cols:
        counts = df[x].value_counts().head(30)
        bar.add_xaxis(counts.index.astype(str).tolist())
        bar.add_yaxis("计数", counts.values.tolist(), label_opts=opts.LabelOpts(is_show=False))

    return bar


def build_line(ctx, df, x: str, y: str, y2: str, title: str, y_cols: list[str]):
    x_data, y_series, x_label, is_date, freq = _prepare_xy(df, x, y_cols)
    x_labels = _axis_label_str(x_data, freq) if is_date else x_data
    line = Line(init_opts=_build_global_opts(title, len(x_labels)))

    line.add_xaxis(x_labels)
    line.set_global_opts(
        xaxis_opts=_gen_label_opts(x_label, len(x_labels)),
        yaxis_opts=_y_axis_opts("" if not y_cols else y),
        **_common_opts(len(x_labels)),
    )

    for yc, values in y_series.items():
        line.add_yaxis(yc, values, label_opts=opts.LabelOpts(is_show=False),
                        areastyle_opts=opts.AreaStyleOpts(opacity=0.08))

    if y2 and y2 in df.columns:
        y2_data = df[y2].apply(_to_json_safe).tolist()
        line.add_yaxis(y2, y2_data, yaxis_index=1, label_opts=opts.LabelOpts(is_show=False))
        line.extend_axis(
            yaxis=opts.AxisOpts(name=y2, type_="value",
                                axislabel_opts=opts.LabelOpts(formatter="{value}")),
        )

    return line


def build_pie(ctx, df, x: str, title: str):
    counts = df[x].value_counts().head(10)
    data_pairs = [[str(k), int(v)] for k, v in zip(counts.index.astype(str), counts.values)]
    pie = Pie(init_opts=_build_global_opts(title))
    pie.add("", data_pairs, radius=["35%", "65%"],
            label_opts=opts.LabelOpts(formatter="{b}: {c} ({d}%)"))
    pie.set_global_opts(
        tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}: {c} ({d}%)"),
        toolbox_opts=opts.ToolboxOpts(
            feature=opts.ToolBoxFeatureOpts(
                save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(title="保存为图片"),
            ),
        ),
    )
    return pie


def build_scatter(ctx, df, x: str, y: str, y_cols: list[str], title: str):
    sc = Scatter(init_opts=_build_global_opts(title))
    data = df[[x] + y_cols].dropna()
    sc.add_xaxis(data[x].astype(str).tolist())

    for yc in y_cols:
        sc.add_yaxis(yc, data[yc].apply(_to_json_safe).tolist(),
                     symbol_size=8,
                     label_opts=opts.LabelOpts(is_show=False))

    sc.set_global_opts(
        xaxis_opts=_gen_label_opts(x),
        yaxis_opts=_y_axis_opts(y_cols[0] if y_cols else ""),
        tooltip_opts=opts.TooltipOpts(trigger="item"),
        toolbox_opts=opts.ToolboxOpts(
            feature=opts.ToolBoxFeatureOpts(
                save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(title="保存为图片"),
            ),
        ),
    )
    return sc


def build_heatmap(ctx, df, title: str):
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()
    labels = [str(c) for c in corr.columns]
    data = []
    for i, col_i in enumerate(corr.columns):
        for j, col_j in enumerate(corr.columns):
            data.append([i, j, round(float(corr.iloc[i, j]), 3)])

    x_min, x_max = 0, len(labels) - 1
    hm = HeatMap(init_opts=_build_global_opts(title))
    hm.add_xaxis(labels)
    hm.add_yaxis("", labels, data,
                 label_opts=opts.LabelOpts(is_show=True, position="inside"),
                 itemstyle_opts=opts.ItemStyleOpts(border_color="#fff", border_width=1))
    hm.set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
        yaxis_opts=opts.AxisOpts(),
        visualmap_opts=opts.VisualMapOpts(min_=-1, max_=1, is_show=True,
                                           range_color=["#e74c3c", "#fff", "#2ecc71"],
                                           pos_right="5%", pos_top="middle"),
        tooltip_opts=opts.TooltipOpts(formatter="相关性: {c}"),
    )
    return hm


def build_hist(ctx, df, x: str, title: str):
    counts, bins = np.histogram(df[x].dropna(), bins=30)
    bar = Bar(init_opts=_build_global_opts(title))
    labels = [f"{bins[i]:.1f}" for i in range(len(bins) - 1)]
    bar.add_xaxis(labels)
    bar.add_yaxis("频数", counts.tolist(), category_gap=0,
                   label_opts=opts.LabelOpts(is_show=False))
    bar.set_global_opts(
        xaxis_opts=opts.AxisOpts(name=x, axislabel_opts=opts.LabelOpts(rotate=30)),
        yaxis_opts=opts.AxisOpts(name="频数"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        toolbox_opts=opts.ToolboxOpts(
            feature=opts.ToolBoxFeatureOpts(
                save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(title="保存为图片"),
            ),
        ),
    )
    return bar


# ═══════════════════════════════════════════════════════════
# 工具入口
# ═══════════════════════════════════════════════════════════


def create_chart_tool(ctx: DataFrameContext):

    @lc_tool
    def generate_chart(chart_type: str = "auto", x: str = "", y: str = "",
                       y2: str = "", title: str = "") -> str:
        """
        对已加载数据生成交互式图表 HTML，保存到 charts/ 目录。
        支持 bar/line/hist/pie/scatter/heatmap/auto。
        ECharts 交互功能：鼠标悬停读数、拖拽缩放、右上角工具箱保存图片。

        Parameters:
        chart_type: 默认"auto"；x: X轴列名；y: Y轴列名(多列用逗号分隔)
        y2: 次Y轴(仅line)；title: 标题(可选)
        """
        df = _resolve_df(ctx)
        if df is None:
            return "错误: 没有已加载的数据。"

        # ── auto 检测 ──
        if chart_type == "auto":
            if _is_datetime_col(df, x):
                chart_type = "line"
            elif pd.api.types.is_numeric_dtype(df[x]) and not y:
                chart_type = "hist"
            else:
                chart_type = "bar"

        # ── heatmap ──
        if chart_type == "heatmap":
            if df.select_dtypes(include=['number']).shape[1] < 2:
                return "错误: 至少需要 2 个数值列。"
            chart = build_heatmap(ctx, df, _chart_title(ctx, chart_type, title))
            if chart is None:
                return "错误: 至少需要 2 个数值列。"
            fname = _make_fname(ctx, "heatmap")
            out_path = config.CHART_DIR / fname
            chart.render(str(out_path))
            ctx.track_chart(str(out_path))
            return (
                f"📊 图表已生成\n"
                f"📐 heatmap ({len(df.select_dtypes(include=['number']).columns)} 列相关性)\n"
                f"🖼️ [交互图表](/charts/{fname})"
            )

        if not x or x not in df.columns:
            return f"错误: X 轴列名无效。可用: {', '.join(df.columns[:12])}"

        # ── 解析 Y 列（自动透视长格式数据：Case 1 y值在分类列 + Case 2 维度列拆分） ──
        raw_y_cols = [yc.strip() for yc in y.split(",")] if y else []
        if raw_y_cols:
            df, x, raw_y_cols = _auto_pivot(df, x, raw_y_cols)
        y_cols = [yc for yc in raw_y_cols if yc in df.columns]
        if not y_cols and y:
            return f"错误: Y 列不存在 — {y}。数据列: {', '.join(df.columns[:12])}"

        # ── 构建图表 ──
        chart_title = _chart_title(ctx, chart_type, title)

        try:
            if chart_type == "bar":
                chart = build_bar(ctx, df, x, y, chart_title, y_cols)
            elif chart_type == "line":
                chart = build_line(ctx, df, x, y, y2, chart_title, y_cols)
            elif chart_type == "pie":
                chart = build_pie(ctx, df, x, chart_title)
            elif chart_type == "scatter":
                chart = build_scatter(ctx, df, x, y, y_cols, chart_title)
            elif chart_type == "hist":
                chart = build_hist(ctx, df, x, chart_title)
            else:
                return f"错误: 不支持的图表类型 '{chart_type}'。支持: bar/line/hist/pie/scatter/heatmap/auto"

            fname = _make_fname(ctx, chart_type)
            out_path = config.CHART_DIR / fname
            chart.render(str(out_path))
            ctx.track_chart(str(out_path))
        except Exception as e:
            return f"错误: 图表生成失败 — {e}"

        y_info = f", y={y}" if y else ""
        y2_info = f", y2={y2}" if y2 else ""
        return (
            f"📊 图表已生成\n"
            f"📐 {chart_type}{y_info}{y2_info} | 槽位'{ctx.active_slot}' — {ctx.loaded_filename} ({len(df):,} 行)\n"
            f"🖼️ [交互图表](/charts/{fname})"
        )

    return generate_chart
