"""
agent — LangChain 工具 + Agent 工厂
"""
from agent.tools_core import create_core_tools
from agent.tools_db import create_db_tools
from agent.tools_chart import create_chart_tool


def create_tools(ctx, db=None):
    """为一个 DataFrameContext 生成绑定的工具列表。

    工具顺序（测试依赖此顺序，不可更改）：
        0: list_files          (core)
        1: read_data_file      (core)
        2: query_loaded_data   (core)
        3: clean_data          (core)
        4: undo                (core)
        5: export_data_file    (core)
        6: merge_file          (core)
        7: switch_data_slot    (core — v4 多槽位)
        8: list_data_slots     (core — v4 多槽位)
        9: copy_to_workspace   (core — 无DB依赖)
       10: drop_data_slot      (core — v4 多槽位)
       11: concat_slots        (core — 纵向拼接多槽位)
       12: sql_to_dataframe    (db — 始终包含)
       13: generate_chart      (chart)
       14: import_data_to_db   (db — 条件)
       15: move_table_to_db    (db — 条件，纯SQL跨库迁移)
       16: backup_database     (db — 条件)
       17: restore_database    (db — 条件)
    """
    db_list = create_db_tools(ctx, db)   # [sql_to_df] 或 [sql_to_df, import, move, backup, restore]

    tools = create_core_tools(ctx)       # indices 0-11 (12 tools)
    tools.append(db_list[0])             # index 12: sql_to_dataframe (始终存在)
    tools.append(create_chart_tool(ctx))  # index 13: generate_chart
    tools.extend(db_list[1:])            # index 14+: db-only tools (可能为空)

    return tools
