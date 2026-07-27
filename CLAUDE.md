# CLAUDE.md — 商品数据分析 Agent

> LangChain + DeepSeek + FastAPI + MySQL。自然语言→SQL→解读/清洗/交互式图表/导出/入库。Plan-and-Execute 工作模式。

## 技术栈

- **LLM**: 可配置 Provider（DeepSeek / OpenAI / Anthropic），LangChain tool-calling agent, max_iterations=100, timeout=120s
- **Web**: FastAPI + SSE 流式 + 单文件前端 `chat.html`（多会话侧边栏 + Skills 系统 + localStorage 持久化）
- **DB**: MySQL 8.0 × SQLAlchemy + PyMySQL
- **缓存/持久化**: Redis（可选 — 会话持久化，未配置则使用内存模式）
- **图表**: pyecharts (ECharts) → 交互式 HTML → 前端 `<iframe>` 嵌入
- **数据**: pandas 3.x, numpy, openpyxl, chardet

## 项目结构

```
sj_agent/
├── api_server.py         # FastAPI：路由 + SSE 流式 + Skills API + 后台清理
├── services.py           # AgentService + SessionManager（session workspace + skill 注入）
├── chat.html             # 前端：侧边栏 + Skills 面板 + 录制 + 编辑器

├── core/                 # 纯函数层
│   ├── config.py         # init_config() 延迟初始化（含 SKILLS_DIR）
│   ├── context.py        # DataFrameContext（v4 多槽位 + session workspace）
│   ├── db.py             # 数据库连接 + refresh_tables
│   ├── backup.py         # mysqldump 备份/恢复（支持跨库）
│   ├── file_io.py        # 编码检测 + 分块扫描 + Welford 算法
│   ├── cleaning.py       # eval 引擎 + 分块聚合 + 友好错误
│   ├── importer.py       # DataFrame→MySQL（采样模式自动加载）
│   └── skill_manager.py  # Skill CRUD + 校验 + System Prompt 生成

├── agent/                # LangChain 层
│   ├── __init__.py       # create_tools() 聚合入口（18 工具）
│   ├── prompts.py        # System prompt 模板（多槽位/跨库/去重指导）
│   ├── tools_core.py     # 12 工具: 文件+数据+槽位管理
│   ├── tools_db.py       # 5 工具: SQL/导入/跨库迁移/备份/恢复
│   ├── tools_chart.py    # pyecharts 图表引擎
│   └── agent_factory.py  # Agent 工厂（支持 skill_name 参数）

├── skills/               # Skill 存储（.skill.json 文件）
├── charts/  logs/  backups/  ← 运行时生成；raw/  workspace/（数据沙箱）
```

## 依赖链

```
config ← db / backup / file_io / cleaning / importer / prompts / tools_* / skill_manager
context ← tools_* / tools_chart（每 session 独立绑定）
skill_manager ← config (SKILLS_DIR)
services → agent(create_tools + create_agent) + core(db/context/skill_manager)
api_server → services(SessionManager) + core(skill_manager)
```

## 18 个工具

### 1. `list_files(directory="")` — 列出目录

列出指定目录下的文件（CSV/Excel），含文件大小。忽略 `~$` 开头的 Office 锁文件。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `directory` | str | `""` | `"raw"` / `"workspace"` / 空(两个都列) |

- 返回格式：`文件名 (大小)`，提取文件名时只取空格前的纯文件名
- 操作的是会话专属 workspace，不显示其他会话的文件
- 始终可用，无依赖

---

### 2. `read_data_file(file_path)` — 读取文件

抽样探索文件结构（仅前 100 行），返回列名/dtype/样本。首次 query/clean/export/import 时自动触发全量加载。

| 参数 | 类型 | 说明 |
|------|------|------|
| `file_path` | str | raw/ 或 workspace/ 下的相对路径，或绝对路径 |

**行为细节**：
- Excel：仅采样前 100 行，进入采样模式（`loaded_df=None`, `source_file_path` 有值）
- CSV ≤100MB：同样采样模式
- CSV >100MB：进入大文件分块模式（`loaded_df=None`, `large_file_path` 有值），后续查询走分块读取
- Excel >100MB：拒绝并提示转 CSV
- **防覆盖**：当前槽位已有数据时，自动以文件核心名创建新槽位并切换
- 采样模式下所有工具（query/clean/export/merge/import）首次调用时触发 `_ensure_full_loaded`

---

### 3. `query_loaded_data(expression)` — 查询已加载数据

对当前槽位的 DataFrame 执行 pandas 表达式（可用变量 `df`, `pd`, `np`, `len`, `bool`）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `expression` | str | pandas 表达式，如 `"df['列名'].sum()"` / `"df.head(10)"` |

**行为细节**：
- 多行表达式和赋值语句均支持（自动切换 eval/exec 模式）
- 结果超过 50 行自动截断
- 返回结果前标注数据来源：`📋 数据来源: 槽位'xxx' — 文件名 (行数)`
- 识别 `.corr()` 调用并标记为相关性矩阵
- 采样模式下自动触发全量加载

---

### 4. `clean_data(expression)` — 数据清洗

对当前槽位的 DataFrame 执行清洗/变换。执行前自动拍快照，可用 undo 撤销。

| 参数 | 类型 | 说明 |
|------|------|------|
| `expression` | str | pandas 表达式，**必须返回 DataFrame** |

**行为细节**：
- 支持单表达式：`"df.fillna(0)"` / `"df.drop_duplicates()"`
- 支持多行/多语句：`"df['a']=...\ndf['b']=...\ndf"`
- 支持 `df.assign(...)` 跨行表达式
- 拒绝空操作：`"df.copy()"` 和裸 `"df"` 直接报错
- 执行前后自动比对数行数/空值变化，返回摘要
- `df.copy()` 被拒绝（不会改变数据，浪费一轮）

---

### 5. `undo()` — 撤销操作

撤销最近一次对当前槽位的数据变更。

| 参数 | 类型 | 说明 |
|------|------|------|
| (无) | — | — |

**行为细节**：
- 可撤销：`clean_data` / `merge_file` / `sql_to_dataframe`
- **不可撤销**：SQL DELETE/UPDATE/DROP（不可逆操作）
- 单次有效：快照一次性，撤销后快照清空
- 按槽位独立：每个槽位有自己的快照

---

### 6. `export_data_file(output_path, expression="")` — 导出文件

将当前槽位的 DataFrame 导出为 CSV/Excel（由后缀决定）。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `output_path` | str | 必填 | 文件名或相对路径，仅写入 session workspace |
| `expression` | str | `""` | 可选，先筛选再导出 |

**行为细节**：
- 仅允许写入 session workspace（安全限制）
- 大文件模式：仅导出前 10 万行，避免 OOM
- 采样模式下自动触发全量加载
- 后缀决定格式：`.csv` / `.xlsx`

---

### 7. `merge_file(file_path="", on="", how="left", from_slot="")` — 数据关联

将当前槽位数据（主表）与另一数据源（副表）按列关联。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `file_path` | str | `""` | 副文件路径（from_slot 为空时必需） |
| `on` | str | `""` | 关联列名 |
| `how` | str | `"left"` | 关联方式：`left` / `right` / `inner` / `outer` |
| `from_slot` | str | `""` | 可选，从指定槽位获取副数据（无需文件路径） |

**行为细节**：
- `from_slot` 非空时：直接从命名槽位读取副 DataFrame，不读文件
- `from_slot` 为空时：从 `file_path` 读取文件作为副数据
- 执行前自动拍快照，可用 undo 撤销
- 关联后新增列自动标注来源

---

### 8. `switch_data_slot(name)` — 切换数据槽位

切换到指定槽位。不存在则自动创建。切换不丢失任何数据。

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | str | 槽位名称（限 12 字符），如 `"m7"` / `"合并"` |

**行为细节**：
- 最多保留 5 个槽位，超限自动删除最旧的非活跃槽位
- 返回切换后槽位信息 + 全部槽位列表
- 所有后续工具操作针对当前活跃槽位

---

### 9. `list_data_slots()` — 列出槽位

列出所有数据槽位及其状态摘要。

| 参数 | 类型 | 说明 |
|------|------|------|
| (无) | — | — |

**行为细节**：
- 显示每个槽位：名称、行数×列数、文件名/来源
- 当前活跃槽位标注 `◀ 当前`
- 空槽位标注 `(空)`，采样模式标注 `采样模式`

---

### 10. `copy_to_workspace(source, target_name="")` — 复制到工作区

将 raw/ 下的文件或文件夹复制到 session workspace。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `source` | str | 必填 | raw/ 下的相对路径 |
| `target_name` | str | `""` | 可选新名称，不传则保持原名 |

- **始终可用**，无 DB 依赖

---

### 11. `drop_data_slot(name)` — 删除槽位

删除指定槽位释放内存。不能删除当前活跃槽位。

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | str | 槽位名称 |

---

### 12. `concat_slots(slot_names)` — 纵向拼接槽位

将多个槽位的 DataFrame 纵向拼接（UNION ALL），结果存入当前槽位。

| 参数 | 类型 | 说明 |
|------|------|------|
| `slot_names` | str | 逗号分隔的槽位名称，如 `"m4_6, m6, m7"` |

**行为细节**：
- 至少需要 2 个槽位
- 自动拍快照，可用 undo 撤销
- 返回各槽位行数和合并后总行数
- 采样模式下自动触发全量加载

---

### 13. `sql_to_dataframe(sql)` — SQL 查询到 DataFrame

执行 SELECT 查询，结果加载为 DataFrame。后续可清洗/作图/导出。

| 参数 | 类型 | 说明 |
|------|------|------|
| `sql` | str | SELECT 语句 |

**行为细节**：
- 自动添加 `LIMIT 10000`（如果 SQL 中未指定）
- `%` → `%%` 转义保护（防 SQLAlchemy pyformat 冲突）
- 仅支持 SELECT
- 执行前自动拍快照

---

### 14. `generate_chart(chart_type, x, y, y2, title)` — 生成图表

基于当前 loaded_df 生成 pyecharts 交互式图表（HTML 输出）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `chart_type` | str | `bar` / `line` / `hist` / `pie` / `scatter` / `heatmap` / `auto` |
| `x` | str | X 轴列名 |
| `y` | str | Y 轴列名（多系列用逗号分隔） |
| `y2` | str | 可选，次 Y 轴列名 |
| `title` | str | 图表标题 |

**行为细节**：
- 输出 `.html` 文件到 `charts/` 目录
- 自动聚合：按密度判断（点数/天）→ 按月/周/日聚合
- 浮点修正：`round(v, 8)` 消除累积误差
- >10 个标签自动开启 DataZoom 滑块
- 返回数据来源标注：`槽位'xxx' — 文件名 (行数)`

---

### 15. `import_data_to_db(table_name, create_table=False)` — 导入数据库

将当前槽位 DataFrame 导入 MySQL 表。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `table_name` | str | 必填 | 目标表名（支持 `库名.表名`） |
| `create_table` | bool | `False` | 表不存在时是否自动建表 |

**行为细节**：
- 表存在：列比较 → 只导入匹配列
- 表不存在 + `create_table=True`：自动根据 DataFrame dtype 建表（VARCHAR/DOUBLE/BIGINT/DATETIME）
- 分块 INSERT（50,000 行/批）
- 采样模式下自动触发全量加载
- NaT/NaN 自动转 NULL
- ⚠️ 仅能写入默认数据库（DB_URI 指定的库）。跨库导入需配合 `move_table_to_db`

---

### 16. `move_table_to_db(source_table, target_table, create_target=False, drop_source=False)` — 跨库迁移

纯 SQL 跨库/同库迁移表数据。不依赖 DataFrame 加载。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `source_table` | str | 必填 | 源表名（支持 `库名.表名`） |
| `target_table` | str | 必填 | 目标表名（支持 `库名.表名`） |
| `create_target` | bool | `False` | 是否先 `CREATE TABLE target LIKE source` |
| `drop_source` | bool | `False` | 迁移成功后是否删除源表 |

**行为细节**：
- 执行 `INSERT INTO target SELECT * FROM source`
- 验证源表和目标表行数
- `drop_source=True` 时自动 DROP 源表

---

### 17. `backup_database(table_name="")` — 备份数据库

通过 mysqldump 备份表/全库到 `backups/` 目录。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `table_name` | str | `""` | 表名（支持 `库名.表名`），空串 = 全库备份 |

**行为细节**：
- 跨库支持：自动解析 `库名.表名`，传正确 db 名给 mysqldump
- 自动轮转：最多保留 10 个备份/表
- 生成时间戳文件名

---

### 18. `restore_database(sql_file)` — 恢复数据库

从 `backups/` 目录中的 `.sql` 备份文件恢复数据。

| 参数 | 类型 | 说明 |
|------|------|------|
| `sql_file` | str | 备份文件名或完整路径 |

**行为细节**：
- mysql CLI 优先
- 失败时回退到逐 SQL 执行
- ⚠️ 会覆盖当前表内容，使用前需用户确认

---

## 多槽位系统（v4）

每个会话支持最多 5 个命名 DataFrame 槽位，切换不丢数据。超限自动删除最旧槽位。

- `switch_data_slot('名称')` — 切换槽位（不存在则自动创建，最多 5 个）
- `list_data_slots()` — 查看所有槽位及状态
- `drop_data_slot('名称')` — 删除指定槽位（不能删活跃槽位）
- `concat_slots('a, b, c')` — 纵向拼接多个槽位
- `merge_file(from_slot='x')` — 从另一槽位关联数据
- 所有工具（read/clean/query/export/merge）操作当前活跃槽位
- undo 和 snapshot 按槽位独立
- 读新文件时当前槽位有数据 → 自动创建新槽位（防覆盖）

## Plan-and-Execute 工作模式

Agent 接到复杂任务时遵循 Plan-and-Execute 模式：

1. **制定计划**（第 1 轮）：列出 5-10 步编号执行计划，不调用任何工具
2. **逐步执行**（第 2 轮起）：按计划顺序执行，每步完成后验证结果
3. **验证驱动**：验证通过→继续下一步；验证失败→修复后重试
4. **计划外禁止**：禁止额外探索、跳过验证、重复已完成步骤

## 状态治理

- **槽位上限**: 最多 5 个，超限自动删除最旧非活跃槽位
- **会话清理**: 启动时自动 `DROP TABLE IF EXISTS temp_*`，清除旧会话遗留
- **数据溯源**: `query_loaded_data` / `generate_chart` 输出自动标注来源槽位和文件名
- **DB 自动建库**: `create_db()` 自动 `CREATE DATABASE IF NOT EXISTS`
- **命名规范**: 槽位名限 12 字符以内，原则：内容简写。合并结果统一用 `合并`

## 会话持久化（Redis）

会话数据（对话历史 + 断点续传检查点）默认仅存内存，服务重启后丢失。配置 `REDIS_URL` 后自动持久化到 Redis：

- **Key 格式**: `sj_agent:session:{session_id}:{messages|checkpoints|skill}`
- **TTL**: 与 `SESSION_TTL_MINUTES` 一致，自动过期
- **容错**: Redis 不可用时自动降级为内存模式，不影响正常使用
- **无需额外配置**: 不配 `REDIS_URL` 则保持原有行为

```env
# .env 中添加（可选）
REDIS_URL=redis://localhost:6379/0
```

## 会话工作区隔离

每个会话自动创建 `workspace/session_<时间戳>_<id>/` 专属子目录。

- `export_data_file` / `copy_to_workspace` → 写入会话子目录
- `list_files('workspace')` → 只列本会话文件
- 会话 TTL 过期 / 手动销毁 → 自动清理子目录
- 跨会话读文件：用绝对路径或 `workspace/session_xxx/file.csv`

## Skills 系统（可复用数据处理工作流）

### 概述

Skills 允许用户将重复性的数据处理流程保存为可复用的"技能"。核心机制是**录制式捕获**——在 Agent 执行过程中实时记录工具调用序列（而非事后让 LLM 总结），确保步骤精确可靠。

### Skill 数据结构

```json
{
  "name": "monthly-sales-cleanup",
  "display_name": "月度销售数据清洗入库",
  "description": "加载CSV→去空值→日期格式化→按月聚合→入库",
  "version": 1,
  "created_at": "2026-07-25T10:00:00",
  "updated_at": "2026-07-25T10:00:00",
  "steps": [
    {"tool": "read_data_file", "input": "{{file_path}}", "description": "加载数据文件"},
    {"tool": "clean_data", "input": "df.dropna(subset=['日期'])", "description": "删除日期为空的行"},
    {"tool": "import_data_to_db", "input": "{{target_table}}", "description": "导入到目标表"}
  ],
  "tags": ["清洗", "入库"],
  "param_hints": {
    "file_path": "数据文件路径（CSV/Excel）",
    "target_table": "目标数据库表名（格式: 库名.表名）"
  }
}
```

### 关键设计

- **存储**: `skills/` 目录，每个 skill 一个 `.skill.json` 文件
- **参数占位符**: `{{参数名}}` 标记可替换的参数，使用 skill 时用户提供实际值
- **注入方式**: skill 的 system prompt 片段插入在主 system prompt **之前**，Agent 启动时即加载
- **会话绑定**: Skill 在会话创建时绑定（首次发送消息时），中途不可切换；需切换则新建对话
- **录制过滤**: 录制时自动跳过探索性工具（list_files / query_loaded_data / generate_chart / backup / restore），只捕获核心操作（read / clean / export / import / merge）

### 前端交互

| 功能 | 位置 | 说明 |
|------|------|------|
| 🔴 录制 | 输入行按钮 | 点击开始录制→执行操作→Agent 工具调用自动捕获→点击停止→弹出编辑器 |
| ✏️ 编辑器 | 模态框 | 预填捕获步骤，可编辑名称/描述/步骤/标签/参数占位符 |
| 📋 Skills 列表 | 侧边栏（可折叠） | 显示所有已保存 skill，支持 ⬇导出 / ✕删除 |
| 🎯 Skill 选择器 | 新建对话时弹出 | 选择已有 skill 或"空白对话"跳过 |
| 📌 Skill Banner | 对话区顶部 | 蓝色提示条显示当前激活的 skill，可 ✕ 取消 |
| ⬆️ 导入 | 侧边栏底部按钮 | 选择 `.skill.json` 文件导入 |

### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/skills` | 列出所有 Skill（元数据摘要） |
| `GET` | `/skills/{name}` | 获取单个 Skill 完整内容 |
| `POST` | `/skills` | 创建/更新 Skill（body 为完整 JSON） |
| `DELETE` | `/skills/{name}` | 删除 Skill |
| `GET` | `/skills/{name}/export` | 导出下载 `.skill.json` 文件 |
| `POST` | `/skills/import` | 导入 Skill（body 为完整 JSON） |

## 图表引擎（pyecharts / ECharts）

- **7 种**: bar / line / hist / pie / scatter / heatmap / auto
- **自动聚合**: 按密度判断（点数/天）→ 自动按月/周/日聚合，标签 `%Y-%m` / `%m-%d` / `%H:%M`
- **长格式自动透视**: `(日期, 产品类型, 销售额)` → pivot 为多系列宽格式
- **浮点修正**: `round(v, 8)` 消除 DOUBLE 累积误差
- **自适应**: >10 标签开 DataZoom 滑块 + 动态高度 + 标签间隔
- **输出**: `.html` → 前端 `markdown 链接` → 自动转 `<iframe>`

## 前端（chat.html）关键设计

- **每个对话独立 DOM 容器**: `<div class="conv-pane" id="pane-xxx">`，切换=显示/隐藏
- **SSE 不中断**: 切换到其他对话时 SSE 流继续写入自己的 pane↓，不会被 kill
- **停止按钮**: A 思考中→A 的发送按钮变红色 ⏹ 停止，B 不受影响
- **侧边栏 🟢**: 正在运行中的对话标题后显示绿色圆点
- **Skills 面板 📋**: 侧边栏可折叠区域，列出已保存 skill，支持导出/删除/导入
- **录制按钮 🔴**: 输入行发送按钮旁，录制时红色脉冲动画，SSE agent_action 事件自动捕获步骤
- **Skill 编辑器**: 录制停止后弹出模态框，可编辑步骤序列、参数占位符、描述、标签
- **存储**: `localStorage` 按 `chat_conv_<id>` 持久化（含 `skillName` 字段），`chat_conv_index` 存储 ID 列表
- **页面刷新**: `/chat/restore` 回放历史到服务端 `AgentService.messages`

## 核心设计决策

- **对话记忆**: `AgentService.messages` 列表手动管理，`_build_prompt()` 拼接（放弃 ConversationBufferMemory）
- **Schema 预注入**: `build_rich_prefix(db, skill_name=None)` 启动时写入 system prompt，可选 skill 片段前置
- **Skill 注入**: `build_rich_prefix()` → `build_skill_prompt()` → 步骤列表 + 参数说明 → 插入主 prompt 之前
- **配置延迟初始化**: `init_config()` 显式创建目录+日志（含 `skills/` 目录）
- **抽样探索 + 延迟加载**: `read_data_file` 仅采样前100行探查结构，首次 query/clean/export/merge/import 时通过 `_ensure_full_loaded()` 自动全量加载；CSV>100MB 走分块查询不触发延迟加载
- **多槽位隔离**: DataFrameContext 支持多个命名槽位，切换不丢数据；undo/snapshot 按槽位独立
- **会话持久化**: 可选 Redis 持久化对话记忆 + 检查点，重启可恢复；未配置则内存模式
- **会话工作区隔离**: 每个会话 `workspace/session_<ts>_<id>/`，过期自动清理
- **SQL `%` 安全**: `sql_to_dataframe` 将 `%` → `%%` 防 pyformat 冲突（`sql_db_query` 由 LangChain 处理，无此保护）
- **corr() 检测**: `shape[0]==shape[1]` → 必须同时含 `.corr(` 才标为相关性矩阵
- **图表切换规则**: prompt 禁止"换成 XX 图"时重查 SQL
- **聚合先行**: prompt 引导 LLM 直接用 SQL GROUP BY
- **录制式捕获**: Skill 录制基于 SSE `agent_action` 事件实时捕获工具调用，非 LLM 事后总结
- **断点续传检查点**: AgentService 自动记录已完成工具调用，当用户说「继续」时注入检查点上下文，消除重复探索
- **数据日期验证**: 加载后必须检查实际日期范围，不依赖文件名判断；多源合并前检查日期重叠并去重
- **操作完成后执行**: 找到匹配数据后必须执行 UPDATE/INSERT，不只报告发现
- **Plan-and-Execute**: 第 1 轮先列计划，第 2 轮起逐步执行，每步验证后进入下一步
- **状态治理**: 槽位上限 5 个、会话启动清理 temp_* 表、数据溯源标注、DB 自动建库
- **防覆盖保护**: `read_data_file` 检测到当前槽位有数据时自动创建新槽位
- **df.copy() 拒绝**: `clean_data` 拒绝无意义的空操作，节省轮次
- **多语句支持**: eval 引擎支持跨行表达式和赋值语句，自动切换 eval/exec 模式

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 + 会话数 |
| GET | `/db/tables` | 表结构 |
| GET | `/charts/{file}` | 图表静态文件 |
| POST | `/chat` | 同步对话（支持 `skill_name` 参数） |
| POST | `/chat/stream` | SSE 流式（支持 `skill_name` 参数） |
| POST | `/chat/reset` | 重置会话 |
| POST | `/chat/restore` | 恢复对话历史 `{session_id, messages: [{role, content}]}` |
| GET | `/skills` | 列出所有 Skill |
| GET | `/skills/{name}` | 获取单个 Skill |
| POST | `/skills` | 创建/更新 Skill |
| DELETE | `/skills/{name}` | 删除 Skill |
| GET | `/skills/{name}/export` | 导出 Skill 下载 |
| POST | `/skills/import` | 导入 Skill |

## 部署启动教程

### 环境要求

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Python | 3.10+ | 推荐 3.12+ |
| MySQL | 8.0+ | 需创建空数据库或使用已有库 |
| pip | 最新版 | `python -m pip install --upgrade pip` |

### 步骤 1：克隆项目

```bash
git clone <repo-url> sj_agent
cd sj_agent
```

### 步骤 2：配置环境变量

```bash
# 复制示例配置
cp .env.example .env
```

编辑 `.env` 文件，填入真实值：

```env
# LLM Provider（必填 — deepseek / openai / anthropic）
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=sk-your-api-key-here

# MySQL 数据库连接（必填）
# 格式: mysql+pymysql://用户名:密码@主机:端口/数据库名
DB_URI=mysql+pymysql://root:yourpassword@localhost:3306/product_analysis
```

### 步骤 3：准备数据库

```bash
# 登录 MySQL 创建数据库
mysql -u root -p

# 在 MySQL 中执行
CREATE DATABASE IF NOT EXISTS product_analysis
  DEFAULT CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;
EXIT;
```

> **注意：** 应用启动时会自动 `CREATE DATABASE IF NOT EXISTS`，但建议提前创建以校验权限正确。

### 步骤 4：安装依赖

```bash
# 方式 A：手动安装（推荐）
python -m venv venv
venv\Scripts\pip install -r requirements.txt      # Windows
# 或 source venv/bin/pip install -r requirements.txt  # macOS/Linux

# 方式 B：一键脚本（仅 Windows）
setup.bat
```

> **切换 LLM Provider：** 默认使用 DeepSeek。如需使用 OpenAI 或 Anthropic（Claude），在 `.env` 中修改 `LLM_PROVIDER`，并安装对应依赖：
> ```bash
> # OpenAI
> pip install langchain-openai
> # Anthropic (Claude)
> pip install langchain-anthropic
> ```
> 同时设置对应的 API Key（`OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY`）和模型名。

### 步骤 5：启动服务

```bash
# 开发模式（推荐入门）
venv\Scripts\python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

# 或一键脚本（仅 Windows）
run_server.bat

# 生产模式（多 worker）
venv\Scripts\python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

### 步骤 6：验证服务

浏览器打开 `http://localhost:8000/chat.html`（静态前端页面），或使用 curl 测试：

```bash
# 健康检查
curl http://localhost:8000/health

# 期望返回: {"status":"ok","service":"商品数据分析 Agent","version":"4.0.0","sessions":0}
```

### 步骤 7：导入数据（可选）

应用支持两种数据源，无需预先建表：

| 方式 | 说明 | 示例指令 |
|------|------|---------|
| **文件分析** | 在聊天中直接上传 CSV/Excel，agent 自动读取分析 | `帮我分析 raw/ 目录下的 sales.csv` |
| **数据库查询** | 已有数据表时，agent 自动探查并查询 | `users 表有多少条记录？` |
| **数据入库** | 清洗文件后导入 MySQL | `将 cleaned_df 导入到 lx.monthly_sales 表` |

### 常见问题排查

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| `环境变量 DB_URI 未设置` | `.env` 不存在或位置不对 | 确认 `.env` 在项目根目录，内容格式正确 |
| `Can't connect to MySQL` | MySQL 未启动或密码错误 | `mysql -u root -p` 测试连接，检查 `DB_URI` |
| `ModuleNotFoundError: xxx` | 依赖未安装 | `pip install -r requirements.txt` |
| `findfont: Font family` 警告 | matplotlib 中文字体缺失 | 不影响功能，可忽略 |
| `mysqldump not found` | 备份功能需要 MySQL 客户端 | 安装 MySQL 并将 bin 目录加入 PATH |
| 图表不显示 | charts/ 目录无写入权限 | 确认启动目录有 charts/ 并检查权限 |

### 生产部署建议

```bash
# 使用 gunicorn + uvicorn workers（Linux）
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# 添加反向代理（nginx 示例）
# location / {
#     proxy_pass http://127.0.0.1:8000;
#     proxy_http_version 1.1;
#     proxy_set_header Upgrade $http_upgrade;
#     proxy_set_header Connection "upgrade";
#     proxy_read_timeout 120s;   # SSE 流式需要较长的超时
# }
```

## 关键环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DB_URI` | (必填) | MySQL 连接 |
| `LLM_PROVIDER` | `deepseek` | Provider: deepseek / openai / anthropic |
| `LLM_MODEL` | `deepseek-v4-flash` | 模型 |
| `LLM_API_KEY` | (必填) | 通用 API Key（优先于专属 Key） |
| `LLM_API_BASE` | (可选) | 自定义 API 地址（代理/兼容接口） |
| `REDIS_URL` | (可选) | Redis 连接，不配则内存模式 |
| `SESSION_TTL_MINUTES` | `30` | 会话过期时间 |
| `LARGE_FILE_MB` | `100` | 大文件分块阈值 |
| `MAX_OUTPUT_ROWS` | `50` | 查询结果截断行数 |
| `SKILLS_DIR` | `skills` | Skill 文件存储目录 |
| `API_HOST` | `0.0.0.0` | API 监听地址 |
| `API_PORT` | `8000` | API 端口 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `MAX_ITERATIONS` | `100` | Agent 最大迭代次数 |
| `MAX_EXECUTION_TIME` | `120` | Agent 超时（秒） |
