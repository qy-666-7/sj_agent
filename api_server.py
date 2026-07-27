"""
商品数据分析 Agent — FastAPI 接口层
启动: uvicorn api_server:app --reload
接口:
  POST /chat     — 同步对话
  POST /chat/stream — SSE 流式对话
  POST /chat/reset  — 清空会话
  GET  /health   — 健康检查
  GET  /db/tables — 数据库表结构
"""
import logging

from dotenv import load_dotenv

load_dotenv(override=True)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import asyncio
import json
import queue

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from langchain_core.callbacks import BaseCallbackHandler

from core.config import init_config, CHART_DIR, REDIS_URL
from core.db import create_db
from core.skill_manager import (
    list_skills, load_skill, save_skill, delete_skill,
    export_skill, import_skill,
)
from services import SessionManager, MemoryStore, RedisStore

init_config()

_log = logging.getLogger(__name__)

# ── 持久化存储 ──────────────────────────────────────────────

_store = RedisStore(REDIS_URL) if REDIS_URL else MemoryStore()

# ── 会话 & DB ──────────────────────────────────────────────

_db = create_db()
session_mgr = SessionManager(store=_store)


# ── 流式回调 ──────────────────────────────────────────────

class StreamCallback(BaseCallbackHandler):
    """捕获 LLM/tool/agent 事件，写入线程安全的 queue.Queue

    事件类型:
      - thinking       LLM 推理文本（从 on_agent_action.log 提取）
      - agent_action   代理决定调用工具（含工具名 + 输入参数）
      - tool_start     工具开始执行
      - tool_end       工具执行完成（含返回预览）
      - agent_finish   代理完成（输出最终答案）
      - done           最终答案（由 event_generator 特殊处理）
      - error          异常
    """

    def __init__(self, q: queue.Queue, agent_service=None):
        self.q = q
        self._agent_actions = 0  # 计数器，用于区分第几轮思考
        self._token_total = 0    # 本轮用户问题的累计 token 消耗
        self._steps: list[dict] = []  # 已完成的工具调用记录 [{tool, input, ok}]
        self._last_tool_name = ""     # 当前正在执行的工具名
        self._last_tool_input = ""    # 当前工具的输入
        self._agent_service = agent_service  # 可选，用于断点续传检查点

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._agent_actions += 1
        self.q.put({
            "type": "llm_start",
            "data": f"第 {self._agent_actions} 轮思考…",
            "round": self._agent_actions,
        })

    def on_agent_action(self, action, **kwargs):
        """★ 核心: LLM 决定使用某个工具时的推理 + 工具选择"""
        tool_name = action.tool
        tool_input = str(action.tool_input) if action.tool_input else ""

        # action.log 包含 LLM 的完整推理过程（为什么选这个工具）
        thinking = (action.log or "").strip()
        # 截断过长的思考文本
        thinking_preview = thinking[:800] if len(thinking) > 800 else thinking

        self._last_tool_name = tool_name
        self._last_tool_input = tool_input

        self.q.put({
            "type": "agent_action",
            "tool": tool_name,
            "tool_input": tool_input[:500],
            "thinking": thinking_preview,
            "thinking_full_len": len(thinking),
            "round": self._agent_actions,
        })

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        self.q.put({
            "type": "tool_start",
            "tool": tool_name,
            "data": f"执行: {tool_name}",
        })

    def on_tool_end(self, output, **kwargs):
        # 记录已完成的步骤（供中断摘要使用）
        ok = not str(output).startswith("错误")
        self._steps.append({
            "tool": self._last_tool_name,
            "input": self._last_tool_input[:120],
            "ok": ok,
        })
        # 同步到 AgentService（用于断点续传检查点）
        if self._agent_service:
            self._agent_service.record_step(self._last_tool_name, self._last_tool_input, ok)
        output_str = str(output)
        preview = output_str[:500]
        is_truncated = len(output_str) > 500
        if isinstance(output, str) and len(output) > 10:
            _log.info("Tool 返回: %s…", output_str[:200].replace("\n", " "))
        self.q.put({
            "type": "tool_end",
            "data": preview,
            "is_truncated": is_truncated,
            "full_len": len(output_str),
        })

    def on_agent_finish(self, finish, **kwargs):
        """代理完成，返回最终答案 + token 汇总"""
        self.q.put({
            "type": "agent_finish",
            "data": "分析完成，生成最终回答…",
        })
        # 发送本轮对话的 token 消耗汇总
        if hasattr(self, '_token_total') and self._token_total > 0:
            self.q.put({
                "type": "token_summary",
                "total_tokens": self._token_total,
                "rounds": self._agent_actions,
            })

    def on_llm_end(self, response, **kwargs):
        # 提取 token 用量（兼容多种 LangChain / DeepSeek 返回格式）
        token_usage = None
        # 方式 1: llm_output.token_usage（最常见）
        if hasattr(response, 'llm_output') and isinstance(response.llm_output, dict):
            token_usage = response.llm_output.get('token_usage')
        # 方式 2: generations[0][0].generation_info（旧版 LangChain）
        if not token_usage and hasattr(response, 'generations') and response.generations:
            try:
                gen = response.generations[0][0]
                gi = getattr(gen, 'generation_info', None) or {}
                token_usage = gi.get('token_usage') if isinstance(gi, dict) else None
            except (IndexError, AttributeError, TypeError):
                pass
        # 方式 3: response_metadata（新版 LangChain / 部分模型）
        if not token_usage and hasattr(response, 'generations') and response.generations:
            try:
                gen = response.generations[0][0]
                rm = getattr(gen, 'response_metadata', None) or {}
                if isinstance(rm, dict):
                    token_usage = rm.get('token_usage') or rm.get('usage')
            except (IndexError, AttributeError, TypeError):
                pass

        if token_usage and isinstance(token_usage, dict):
            prompt_tokens = token_usage.get('prompt_tokens', 0) or 0
            completion_tokens = token_usage.get('completion_tokens', 0) or 0
            total_tokens = token_usage.get('total_tokens', 0) or (prompt_tokens + completion_tokens)
            self._token_total += total_tokens
            self.q.put({
                "type": "token_usage",
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cumulative": self._token_total,
                "round": self._agent_actions,
            })
        else:
            self.q.put({"type": "llm_end", "data": "思考完成"})


# ═══════════════════════════════════════════════════════════
# 中断恢复：当 Agent 因 max_iterations 截断时生成进度摘要
# ═══════════════════════════════════════════════════════════

def _build_progress_summary(output: str, steps: list, rounds: int,
                            max_rounds: int) -> str:
    """Agent 被 max_iterations 中断时，从 callback 记录的步骤生成摘要。"""
    TOOL_ICONS = {
        "list_files": "📋", "read_data_file": "📂", "query_loaded_data": "🔍",
        "clean_data": "🧹", "undo": "↩️", "export_data_file": "💾", "merge_file": "🔗",
        "sql_to_dataframe": "📥", "sql_db_query": "📊", "generate_chart": "📈",
        "import_data_to_db": "🗄️", "backup_database": "📋", "restore_database": "♻️",
        "copy_to_workspace": "📁",
    }

    # 去重 + 生成可读摘要
    seen = set()
    done_lines = []
    last_file = ""
    for s in steps:
        tool = s["tool"]
        inp = s["input"]
        ok = s["ok"]
        mark = "" if ok else " ⚠️失败"

        if tool == "list_files":
            key = f"list:{inp}"
            if key not in seen:
                seen.add(key)
                done_lines.append(f"{TOOL_ICONS.get(tool, '🔧')} 列出 {inp} 目录文件")
        elif tool == "read_data_file":
            fname = inp.split("/")[-1] if "/" in inp else inp
            if fname not in seen:
                seen.add(fname)
                done_lines.append(f"{TOOL_ICONS.get(tool, '🔧')} 读取 {fname}{mark}")
                last_file = fname
        elif tool == "clean_data":
            done_lines.append(f"{TOOL_ICONS.get(tool, '🔧')} 清洗数据: {inp[:80]}{mark}")
        elif tool == "export_data_file":
            done_lines.append(f"{TOOL_ICONS.get(tool, '🔧')} 导出: {inp}{mark}")
        elif tool == "merge_file":
            done_lines.append(f"{TOOL_ICONS.get(tool, '🔧')} 合并: {inp}{mark}")
        elif tool == "import_data_to_db":
            done_lines.append(f"{TOOL_ICONS.get(tool, '🔧')} 导入数据库: {inp}{mark}")
        elif tool == "sql_to_dataframe":
            done_lines.append(f"{TOOL_ICONS.get(tool, '🔧')} SQL→DataFrame{mark}")
        elif tool == "sql_db_query":
            done_lines.append(f"{TOOL_ICONS.get(tool, '🔧')} SQL 查询: {inp[:60]}{mark}")
        else:
            done_lines.append(f"{TOOL_ICONS.get(tool, '🔧')} {tool}: {inp[:60]}{mark}")

    lines = [f"\n\n---\n", f"## ⚠️ 分析被中断 — 已用 {rounds}/{max_rounds} 轮\n"]

    if done_lines:
        lines.append(f"### ✅ 已完成 {len(done_lines)} 步\n")
        for i, dl in enumerate(done_lines, 1):
            lines.append(f"{i}. {dl}")

    # 生成简洁的"继续"提示
    context_parts = []
    if last_file:
        context_parts.append(f"最后处理了 {last_file}")
    lines.append("\n### 💡 继续方式\n")
    lines.append("直接回复「继续」+ 下一步任务。例如：\n")
    if context_parts:
        hint = "，".join(context_parts)
        lines.append(f"> 继续：{hint}，现在需要完成剩余步骤\n")
    else:
        lines.append("> 继续完成任务\n")

    return "\n".join(lines)


# ── FastAPI 应用 ──────────────────────────────────────────

app = FastAPI(
    title="商品数据分析 Agent",
    description="LangChain + MySQL + CSV/Excel 智能分析接口，支持数据清洗",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/charts", StaticFiles(directory=str(CHART_DIR)), name="charts")


# ── 请求/响应模型 ─────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(
        ..., description="自然语言问题", min_length=1, max_length=2000,
    )
    session_id: str = Field(
        default="default", description="会话ID，不传则使用默认会话",
    )
    skill_name: str | None = Field(
        default=None, description="可选，绑定的 skill 名称（仅新建会话时生效）",
    )


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str


class RestoreRequest(BaseModel):
    session_id: str = "default"
    messages: list[dict] = Field(
        default=[], description="[{role: 'user'|'assistant', content: '...'}, ...]"
    )


# ── 接口 ──────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "商品数据分析 Agent",
        "version": "4.0.0",
        "sessions": session_mgr.count,
    }


@app.get("/db/tables")
async def list_tables():
    """列出数据库所有表及结构"""
    tables = _db.get_usable_table_names()
    result = {}
    for t in tables:
        result[t] = _db.get_table_info([t])
    return {"tables": tables, "schemas": result}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """同步对话：发送问题，等待完整结果返回（带对话记忆）"""
    svc = session_mgr.get(req.session_id, skill_name=req.skill_name)
    try:
        result = await asyncio.to_thread(svc.invoke_str, req.question)
        return ChatResponse(
            session_id=svc.id,
            question=req.question,
            answer=result,
        )
    except Exception as e:
        _log.error("对话出错: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式对话：SSE 逐事件返回 Agent 的思考过程（带对话记忆）"""
    svc = session_mgr.get(req.session_id, skill_name=req.skill_name)

    async def event_generator():
        q: queue.Queue = queue.Queue()
        callback = StreamCallback(q, agent_service=svc)

        # 记录本轮对话前已有的图表文件
        charts_before = set(svc.ctx.chart_files)
        prompt = svc._build_prompt(req.question)

        async def run_agent():
            try:
                result = await asyncio.to_thread(
                    svc.agent.invoke,
                    prompt,
                    {"callbacks": [callback]},
                )
                output = result["output"]

                # 检测是否因 max_iterations 被截断，生成进度摘要
                if ("stopped due to" in str(output).lower()
                        or "max iterations" in str(output).lower()
                        or "agent stopped" in str(output).lower()):
                    output = _build_progress_summary(
                        output, callback._steps,
                        rounds=callback._agent_actions,
                        max_rounds=callback._agent_actions,
                    )

                svc._record(req.question, output)
                # 注入本轮新生成的图表——按"图N"标记插入到对应文字下方
                new_charts = [c for c in svc.ctx.chart_files if c not in charts_before]
                if new_charts:
                    import re
                    markers = list(re.finditer(r'图\s*(\d+)\s*[：:〉》]', output))
                    n_matched = min(len(new_charts), len(markers))
                    # 匹配到的图表：从后往前插入标记处（避免位置偏移）
                    for i in range(n_matched - 1, -1, -1):
                        fname = new_charts[i].replace("\\", "/").split("/")[-1]
                        if f"/charts/{fname}" in output:
                            continue
                        chart_md = f"\n\n[交互图表](/charts/{fname})"
                        pos = markers[i].end()
                        output = output[:pos] + chart_md + output[pos:]
                    # 多余的图表：按原始顺序追加到末尾
                    for i in range(n_matched, len(new_charts)):
                        fname = new_charts[i].replace("\\", "/").split("/")[-1]
                        if f"/charts/{fname}" in output:
                            continue
                        output += f"\n\n[交互图表](/charts/{fname})"
                q.put({"type": "done", "data": output})
            except Exception as e:
                q.put({"type": "error", "data": str(e)})

        task = asyncio.create_task(run_agent())

        yield (
            f"data: {json.dumps({'type': 'start', 'data': req.question, 'session_id': svc.id}, ensure_ascii=False)}\n\n"
        )

        loop = asyncio.get_event_loop()
        while not task.done() or not q.empty():
            try:
                event = await loop.run_in_executor(None, q.get, True, 0.5)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

        await task
        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat/reset")
async def chat_reset(session_id: str = "default"):
    """清空对话记忆 + 释放数据资源"""
    session_mgr.reset(session_id)
    return {"status": "ok", "message": f"会话 {session_id} 已清空", "session_id": session_id}


@app.post("/chat/restore")
async def chat_restore(req: RestoreRequest):
    """恢复对话历史到服务器内存（页面刷新后调用）"""
    svc = session_mgr.get(req.session_id)
    svc.restore_history(req.messages)
    _log.info("恢复对话记忆: %d 条消息, session=%s", len(req.messages), req.session_id)
    return {"status": "ok", "restored": len(req.messages), "session_id": svc.id}


# ── Skills 管理 ───────────────────────────────────────────

@app.get("/skills")
async def skills_list():
    """列出所有已保存的 Skill（元数据摘要）"""
    try:
        return {"skills": list_skills()}
    except Exception as e:
        _log.error("列出 skills 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/skills/{name}")
async def skills_get(name: str):
    """获取单个 Skill 完整内容"""
    try:
        return load_skill(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' 不存在")
    except Exception as e:
        _log.error("读取 skill 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/skills")
async def skills_save(data: dict):
    """创建或更新 Skill（body 为完整 skill JSON）"""
    try:
        name = save_skill(data)
        return {"status": "ok", "name": name, "message": f"Skill '{name}' 已保存"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.error("保存 skill 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/skills/{name}")
async def skills_delete(name: str):
    """删除 Skill"""
    try:
        ok = delete_skill(name)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Skill '{name}' 不存在")
        return {"status": "ok", "message": f"Skill '{name}' 已删除"}
    except HTTPException:
        raise
    except Exception as e:
        _log.error("删除 skill 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/skills/{name}/export")
async def skills_export(name: str):
    """导出 Skill 为 .skill.json 文件下载"""
    try:
        data = export_skill(name)
        content = json.dumps(data, ensure_ascii=False, indent=2)
        return Response(
            content=content,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={name}.skill.json",
            },
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' 不存在")
    except Exception as e:
        _log.error("导出 skill 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/skills/import")
async def skills_import(data: dict):
    """导入 Skill（body 为完整 skill JSON）"""
    try:
        name = import_skill(data)
        return {"status": "ok", "name": name, "message": f"Skill '{name}' 已导入"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.error("导入 skill 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── 优雅启动 & 关闭 ────────────────────────────────────

@app.on_event("startup")
async def startup():
    """启动后台清理任务"""
    async def _bg_cleanup():
        while True:
            await asyncio.sleep(60)
            session_mgr._evict_expired()
    asyncio.create_task(_bg_cleanup())
    _log.info("后台会话清理已启动（每60秒）")


@app.on_event("shutdown")
async def shutdown():
    """服务关闭时清理所有会话"""
    session_mgr.shutdown()


# ── 启动入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
