"""
services.py — Agent 服务层
AgentService: 装配 ctx + tools + agent，手动管理对话历史（代替已弃用的 ConversationBufferMemory）
SessionManager: 管理多个带 TTL 的会话
SessionStore: 会话持久化适配器（支持 Redis / 内存）
"""
import json
import logging
import time
import uuid

from core.config import SESSION_TTL_MINUTES
from core.context import DataFrameContext
from core.db import create_db, refresh_tables
from agent import create_tools
from agent.agent_factory import create_agent

_log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 会话持久化适配器
# ═══════════════════════════════════════════════════════════

class SessionStore:
    """会话持久化抽象基类。MemoryStore 为空操作，RedisStore 持久化到 Redis。"""

    def load_messages(self, session_id: str) -> list[dict]:
        return []

    def save_messages(self, session_id: str, messages: list[dict], ttl: int):
        pass

    def load_checkpoints(self, session_id: str) -> list[dict]:
        return []

    def save_checkpoints(self, session_id: str, steps: list[dict], ttl: int):
        pass

    def load_skill(self, session_id: str) -> str | None:
        return None

    def save_skill(self, session_id: str, skill_name: str, ttl: int):
        pass

    def delete(self, session_id: str):
        pass


class MemoryStore(SessionStore):
    """内存模式：不持久化任何数据。服务重启后会话丢失。"""
    pass


class RedisStore(SessionStore):
    """Redis 持久化模式：会话数据写入 Redis，服务重启可恢复。"""

    def __init__(self, redis_url: str):
        import redis
        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._prefix = "sj_agent:session"
        # 启动时验证连接
        try:
            self._redis.ping()
            _log.info("Redis 会话持久化已启用: %s", redis_url)
        except Exception as e:
            _log.warning("Redis 连接失败，回退到内存模式: %s", e)
            self._redis = None

    @property
    def _r(self):
        """获取 Redis 客户端；连接断开时返回 None"""
        if self._redis is None:
            return None
        try:
            self._redis.ping()
            return self._redis
        except Exception:
            return None

    def _key(self, session_id: str, field: str) -> str:
        return f"{self._prefix}:{session_id}:{field}"

    def _save(self, session_id: str, field: str, data, ttl: int):
        r = self._r
        if r is None:
            return
        key = self._key(session_id, field)
        try:
            r.set(key, json.dumps(data, ensure_ascii=False), ex=ttl)
        except Exception as e:
            _log.debug("Redis 写入失败 (%s): %s", field, e)

    def _load(self, session_id: str, field: str, default):
        r = self._r
        if r is None:
            return default
        key = self._key(session_id, field)
        try:
            raw = r.get(key)
            if raw is None:
                return default
            return json.loads(raw)
        except Exception as e:
            _log.debug("Redis 读取失败 (%s): %s", field, e)
            return default

    def load_messages(self, session_id: str) -> list[dict]:
        return self._load(session_id, "messages", [])

    def save_messages(self, session_id: str, messages: list[dict], ttl: int):
        self._save(session_id, "messages", messages, ttl)

    def load_checkpoints(self, session_id: str) -> list[dict]:
        return self._load(session_id, "checkpoints", [])

    def save_checkpoints(self, session_id: str, steps: list[dict], ttl: int):
        self._save(session_id, "checkpoints", steps, ttl)

    def load_skill(self, session_id: str) -> str | None:
        r = self._r
        if r is None:
            return None
        key = self._key(session_id, "skill")
        try:
            return r.get(key)
        except Exception:
            return None

    def save_skill(self, session_id: str, skill_name: str, ttl: int):
        self._save(session_id, "skill", skill_name, ttl)

    def delete(self, session_id: str):
        r = self._r
        if r is None:
            return
        pattern = self._key(session_id, "*")
        try:
            keys = list(r.scan_iter(match=pattern, count=10))
            if keys:
                r.delete(*keys)
        except Exception as e:
            _log.debug("Redis 删除失败: %s", e)


# ═══════════════════════════════════════════════════════════
# AgentService
# ═══════════════════════════════════════════════════════════


class AgentService:
    """为单个会话装配并持有完整的 agent 栈。

    对话记忆：手动维护 messages 列表，每次 invoke 时拼接历史到 input 中。
    不再依赖 ConversationBufferMemory（已弃用且与 create_sql_agent 不兼容）。
    """

    def __init__(self, db=None, skill_name=None, store: SessionStore | None = None):
        import time
        from pathlib import Path
        from core.config import WORKSPACE_DIR

        self.id = str(uuid.uuid4())[:8]
        self.db = db or create_db()
        self.ctx = DataFrameContext()
        self.skill_name = skill_name
        self._store = store or MemoryStore()

        # 创建会话专属 workspace 子目录
        ts = time.strftime("%Y%m%d_%H%M%S")
        session_dir = Path(WORKSPACE_DIR) / f"session_{ts}_{self.id}"
        session_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.session_ws = str(session_dir.resolve())
        _log.info("会话工作区: %s", self.ctx.session_ws)

        # 清理旧临时表（避免 Agent 混淆数据来源）
        try:
            tables = self.db.get_usable_table_names()
            for t in tables:
                if t.startswith("temp_"):
                    self.db.run(f"DROP TABLE IF EXISTS `{t}`")
            _log.info("已清理旧临时表")
        except Exception:
            pass

        refresh_tables(self.db)
        tools = create_tools(self.ctx, self.db)
        self.agent = create_agent(db=self.db, extra_tools=tools, skill_name=skill_name)
        self.messages: list[dict] = []  # [{"role":"user","content":...}, {"role":"assistant","content":...}]
        self._completed_steps: list[dict] = []  # 断点续传检查点

        # 尝试从持久化存储恢复
        if not skill_name:
            restored_skill = self._store.load_skill(self.id)
            if restored_skill:
                self.skill_name = restored_skill
        restored_msgs = self._store.load_messages(self.id)
        if restored_msgs:
            self.messages = restored_msgs
            _log.info("从持久化恢复 %d 条消息: %s", len(self.messages), self.id)
        restored_steps = self._store.load_checkpoints(self.id)
        if restored_steps:
            self._completed_steps = restored_steps
            _log.info("从持久化恢复 %d 个检查点: %s", len(self._completed_steps), self.id)

        _log.info("AgentService 创建: %s, skill=%s", self.id, self.skill_name or "-")

    def invoke(self, question: str) -> dict:
        """同步调用 agent（带对话历史）。"""
        prompt = self._build_prompt(question)
        result = self.agent.invoke(prompt)
        self._record(question, result["output"])
        return result

    def invoke_str(self, question: str) -> str:
        """同步调用 agent，仅返回输出文本。"""
        return self.invoke(question)["output"]

    def restore_history(self, messages: list[dict]):
        """从前端恢复对话历史。每对 user/assistant 消息还原到 messages 列表。"""
        self.messages.clear()
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if not content:
                continue
            if role in ("user", "assistant"):
                self.messages.append({"role": role, "content": content})

    def record_step(self, tool: str, input_summary: str, ok: bool = True):
        """记录一个已完成的工具调用（供 SSE callback 调用，用于断点续传检查点）。"""
        self._completed_steps.append({
            "tool": tool,
            "input": input_summary[:120],
            "ok": ok,
        })
        # 只保留最近 50 步
        if len(self._completed_steps) > 50:
            self._completed_steps = self._completed_steps[-50:]
        self._store.save_checkpoints(self.id, self._completed_steps, SESSION_TTL_MINUTES * 60)

    def _is_continue(self, question: str) -> bool:
        """检测用户问题是否为断点续传指令。"""
        keywords = ["继续", "接着做", "接着上次", "断点续传", "继续做"]
        return any(kw in question for kw in keywords)

    def _build_checkpoint_context(self) -> str:
        """从已完成步骤和对话历史中构建断点续传上下文。"""
        if not self._completed_steps and not self.messages:
            return ""

        lines = ["## ⚠️ 断点续传检查点（重要：此任务之前已执行过，以下是已完成的工作）\n"]

        # 已完成步骤摘要
        if self._completed_steps:
            lines.append("### 已完成步骤（按时间顺序）")
            for i, step in enumerate(self._completed_steps[-30:], 1):
                status = "✅" if step["ok"] else "⚠️"
                lines.append(f"{status} {step['tool']}: {step['input']}")
            lines.append("")

        # 从对话历史中提取文件/表信息
        files_done = set()
        tables_mentioned = set()
        for m in self.messages:
            content = m.get("content", "")
            # 提取清洗完成的文件
            if "_清洗" in content or "_cleaned" in content:
                for word in content.split():
                    if any(kw in word for kw in ("清洗", "cleaned", "清洗完成", "清洗后")):
                        files_done.add(word.strip("'\".,;*`[]()"))
            # 提取提到的表名
            for prefix in ("lx.", "product_analysis."):
                for word in content.replace("\n", " ").split():
                    if word.startswith(prefix):
                        tables_mentioned.add(word.strip("'\".,;*`[]()"))

        if files_done:
            lines.append(f"### 已有清洗文件 (workspace/): {', '.join(sorted(files_done)[:10])}")
        if tables_mentioned:
            lines.append(f"### 涉及数据库表: {', '.join(sorted(tables_mentioned)[:10])}")

        lines.append("\n⚠️ 以上步骤已执行完毕，禁止重新探查 raw/ 目录或重复 SHOW CREATE TABLE。")
        lines.append("直接从未完成的下一步开始执行。\n")
        return "\n".join(lines)

    def destroy(self):
        import shutil
        from pathlib import Path
        self.ctx.cleanup()
        self.messages.clear()
        self._completed_steps.clear()
        self._store.delete(self.id)
        # 清理会话专属 workspace 子目录
        if self.ctx.session_ws:
            ws_path = Path(self.ctx.session_ws)
            if ws_path.exists():
                shutil.rmtree(ws_path, ignore_errors=True)
                _log.info("会话工作区已清理: %s", self.ctx.session_ws)
        _log.debug("AgentService 销毁: %s", self.id)

    def _build_prompt(self, question: str) -> str:
        """将对话历史拼接到当前问题中，帮助 LLM 理解上下文。"""
        if not self.messages:
            return question

        lines = []

        # 如果是断点续传，注入检查点上下文
        if self._is_continue(question) and self._checkpoint_context:
            checkpoint = self._build_checkpoint_context()
            if checkpoint:
                lines.append(checkpoint)

        lines.append("## 对话历史（重要：上文已经说过的话，请根据此历史理解当前问题）\n")
        for m in self.messages[-20:]:  # 最多保留最近 20 条消息
            role_label = "用户" if m["role"] == "user" else "助手"
            lines.append(f"**{role_label}**: {m['content']}")
        lines.append(f"\n---\n当前问题: {question}")
        return "\n".join(lines)

    @property
    def _checkpoint_context(self) -> bool:
        """是否有检查点数据可用。"""
        return bool(self._completed_steps) or bool(self.messages)

    def _record(self, question: str, answer: str):
        """记录一轮对话。"""
        self.messages.append({"role": "user", "content": question})
        self.messages.append({"role": "assistant", "content": answer})
        self._store.save_messages(self.id, self.messages, SESSION_TTL_MINUTES * 60)


class SessionManager:
    """管理带 TTL 驱逐的 AgentService 实例字典。"""

    def __init__(self, ttl_seconds: int | None = None, store: SessionStore | None = None):
        self._sessions: dict[str, AgentService] = {}
        self._access_times: dict[str, float] = {}
        self._ttl = ttl_seconds if ttl_seconds is not None else SESSION_TTL_MINUTES * 60
        self._store = store or MemoryStore()

    def get(self, session_id: str = "default", skill_name: str | None = None) -> AgentService:
        """获取或创建会话（带 TTL 清理）。

        Args:
            session_id: 会话 ID
            skill_name: 可选，新创建会话时绑定的 skill
        """
        self._evict_expired()
        if session_id not in self._sessions:
            self._sessions[session_id] = AgentService(skill_name=skill_name, store=self._store)
            _log.info("会话创建: %s → %s", session_id, self._sessions[session_id].id)
        self._access_times[session_id] = time.time()
        return self._sessions[session_id]

    def reset(self, session_id: str = "default"):
        """重置会话：销毁并移除。"""
        if session_id in self._sessions:
            self._sessions[session_id].destroy()
            del self._sessions[session_id]
            self._access_times.pop(session_id, None)
            _log.info("会话已重置: %s", session_id)

    def shutdown(self):
        """销毁所有会话。"""
        _log.info("关闭 %d 个会话", len(self._sessions))
        for s in list(self._sessions.values()):
            s.destroy()
        self._sessions.clear()
        self._access_times.clear()

    @property
    def count(self) -> int:
        return len(self._sessions)

    def _evict_expired(self):
        """清理过期会话。"""
        now = time.time()
        expired = [
            sid for sid, ts in self._access_times.items()
            if now - ts > self._ttl
        ]
        for sid in expired:
            if sid in self._sessions:
                self._sessions[sid].destroy()
                del self._sessions[sid]
            self._access_times.pop(sid, None)
        if expired:
            _log.info("清理 %d 个过期会话", len(expired))
