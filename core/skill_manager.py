"""
skill_manager.py — Skill CRUD 引擎

Skill 以 .skill.json 文件形式存储在 skills/ 目录中。
每个 skill 定义一组可复用的数据处理步骤（工具调用序列），
支持参数占位符 {{param_name}}。
"""
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from core.config import SKILLS_DIR

_log = logging.getLogger(__name__)

# Skill 文件名规则：只允许字母、数字、下划线、连字符
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

# 必填字段
_REQUIRED_FIELDS = {"name", "steps"}


# ── 输入校验 ──────────────────────────────────────────────

def _validate_skill(data: dict):
    """校验 skill 数据，不合法时抛出 ValueError。"""
    missing = _REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ValueError(f"缺少必填字段: {', '.join(sorted(missing))}")

    name = data.get("name", "")
    if not name or not _NAME_RE.match(name):
        raise ValueError(
            f"Skill 名称 '{name}' 不合法，只允许字母、数字、下划线、连字符"
        )

    steps = data.get("steps", [])
    if not isinstance(steps, list) or len(steps) == 0:
        raise ValueError("steps 必须是非空列表")

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"步骤 {i + 1} 必须是对象")
        if "tool" not in step:
            raise ValueError(f"步骤 {i + 1} 缺少 'tool' 字段")
        if "input" not in step:
            raise ValueError(f"步骤 {i + 1} 缺少 'input' 字段")


def _skill_path(name: str) -> Path:
    """返回 skill 文件的完整路径。"""
    return SKILLS_DIR / f"{name}.skill.json"


# ── CRUD ──────────────────────────────────────────────────

def list_skills() -> list[dict]:
    """列出所有 skill 的元数据（名称、描述、标签、版本、步骤数）。"""
    skills = []
    if not SKILLS_DIR.exists():
        return skills

    for f in sorted(SKILLS_DIR.glob("*.skill.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            skills.append({
                "name": data.get("name", f.stem.replace(".skill", "")),
                "display_name": data.get("display_name", ""),
                "description": data.get("description", ""),
                "version": data.get("version", 1),
                "tags": data.get("tags", []),
                "step_count": len(data.get("steps", [])),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
            })
        except (json.JSONDecodeError, OSError) as e:
            _log.warning("跳过无效 skill 文件 %s: %s", f.name, e)

    return skills


def load_skill(name: str) -> dict:
    """读取单个 skill 的完整内容。"""
    path = _skill_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Skill '{name}' 不存在")
    return json.loads(path.read_text(encoding="utf-8"))


def save_skill(data: dict) -> str:
    """保存或更新 skill。返回 skill name。"""
    _validate_skill(data)

    name = data["name"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    # 确保 display_name 有默认值
    data.setdefault("display_name", name)
    data.setdefault("description", "")
    data.setdefault("version", 1)
    data.setdefault("tags", [])
    data.setdefault("param_hints", {})

    # 自动扫描步骤中的 {{param}} 占位符，补全 param_hints
    existing_params = set(data.get("param_hints", {}).keys())
    for step in data.get("steps", []):
        for match in re.finditer(r"\{\{(\w+)\}\}", step.get("input", "")):
            pname = match.group(1)
            if pname not in existing_params:
                data["param_hints"][pname] = f"参数: {pname}"
                existing_params.add(pname)

    # 更新时间戳
    if "created_at" not in data or not data["created_at"]:
        data["created_at"] = now
    data["updated_at"] = now

    path = _skill_path(name)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _log.info("Skill 已保存: %s (%d 步骤)", name, len(data.get("steps", [])))
    return name


def delete_skill(name: str) -> bool:
    """删除 skill 文件。返回 True 表示成功删除，False 表示不存在。"""
    path = _skill_path(name)
    if not path.exists():
        return False
    path.unlink()
    _log.info("Skill 已删除: %s", name)
    return True


def export_skill(name: str) -> dict:
    """导出 skill 完整 JSON（供下载）。同 load_skill。"""
    return load_skill(name)


def import_skill(data: dict) -> str:
    """校验并导入 skill。返回 skill name。"""
    _validate_skill(data)

    name = data["name"]
    path = _skill_path(name)

    # 如果已存在，递增版本号
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            old_version = existing.get("version", 1)
            data["version"] = old_version + 1
            data["created_at"] = existing.get("created_at", "")
        except (json.JSONDecodeError, OSError):
            data["version"] = 1

    return save_skill(data)


# ── System Prompt 生成 ───────────────────────────────────

def build_skill_prompt(name: str) -> str:
    """从 skill 生成注入到 system prompt 的文本片段。"""
    skill = load_skill(name)

    display_name = skill.get("display_name", name)
    description = skill.get("description", "")
    steps = skill.get("steps", [])
    param_hints = skill.get("param_hints", {})

    lines = [
        "## 🎯 当前激活的 Skill",
        f"**技能名称**: {display_name}",
    ]
    if description:
        lines.append(f"**描述**: {description}")

    lines.append("\n### 📋 执行步骤")
    lines.append("请严格按以下步骤顺序执行。`{{参数名}}` 表示需要用户提供的参数，" +
                 "如用户未提供，请主动询问。\n")

    for i, step in enumerate(steps, 1):
        tool = step.get("tool", "unknown")
        inp = step.get("input", "")
        desc = step.get("description", "")

        # 截断过长的输入
        inp_display = inp if len(inp) <= 200 else inp[:200] + "…"

        line = f"{i}. **{tool}**"
        if desc:
            line += f" — {desc}"
        line += f"\n   `{inp_display}`"
        lines.append(line)

    # 参数说明
    if param_hints:
        lines.append("\n### 🔧 参数说明")
        for pname, phint in param_hints.items():
            lines.append(f"- `{{{{{pname}}}}}`: {phint}")

    # 使用提示
    lines.append("\n### ⚠️ 重要提示")
    lines.append("- 用户可能只提供参数值（如文件路径、表名），其余步骤按本 Skill 定义执行。")
    lines.append("- 如果某个步骤不适用当前数据，请说明原因并跳过，不要强行执行。")
    lines.append("- 所有 Skill 步骤执行完毕后，汇总结果告知用户。")

    return "\n".join(lines)
