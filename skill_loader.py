"""
Markdown Skills 加载器

从 skills/ 目录读取 .md 文件并解析为可执行的 Skill
支持热重载，修改 md 文件后无需重启服务
"""
import os
import re
import json
import math
from typing import Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SkillParameter:
    """技能参数"""
    name: str
    type: str
    required: bool = True
    description: str = ""
    enum: list = field(default_factory=list)


@dataclass
class MarkdownSkill:
    """Markdown 技能"""
    name: str
    description: str
    trigger_words: list[str]
    parameters: list[SkillParameter]
    execute_code: str
    file_path: str
    examples: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def get_schema(self) -> dict:
        """返回 OpenAI Function Calling 格式的 schema"""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    def execute(self, **kwargs) -> str:
        """执行技能"""
        # 创建安全的执行环境
        safe_globals = {
            "__builtins__": {
                "abs": abs,
                "all": all,
                "any": any,
                "bool": bool,
                "dict": dict,
                "enumerate": enumerate,
                "eval": eval,
                "float": float,
                "int": int,
                "isinstance": isinstance,
                "len": len,
                "list": list,
                "max": max,
                "min": min,
                "pow": pow,
                "print": print,
                "range": range,
                "round": round,
                "set": set,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "zip": zip,
            },
            # 常用模块
            "math": math,
            "json": json,
        }

        # 执行代码并获取 execute 函数
        try:
            local_vars = {}
            exec(self.execute_code, safe_globals, local_vars)

            if "execute" not in local_vars:
                return f"❌ 技能 {self.name} 的执行代码中未定义 execute 函数"

            result = local_vars["execute"](**kwargs)

            # 格式化结果
            if isinstance(result, str):
                return result
            elif isinstance(result, dict):
                return self._format_dict_result(result)
            else:
                return str(result)

        except Exception as e:
            return f"❌ 执行技能 {self.name} 失败: {str(e)}"

    def _format_dict_result(self, data: dict) -> str:
        """格式化字典结果为可读文本"""
        lines = []
        for key, value in data.items():
            key_display = key.replace("_", " ").title()
            lines.append(f"- {key_display}: {value}")
        return "\n".join(lines)


class MarkdownSkillLoader:
    """Markdown 技能加载器"""

    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = Path(skills_dir)
        self._skills: dict[str, MarkdownSkill] = {}
        self._load_all_skills()

    def _load_all_skills(self):
        """加载所有技能文件"""
        if not self.skills_dir.exists():
            print(f"⚠️  Skills 目录不存在: {self.skills_dir}")
            return

        for md_file in self.skills_dir.glob("*.md"):
            try:
                skill = self._parse_skill_file(md_file)
                if skill:
                    self._skills[skill.name] = skill
                    print(f"✅ 加载技能: {skill.name} ({md_file.name})")
            except Exception as e:
                print(f"❌ 加载技能文件失败 {md_file}: {e}")

    def _parse_skill_file(self, file_path: Path) -> Optional[MarkdownSkill]:
        """解析技能文件"""
        content = file_path.read_text(encoding="utf-8")

        # 解析名称
        name_match = re.search(r"- \*\*名称\*\*:\s*(\w+)", content)
        if not name_match:
            name_match = re.search(r"name:\s*(\w+)", content)
        name = name_match.group(1) if name_match else file_path.stem

        # 解析描述
        desc_match = re.search(r"- \*\*描述\*\*:\s*(.+)", content)
        description = desc_match.group(1).strip() if desc_match else ""

        # 解析触发词
        trigger_match = re.search(r"- \*\*触发词\*\*:\s*(.+)", content)
        trigger_words = []
        if trigger_match:
            trigger_words = [w.strip() for w in trigger_match.group(1).split(",")]

        # 解析参数表格
        parameters = self._parse_parameters(content)

        # 解析执行代码
        code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
        execute_code = code_match.group(1) if code_match else ""

        # 解析示例
        examples = self._parse_examples(content)

        return MarkdownSkill(
            name=name,
            description=description,
            trigger_words=trigger_words,
            parameters=parameters,
            execute_code=execute_code,
            file_path=str(file_path),
            examples=examples,
        )

    def _parse_parameters(self, content: str) -> list[SkillParameter]:
        """解析参数表格"""
        parameters = []

        # 查找参数表格
        param_section = re.search(r"## 参数\n\n\|.*?\n\|.*?\n([\s\S]*?)(?=\n##|\Z)", content)
        if not param_section:
            return parameters

        table_content = param_section.group(1)
        for line in table_content.strip().split("\n"):
            if line.startswith("|"):
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) >= 4:
                    param = SkillParameter(
                        name=parts[0],
                        type=parts[1] or "string",
                        required=parts[2] == "是",
                        description=parts[3] if len(parts) > 3 else "",
                        enum=[e.strip() for e in parts[4].split(",")] if len(parts) > 4 and parts[4] else []
                    )
                    parameters.append(param)

        return parameters

    def _parse_examples(self, content: str) -> list[dict]:
        """解析示例"""
        examples = []
        example_section = re.search(r"## 示例\n\n([\s\S]*?)(?=\n##|\Z)", content)
        if example_section:
            lines = example_section.group(1).strip().split("\n")
            current_example = {}
            for line in lines:
                if line.startswith("**用户**:"):
                    current_example["user"] = line.replace("**用户**:", "").strip()
                elif line.startswith("**调用**:"):
                    current_example["call"] = line.replace("**调用**:", "").strip()
                elif line.startswith("**返回**:"):
                    current_example["result"] = line.replace("**返回**:", "").strip()
                    if current_example:
                        examples.append(current_example)
                        current_example = {}
        return examples

    def reload(self):
        """重新加载所有技能"""
        self._skills.clear()
        self._load_all_skills()

    def get(self, name: str) -> Optional[MarkdownSkill]:
        """获取技能"""
        return self._skills.get(name)

    def list_all(self) -> list[MarkdownSkill]:
        """列出所有技能"""
        return list(self._skills.values())

    def get_all_schemas(self) -> list[dict]:
        """获取所有技能的 schema"""
        return [skill.get_schema() for skill in self._skills.values()]

    def find_by_trigger(self, text: str) -> Optional[MarkdownSkill]:
        """根据触发词查找技能"""
        text_lower = text.lower()
        for skill in self._skills.values():
            for trigger in skill.trigger_words:
                if trigger.lower() in text_lower:
                    return skill
        return None

    def execute(self, name: str, arguments: dict) -> str:
        """执行技能"""
        skill = self.get(name)
        if not skill:
            return f"❌ 未找到技能: {name}"
        return skill.execute(**arguments)


# 全局技能加载器
skill_loader = MarkdownSkillLoader()