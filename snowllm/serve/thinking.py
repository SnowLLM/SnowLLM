# SPDX-License-Identifier: Apache-2.0
"""Discover controls from the loaded tokenizer's actual Jinja chat template.

There is no standard thinking-level field in model config.json. Read template
expressions instead of maintaining a model-name registry. Only advertise values
that the template both names and accepts when rendering a user message.
"""
from jinja2 import TemplateError, nodes
from transformers.utils.chat_template_utils import _compile_jinja_template


def thinking_capabilities(tokenizer, *, required: bool = False) -> dict:
    if required:
        return {"switch": False, "levels": [], "required": True, "source": "model"}
    controls = getattr(tokenizer, "thinking_controls", None)
    if controls is not None:
        return dict(controls)
    result = {"switch": False, "levels": [], "source": "chat_template"}
    try:
        template = tokenizer.get_chat_template()
        tree = _compile_jinja_template(template).environment.parse(template)
    except (ValueError, TypeError, TemplateError, AttributeError):
        return result

    effort_names = {"reasoning_effort"}

    def effort(node):
        if isinstance(node, nodes.CondExpr) and isinstance(node.expr2, nodes.Const):
            return effort(node.expr1)
        if isinstance(node, nodes.Filter) and node.name in ("default", "trim", "lower"):
            return effort(node.node)
        return isinstance(node, nodes.Name) and node.name in effort_names

    constants = {}
    for assignment in tree.find_all(nodes.Assign):
        if isinstance(assignment.target, nodes.Name):
            constants[assignment.target.name] = assignment.node
            if effort(assignment.node):
                effort_names.add(assignment.target.name)

    def literals(node, seen=frozenset()):
        if isinstance(node, nodes.Const):
            if isinstance(node.value, str):
                return [node.value]
            if isinstance(node.value, (list, tuple)):
                return [v for v in node.value if isinstance(v, str)]
        if isinstance(node, (nodes.List, nodes.Tuple)):
            return [v for item in node.items for v in literals(item, seen)]
        if isinstance(node, nodes.Dict):
            return [v for pair in node.items for v in literals(pair.key, seen)]
        if isinstance(node, nodes.Name) and node.name in constants and node.name not in seen:
            return literals(constants[node.name], seen | {node.name})
        return []

    allowed = []
    candidates = []
    for comparison in tree.find_all(nodes.Compare):
        left = comparison.expr
        for operand in comparison.ops:
            if effort(left):
                values = allowed if operand.op in ("in", "notin") else candidates
                values.extend(literals(operand.expr))
            elif effort(operand.expr):
                candidates.extend(literals(left))
            left = operand.expr
    for conditional in tree.find_all(nodes.CondExpr):
        if effort(conditional.expr1):
            candidates.extend(literals(conditional.expr2))
    for lookup in tree.find_all(nodes.Getitem):
        if effort(lookup.arg):
            candidates.extend(literals(lookup.node))

    for call in tree.find_all(nodes.Call):
        if (isinstance(call.node, nodes.Getattr) and call.node.attr == "get"
                and call.args and effort(call.args[0])):
            candidates.extend(literals(call.node.node))

    def render(**kwargs):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}], tokenize=False,
            add_generation_prompt=True, **kwargs)

    names = {node.name for node in tree.find_all(nodes.Name)}
    if names & {"enable_thinking", "thinking"}:
        try:
            result["switch"] = render(enable_thinking=True, thinking=True) != render(
                enable_thinking=False, thinking=False)
        except (ValueError, TypeError, TemplateError):
            pass
    allowed_prompts = set()
    for value in dict.fromkeys(allowed + candidates):
        try:
            prompt = render(reasoning_effort=value, enable_thinking=True, thinking=True)
        except (ValueError, TypeError, TemplateError):
            continue
        if value in allowed:
            allowed_prompts.add(prompt)
        elif prompt in allowed_prompts:
            # Prefer listed spellings over aliases that render the same prompt.
            continue
        result["levels"].append(value)
    return result


def thinking_disabled(options: dict) -> bool:
    for name in ("enable_thinking", "thinking"):
        value = options.get(name)
        if isinstance(value, dict):
            value = value.get("type")
        if (value is False or value == 0 or isinstance(value, str)
                and value.strip().lower() in ("false", "off", "disabled")):
            return True
    effort = options.get("reasoning_effort")
    return isinstance(effort, str) and effort.strip().lower() == "none"
