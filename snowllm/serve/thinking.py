# SPDX-License-Identifier: Apache-2.0
"""Discover controls from the loaded tokenizer's actual Jinja chat template.

There is no standard thinking-level field in model config.json. Read template
expressions instead of maintaining a model-name registry. Only advertise values
that the template both names and accepts when rendering a user message.
"""
from jinja2 import Environment, TemplateError, nodes


def thinking_capabilities(tokenizer) -> dict:
    controls = getattr(tokenizer, "thinking_controls", None)
    if controls is not None:
        return dict(controls)
    result = {"switch": False, "levels": [], "source": "chat_template"}
    try:
        template = tokenizer.get_chat_template()
        tree = Environment(extensions=["jinja2.ext.loopcontrols"]).parse(template)
    except (ValueError, TypeError, TemplateError, AttributeError):
        return result

    constants = {}
    for assignment in tree.find_all(nodes.Assign):
        if isinstance(assignment.target, nodes.Name):
            constants[assignment.target.name] = assignment.node

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

    def effort(node):
        if isinstance(node, nodes.Filter) and node.name == "default":
            return effort(node.node)
        return isinstance(node, nodes.Name) and node.name == "reasoning_effort"

    candidates = []
    for comparison in tree.find_all(nodes.Compare):
        left = comparison.expr
        for operand in comparison.ops:
            if effort(left):
                candidates.extend(literals(operand.expr))
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

    def accepts(**kwargs):
        try:
            render(**kwargs)
            return True
        except (ValueError, TypeError, TemplateError):
            return False

    names = {node.name for node in tree.find_all(nodes.Name)}
    if names & {"enable_thinking", "thinking"}:
        try:
            result["switch"] = render(enable_thinking=True, thinking=True) != render(
                enable_thinking=False, thinking=False)
        except (ValueError, TypeError, TemplateError):
            pass
    result["levels"] = [value for value in dict.fromkeys(candidates)
                        if accepts(reasoning_effort=value, enable_thinking=True, thinking=True)]
    return result
