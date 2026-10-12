# SPDX-License-Identifier: Apache-2.0
"""CPU-only template capability tests; no model weights or GPU imports."""
import unittest
from jinja2 import Environment, TemplateError
from snowllm.serve.thinking import thinking_capabilities


class Tokenizer:
    def __init__(self, template):
        self.chat_template = template

    def get_chat_template(self):
        return self.chat_template

    def apply_chat_template(self, messages, **kwargs):
        def reject(message):
            raise TemplateError(message)
        return Environment().from_string(self.chat_template).render(
            messages=messages, raise_exception=reject, **kwargs)


class ThinkingTests(unittest.TestCase):
    def test_unknown_model_levels_and_validation(self):
        template = """{% set allowed = ['tiny', 'future', 'huge'] %}
        {% if reasoning_effort is defined and reasoning_effort not in allowed %}
          {{ raise_exception('invalid') }}
        {% endif %}
        {% if reasoning_effort == 'invalid' %}invalid{% endif %}
        {{ reasoning_effort | default('future') }}
        {% if enable_thinking | default(true) %}think{% else %}answer{% endif %}"""
        self.assertEqual(thinking_capabilities(Tokenizer(template)), {
            'switch': True, 'levels': ['tiny', 'future', 'huge'], 'source': 'chat_template'})

    def test_switch_only_and_no_false_positive(self):
        self.assertTrue(thinking_capabilities(Tokenizer(
            '{% if thinking %}think{% else %}answer{% endif %}'))['switch'])
        self.assertFalse(thinking_capabilities(Tokenizer(
            '{% if enable_thinking %}{% set unused = true %}{% endif %}answer'))['switch'])
        self.assertEqual(thinking_capabilities(Tokenizer('plain chat'))['levels'], [])

if __name__ == '__main__':
    unittest.main()
