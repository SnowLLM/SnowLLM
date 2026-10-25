# SPDX-License-Identifier: Apache-2.0

import copy

from transformers import PreTrainedTokenizerFast

from .encoding_v41 import _encode_messages_text, merge_tool_messages, sort_tool_results_by_call_order


class DeepSeekV41Tokenizer(PreTrainedTokenizerFast):
    thinking_controls = {"switch": True, "levels": ["low", "high", "max"], "source": "model_encoding"}

    def apply_chat_template(self, conversation, tools=None, chat_template=None,
                            add_generation_prompt=False, tokenize=True, padding=False,
                            truncation=False, max_length=None, return_tensors=None,
                            return_dict=False, **kwargs):
        batched = bool(conversation and isinstance(conversation[0], (list, tuple)))
        conversations = conversation if batched else [conversation]
        enabled = kwargs.get("enable_thinking", kwargs.get("thinking", True))
        effort = kwargs.get("reasoning_effort")
        if effort == "none":
            enabled, effort = False, None
        if effort is not None and not (type(effort) is int and 1 <= effort <= 100
                                       or type(effort) is str and effort in ("low", "high", "max")):
            raise ValueError("DeepSeek V4.1 reasoning_effort must be 1–100, low, high, max or none")
        prompts = []
        for messages in conversations:
            messages = copy.deepcopy(messages)
            for message in messages:
                content = message.get("content")
                if isinstance(content, list):
                    if any(block.get("type") != "text" for block in content):
                        raise ValueError("DeepSeek V4.1 currently accepts text content only")
                    message["content"] = "\n\n".join(block.get("text", "") for block in content)
                if "<｜deepseek_image｜>" in (message.get("content") or ""):
                    raise ValueError("DeepSeek V4.1 vision is not supported yet")
            if tools:
                if not messages or messages[0]["role"] != "system":
                    messages.insert(0, {"role": "system", "content": ""})
                messages[0]["tools"] = tools
            messages = sort_tool_results_by_call_order(merge_tool_messages(messages))
            prompt = _encode_messages_text(messages, "thinking" if enabled else "chat",
                drop_thinking=not kwargs.get("preserve_thinking", False), reasoning_effort=effort)
            if not add_generation_prompt:
                for suffix in ("<｜Assistant｜><think>", "<｜Assistant｜></think>"):
                    if prompt.endswith(suffix):
                        prompt = prompt[:-len(suffix)]
                        break
            prompts.append(prompt)
        if not tokenize:
            return prompts if batched else prompts[0]
        encoded = self(prompts if batched else prompts[0], add_special_tokens=False,
            padding=padding, truncation=truncation, max_length=max_length, return_tensors=return_tensors)
        return encoded if return_dict else encoded["input_ids"]
