# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from pydantic import BaseModel, Field


class StreamOptions(BaseModel):
    include_usage: bool = False


class Common(BaseModel):
    model: str | None = None
    max_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(1.0, gt=0.0)
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    seed: int | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict] | None = ""
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


THINK_FIELDS = ("enable_thinking", "thinking", "reasoning_effort", "preserve_thinking")


class ChatRequest(Common):
    messages: list[ChatMessage]
    max_completion_tokens: int | None = None
    chat_template_kwargs: dict = Field(default_factory=dict)
    enable_thinking: bool | None = None
    thinking: object = None
    reasoning_effort: object = None
    preserve_thinking: object = None
    tools: list[dict] | None = None
    tool_choice: str | dict = "auto"

    def template_kwargs(self) -> dict:
        kw = dict(self.chat_template_kwargs)
        for f in THINK_FIELDS:
            v = getattr(self, f, None)
            if isinstance(v, dict):
                v = v.get("type") == "enabled"
            if v is not None:
                kw.setdefault(f, v)
        if str(kw.get("reasoning_effort", "")).lower() == "none" \
                and "enable_thinking" not in kw and "thinking" not in kw:
            kw["enable_thinking"] = kw["thinking"] = False
        return kw


class CompletionRequest(Common):
    prompt: str | list[str]


class ResponsesRequest(Common):
    input: str | list[dict]
    instructions: str | None = None
    max_output_tokens: int | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict = "auto"
    parallel_tool_calls: bool = True
    chat_template_kwargs: dict = Field(default_factory=dict)
    store: bool | None = None
    reasoning: dict | None = None
    text: dict | None = None
    previous_response_id: str | None = None
    background: bool | None = None
    metadata: dict | None = Field(default=None)
