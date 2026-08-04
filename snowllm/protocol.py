# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from pydantic import BaseModel, Field


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
    seed: int | None = None


class ChatMessage(BaseModel):
    role: str
    content: "str | list[dict] | None" = ""
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatRequest(Common):
    messages: list[ChatMessage]
    chat_template_kwargs: dict = Field(default_factory=dict)
    tools: list[dict] | None = None
    tool_choice: str | dict = "auto"


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
