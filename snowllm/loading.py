# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, ConfigDict, Field

from .targets import Target


MODEL_LOADING_GUIDE = 'https://snowllm.dev/docs/webui#loading-multiple-models'


class ModelLoad(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str = Field(min_length=1)
    options: list[str] = Field(default_factory=list)
    id: str | None = Field(default=None, min_length=1, max_length=256)
    target: Target | None = None


class LoadBatch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    models: list[ModelLoad] = Field(min_length=1)
