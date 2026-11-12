# SPDX-License-Identifier: Apache-2.0
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Positive = Annotated[int, Field(gt=0, strict=True)]


def sizes(value):
    result = []
    for item in value.split(','):
        match = re.fullmatch(r'\s*(\d+)\s*([kKmM]?)\s*', item)
        if not match:
            raise ValueError('use comma-separated lengths such as 1k,8k,32k')
        result.append(int(match[1]) * {'': 1, 'k': 1024, 'm': 1024**2}[match[2].lower()])
    return result


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model_id: str = Field(min_length=1, max_length=256)
    isl: list[Positive] = Field(default_factory=lambda: [1024, 8192, 32768, 131072, 225280], min_length=1, max_length=32)
    osl: Positive = 256
    conc: list[Positive] = Field(default_factory=lambda: [1], min_length=1, max_length=32)
    reps: Positive = 4
    warm: Annotated[int, Field(ge=0, strict=True)] = 1
    thinking: Literal['auto', 'on', 'off'] = 'auto'
    label: str = Field(default='', max_length=256)

    @field_validator('isl', 'conc')
    @classmethod
    def unique(cls, values):
        if len(set(values)) != len(values):
            raise ValueError('duplicate lengths or concurrency values')
        return values

    def arguments(self):
        return ['--isl', ','.join(map(str, self.isl)), '--osl', str(self.osl),
                '--conc', ','.join(map(str, self.conc)), '--reps', str(self.reps),
                '--warm', str(self.warm), '--thinking', self.thinking,
                '--label', self.label or self.model_id]
