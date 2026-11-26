# SPDX-License-Identifier: Apache-2.0
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    workload: Literal['pg19', 'file-edit'] = 'pg19'
    isl: list[Positive] = Field(default_factory=lambda: [1024, 8192, 32768, 131072, 225280], min_length=1, max_length=32)
    osl: Positive = Field(default=256, description='Output cap in tokens. Defaults to 256 for PG-19 and 1024 for File Edit.')
    conc: list[Positive] = Field(default_factory=lambda: [1], min_length=1, max_length=32)
    reps: Positive = 4
    warm: Annotated[int, Field(ge=0, strict=True)] = 1
    thinking: Literal['auto', 'on', 'off'] = 'auto'
    label: str = Field(default='', max_length=256)

    @model_validator(mode='before')
    @classmethod
    def workload_defaults(cls, values):
        if isinstance(values, dict) and values.get('workload') == 'file-edit' and 'osl' not in values:
            return {**values, 'osl': 1024}
        return values

    @field_validator('isl', 'conc')
    @classmethod
    def unique(cls, values):
        if len(set(values)) != len(values):
            raise ValueError('duplicate lengths or concurrency values')
        return values

    def arguments(self):
        return ['--workload', self.workload, '--isl', ','.join(map(str, self.isl)), '--osl', str(self.osl),
                '--conc', ','.join(map(str, self.conc)), '--reps', str(self.reps),
                '--warm', str(self.warm), '--thinking', self.thinking,
                '--label', self.label or self.model_id]
