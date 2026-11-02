# SPDX-License-Identifier: Apache-2.0
import base64
import binascii
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field


class ParseDocument(BaseModel):
    model: str | None = None
    filename: str = Field(min_length=1, max_length=255)
    file: str = Field(min_length=1)
    tier: Literal["basic", "standard", "advanced"] = "standard"
    page_range: str = ""


def configure(app):
    @app.post("/v1/documents/parse")
    async def parse_document(body: ParseDocument):
        from ..serve.state import serving

        runtime = getattr(serving().engine, "mineru", None)
        if runtime is None:
            raise HTTPException(400, "select a loaded MinerU recipe to parse documents")
        filename = Path(body.filename.replace("\\", "/")).name
        suffix = Path(filename).suffix.lower()
        if suffix not in (
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".tif",
            ".tiff",
            ".bmp",
        ):
            raise HTTPException(400, "MinerU document parsing accepts PDF and images")
        try:
            content = base64.b64decode(body.file, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(400, "file must be base64 encoded") from exc
        if not content:
            raise HTTPException(400, "file must not be empty")
        with tempfile.TemporaryDirectory(prefix="snowllm-mineru-") as temporary:
            path = Path(temporary) / ("input" + suffix)
            path.write_bytes(content)
            try:
                result = await runtime.parse(
                    path, tier=body.tier, page_range=body.page_range
                )
            except (ValueError, RuntimeError, MemoryError, OSError) as exc:
                raise HTTPException(
                    503 if isinstance(exc, MemoryError) else 400, str(exc)
                ) from exc
        return dict(model=serving().model_name, filename=filename, **result)
