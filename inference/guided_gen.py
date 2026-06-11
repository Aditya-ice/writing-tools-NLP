"""Schema-constrained generation for Smart Reply (outlines + Pydantic)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class SmartReply(BaseModel):
    subject: str
    body: str
    tone: Literal["formal", "casual", "friendly"]


def generate_smart_reply(prompt: str, model, tokenizer) -> SmartReply:
    """Generate a schema-valid SmartReply JSON object."""
    raise NotImplementedError
