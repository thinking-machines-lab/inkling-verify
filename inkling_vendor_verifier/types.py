"""The shared boundary between benchmarks and API protocols."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

Api = Literal["openai-completions", "openai-responses", "anthropic"]


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Image:
    data: bytes
    media_type: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: tuple[Text | Image, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    native: dict[str, Any] | None = None

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.content if isinstance(part, Text))


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class Generation:
    max_tokens: int
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    reasoning_effort: Literal["none", "low", "medium", "high", "max"] | None = "max"

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0


@dataclass(frozen=True)
class Response:
    message: Message
    usage: Usage
    stop_reason: Literal["stop", "tool_calls", "length", "refusal"]
    raw: dict[str, Any]
    request: dict[str, Any]

    @property
    def text(self) -> str:
        return self.message.text


class Provider(Protocol):
    async def generate(
        self,
        messages: Sequence[Message],
        generation: Generation,
        *,
        tools: Sequence[Tool] = (),
        response_schema: dict[str, Any] | None = None,
    ) -> Response: ...


@dataclass(frozen=True)
class Example:
    id: str
    messages: tuple[Message, ...]
    target: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evaluation:
    score: float
    answer: str
    details: dict[str, Any] = field(default_factory=dict)


class Benchmark(Protocol):
    name: str
    cache_name: str

    def load(self, data_dir: Path) -> list[Example]: ...

    async def evaluate(
        self,
        example: Example,
        provider: Provider,
        generation: Generation,
        work_dir: Path,
        *,
        judge: Provider | None = None,
    ) -> Evaluation: ...
