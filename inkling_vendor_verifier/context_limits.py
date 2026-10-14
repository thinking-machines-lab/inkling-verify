"""Recognize explicit context bounds without estimating or truncating prompts."""

import re

OUTPUT_TOKEN_FIELDS = {
    "openai-completions": "max_completion_tokens",
    "openai-responses": "max_output_tokens",
    "anthropic": "max_tokens",
}

_TOTAL_LIMIT = re.compile(
    r"Requested token count exceeds the model's maximum context length of "
    r"(?P<context>[1-9][0-9]*) tokens\. You requested a total of "
    r"(?P<total>[1-9][0-9]*) tokens: (?P<input>[0-9]+) tokens from the input messages "
    r"and (?P<requested>[1-9][0-9]*) tokens for the completion\. Please reduce the number "
    r"of tokens in the input messages or the completion to fit within the limit\."
)
_INPUT_LIMIT = re.compile(
    r"The input \((?P<input>[0-9]+) tokens\) is longer than the "
    r"model's context length \((?P<context>[1-9][0-9]*) tokens\)\."
)


class ContextExhausted(RuntimeError):
    def __init__(self, input_tokens: int, context_limit: int) -> None:
        self.input_tokens = input_tokens
        self.context_limit = context_limit
        super().__init__(
            f"No output capacity remains: {input_tokens} input tokens, "
            f"{context_limit} context limit"
        )


def remaining_output_tokens(body: object, requested: int) -> int | None:
    if not isinstance(body, dict):
        return None
    error = body.get("error", body)
    if not isinstance(error, dict) or not isinstance(message := error.get("message"), str):
        return None
    if match := _INPUT_LIMIT.fullmatch(message):
        input_tokens, context = int(match["input"]), int(match["context"])
        if input_tokens >= context:
            raise ContextExhausted(input_tokens, context)
        return None
    match = _TOTAL_LIMIT.fullmatch(message)
    if match is None:
        return None
    context, total, input_tokens, completion = (
        int(match[key]) for key in ("context", "total", "input", "requested")
    )
    if (
        completion != requested
        or total != input_tokens + completion
        or total <= context
        or input_tokens >= context
    ):
        return None
    # This server error comes from SGLang, which reserves one final context position.
    maximum = context - input_tokens - 1
    if maximum <= 0:
        raise ContextExhausted(input_tokens, context)
    return maximum
