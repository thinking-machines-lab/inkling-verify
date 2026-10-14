"""Exercise benchmark logic through each real SDK without contacting a model server."""

from __future__ import annotations

import base64
import hashlib
import json
from contextlib import AsyncExitStack
from functools import partial
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from inkling_vendor_verifier import code_agent, python_tools
from inkling_vendor_verifier.benchmarks import hle_no_tools as hle
from inkling_vendor_verifier.benchmarks import mmmu_pro as mmmu
from inkling_vendor_verifier.benchmarks import niah
from inkling_vendor_verifier.benchmarks.deepswe import DeepSWE
from inkling_vendor_verifier.runner import run
from inkling_vendor_verifier.sandbox import CommandResult
from inkling_vendor_verifier.types import Example, Generation, Message, Text, ToolCall
from inkling_vendor_verifier.types import Image as DisplayImage

APIS = ("openai-completions", "openai-responses", "anthropic")
GENERATION = Generation(131072, temperature=None, top_p=None)


@pytest.fixture
def reply(response_body):
    return partial(response_body, cached=False)


def conversation(body, api):
    return body["input"] if api == "openai-responses" else body["messages"]


def user_text(body, api):
    content = conversation(body, api)[-1]["content"]
    if isinstance(content, str):
        return content
    return "".join(part["text"] for part in content if "text" in part)


@pytest.mark.parametrize("api", APIS)
async def test_niah_uses_same_prompt_and_visible_answer_across_apis(
    sdk_provider, reply, tmp_path, api
):
    target = "77986351-01ca-4fe3-af26-9e2c7b8b6d32"
    prompt = niah.build_prompt([("key", target)], "key").encode()
    path = tmp_path / "prompt.txt"
    path.write_bytes(prompt)
    example = Example(
        "niah_uuid_key_to_value_2000_50_1234",
        (),
        target,
        {
            "prompt_path": str(path),
            "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
            "context_length": 2000,
            "depth_percent": 50,
            "sample_index": 0,
            "prompt_tokens": 70,
            "target_key": "key",
        },
    )
    provider, requests = sdk_provider(api, [reply(api, f"The value is:\n{target.upper()}")])
    generation = Generation(128, 0.0, None, "none")
    output = tmp_path / "results"
    configuration = {"dataset": {"fixture": "niah"}}
    async with provider:
        for resume in (False, True):
            summary = await run(
                niah.NIAH(),
                [example],
                provider,
                generation,
                output,
                configuration=configuration,
                concurrency=1,
                resume=resume,
            )
            assert summary["accuracy"] == 1
        path.write_bytes(b"corrupted")
        with pytest.raises(ValueError, match="checksum"):
            await niah.NIAH().evaluate(example, provider, generation, tmp_path)
    result = json.loads(next(output.glob("samples/*/result.json")).read_text())
    assert result["score"] == 1 and result["answer"] == target.upper()
    assert len(list(output.glob("samples/*/attempt-*/call-*.json"))) == 1
    niah.write_grid(output)
    assert json.loads((output / "grid.json").read_text())[0]["accuracy"] == 1
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert user_text(body, api) == prompt.decode()
    assert "tools" not in body and "top_p" not in body
    assert body["temperature"] == 0
    if api == "openai-completions":
        assert body["reasoning_effort"] == "none" and body["max_completion_tokens"] == 128
    elif api == "openai-responses":
        assert body["reasoning"] == {"effort": "none"} and body["max_output_tokens"] == 128
    else:
        assert body["thinking"] == {"type": "disabled"} and body["max_tokens"] == 128


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize(
    "judge_api, generation, judge_generation, answer",
    [
        (None, Generation(131072, reasoning_effort="max"), None, "Exact Answer: 4"),
        (None, GENERATION, None, ""),
    ]
    + [
        (judge, GENERATION, settings, "Exact Answer: 4")
        for judge in APIS
        for settings in (None, Generation(50, 0.5, 0.9, "low"))
    ],
)
async def test_hle_generates_and_judges_across_apis(
    sdk_provider, reply, tmp_path, api, judge_api, generation, judge_generation, answer
):
    verdict = {
        "extracted_final_answer": "4",
        "reasoning": "The answers match.",
        "correct": "yes",
        "confidence": 100,
        "strict": True,
    }
    responses = [reply(api, answer)]
    if judge_api is None:
        responses.append(reply(api, json.dumps(verdict)))
    candidate, candidate_requests = sdk_provider(
        api,
        responses,
        model="candidate",
        base_url="https://candidate.test/v1/",
        api_key="candidate-key",
    )
    example = Example(
        "math",
        (
            Message("system", (Text(hle.SYSTEM_PROMPTS["exactMatch"]),)),
            Message("user", (Text("What is two plus two?"),)),
        ),
        "4",
        {"question": "What is two plus two?"},
    )
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(candidate)
        judge = None
        judge_requests = candidate_requests
        if judge_api is not None:
            judge, judge_requests = sdk_provider(
                judge_api,
                [reply(judge_api, json.dumps(verdict))],
                model="judge",
                base_url="https://judge.test/v1/",
                api_key="judge-key",
            )
            await stack.enter_async_context(judge)
        result = await hle.HLE(judge_generation).evaluate(
            example, candidate, generation, tmp_path, judge=judge
        )
    expected_judge = judge_generation or (
        Generation(262144, None, None, "high")
        if judge_api is None and generation.temperature is None
        else hle.JUDGE_GENERATION
    )
    if judge_generation is None:
        assert expected_judge.reasoning_effort == "high"
        assert expected_judge.max_tokens == 262144
    assert result.score == 1 and result.answer == "4"
    assert result.details["verdict"] == verdict
    assert result.details["judge_temperature"] == expected_judge.temperature
    assert result.details["judge_top_p"] == expected_judge.top_p
    assert len(candidate_requests) == (2 if judge_api is None else 1)
    assert len(judge_requests) == (2 if judge_api is None else 1)
    candidate_request, judge_request = candidate_requests[0], judge_requests[-1]
    assert candidate_request.url.host == "candidate.test"
    judge_name = "candidate" if judge_api is None else "judge"
    assert judge_request.url.host == f"{judge_name}.test"
    assert "candidate-key" in " ".join(candidate_request.headers.values())
    assert f"{judge_name}-key" in " ".join(judge_request.headers.values())
    if judge_api is not None:
        assert "candidate-key" not in " ".join(judge_request.headers.values())
    candidate_body = json.loads(candidate_request.content)
    judge_body = json.loads(judge_request.content)
    assert candidate_body["model"] == "candidate"
    assert judge_body["model"] == judge_name
    system = (
        candidate_body["system"][0]["text"]
        if api == "anthropic"
        else conversation(candidate_body, api)[0]["content"]
    )
    assert system == hle.SYSTEM_PROMPTS["exactMatch"]
    assert user_text(candidate_body, api) == example.metadata["question"]
    assert user_text(judge_body, judge_api or api) == hle.JUDGE_PROMPT.format(
        question=example.metadata["question"], response=answer, correct_answer="4"
    )
    for private_state in ("native reasoning", "opaque reasoning", "opaque signature"):
        assert private_state not in json.dumps(judge_body)
    judge_api = judge_api or api
    messages = conversation(judge_body, judge_api)
    assert [message["role"] for message in messages] == (
        ["user"] if judge_api == "anthropic" else ["system", "user"]
    )
    system = judge_body["system"][0]["text"] if judge_api == "anthropic" else messages[0]["content"]
    assert system == hle.SCHEMA_SYSTEM_INSTRUCTION
    schema = (
        judge_body["response_format"]["json_schema"]["schema"]
        if judge_api == "openai-completions"
        else judge_body["text"]["format"]["schema"]
        if judge_api == "openai-responses"
        else judge_body["output_config"]["format"]["schema"]
    )
    assert schema == hle.JUDGE_SCHEMA
    for body, protocol, settings in (
        (candidate_body, api, generation),
        (judge_body, judge_api, expected_judge),
    ):
        assert "tools" not in body
        for field in ("temperature", "top_p"):
            value = getattr(settings, field)
            if value is None:
                assert field not in body
            else:
                assert body[field] == value
        if protocol == "openai-completions":
            assert body["reasoning_effort"] == settings.reasoning_effort
            assert body["max_completion_tokens"] == settings.max_tokens
        elif protocol == "openai-responses":
            assert body["reasoning"] == {"effort": settings.reasoning_effort}
            assert body["max_output_tokens"] == settings.max_tokens
        else:
            assert body["max_tokens"] == settings.max_tokens
            if settings.reasoning_effort == "none":
                assert body["thinking"] == {"type": "disabled"}
            else:
                assert body["thinking"] == {"type": "adaptive"}
                assert body["output_config"]["effort"] == settings.reasoning_effort


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize(
    "answer, expected, method",
    [("The shape is a rectangle.\nAnswer: B", "B", "answer_prefix"), ("", "", "empty")],
)
async def test_mmmu_sends_original_png_and_scores_shared_parser(
    sdk_provider, reply, tmp_path, api, answer, expected, method
):
    image_path = tmp_path / "question.png"
    Image.new("RGB", (3, 2), color=(9, 18, 27)).save(image_path)
    image_bytes = image_path.read_bytes() + (bytes(range(256)) * 43000 if answer else b"")
    image_path.write_bytes(image_bytes)
    example = Example(
        "vision",
        (),
        "B",
        {
            "image_path": str(image_path),
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "options": {"A": "triangle", "B": "rectangle"},
        },
    )
    provider, requests = sdk_provider(
        api,
        [reply(api, answer)],
        model="candidate",
        base_url="https://candidate.test/v1/",
        api_key="test-key",
    )
    async with provider:
        result = await mmmu.MMMUPro().evaluate(example, provider, GENERATION, tmp_path)
        image_path.write_bytes(b"changed")
        with pytest.raises(ValueError, match="checksum"):
            await mmmu.MMMUPro().evaluate(example, provider, GENERATION, tmp_path)
    assert result.score == (expected == "B") and result.answer == expected
    assert result.details["parse_method"] == method
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert "tools" not in body
    parts = conversation(body, api)[0]["content"]
    assert len(parts) == 2
    assert parts[1]["text"] == mmmu.PROMPT
    if api == "openai-completions":
        encoded = parts[0]["image_url"]["url"].split(",", 1)[1]
    elif api == "openai-responses":
        encoded = parts[0]["image_url"].split(",", 1)[1]
    else:
        encoded = parts[0]["source"]["data"]
    assert base64.b64decode(encoded) == image_bytes


class Sandbox:
    config = {"agent": {"network_mode": "no-network"}, "verifier": {"network_mode": "no-network"}}

    def __init__(self):
        self.commands = []
        self.graded = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def run(self, command, *, workdir, timeout):
        self.commands.append((command, workdir, timeout))
        return CommandResult("/repo" if command == "pwd" else "fixed", "", 0, 0.01)

    async def grade(self):
        self.graded = True
        return 1.0, {"verifier_exit_code": 0}


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("tool_text", ["I will inspect the repository.", None])
async def test_deepswe_native_tool_loop_and_grading(
    monkeypatch, sdk_provider, reply, tmp_path, api, tool_text
):
    sandbox = Sandbox()
    monkeypatch.setattr(code_agent, "DockerSandbox", lambda *args, **kwargs: sandbox)
    first_response = reply(api, tool_text, tools=("pwd", "printf fixed"))
    provider, requests = sdk_provider(
        api,
        [first_response, reply(api, "Fixed.")],
        model="candidate",
        base_url="https://candidate.test/v1/",
        api_key="test-key",
    )
    example = Example(
        "fix",
        (Message("user", (Text("Fix the defect."),)),),
        "",
        {
            "task_dir": str(tmp_path),
            "suite": "deepswe",
        },
    )
    async with provider:
        result = await DeepSWE().evaluate(example, provider, GENERATION, tmp_path)
    assert result.score == 1 and result.answer == "Fixed."
    assert result.details["turns"] == 2 and result.details["tool_calls"] == 2
    assert result.details["generated_tokens"] == 10
    assert sandbox.commands == [("pwd", None, 300), ("printf fixed", None, 300)]
    assert sandbox.graded
    bodies = [json.loads(request.content) for request in requests]
    assert len(bodies) == 2
    expected_tool = code_agent.bash_tool()
    for body in bodies:
        tool = body["tools"][0]
        if api == "openai-completions":
            tool = tool["function"]
        assert tool["name"] == expected_tool.name
        assert tool["description"] == expected_tool.description
        assert (
            tool["input_schema" if api == "anthropic" else "parameters"] == expected_tool.parameters
        )
    messages = conversation(bodies[1], api)
    if api == "openai-completions":
        assert messages[1] == first_response["choices"][0]["message"]
        results = [(message["tool_call_id"], message["content"]) for message in messages[2:]]
        budget = bodies[1]["max_completion_tokens"]
    elif api == "openai-responses":
        end = 1 + len(first_response["output"])
        assert messages[1:end] == first_response["output"]
        results = [(message["call_id"], message["output"]) for message in messages[end:]]
        budget = bodies[1]["max_output_tokens"]
    else:
        assert messages[1]["content"] == first_response["content"]
        assert len(messages) == 3
        results = [(block["tool_use_id"], block["content"]) for block in messages[2]["content"]]
        budget = bodies[1]["max_tokens"]
    assert budget == GENERATION.max_tokens - 5
    assert [call_id for call_id, _ in results] == ["call_1", "call_2"]
    assert [json.loads(content)["stdout"] for _, content in results] == ["/repo", "fixed"]


@pytest.mark.parametrize("api", APIS)
async def test_hle_invalid_judge_is_error_across_protocols(sdk_provider, reply, tmp_path, api):
    provider, requests = sdk_provider(
        api,
        [reply(api, "4"), reply(api, "not JSON")],
        model="candidate",
        base_url="https://candidate.test/v1/",
        api_key="test-key",
    )
    example = Example(
        "bad-grade",
        (Message("user", (Text("Two plus two?"),)),),
        "4",
        {
            "question": "Two plus two?",
        },
    )
    async with provider:
        with pytest.raises(ValueError, match="invalid ExtractedAnswer"):
            await hle.HLE().evaluate(example, provider, GENERATION, tmp_path)
    assert len(requests) == 2


@pytest.mark.parametrize("api", APIS)
async def test_deepswe_nul_command_returns_tool_error_and_continues(
    monkeypatch, sdk_provider, reply, tmp_path, api
):
    sandbox = Sandbox()
    monkeypatch.setattr(code_agent, "DockerSandbox", lambda *args, **kwargs: sandbox)
    invalid = "printf 'prefix\x00suffix'"
    provider, requests = sdk_provider(
        api, [reply(api, None, tools=(invalid, "pwd")), reply(api, "Fixed.")]
    )
    example = Example("nul", (), "", {"task_dir": str(tmp_path), "suite": "deepswe"})
    async with provider:
        result = await DeepSWE().evaluate(example, provider, GENERATION, tmp_path)
    assert sandbox.commands == [("pwd", None, 300)]
    assert sandbox.graded and result.score == 1
    assert result.details["turns"] == 2 and result.details["tool_calls"] == 2
    second_request = json.loads(requests[1].content)
    assert "command contains a NUL character; command was not executed" in json.dumps(
        second_request
    )


@pytest.mark.parametrize("field", ["command", "workdir"])
async def test_bash_rejects_nul_arguments_without_changing_valid_escapes(field):
    sandbox = Sandbox()
    args = {"command": r"printf '\\0'", "workdir": "/repo"}
    original = args.copy()
    args[field] += "\x00"
    result = await code_agent.run_bash(sandbox, json.dumps(args))
    assert json.loads(result[0].text) == {
        "error": f"{field} contains a NUL character; command was not executed"
    }
    assert sandbox.commands == []
    await code_agent.run_bash(sandbox, json.dumps(original))
    assert sandbox.commands == [(original["command"], "/repo", 300)]


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("subset", ["standard-4", "standard-10"])
async def test_mmmu_standard_interleaved_images_across_apis(
    sdk_provider, reply, tmp_path, api, subset
):
    image_path = tmp_path / "question.png"
    Image.new("RGB", (3, 2), color=(9, 18, 27)).save(image_path)
    image_bytes = image_path.read_bytes()
    example = Example(
        "standard",
        (Message("user", (Text("First <image 2> then <image 2> choose."),)),),
        "B",
        {
            "images": {
                "image_2": {
                    "image_path": str(image_path),
                    "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                }
            },
            "options": {"A": "triangle", "B": "rectangle"},
        },
    )
    provider, requests = sdk_provider(api, [reply(api, "Answer: B")])
    async with provider:
        result = await mmmu.MMMUPro(subset).evaluate(example, provider, GENERATION, tmp_path)
    assert result.score == 1
    parts = conversation(json.loads(requests[0].content), api)[0]["content"]
    assert [part["text"] for part in parts if "text" in part] == ["First ", " then ", " choose."]
    for index in (1, 3):
        if api == "openai-completions":
            encoded = parts[index]["image_url"]["url"].split(",", 1)[1]
        elif api == "openai-responses":
            encoded = parts[index]["image_url"].split(",", 1)[1]
        else:
            encoded = parts[index]["source"]["data"]
        assert base64.b64decode(encoded) == image_bytes


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("subset", mmmu.SUBSETS)
async def test_mmmu_tools_multi_turn_images_scoring_and_resume(
    monkeypatch, sdk_provider, reply, tmp_path, api, subset
):
    image_path = tmp_path / "question.png"
    Image.new("RGB", (3, 2), color=(9, 18, 27)).save(image_path)
    image_bytes = image_path.read_bytes()
    image_record = {
        "image_path": str(image_path),
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
    }
    example = Example(
        "with-tools",
        (Message("user", (Text("Read <image 1>."),)),),
        "B",
        {"options": {"A": "triangle", "B": "rectangle"}}
        | (image_record if subset == "vision" else {"images": {"image_1": image_record}}),
    )
    calls = [
        ToolCall("crop", "ipython", json.dumps({"code": "display(image)"})),
        ToolCall("measure", "ipython", json.dumps({"code": "image.size"})),
    ]
    executed, closed = [], []

    class PythonSandbox(python_tools.PythonSandbox):
        async def __aenter__(self):
            self.work_dir.mkdir(parents=True, exist_ok=True)
            self._docker = AsyncMock()
            return self

        async def __aexit__(self, *args):
            closed.append(True)

        async def execute(self, arguments):
            executed.append(arguments)
            return (
                (DisplayImage(image_bytes, "image/png"),)
                if len(executed) == 1
                else (Text("(3, 2)"),)
            )

    monkeypatch.setattr(python_tools, "PythonSandbox", PythonSandbox)
    provider, requests = sdk_provider(
        api,
        [
            reply(api, None, tools=(calls[0],)),
            reply(api, None, tools=(calls[1],)),
            reply(api, "Answer: B"),
        ],
    )
    benchmark = mmmu.MMMUPro(subset, with_tools=True)
    output = tmp_path / "results"
    generation = Generation(131075)
    async with provider:
        for resume in (False, True):
            summary = await run(
                benchmark,
                [example],
                provider,
                generation,
                output,
                configuration={"dataset": {"fixture": "mmmu"}},
                concurrency=1,
                resume=resume,
            )
            assert summary["accuracy"] == 1
        with pytest.raises(ValueError, match="Resume requires identical"):
            await run(
                mmmu.MMMUPro(subset),
                [example],
                provider,
                generation,
                output,
                configuration={"dataset": {"fixture": "mmmu"}},
                concurrency=1,
                resume=True,
            )
    assert executed == [call.arguments for call in calls] and closed == [True]
    result = json.loads(next(output.glob("samples/*/result.json")).read_text())
    assert result["answer"] == "B"
    assert result["details"] == {
        "parse_method": "answer_prefix",
        "stop_reason": "stop",
        "turns": 3,
        "tool_calls": 2,
        "generated_tokens": 15,
        "with_tools": True,
        "allow_network": False,
        "token_budget_kind": "cumulative_generated_tokens",
    }
    assert len(list(output.glob("samples/*/attempt-*/call-*.json"))) == 3
    bodies = [json.loads(request.content) for request in requests]
    budget_key = {
        "openai-completions": "max_completion_tokens",
        "openai-responses": "max_output_tokens",
        "anthropic": "max_tokens",
    }[api]
    assert [body[budget_key] for body in bodies] == [131072, 131070, 131065]
    first = bodies[0]
    system = (
        first["system"][0]["text"] if api == "anthropic" else conversation(first, api)[0]["content"]
    )
    assert system == python_tools.SYSTEM_PROMPT
    assert "/sandbox/images/image_0.png" in json.dumps(first)
    for body in bodies:
        tool = body["tools"][0]
        assert (tool["function"] if api == "openai-completions" else tool)["name"] == "ipython"
    encoded = base64.b64encode(image_bytes).decode()
    assert [json.dumps(body).count(encoded) for body in bodies] == [1, 2, 2]
    assert "(3, 2)" in json.dumps(bodies[2])
