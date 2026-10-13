# Inkling vendor verifier

Run pinned evaluation benchmarks against OpenAI Chat Completions, OpenAI
Responses, or Anthropic Messages endpoints. Benchmark prompts, tool loops and
grading are shared across APIs. Each API adapter is shared across all benchmarks.

| Script | Subsets / cases | Grading |
| --- | --- | --- |
| `swe_bench.py` | Signal (250), full Verified (500), Pro v1 (731), Multilingual (300), DeepSWE v1.1 (113) | Each task's executable tests after native bash tool calls |
| `mmmu_pro.py` | Vision, Standard-4, Standard-10 (1,730 each) | Local multiple-choice answer extraction |
| `hle.py` | Text-only HLE (2,158) | Structured JSON judgment, with a configurable judge |
| `niah.py` | UUID needle retrieval: 5 context lengths × 11 depths × 10 samples | Case-insensitive UUID substring in the visible answer |

Defaults remain SWE Signal, MMMU-Pro Vision, and HLE self-judging. HLE
self-judging is a diagnostic workload, not an official HLE score from an
independent judge. This repository runs native API conversations: it does not
reproduce a particular server's private renderer or raw-token endpoint.
Assistant history replays the original API response, preserving reasoning and
tool-call state; manually constructed assistant history is unsupported.

## Install

Python 3.12 or newer on Linux or macOS is required. SWE also needs a local Linux
Docker daemon with access to the task image registries (Docker Hub and, for
DeepSWE, public ECR). Its task images
target Linux x86-64; use an x86-64 host for benchmark runs. No GPUs are required
on the client host.

```bash
git clone https://github.com/thinking-machines-lab/inkling-vendor-verifier.git
cd inkling-vendor-verifier
python -m pip install -r requirements.txt
python swe_bench.py --help
```

Each benchmark has its own script with `prepare` and `run` subcommands. Run the
scripts from the checkout; installing the project is optional.

For an editable installation, use:

```bash
python -m pip install -e .
swe-bench --help
mmmu-pro --help
hle --help
niah --help
```

The installed commands accept the same arguments as their corresponding Python
scripts. Both installation methods use the dependencies in `requirements.txt`.

## Prepare the data

```bash
python mmmu_pro.py prepare --subset vision
python mmmu_pro.py prepare --subset standard-4
python mmmu_pro.py prepare --subset standard-10

# Download the full public source datasets, independently of the Signal bundle.
python swe_bench.py prepare --subset verified
python swe_bench.py prepare --subset pro
python swe_bench.py prepare --subset multilingual
python swe_bench.py prepare --subset deepswe

# Accept the cais/hle dataset terms on Hugging Face and authenticate first.
export HF_TOKEN=...
python hle.py prepare

# Generate NIAH using the model's Hugging Face tokenizer.json.
python niah.py prepare --tokenizer MODEL_OR_DIRECTORY

# Obtain the pinned task bundle from this repository's swe-bench-signal-v1 release.
mkdir -p data
gh release download swe-bench-signal-v1 \
  --repo thinking-machines-lab/inkling-vendor-verifier \
  --pattern 'swe-bench-signal-v1.tar.gz' --dir data
python swe_bench.py prepare \
  --bundle data/swe-bench-signal-v1.tar.gz
```

Preparation writes `data/<benchmark>/manifest.json` and checks pinned revisions,
case counts and checksums. HLE retains its existing `data/hle-self-judge` cache.
Downloads and results stay outside Git. HLE requires
authorized dataset access; downloaded questions, answers and result logs must
be handled under its dataset terms.

MMMU-Pro uses the **test** split for each selected subset at revision
[`1ba55708b8588a8f9b180b8fec9e6435c88ce363`](https://huggingface.co/datasets/MMMU/MMMU_Pro/tree/1ba55708b8588a8f9b180b8fec9e6435c88ce363).
The standard subsets interleave question and option text with their referenced
images. `standard-4` and `standard-10` select the upstream configurations named
`standard (4 options)` and `standard (10 options)`; individual questions can have
different option counts. Images retain their original dimensions and pixels
through lossless PNG encoding. The API adapters
send those prepared PNG bytes unchanged. Some images exceed 10 MB before base64
encoding; an endpoint that cannot accept them fails explicitly, without resizing.

HLE uses the **text-only** test rows at revision
[`5a81a4c7271a2a2a312b9a690f0c2fde837e4c29`](https://huggingface.co/datasets/cais/hle/tree/5a81a4c7271a2a2a312b9a690f0c2fde837e4c29).
Every question is preserved verbatim.

SWE-bench-Signal is a fixed mixture of 63 SWE-bench Verified, 121 SWE-bench Pro
and 66 SWE-bench Multilingual tasks. The package contains the selection and
canonical bundle checksums; the release bundle supplies instructions, Docker build
contexts and verifier tests. Solutions are excluded. It removes an optional
organization-specific package-cache helper from Dockerfiles and configures the
six affected Maven tasks to use the already-allowlisted public Maven Central
hostname. Task instructions, verifier scripts and network policies remain pinned.
Public base-image tags are resolved to immutable image digests in the bundle so
installations share the same base images. Subsequent Docker build commands can
still download packages from public repositories.

The pinned bundle's SHA-256 is
`96ffb65777436e561e84e1e77d6187bcfb7300ab0dc109d4ffc1841c95b5b38b`.
Preparation verifies it before extraction and checks the canonical asset manifest.

The Verified, Pro and Multilingual source subsets come from the pinned
[Harbor registry](https://github.com/harbor-framework/harbor/blob/5fcf07f5af37eba19db389c7d8916743e4591ff5/registry.json).
`pro` selects the original 731-task Pro v1 release, not the later Pro v2 dataset.
All source tasks are included, with no filtering based on verifier results.
Preparation downloads the pinned instructions, environment and test assets,
omits reference solutions, and records file checksums. These source tasks retain
their upstream timeouts and public internet policy. Their Docker base-image tags
are upstream tags and can move; only the Signal bundle pins base-image digests.

`deepswe` includes all 113 tasks from the public
[Datacurve DeepSWE v1.1 release](https://github.com/datacurve-ai/deep-swe/tree/0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea).
Preparation verifies the pinned Git tree and blobs, excludes reference solutions,
and preserves the task instructions, collection hooks and graders. Solver images
use upstream tags; verifier images are built from the pinned `tests/Dockerfile`.
Image pulls and builds need network access; both execution phases run offline.

## Run

Use the same benchmark, model, generation settings and concurrency when comparing
two implementations. Supply a different output directory for each run.

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...

python hle.py run \
  --api openai-completions --model MODEL \
  --base-url https://VENDOR/v1 --output results/chat-hle

python hle.py run \
  --api openai-responses --model MODEL \
  --base-url https://VENDOR/v1 --output results/responses-hle

python hle.py run \
  --api anthropic --model MODEL \
  --base-url https://VENDOR --output results/anthropic-hle
```

Replace `hle.py` with `mmmu_pro.py`, `swe_bench.py`, or `niah.py` for another
benchmark; the API options do not change. Pass the same `--subset` used
for preparation, for example:

```bash
python swe_bench.py run --subset verified \
  --api openai-completions --model MODEL --output results/verified
python mmmu_pro.py run --subset standard-10 \
  --api anthropic --model MODEL --output results/standard-10
```

`openai-completions` means
`/v1/chat/completions`, not the legacy text-only `/v1/completions` endpoint.
Without `--base-url`, the client uses `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL`
when set, then the respective official API. Credentials come from the matching
API key environment variable, or `--api-key-env NAME`, and are not written to results.

For an initial smoke test, add `--limit 2 --concurrency 1`. Limited runs are
marked `full_dataset: false` and are not full benchmark scores. To continue an
interrupted run, repeat the identical command with `--resume`. Completed cases
are reused; failed cases restart from their first request, with previous attempts
retained. Resume requires the same prepared dataset, selected cases, candidate
and judge configuration, generation settings, and concurrency. Source code, SDK
versions, and environment headers are not compared; use a new output directory
when changes to them could affect results. Only one process can use a given
output directory at a time. Runs written with the earlier fingerprint-based
format require a new output directory.

### Generation settings and compatibility

| Setting | SWE-bench (all subsets) | MMMU-Pro (all subsets) / HLE candidate |
| --- | --- | --- |
| Samples per case | 1 | 1 |
| Temperature / top-p | 1 / 1 | 1 / 1 |
| Reasoning effort | high | high |
| Maximum generated output tokens | 262,144 across the entire tool trajectory | 131,072 per candidate |
| Concurrent cases | 32 | 64 |

`--max-output-tokens`, `--temperature`, `--top-p`, `--reasoning-effort` and
`--concurrency` explicitly override these defaults and are recorded. Token limits
are **output budgets**, using each endpoint's reported output usage, including
reasoning where the endpoint includes it. They are not total-context limits;
prompt and tool-result tokens do not consume these output budgets. No common
random seed or identical tokenization is assumed across protocols. Consequently,
these runs are not a claim of bitwise equivalence to historical raw-token runs.

Some endpoints do not accept sampling controls or high reasoning. Use
`--temperature none --top-p none` to omit sampling fields, and
`--reasoning-effort default` to omit the candidate's reasoning controls. `none`
explicitly disables reasoning. On Anthropic, other effort levels request adaptive
thinking plus `output_config.effort`; the endpoint must implement those features.
Unsupported settings fail rather than silently falling back to another request.

HLE judgment uses a fresh conversation. By default, the candidate's provider,
model and endpoint judge their own answer, with a 32,768 output-token budget,
temperature 0, top-p 1 and reasoning disabled. Explicit candidate sampling
omissions also apply to this default self-judge.

To use an independent judge, set any judge API option:

```bash
export JUDGE_API_KEY=...
python hle.py run \
  --api openai-completions --model CANDIDATE \
  --base-url https://CANDIDATE_VENDOR/v1 --output results/hle \
  --judge-api anthropic --judge-model JUDGE \
  --judge-base-url https://JUDGE_VENDOR --judge-api-key-env JUDGE_API_KEY
```

`--judge-api` and `--judge-model` default to the candidate's values. With the same
API, the judge inherits the candidate's base URL and key environment variable
unless overridden. With a different API, omitted connection options use that
API's normal environment variables and defaults. A separate judge has independent
sampling defaults. Override them with `--judge-max-output-tokens`,
`--judge-temperature`, `--judge-top-p`, and `--judge-reasoning-effort`; `none` and
`default` have the same meanings as the candidate options. Generation-only judge
overrides still use the candidate's provider.

HLE uses native structured-output controls and validates the full JSON locally;
the schema requires an actual `"strict": true` field. A malformed, refused or
truncated judgment fails the run instead of counting as an incorrect answer.
Only the candidate's visible answer is judged; separate reasoning is excluded.
Both stages' resolved configuration, requests and usage are recorded, and changing
the judge invalidates resume.

For vendor extensions, `--extra-body path/to/settings.json` adds a JSON object
to candidate requests and default self-judgment. A separate judge uses only
`--judge-extra-body`, so candidate-specific extensions do not reach another
provider. These fields cannot override core protocol, model, prompt, tool,
sampling or schema fields. Do not put credentials there: request bodies are
intentionally saved for reproducibility.

### Needle-in-a-haystack

NIAH ports the Inkling release's deterministic UUID key-to-value retrieval test:
seed 42, context lengths of 4,000, 16,000, 64,000, 256,000, and 1,024,000 tokens,
depths 0–100% in 10% steps, and 10 samples per cell (550 cases). It preserves the release prompt,
random-number sequence, line-fitting algorithm, and case-insensitive UUID
substring scoring across the entire visible answer. It requires no judge.

```bash
python niah.py prepare --tokenizer MODEL_OR_DIRECTORY
python niah.py run --api openai-completions --model MODEL \
  --base-url https://VENDOR/v1 --output results/niah

# Smaller grid for an endpoint with a shorter context window.
python niah.py prepare --tokenizer MODEL_OR_DIRECTORY --data-dir data/niah-smoke \
  --context-lengths 2000 8000 --depths 0 50 100 --samples-per-cell 1
```

Preparation accepts a Hugging Face model ID, local tokenizer directory, or
`tokenizer.json` file. It uses the fast tokenizer without special tokens,
padding, or truncation, records its checksum, and saves prompts incrementally
instead of holding the whole grid in memory. Use the same prepared data for all
API comparisons; running it needs no tokenizer download. Changing the grid or
tokenizer requires another `--data-dir`.

NIAH defaults differ from the reasoning benchmarks: **128 output tokens,
temperature 0, no top-p override, reasoning disabled, concurrency 4**. The grid
reserves 128 tokens for output but excludes API chat-template overhead. Its
context lengths and depths are nominal: fitting preserves the original needle
selection, so added distractors can shift the final depth slightly. An over-budget
prompt fails preparation instead of silently proceeding. Endpoints need room for
the prepared prompt, their rendering overhead, and the chosen output budget;
requests are never truncated to fit. A complete run also writes `grid.json` with
accuracy by context length and depth, alongside the shared run artifacts.

### SWE execution

Each case builds or pulls its prepared environment, runs a bash-only agent, then runs the
task's verifier. Limits are 4,096 turns, 4,096 tool calls, the configured output
budget, and the task's agent timeout (eight hours for Signal; upstream values
for full source subsets). Exceeding an agent limit scores
zero even if tests subsequently pass, except for DeepSWE, which grades the
committed patch even when a limit is reached. Bash output is capped at 16 KiB per stream;
the default command timeout is 300 seconds. Containers receive at least 8 GiB
RAM, so choose concurrency to fit the host. Budget additional time and disk space
for the public base images and Docker builds.

For Signal, Verified, Pro and Multilingual, the model's container initially
contains only the build context. Verifier files
are copied in after the agent exits and background processes are stopped. Task
containers have no direct external network; an HTTP(S) proxy enforces the pinned
solve and verifier host allowlists. It supports HTTP port 80 and HTTPS port 443,
and blocks private and link-local destinations. Of the 250 Signal tasks, 246
disable solve-time internet and four allow selected hosts; 89 verifiers need
allowlisted access. Verified, Pro and Multilingual allow public HTTP(S) hosts during both
solving and verification, following their upstream default. This is a portable
Docker execution environment, not a reproduction of
another sandbox service's kernel or networking implementation.

DeepSWE uses Docker `--network none` for both solving and grading, with no proxy
or shared network. Its pinned collection command exports committed changes
only; uncommitted edits are not submitted. The solver is removed before starting
a fresh verifier container with the held-out tests. Only `model.patch` is copied
across. The verifier applies it, runs tests and writes a binary `reward.json`.
Both containers disable Cargo cache expiry and set `TERM=xterm-256color` so
cached dependencies and Node readline tests work without network access or a TTY.
The upstream limits are three hours for solving and 30 minutes for grading.

The verifier saves the candidate patch when Git metadata is available, test
stdout/stderr and reward artifacts. Patch-capture failures are recorded separately
from the test result.
A missing or invalid `reward.txt` (`reward.json` for DeepSWE), or a failed DeepSWE
patch collection, is an infrastructure failure, not a model score of
zero. Containers and per-case networks are removed on exit; built images remain
available for subsequent runs.

## Results

Each output directory contains:

- `run.json`: effective candidate/judge and generation settings, prepared dataset
  manifest, and ordered case IDs. Dataset identity is independent of local paths.
- `summary.json`: completed/expected cases, coverage, provisional accuracy and
  final accuracy only after every selected case finishes. `usage` counts scored
  cases; `recorded_usage` also includes successful API responses from failed or
  abandoned attempts. Calls that fail before returning usage cannot be counted.
- `samples/<id-hash>/result.json`: score, parsed answer, grading details, usage
  and timing for each completed case.
- `samples/<id-hash>/attempt-*/call-*.json`: exact API request/response bodies,
  usage, stop reason and latency, persisted before grading proceeds. Errors and
  task artifacts are saved alongside them.

Concurrency includes both generation and judgment. Report whole-run elapsed
time when comparing HLE throughput; comparing only candidate generation omits
part of the workload. Current-session throughput excludes tokens from previously
completed cases when resuming. SDK transport retries are bounded; request errors
cancel the run and leave its coverage explicitly incomplete.

## Code and validation

```text
swe_bench.py      SWE-bench Signal, full source subsets and DeepSWE
mmmu_pro.py       MMMU-Pro Vision and standard subsets
hle.py            HLE with self-judging or an independent judge
niah.py           Long-context UUID needle retrieval
inkling_vendor_verifier/
  types.py         shared message, tool, generation and evaluation contract
  providers.py     the three native API adapters
  runner.py        concurrency, recordings, resume and score accounting
  cli.py           common prepare/run commands
  benchmarks/      one implementation of each benchmark
  sandbox.py       SWE container lifecycle and grading
  _proxy.py        task network allowlists
```

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Tests exercise all twelve benchmark/API combinations through the official SDKs
with mocked HTTP responses, including native reasoning state across tool turns,
unchanged image payloads and strict judgment. Separate-judge tests cover
independent configuration, request recording and failure accounting.
Benchmark tests cover parser
corner cases, corrupted data, token limits and sandbox lifecycle. These tests do
not substitute for validating a particular vendor endpoint or Docker host.

MMMU answer extraction follows the
[Kimi vendor verifier](https://github.com/MoonshotAI/Kimi-Vendor-Verifier/blob/main/mmmu_pro_vision.py),
including stable per-task random fallback for nonempty unparseable answers;
empty answers score zero. HLE judgment follows the
[published equality prompt](https://github.com/centerforaisafety/hle/blob/73ae974b1844c3ffa64c3f4343d9f1f259575700/hle_eval/run_judge_results.py).
Dataset and task content remain subject to their upstream terms; this repository
does not relicense them.
