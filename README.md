# Inkling vendor verifier

Run pinned benchmarks against OpenAI Chat Completions, OpenAI Responses, or
Anthropic Messages endpoints. Prompts, tool loops and grading are shared across
APIs; each API adapter serves every benchmark.

| Script | Subsets / cases | Grading |
| --- | --- | --- |
| `deepswe.py` | DeepSWE v1.1 (113) | Committed patch tested in a fresh container; offline by default |
| `mmmu_pro.py` | Vision, Standard-4, Standard-10 (1,730 each) | Local multiple-choice answer extraction |
| `hle_no_tools.py` | Text-only HLE without tools (2,158) | Structured judgment by the candidate or a separate judge |
| `niah.py` | 5 context lengths × 11 depths × 10 samples | UUID retrieval from the visible answer |

Defaults are MMMU-Pro Vision and HLE self-judging. HLE self-judging is diagnostic,
not an official score from an independent judge. Conversations
use native APIs and preserve original assistant responses, including reasoning
and tool state. They do not reproduce private renderers or raw-token endpoints.

## Install

Use Python 3.12 or newer on Linux or macOS. DeepSWE and MMMU-Pro with tools also
require a Linux Docker daemon and build natively on ARM64 or x86-64. A compatible
Podman `docker` command works too. Image builds need access to public registries
and package repositories. The client host needs no GPU.

```bash
git clone https://github.com/thinking-machines-lab/inkling-vendor-verifier.git
cd inkling-vendor-verifier
python -m pip install -r requirements.txt
python deepswe.py --help
```

Each script has `prepare` and `run` subcommands and works directly from the
checkout. Alternatively, `python -m pip install -e .` installs the equivalent
`deepswe`, `mmmu-pro`, `hle-no-tools`, and `niah` commands. Both installation
methods use `requirements.txt`; uv is not required.

## Prepare data

DeepSWE preparation automatically downloads its pinned task archive and verifies
its checksums. The archive contains instructions, Docker build files, and grading
tests. Private release access uses `GH_TOKEN`,
`GITHUB_TOKEN`, or an existing `gh auth login`; GitHub CLI is optional.

```bash
python deepswe.py prepare

python mmmu_pro.py prepare --subset vision
# Other MMMU-Pro subsets: standard-4, standard-10.

# Accept the cais/hle dataset terms and authenticate on Hugging Face first.
export HF_TOKEN=...
python hle_no_tools.py prepare
python niah.py prepare --tokenizer MODEL_OR_DIRECTORY
```

Preparation checks pinned data and writes `data/<benchmark>/manifest.json`;
HLE uses `data/hle-self-judge`. The DeepSWE archive requires both the expected
archive checksum and canonical manifest/file checksums. Its manifest records
source revisions and task provenance. Reference solutions are excluded. Downloads and
results stay outside Git; HLE data and logs remain subject to its access terms.

MMMU-Pro uses the test split at revision
[`1ba55708b8588a8f9b180b8fec9e6435c88ce363`](https://huggingface.co/datasets/MMMU/MMMU_Pro/tree/1ba55708b8588a8f9b180b8fec9e6435c88ce363).
Standard subsets interleave text with referenced images; their names correspond
to the upstream configurations, though individual option counts can differ.
Images retain original pixels and dimensions as lossless PNGs, sent unchanged
through every API. Endpoints must accept these payloads, some exceeding 10 MB
before base64 encoding; images are never resized to fit.

HLE preserves the text-only test questions verbatim at revision
[`5a81a4c7271a2a2a312b9a690f0c2fde837e4c29`](https://huggingface.co/datasets/cais/hle/tree/5a81a4c7271a2a2a312b9a690f0c2fde837e4c29).

## Run

Keep benchmark, model, generation settings and concurrency the same when
comparing implementations, with a different output directory per run.

```bash
export OPENAI_API_KEY=...
python mmmu_pro.py run --subset vision \
  --api openai-completions --model MODEL --base-url https://VENDOR/v1 \
  --output results/mmmu-smoke --limit 2 --concurrency 1
```

Replace the script for another benchmark and use the subset prepared earlier.
All four scripts share these API arguments:

| Argument | Meaning |
| --- | --- |
| `--api` | `openai-completions`, `openai-responses`, or `anthropic` |
| `--model` | Endpoint model name |
| `--base-url` | Override the provider's base URL |
| `--api-key-env NAME` | Read credentials from a named environment variable |
| `--output` | Run output directory |

`openai-completions` uses `/v1/chat/completions`, not legacy `/v1/completions`.
OpenAI APIs normally use `OPENAI_API_KEY` and `OPENAI_BASE_URL`; Anthropic uses
`ANTHROPIC_API_KEY` and `ANTHROPIC_BASE_URL`. Without a URL override or environment
setting, the SDK uses its official API. Credentials are not written to results.

Limited runs are marked `full_dataset: false`. Resume an interrupted run by
repeating the identical command with `--resume`: completed cases are reused,
failed cases restart from their first request, and prior attempts are retained.
Dataset, selected cases, candidate/judge configuration, generation settings,
MMMU-Pro tool mode, DeepSWE network access and concurrency must match. Code, SDK versions and
environment headers are not compared; use a new output directory when changing
them could affect results.
Only one process may use an output directory at a time.

### Generation and judging

| Default | DeepSWE | MMMU-Pro | HLE candidate | NIAH |
| --- | --- | --- | --- | --- |
| Output tokens | 262,144 per tool trajectory | 131,072 | 262,144 | 128 |
| Temperature / top-p | 1 / 1 | 1 / 1 | 1 / 1 | 0 / omitted |
| Reasoning effort | high | high | max (0.99 for Inkling) | none |
| Concurrency | 32 | 64 | 64 | 4 |

There is one sample per case. Override defaults with `--max-output-tokens`,
`--temperature`, `--top-p`, `--reasoning-effort` and `--concurrency`.
Output budgets use reported output usage, including reasoning where reported;
prompt and tool-result tokens do not consume them. They are not context limits.
No common random seed, identical tokenization or bitwise output parity across
protocols is assumed.

With `mmmu_pro.py run --with-tools`, defaults change to 524,288 output tokens
across the entire conversation and concurrency 4, with at most 131,072 output
tokens per turn. The table above describes MMMU-Pro without tools.

Use `--temperature none --top-p none` to omit sampling controls.
`--reasoning-effort default` omits reasoning controls; `none` disables reasoning.
Other Anthropic effort levels use adaptive thinking and `output_config.effort`.
Unsupported settings fail explicitly instead of falling back.

HLE judges each visible candidate answer in a fresh conversation, excluding
separate reasoning. Default self-judging uses the candidate endpoint/model,
32,768 output tokens, temperature 0, top-p 1 and reasoning disabled. Explicit
candidate sampling omissions also apply to the default self-judge.

```bash
export JUDGE_API_KEY=...
python hle_no_tools.py run --api openai-completions --model CANDIDATE \
  --base-url https://CANDIDATE_VENDOR/v1 --output results/hle \
  --judge-api anthropic --judge-model JUDGE \
  --judge-base-url https://JUDGE_VENDOR --judge-api-key-env JUDGE_API_KEY
```

Judge API/model default to the candidate's. With the same API, omitted judge URL
and key options inherit the candidate's; a different API uses its own normal
environment variables/defaults. Separate judges have independent sampling defaults.
Override them with `--judge-max-output-tokens`, `--judge-temperature`,
`--judge-top-p` and `--judge-reasoning-effort`; generation-only judge overrides
still use the candidate provider. Native structured-output controls and local
validation require the complete judgment schema, including `"strict": true`.
Malformed, refused or truncated judgments fail the run, not the answer's score.
Both stages' settings, requests and usage are recorded; changing the judge
invalidates resume. Compare whole-run HLE timing, including judgment.

### MMMU-Pro image tools

Add `--with-tools` to any MMMU-Pro subset to enable one persistent `ipython` tool:

```bash
python mmmu_pro.py run --with-tools --subset vision \
  --api openai-responses --model MODEL --base-url https://VENDOR/v1 \
  --output results/mmmu-tools --limit 2 --concurrency 1
```

Each question gets a fresh offline container. Original question images are
available under `/sandbox/images/`, with paths included beside the images in the
prompt. Python variables and imports persist between calls; `display(...)`, PIL
images and matplotlib plots return inline images. The sandbox includes NumPy,
SciPy, pandas, SymPy, scikit-learn, scikit-image, OpenCV, Shapely and Tesseract.
This mode adds image-analysis instructions while retaining the subset's question
prompt and answer grading. Tools remain optional for the model; their use is
recorded per question. Prepared data is shared with no-tools runs.

Tool results carry images directly on Anthropic. For compatibility with current
serving APIs, Chat Completions and Responses attach images in a follow-up user
message labeled with the tool-call ID, after all tool results in that turn.

Limits are 128 turns and 128 tool calls per question, 60 seconds per Python call,
16 KiB of output text per stream, and eight displayed images per call (8,192
pixels per side and 32 MiB of PNG data combined). Each container requests 8 GiB
RAM and two CPUs. Images are built once and cached; builds need internet, but
model-executed Python always runs with networking disabled. No host directories,
API credentials or container sockets are mounted into the sandbox. Containers
are removed on exit; the built image remains. Start with concurrency 1 on a new
host and increase it to fit available memory.

### Needle-in-a-haystack

NIAH preserves the Inkling release's deterministic UUID key-to-value retrieval
prompt, random sequence, line-fitting algorithm and case-insensitive UUID
substring scoring over the entire visible answer. It requires no judge.
Defaults are seed 42; context lengths **4,000, 16,000, 64,000, 256,000, 1,024,000**;
depths 0–100% in 10% steps; and 10 samples per cell (550 cases).

Preparation accepts a Hugging Face model ID, local tokenizer directory, or
`tokenizer.json`. It uses the fast tokenizer without special tokens, padding or
truncation, records its checksum, and saves prompts incrementally. Run all APIs
against the same prepared data; evaluation needs no tokenizer download.
For a smaller grid, add `--context-lengths 2000 8000 --depths 0 50 100
--samples-per-cell 1` to preparation. Changing the grid or tokenizer requires a
separate `--data-dir`.

The grid reserves 128 output tokens but excludes API rendering overhead. Context
lengths and depths are nominal: added distractors can shift depth slightly.
Over-budget prompts fail preparation; requests are never truncated. Endpoints
need room for the prepared prompt, rendering overhead and selected output budget.
A complete run writes `grid.json` with accuracy by context length and depth.

### DeepSWE sandbox behavior

Cases build their environment, run a bash-only agent, then execute the task's
verifier. Limits are 4,096 turns, 4,096 tool calls, the output budget and the
task's timeout. The committed patch is graded even if the agent hits a limit.
Bash output is capped at 16 KiB per stream; commands default to 300 seconds.
Containers receive at least 8 GiB RAM, so size concurrency for the host and budget
space for images and builds.

The solver and grader default to Docker `--network none`, with no internet
access or proxy. Add `--with-network` to `python deepswe.py run ...` to enable
network access for **both** through Docker's bridge network. This overrides the
upstream offline policy and is recorded in results and resume settings.
The model API client runs outside the sandbox and sends tool commands into it.
The collection hook exports **committed changes only**;
uncommitted edits are not submitted. The solver is removed and only `model.patch` enters a fresh verifier
container. Upstream limits are three hours solving and 30 minutes grading.
Containers disable Cargo cache expiry and set `TERM=xterm-256color` for offline
Rust dependencies and Node readline tests. Setup downloads and image builds
need internet regardless of this flag.

DeepSWE uses the same build path on ARM64 and x86-64, without a platform flag or
emulation. Deno and cargo-nextest downloads select the native binary at their
original versions; instructions and grading tests are unchanged. The verifier
image extends the same pristine task image used to start the solver. Initial
builds can take time and disk space; Docker caches them for later runs. Start
with `--limit 1 --concurrency 1` on a new host. Generated Dockerfiles and build
logs are saved with each attempt; the checksummed task bundle is never modified.

Missing/invalid rewards or failed DeepSWE patch collection are infrastructure
failures, not zero scores. Patches, verifier output and reward artifacts are
saved. Containers are removed on exit; built images remain.

## Results

- `run.json`: resolved settings, dataset manifest and ordered case IDs; dataset
  identity is independent of local paths.
- `summary.json`: coverage, provisional accuracy, and final accuracy after every
  selected case finishes. `usage` counts scored cases; `recorded_usage` also
  includes successful responses from failed/abandoned attempts.
- `samples/<id-hash>/result.json`: score, answer, grading details, usage and timing.
- `samples/<id-hash>/attempt-*/call-*.json`: exact API request/response bodies,
  usage, stop reason and latency, persisted before grading. Errors and task
  artifacts are saved alongside them.

Concurrency includes generation and judgment. Resume throughput excludes prior
sessions' completed cases. SDK retries are bounded; request errors cancel the
run with incomplete coverage. Calls failing before returning usage cannot be counted.

MMMU answer extraction follows the
[Kimi verifier](https://github.com/MoonshotAI/Kimi-Vendor-Verifier/blob/main/mmmu_pro_vision.py):
nonempty unparseable answers use stable per-task random fallback; empty answers
score zero. HLE uses the
[published equality prompt](https://github.com/centerforaisafety/hle/blob/73ae974b1844c3ffa64c3f4343d9f1f259575700/hle_eval/run_judge_results.py).
Dataset and task content remain subject to their upstream terms.
