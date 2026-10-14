# Inkling vendor verifier

Run the same evaluations against OpenAI Chat Completions, OpenAI Responses, and
Anthropic Messages endpoints, with shared prompts, grading, and result recording.

## Evaluations

| Script | Workload | Grading |
| --- | --- | --- |
| `deepswe.py` | 113 DeepSWE v1.1 coding tasks | Committed patches tested in fresh containers |
| `mmmu_pro.py` | MMMU-Pro Vision, Standard-4, or Standard-10; 1,730 questions each | Multiple-choice answers; optional Python image tools |
| `hle_no_tools.py` | 2,158 text-only HLE questions | Candidate self-judging or a separate judge |
| `niah.py` | Needle-in-a-haystack UUID retrieval; 550 cases by default | Exact UUID retrieval |

HLE self-judging is diagnostic; use a separate judge for independent grading.

## Installation

Use Python 3.12 or newer on Linux or macOS. The client needs no GPU.

```bash
git clone https://github.com/thinking-machines-lab/inkling-vendor-verifier.git
cd inkling-vendor-verifier
python -m pip install -r requirements.txt
```

Run scripts directly from the checkout, or use `python -m pip install -e .` to
install the `deepswe`, `mmmu-pro`, `hle-no-tools`, and `niah` commands.

DeepSWE and MMMU-Pro with tools also need a Linux Docker daemon or a compatible
Podman `docker` command. Both ARM64 and x86-64 are supported. Initial image builds
need internet access and disk space; start with concurrency 1.

## Usage

### Prepare data

Prepare each evaluation you want to run. Data is saved under `data/` by default.

```bash
python deepswe.py prepare
python mmmu_pro.py prepare --subset vision
python hle_no_tools.py prepare
python niah.py prepare --tokenizer MODEL_OR_DIRECTORY
```

HLE requires accepting the [dataset terms](https://huggingface.co/datasets/cais/hle)
and authenticating with Hugging Face, for example via `HF_TOKEN`. Private DeepSWE
downloads use `GH_TOKEN`, `GITHUB_TOKEN`, or an existing `gh auth login`.

NIAH accepts a Hugging Face model ID, local tokenizer directory, or `tokenizer.json`.
Its default grid uses context lengths 4,000, 16,000, 64,000, 256,000, and 1,024,000;
depths 0–100% in 10% steps; and 10 samples per cell. For shorter-context models,
prepare a smaller grid in a separate directory:

```bash
python niah.py prepare --tokenizer MODEL_OR_DIRECTORY --data-dir data/niah-small \
  --context-lengths 4000 16000 --depths 0 50 100 --samples-per-cell 1
```

Pass the same `--data-dir` when running. Allow room in the endpoint's context
window for both the prompt and the output budget.

### Run an evaluation

Start with a small run against your endpoint:

```bash
export OPENAI_API_KEY=YOUR_API_KEY
python mmmu_pro.py run --subset vision \
  --api openai-completions --model MODEL --base-url https://VENDOR/v1 \
  --output results/mmmu-smoke --limit 2 --concurrency 1
```

To run another evaluation, replace `mmmu_pro.py` and omit `--subset vision`.
Remove `--limit` for the full dataset and increase `--concurrency` as capacity
allows. Use the same prepared data and settings, with separate output directories,
when comparing endpoints.

| `--api` | Endpoint | Default credential variable |
| --- | --- | --- |
| `openai-completions` | `/v1/chat/completions` | `OPENAI_API_KEY` |
| `openai-responses` | `/v1/responses` | `OPENAI_API_KEY` |
| `anthropic` | `/v1/messages` | `ANTHROPIC_API_KEY` |

OpenAI base URLs include `/v1`; Anthropic base URLs normally do not. Use
`--api-key-env NAME` for a different credential variable. Local endpoints without
authentication still require a nonempty key, such as `OPENAI_API_KEY=local`.

### Evaluation options

- **MMMU-Pro:** use `--subset vision`, `standard-4`, or `standard-10` for both
  preparation and execution. Add `--with-tools` when running to enable a persistent
  offline Python sandbox for image analysis. The endpoint must support images.
- **HLE:** self-judges by default. Add `--judge-model MODEL` to use another model,
  with `--judge-api`, `--judge-base-url`, and `--judge-api-key-env` as needed.
- **DeepSWE:** solver and grader networking is disabled by default. Add
  `--with-network` to enable it for both. Setup downloads and image builds still
  need internet without this flag.

All scripts accept `--max-output-tokens`, `--reasoning-effort`, `--temperature`,
and `--top-p`. Candidate reasoning defaults to `max` (0.99 for Inkling), except
NIAH uses `none`. HLE candidate and judge budgets are each 262,144 output tokens;
the judge defaults to `high` (0.9). NIAH allows 4,096 output tokens.
Use `--reasoning-effort default` to omit reasoning controls, or
`--temperature none --top-p none` to omit sampling controls.

See `python SCRIPT.py prepare --help` or `python SCRIPT.py run --help` for all options.

### Results and resuming

The output directory contains `summary.json` with scores and coverage, `run.json`
with settings, and `samples/` with individual results and API request/response logs.
NIAH also writes `grid.json` with accuracy by context length and depth.

Resume an interrupted run by repeating the same command with `--resume`.
Completed cases are reused; failed cases restart. Keep data and run settings
unchanged, and use a new output directory when changing the workload or implementation.
