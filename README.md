# Inkling vendor verifier

Run shared evaluation workloads against OpenAI Chat Completions, OpenAI Responses,
and Anthropic Messages endpoints.

## Installation

Requires Python 3.12+ on Linux or macOS. DeepSWE and MMMU-Pro with tools also require
Docker or a compatible Podman `docker` command, on ARM64 or x86-64.

```bash
git clone https://github.com/thinking-machines-lab/inkling-vendor-verifier.git
cd inkling-vendor-verifier
python -m pip install -r requirements.txt
```

Alternatively, `python -m pip install -e .` installs the `deepswe`, `mmmu-pro`,
`hle-no-tools`, and `niah` commands.

## Evaluations

The examples below use `OPENAI_API_KEY` and Chat Completions. For Responses, use
`--api openai-responses`. For Anthropic, use `--api anthropic`, `ANTHROPIC_API_KEY`,
and an Anthropic base URL.

Each script has `prepare` and `run` commands. Use `run --help` for options,
`--limit N` for a smaller run, and `--resume` to continue an interrupted run.
Scores are saved in `summary.json` under the output directory.

### DeepSWE

113 coding tasks, graded by testing the model's committed patch in a fresh
container. Networking is disabled by default; add `--with-network` to enable it.

```bash
python deepswe.py prepare
python deepswe.py run \
  --api openai-completions --model MODEL --base-url https://VENDOR/v1 \
  --output results/deepswe
```

### MMMU-Pro

Multimodal multiple-choice questions: `vision`, `standard-4`, or `standard-10`,
with 1,730 questions each. Add `--with-tools` to enable offline Python image tools.

```bash
python mmmu_pro.py prepare --subset vision
python mmmu_pro.py run --subset vision \
  --api openai-completions --model MODEL --base-url https://VENDOR/v1 \
  --output results/mmmu-pro
```

### HLE without tools

2,158 text-only questions. The model judges its own answers by default;
use `--judge-model` to select a separate judge.

```bash
python hle_no_tools.py prepare
python hle_no_tools.py run \
  --api openai-completions --model MODEL --base-url https://VENDOR/v1 \
  --output results/hle
```

### Needle-in-a-haystack

UUID retrieval across 550 cases, with context lengths from 4,000 to 1,024,000
tokens. Prepare prompts using the model's Hugging Face ID or local tokenizer.

```bash
python niah.py prepare --tokenizer MODEL_OR_DIRECTORY
python niah.py run \
  --api openai-completions --model MODEL --base-url https://VENDOR/v1 \
  --output results/niah
```
