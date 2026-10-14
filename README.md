# Inkling vendor verifier

Evaluation scripts for verifying the accuracy of Inkling model implementations.
Supports OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages endpoints.

## Installation

Requires Python 3.12+ on Linux or macOS. DeepSWE and MMMU-Pro with tools also require
Docker or a compatible Podman `docker` command, on ARM64 or x86-64.
DeepSWE uses pinned `linux/amd64` images; ARM hosts need x86 emulation enabled in
their container runtime. MMMU-Pro's image tools run natively on either architecture.

```bash
git clone https://github.com/thinking-machines-lab/inkling-vendor-verifier.git
cd inkling-vendor-verifier
python -m pip install -r requirements.txt
```

Alternatively, use `python -m pip install -e .` to run the modules from any directory.

## Evaluations

The examples below use `OPENAI_API_KEY` and Chat Completions. For Responses, use
`--api openai-responses`. For Anthropic, use `--api anthropic`, `ANTHROPIC_API_KEY`,
and an Anthropic base URL.

Each evaluation has `prepare` and `run` commands. Use `run --help` for options,
`--limit N` for a smaller run, and `--resume` to continue an interrupted run.
Scores are saved in `summary.json` under the output directory.
Reasoning defaults to `max` (0.99 for Inkling), except for NIAH, which disables it.

### DeepSWE

113 coding tasks, graded by testing the model's committed patch in a fresh
container. Networking is disabled by default; add `--with-network` to enable it.
Each task allows 1,000,000 cumulative output tokens and 12 hours. Output is bounded
by the remaining context window; conversation history is never truncated.

```bash
python -m inkling_vendor_verifier.benchmarks.deepswe prepare
python -m inkling_vendor_verifier.benchmarks.deepswe run \
  --api openai-completions --model MODEL --base-url https://API_HOST/v1 \
  --output results/deepswe
```

### MMMU-Pro

Multimodal multiple-choice questions: `vision`, `standard-4`, or `standard-10`,
with 1,730 questions each. The example enables offline Python image tools.
With tools, the output budget is 524,288 tokens total, with 131,072 per turn.

```bash
python -m inkling_vendor_verifier.benchmarks.mmmu_pro prepare --subset vision
python -m inkling_vendor_verifier.benchmarks.mmmu_pro run --subset vision --with-tools \
  --api openai-completions --model MODEL --base-url https://API_HOST/v1 \
  --output results/mmmu-pro
```

### HLE without tools

2,158 text-only questions. The model judges its own answers by default;
use `--judge-model` to select a separate judge. Answers and judgments each allow
262,144 output tokens; judging uses `high` reasoning (0.9 for Inkling).
Invalid or unfinished judgments are saved as unscored failures without stopping
the remaining questions. `--resume` preserves those outcomes.

```bash
python -m inkling_vendor_verifier.benchmarks.hle_no_tools prepare
python -m inkling_vendor_verifier.benchmarks.hle_no_tools run \
  --api openai-completions --model MODEL --base-url https://API_HOST/v1 \
  --output results/hle
```

### Needle-in-a-haystack

UUID retrieval across 550 cases, with context lengths from 4,000 to 1,024,000
tokens and a 4,096-token output limit. Prepare prompts using the model's Hugging
Face ID or local tokenizer.

```bash
python -m inkling_vendor_verifier.benchmarks.niah prepare --tokenizer MODEL_OR_DIRECTORY
python -m inkling_vendor_verifier.benchmarks.niah run \
  --api openai-completions --model MODEL --base-url https://API_HOST/v1 \
  --output results/niah
```
