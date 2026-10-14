# Inkling Verify

Evaluation scripts for verifying the accuracy of Inkling model implementations.
Supports OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages endpoints.

## Installation

Requires Python 3.12+ on Linux or macOS. DeepSWE and MMMU-Pro with tools also require
Docker or a compatible Podman `docker` command, on ARM64 or x86-64.

```bash
git clone https://github.com/thinking-machines-lab/inkling-verify.git
cd inkling-verify
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

### DeepSWE

113 coding tasks, graded by testing the model's committed patch in a fresh
container. Networking is disabled by default; add `--with-network` to enable it.

```bash
python -m inkling_verify.benchmarks.deepswe prepare
python -m inkling_verify.benchmarks.deepswe run \
  --api openai-completions --model MODEL --base-url https://API_HOST/v1 \
  --output results/deepswe
```

### MMMU-Pro

Multimodal multiple-choice questions: `vision`, `standard-4`, or `standard-10`,
with 1,730 questions each. The example enables offline Python image tools.

```bash
python -m inkling_verify.benchmarks.mmmu_pro prepare --subset vision
python -m inkling_verify.benchmarks.mmmu_pro run --subset vision --with-tools \
  --api openai-completions --model MODEL --base-url https://API_HOST/v1 \
  --output results/mmmu-pro
```

### MMAU

1,000 audio multiple-choice questions covering sound, music, and speech. Requires
an audio-capable Inkling model through Chat Completions. Use `--subset sound`,
`music`, or `speech` to run one category.

```bash
python -m inkling_verify.benchmarks.mmau prepare
python -m inkling_verify.benchmarks.mmau run \
  --api openai-completions --model MODEL --base-url https://API_HOST/v1 \
  --output results/mmau
```

### HLE without tools

2,158 text-only questions. The model judges its own answers by default;
use `--judge-model` to select a separate judge.

```bash
python -m inkling_verify.benchmarks.hle_no_tools prepare
python -m inkling_verify.benchmarks.hle_no_tools run \
  --api openai-completions --model MODEL --base-url https://API_HOST/v1 \
  --output results/hle
```

### Needle-in-a-haystack

UUID retrieval across 550 cases, with context lengths from 4,000 to 1,024,000
tokens. Prepare prompts using the model's Hugging Face ID or local tokenizer.

```bash
python -m inkling_verify.benchmarks.niah prepare --tokenizer MODEL_OR_DIRECTORY
python -m inkling_verify.benchmarks.niah run \
  --api openai-completions --model MODEL --base-url https://API_HOST/v1 \
  --output results/niah
```

## Tinker

Set `TINKER_API_KEY`, then use Tinker's [Chat Completions](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/)
or [Anthropic Messages](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/anthropic/)
endpoint. After preparing the dataset, for example:

```bash
python -m inkling_verify.benchmarks.mmmu_pro run --subset vision --with-tools \
  --api openai-completions --model thinkingmachines/Inkling \
  --base-url https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1 \
  --api-key-env TINKER_API_KEY --output results/tinker-openai

python -m inkling_verify.benchmarks.mmmu_pro run --subset vision --with-tools \
  --api anthropic --model thinkingmachines/Inkling \
  --base-url https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api \
  --api-key-env TINKER_API_KEY --output results/tinker-anthropic
```

Replace `thinkingmachines/Inkling` with another base model or a
`tinker://YOUR_RUN/sampler_weights/YOUR_CHECKPOINT` path.
