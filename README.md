# tinker-agent

An intelligent agent for Anthropic's Tinker fine-tuning platform. Automates the creation of fine-tuning datasets and configuration with interactive and programmatic interfaces.

## Installation

```bash
pip install tinker-agent
```

Or with `uv`:

```bash
uv pip install tinker-agent
```

## Quick Start

### Interactive Mode

Simply run the command without arguments for an interactive setup:

```bash
tinker-agent
```

This will guide you through:
1. **Environment setup** - Configure API keys on first run
2. **Task selection** - Choose between SFT, RL, or CPT
3. **Model selection** - Pick from available Claude models
4. **Dataset configuration** - Enter a HuggingFace dataset with validation

### Non-Interactive Mode

Pass configuration directly via command-line arguments:

```bash
tinker-agent \
  config.dataset=HuggingFaceFW/fineweb \
  config.task_type=sft \
  config.model=Qwen/Qwen3-8B
```

## Configuration

### Environment Setup

Run the setup command to configure your environment:

```bash
tinker-agent setup
```

This creates a `.env` file with:

- **TINKER_API_KEY** - Your Tinker API key ([Get it here](https://tinker.anthropic.com/settings/api-keys))
- **WANDB_API_KEY** - Weights & Biases API key for tracking ([Get it here](https://wandb.ai/authorize))
- **WANDB_PROJECT** - W&B project name for organizing experiments

Alternatively, set these as environment variables before running.

### Task Types

- **sft** - Supervised Fine-Tuning (instruction-response pairs)
- **rl** - Reinforcement Learning (reward-based training)
- **cpt** - Continued Pre-Training (raw text data)

### Available Models

| Model                                | Type        | Size    |
| ------------------------------------ | ----------- | ------- |
| `Qwen/Qwen3-VL-235B-A22B-Instruct`   | Vision      | Large   |
| `Qwen/Qwen3-VL-30B-A3B-Instruct`     | Vision      | Medium  |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | Instruction | Large   |
| `Qwen/Qwen3-30B-A3B-Instruct-2507`   | Instruction | Medium  |
| `Qwen/Qwen3-30B-A3B`                 | Hybrid      | Medium  |
| `Qwen/Qwen3-30B-A3B-Base`            | Base        | Medium  |
| `Qwen/Qwen3-32B`                     | Hybrid      | Medium  |
| `Qwen/Qwen3-8B`                      | Hybrid      | Small   |
| `Qwen/Qwen3-8B-Base`                 | Base        | Small   |
| `Qwen/Qwen3-4B-Instruct-2507`        | Instruction | Compact |
| `openai/gpt-oss-120b`                | Reasoning   | Medium  |
| `openai/gpt-oss-20b`                 | Reasoning   | Small   |
| `deepseek-ai/DeepSeek-V3.1`          | Hybrid      | Large   |
| `deepseek-ai/DeepSeek-V3.1-Base`     | Base        | Large   |
| `meta-llama/Llama-3.1-70B`           | Base        | Large   |
| `meta-llama/Llama-3.3-70B-Instruct`  | Instruction | Large   |
| `meta-llama/Llama-3.1-8B`            | Base        | Small   |
| `meta-llama/Llama-3.1-8B-Instruct`   | Instruction | Small   |
| `meta-llama/Llama-3.2-3B`            | Base        | Compact |
| `meta-llama/Llama-3.2-1B`            | Base        | Compact |
| `moonshotai/Kimi-K2-Thinking`        | Reasoning   | Large   |

## Usage Examples

### Interactive Mode Example

```bash
$ tinker-agent

╭──────────────────────────────────────────╮
│            tinker-agent                  │
│     Fine-tuning configuration            │
╰──────────────────────────────────────────╯

Select task type:
  1    sft    Supervised Fine-Tuning
  2    rl     Reinforcement Learning
  3    cpt    Continued Pre-Training
Choice [1/2/3/sft/rl/cpt]: 1

Select model:
Key   Model                                Type         Size
1     Qwen/Qwen3-VL-235B-A22B-Instruct     Vision       Large
2     Qwen/Qwen3-VL-30B-A3B-Instruct       Vision       Medium
3     Qwen/Qwen3-235B-A22B-Instruct-2507   Instruction  Large
...
Choice (number or model name) [1]: 8

HuggingFace dataset: HuggingFaceFW/fineweb
```

### Non-Interactive Example

```bash
# Basic usage
tinker-agent config.dataset=my-org/my-dataset config.task_type=sft

# With all options
tinker-agent \
  config.dataset=HuggingFaceFW/fineweb \
  config.task_type=sft \
  config.model=meta-llama/Llama-3.3-70B-Instruct
```

### Environment Variables

```bash
# Set via environment
export TINKER_API_KEY="your-api-key"
export WANDB_API_KEY="your-wandb-key"
export WANDB_PROJECT="my-finetuning-project"

# Run with config
tinker-agent config.dataset=my-dataset config.task_type=rl
```

## Features

- ✅ **Interactive CLI** - Beautiful rich terminal UI for configuration
- ✅ **Non-interactive mode** - Scriptable with command-line arguments
- ✅ **Dataset validation** - Verifies HuggingFace datasets exist before use
- ✅ **Model selection** - Choose from available Claude models
- ✅ **Environment management** - Simple .env-based configuration
- ✅ **Sandboxed execution** - Agent runs in isolated directory with path validation
- ✅ **Trace viewer** - Streamlit-based viewer for execution traces

## Sandboxing

The agent runs in a sandboxed environment with strict path validation:

- **Root directory isolation** - Agent can only access files within its working directory
- **Path validation** - Blocks access to `~`, `$HOME`, absolute paths, and `..` escapes
- **No system access** - Cannot read sensitive files like `/etc/passwd` or user home directories

This ensures the agent operates safely without requiring Docker, making it more scalable for production use.

## Additional Commands

### View Execution Traces

```bash
tinker-viewer
```

Opens a Streamlit interface to view and analyze agent execution traces.

## Development

### Setup Development Environment

```bash
git clone https://github.com/anthropics/tinker-agent.git
cd tinker-agent
uv sync --extra dev
```

### Run Tests

```bash
uv run pytest
```

### Build Package

```bash
uv build
```

### Deploy to PyPI

```bash
python deploy.py
```

This will:
1. Ask for confirmation
2. Request your PyPI API token ([create one here](https://pypi.org/manage/account/token/))
3. Clean old builds
4. Build the package
5. Upload to PyPI

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.