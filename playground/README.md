# Local Playground

Put your local prototype scripts in this directory.

This repo already defines its environment in `pyproject.toml`, so the simplest setup is to sync the local `uv` environment from the repository root and run scripts with `uv run`.

For this playground, prefer Python 3.12 to avoid Python 3.13 compatibility issues in one of the current dependencies.

## Recommended setup

From the repository root:

```bash
uv sync --python 3.12
```

## Run a script

From the repository root:

```bash
uv run python playground/my_script.py
```

## Apple Silicon note

On a Mac, start with small models and use `mps` when available instead of CUDA:

```python
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
```

Avoid examples that explicitly require `cuda` or `flash_attention_2`, since those are typically for NVIDIA GPUs.

## Suggested workflow

1. Create scripts in `playground/`.
2. Keep them small and focused while learning.
3. Start with smaller Hugging Face models before trying larger contexts or heavier evaluations.
