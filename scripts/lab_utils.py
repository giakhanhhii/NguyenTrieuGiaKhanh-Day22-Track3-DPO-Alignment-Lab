from __future__ import annotations

import os
import zipfile
from pathlib import Path


def get_repo_root() -> Path:
    cwd = Path.cwd()
    return cwd.parent if cwd.name == "notebooks" else cwd


def load_lab_env(repo_root: Path | None = None) -> Path:
    repo_root = repo_root or get_repo_root()
    env_path = repo_root / ".env"
    if not env_path.exists():
        return env_path

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.split(" #", 1)[0].strip().strip("'").strip('"')
        os.environ.setdefault(key, value)
    return env_path


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def choose_mixed_precision() -> tuple[bool, bool, str]:
    """Pick a safer bf16/fp16 setting for Colab GPUs.

    L4/Ada runtimes have been observed to hit xformers backward failures with
    bf16 in TRL DPO training. Default to fp16 there, while keeping bf16 on
    A100/H100-class hardware where it is usually stable and faster.
    """
    import torch

    if env_flag("FORCE_FP16", False):
        return False, True, "fp16 forced by FORCE_FP16"
    if env_flag("FORCE_BF16", False):
        return True, False, "bf16 forced by FORCE_BF16"

    if not torch.cuda.is_available():
        return False, True, "fp16 fallback (no CUDA)"

    if not torch.cuda.is_bf16_supported():
        return False, True, "fp16 fallback (bf16 unsupported)"

    gpu_name = torch.cuda.get_device_name(0).upper()
    if "L4" in gpu_name:
        return False, True, "fp16 on L4 to avoid xformers bf16 backward failures"

    return True, False, f"bf16 on {gpu_name}"


def configure_attention_backend() -> tuple[str | None, str]:
    """Pick a safer attention backend for Colab GPUs.

    Some Colab L4 sessions hit xformers backward kernel failures in DPO even
    after switching precision. Prefer SDPA there and disable xformers eagerly
    before importing Unsloth / Transformers model code.
    """
    import torch

    if env_flag("FORCE_SDPA_ATTENTION", False):
        os.environ["XFORMERS_DISABLED"] = "1"
        os.environ["DISABLE_XFORMERS"] = "1"
        os.environ["UNSLOTH_DISABLE_XFORMERS"] = "1"
        return "eager", "eager attention forced by FORCE_SDPA_ATTENTION"

    if not torch.cuda.is_available():
        return None, "default attention backend (no CUDA)"

    gpu_name = torch.cuda.get_device_name(0).upper()
    if "L4" in gpu_name:
        os.environ["XFORMERS_DISABLED"] = "1"
        os.environ["DISABLE_XFORMERS"] = "1"
        os.environ["UNSLOTH_DISABLE_XFORMERS"] = "1"
        return "eager", "eager attention on L4 to avoid xformers DPO backward failures"

    return None, f"default attention backend on {gpu_name}"


def choose_gradient_checkpointing() -> str | bool:
    """Avoid Unsloth's checkpointing path on L4 where xformers can still be selected."""
    import torch

    if env_flag("FORCE_UNSLOTH_CHECKPOINTING", False):
        return "unsloth"
    if not torch.cuda.is_available():
        return "unsloth"
    gpu_name = torch.cuda.get_device_name(0).upper()
    if "L4" in gpu_name:
        return False
    return "unsloth"


LLAMA3_CHAT_TEMPLATE = """{% for message in messages %}{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + (message['content'] | trim) + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"""


def ensure_chat_template(tokenizer, model_name: str | None = None):
    """Ensure Llama/Qwen chat-format calls work even when the tokenizer omits a template."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer

    model_hint = (model_name or getattr(tokenizer, "name_or_path", "") or "").lower()
    if "llama" in model_hint or "sft-mini" in model_hint or "adapter" in model_hint:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
        print("Set tokenizer.chat_template = Llama-3 fallback template")
    else:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
        print("Set tokenizer.chat_template = fallback chat template")
    return tokenizer


def render_text_card(
    output_path: Path,
    title: str,
    lines: list[str],
    *,
    width: float = 11,
    height: float = 4.5,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.text(
        0.01,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f6f8fa", "edgecolor": "#d0d7de"},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def package_submission_artifacts(
    repo_root: Path | None = None,
    *,
    output_name: str = "submission-package.zip",
    include_eval_json: bool = True,
) -> Path:
    repo_root = repo_root or get_repo_root()
    output_path = repo_root / output_name

    screenshots_dir = repo_root / "submission" / "screenshots"
    reflection_path = repo_root / "submission" / "REFLECTION.md"
    eval_dir = repo_root / "data" / "eval"

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if screenshots_dir.exists():
            for path in sorted(screenshots_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(repo_root))
        if reflection_path.exists():
            zf.write(reflection_path, reflection_path.relative_to(repo_root))
        if include_eval_json and eval_dir.exists():
            for path in sorted(eval_dir.glob("*.json")):
                zf.write(path, path.relative_to(repo_root))

    return output_path
