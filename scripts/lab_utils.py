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
