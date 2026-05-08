from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

try:
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
except ImportError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "packaging>=24,<26"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    from packaging.markers import default_environment
    from packaging.requirements import Requirement


REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = REPO_ROOT / ".colab"
STATE_FILE = STATE_DIR / "setup_state.json"
OUTPUT_DIRS = [
    REPO_ROOT / "data" / "pref",
    REPO_ROOT / "adapters" / "sft-mini",
    REPO_ROOT / "adapters" / "dpo",
    REPO_ROOT / "gguf",
]


def log(message: str) -> None:
    print(f"[colab] {message}")


def run(cmd: list[str], *, check: bool = True, capture: bool = False, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        check=check,
        text=True,
        capture_output=capture,
    )
    return completed.stdout.strip() if capture else ""


def detect_tier() -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to auto-detect the Colab tier.") from exc

    if not torch.cuda.is_available():
        return "CPU"
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return "BIGGPU" if total_gb >= 22 else "T4"


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_state(payload: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def requirement_texts(path: Path) -> list[str]:
    items: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            items.append(line)
    return items


def requirement_applies(req: Requirement, extra: str | None = None) -> bool:
    if req.marker is None:
        return True
    env = default_environment()
    if extra is not None:
        env["extra"] = extra
    return req.marker.evaluate(env)


def requirement_satisfied(req: Requirement, seen: set[str] | None = None) -> bool:
    if not requirement_applies(req):
        return True

    seen = seen or set()
    normalized_name = req.name.lower()
    if normalized_name in seen:
        return True
    seen.add(normalized_name)

    try:
        version = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        return False

    if req.specifier and not req.specifier.contains(version, prereleases=True):
        return False

    if not req.extras:
        return True

    try:
        dist = metadata.distribution(req.name)
    except metadata.PackageNotFoundError:
        return False

    for dep_text in dist.requires or []:
        dep = Requirement(dep_text)
        if not any(requirement_applies(dep, extra) for extra in req.extras):
            continue
        if not requirement_satisfied(dep, seen.copy()):
            return False
    return True


def unmet_requirements(path: Path) -> list[str]:
    missing: list[str] = []
    for req_text in requirement_texts(path):
        req = Requirement(req_text)
        if not requirement_satisfied(req):
            missing.append(req_text)
    return missing


def install_requirements(req_file: Path, *, force: bool = False) -> None:
    missing = requirement_texts(req_file) if force else unmet_requirements(req_file)
    if not missing:
        log(f"{req_file.name}: all requirements already satisfy the requested versions")
        return

    log(f"{req_file.name}: installing {len(missing)} requirement(s)")
    run([sys.executable, "-m", "pip", "install", "--progress-bar", "on", *missing])


def notebook_fingerprint(tier: str) -> dict:
    payload = {
        "tier": tier,
        "requirements": file_sha256(REPO_ROOT / "requirements.txt"),
        "python": sys.version.split()[0],
    }
    biggpu_file = REPO_ROOT / "requirements-biggpu.txt"
    if biggpu_file.exists():
        payload["requirements_biggpu"] = file_sha256(biggpu_file)
    return payload


def update_env_file(tier: str) -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        shutil.copyfile(REPO_ROOT / ".env.example", env_path)

    lines = env_path.read_text(encoding="utf-8").splitlines()
    updated = False
    for index, line in enumerate(lines):
        if line.startswith("COMPUTE_TIER="):
            lines[index] = f"COMPUTE_TIER={tier}"
            updated = True
            break
    if not updated:
        lines.append(f"COMPUTE_TIER={tier}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_pull_latest() -> None:
    git_dir = REPO_ROOT / ".git"
    if not git_dir.exists():
        log("Skipping git pull because this folder is not a git checkout")
        return

    dirty_tracked = run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture=True,
        check=True,
    )
    if dirty_tracked:
        log("Skipping git pull because tracked files have local edits")
        return

    current_branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture=True)
    upstream = run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], capture=True, check=False)
    if not upstream:
        log(f"Skipping git pull because branch {current_branch} has no upstream")
        return

    before = run(["git", "rev-parse", "HEAD"], capture=True)
    log(f"Pulling latest code for {current_branch} with --ff-only")
    run(["git", "pull", "--ff-only"])
    after = run(["git", "rev-parse", "HEAD"], capture=True)
    if before == after:
        log("Repository already up to date")
    else:
        log(f"Updated repository to commit {after[:10]}")


def convert_notebooks() -> None:
    notebook_sources = sorted((REPO_ROOT / "notebooks").glob("*.py"))
    if not notebook_sources:
        return
    log("Refreshing notebooks from Jupytext sources")
    run([sys.executable, "-m", "jupytext", "--to", "notebook", "--update", *map(str, notebook_sources)], check=False)


def ensure_outputs() -> None:
    for directory in OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def print_next_steps(tier: str) -> None:
    stitched_nb = "colab/Lab22_DPO_BigGPU.ipynb" if tier == "BIGGPU" else "colab/Lab22_DPO_T4.ipynb"
    log(f"Done. tier={tier}")
    print()
    print("Use one of these in Colab:")
    print(f"  !bash setup-colab.sh                 # initial setup or safe re-run")
    print(f"  !bash colab-refresh.sh               # pull latest code + reuse installed deps")
    print(f"  Open {stitched_nb}                   # stitched notebook entrypoint")
    print("  !make smoke                          # quick verification")


def main() -> int:
    parser = argparse.ArgumentParser(description="Idempotent Colab setup for the Day 22 lab.")
    parser.add_argument("--pull", action="store_true", help="Pull the latest code before refreshing the environment.")
    parser.add_argument("--force-install", action="store_true", help="Reinstall requirement groups even if they already satisfy the version constraints.")
    args = parser.parse_args()

    log("Day 22 lab - Colab setup")
    tier = detect_tier()
    log(f"Detected tier: {tier}")
    if tier == "CPU":
        log("No GPU detected. Switch the Colab runtime to T4/L4/A100 and rerun.")
        return 1

    if args.pull:
        maybe_pull_latest()

    state = load_state()
    fingerprint = notebook_fingerprint(tier)
    if state.get("fingerprint") == fingerprint and not args.force_install:
        log("Requirement fingerprint unchanged; skipping dependency install")
    else:
        install_requirements(REPO_ROOT / "requirements.txt", force=args.force_install)
        if tier == "BIGGPU":
            try:
                install_requirements(REPO_ROOT / "requirements-biggpu.txt", force=args.force_install)
            except subprocess.CalledProcessError:
                log("BigGPU extras failed to install; the vLLM notebook cell may be skipped")
        save_state({"fingerprint": fingerprint})

    convert_notebooks()
    update_env_file(tier)
    ensure_outputs()
    print_next_steps(tier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
