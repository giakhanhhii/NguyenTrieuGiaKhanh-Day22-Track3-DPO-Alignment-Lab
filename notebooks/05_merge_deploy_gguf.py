# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB5 — Merge + Deploy + GGUF
#
# **Stack:** Unsloth `merge_and_unload` + `save_pretrained_gguf(quantization='Q4_K_M')`
# + llama-cpp-python smoke test.
# Maps to deck §7.1 lab brief: "merge adapter, quantize GGUF, serve với vLLM".
#
# > **Mục tiêu:** export the SFT+DPO adapter as a deployable GGUF Q4_K_M file
# > (~1.5 GB on 3B / ~4 GB on 7B), then smoke-test it through llama-cpp-python.
# > Final cell shows the optional vLLM serving command (BigGPU only).

# %% [markdown]
# ## 0. Setup

# %%
import os
import json
import sys
from pathlib import Path

ROOT_FOR_IMPORT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from scripts.lab_utils import ensure_chat_template, get_repo_root, load_lab_env, render_text_card

REPO_ROOT = get_repo_root()
load_lab_env(REPO_ROOT)
COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()
BASE_MODEL = (
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit" if COMPUTE_TIER == "T4"
    else "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
)
MAX_LEN = 512 if COMPUTE_TIER == "T4" else 1024

DPO_PATH = REPO_ROOT / "adapters" / "dpo"
MERGED_PATH = REPO_ROOT / "adapters" / "merged-fp16"
GGUF_DIR = REPO_ROOT / "gguf"
MERGED_PATH.mkdir(parents=True, exist_ok=True)
GGUF_DIR.mkdir(parents=True, exist_ok=True)

assert DPO_PATH.exists(), "NB3 must run first"

print(f"COMPUTE_TIER:    {COMPUTE_TIER}")
print(f"DPO adapter:     {DPO_PATH}")
print(f"merged output:   {MERGED_PATH}")
print(f"GGUF output:     {GGUF_DIR}")

# %%
import torch

assert torch.cuda.is_available()

# %% [markdown]
# ## 1-3. Merge adapters and quantize to GGUF Q4_K_M
#
# The training base is a bitsandbytes 4-bit model, but llama.cpp cannot convert
# HF folders that still carry `quantization_config: bitsandbytes`. The helper
# script below therefore uses a full/non-bnb base for GGUF export when available
# (`GGUF_BASE_MODEL`, default: `unsloth/Llama-3.2-1B-Instruct`). If that base is
# unavailable or gated, it writes a clear pending screenshot instead of crashing
# the notebook.

# %%
import subprocess

cmd = [
    sys.executable,
    str(REPO_ROOT / "scripts" / "merge_and_gguf.py"),
    "--sft-path",
    str(REPO_ROOT / "adapters" / "sft-mini"),
    "--dpo-path",
    str(DPO_PATH),
    "--merged-output",
    str(REPO_ROOT / "adapters" / "merged-clean"),
    "--gguf-output",
    str(GGUF_DIR),
    "--quant",
    "Q4_K_M",
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)

# %% [markdown]
# ### 3a. Optional — additional quantization tiers (for the +3 rigor add-on)

# %%
# Uncomment if you want Q5_K_M + Q8_0 too (~2× total disk space).
# Each adds ~30s for an extra GGUF file.
#
# model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method="q5_k_m")
# model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method="q8_0")

# %%
import os

print("GGUF files:")
for p in sorted(GGUF_DIR.iterdir()):
    if p.suffix == ".gguf":
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name:50s}  {size_mb:>8.1f} MB")

torch.cuda.empty_cache()

# %% [markdown]
# ## 4. Smoke test with llama-cpp-python

# %%
from llama_cpp import Llama

# Find the Q4_K_M GGUF
gguf_files = list(GGUF_DIR.glob("*Q4_K_M*.gguf")) + list(GGUF_DIR.glob("*q4_k_m*.gguf"))
if not gguf_files:
    gguf_path = None
    llm = None
    print("No Q4_K_M GGUF found. A pending deploy artifact should already be in submission/screenshots/.")
else:
    gguf_path = gguf_files[0]
    print(f"Loading: {gguf_path.name}")

    # n_gpu_layers=-1 offloads all layers to GPU if compiled with CUDA/Metal/Vulkan
    llm = Llama(
        model_path=str(gguf_path),
        n_ctx=MAX_LEN,
        n_gpu_layers=-1,           # all layers on GPU; falls back to CPU if no GPU compile
        verbose=False,
    )
    print("Loaded.")

# %% [markdown]
# ### 4a. Smoke prompt + response (deliverable: `06-gguf-smoke.png`)

# %%
screenshot_dir = REPO_ROOT / "submission" / "screenshots"
screenshot_dir.mkdir(parents=True, exist_ok=True)
SMOKE_PROMPT = "Giải thích ngắn gọn (3 câu) cách thuật toán Bubble sort hoạt động."

if llm is None:
    response = None
    render_text_card(
        screenshot_dir / "06-gguf-smoke.png",
        "GGUF smoke test - pending",
        [
            "Status : GGUF file not available",
            "Reason : converter skipped or failed in step 1-3",
            "",
            "Completed before this deploy step:",
            "- SFT adapter",
            "- Preference data",
            "- DPO adapter / reward analysis if available",
        ],
        height=5.0,
    )
else:
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": SMOKE_PROMPT}],
        max_tokens=200,
        temperature=0.0,
    )

    print(f"PROMPT:\n  {SMOKE_PROMPT}\n")
    print(f"RESPONSE (Q4_K_M GGUF, llama-cpp-python):\n  {response['choices'][0]['message']['content']}")
    print(f"\nTokens used: {response['usage']}")

    render_text_card(
        screenshot_dir / "06-gguf-smoke.png",
        "GGUF smoke test",
        [
            f"GGUF file    : {gguf_path.name}",
            f"Quantization : Q4_K_M",
            f"Prompt       : {SMOKE_PROMPT}",
            "",
            "Response:",
            response["choices"][0]["message"]["content"],
        ],
        height=5.3,
    )
print(f"Saved GGUF smoke screenshot to {screenshot_dir / '06-gguf-smoke.png'}")

# %% [markdown]
# ## 5. Optional — vLLM serving (BigGPU only)
#
# vLLM provides production-grade OpenAI-compatible serving. **Requires CUDA GPU
# with ≥ 16 GB VRAM** and `vllm` installed (see `requirements-biggpu.txt`).
# On T4 tier this cell will OOM. Skip on T4.
#
# Run in a SEPARATE terminal (NOT in the notebook — vLLM blocks until killed):
#
# ```bash
# pip install vllm                         # once
# vllm serve adapters/merged-fp16 \
#   --port 8000 \
#   --max-model-len 1024 \
#   --gpu-memory-utilization 0.9
# ```
#
# Then test:
#
# ```bash
# curl http://localhost:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{"model": "merged-fp16", "messages": [{"role": "user", "content": "Hello"}]}'
# ```
#
# **Why not in the notebook?** vLLM's process model doesn't play nicely with
# Jupyter — it expects to own the GPU + a long-running HTTP server. Run it as
# a sidecar process. The deck mentions vLLM as the deploy target; for actual
# production you'd containerize this command. For the lab, llama-cpp-python in
# step 4 is the graded artifact.

# %% [markdown]
# ## 6. Save deployment metadata

# %%
deploy_meta = {
    "compute_tier": COMPUTE_TIER,
    "base_model": BASE_MODEL,
    "merged_path": str(MERGED_PATH),
    "gguf_path": str(gguf_path) if gguf_path else None,
    "gguf_size_mb": round(gguf_path.stat().st_size / 1e6, 1) if gguf_path else None,
    "quantization": "q4_k_m",
    "smoke_prompt": SMOKE_PROMPT,
    "smoke_response": response["choices"][0]["message"]["content"] if response else None,
    "status": "ok" if gguf_path else "pending",
}
(REPO_ROOT / "data" / "eval" / "deploy_meta.json").parent.mkdir(parents=True, exist_ok=True)
(REPO_ROOT / "data" / "eval" / "deploy_meta.json").write_text(
    json.dumps(deploy_meta, ensure_ascii=False, indent=2)
)
print("Saved data/eval/deploy_meta.json")

# %% [markdown]
# ## 7. Submission checklist
#
# Bạn vừa hoàn thành core lab. Trước khi submit:
#
# 1. **Run** `make verify` — gatekeeper sẽ list missing artifacts.
# 2. **Take screenshots** vào `submission/screenshots/` (xem `submission/screenshots/README.md`).
# 3. **Fill** `submission/REFLECTION.md` — đặc biệt là § 3 (reward curves analysis,
#    cross-reference deck §3.4) và § 6 (single change that mattered most).
# 4. **(Optional)** Pick a rigor add-on từ rubric.md (β-sweep, HF push, GGUF
#    release, W&B link, cross-judge).
# 5. **(Optional)** Pick a `BONUS-CHALLENGE.md` provocation cho creative bonus.
#
# Push public repo + paste URL vào VinUni LMS Day-22 box.
#
# Câu hỏi cuối để brainstorm trước khi đóng laptop:
#
# > **The deck says:** "DPO + 30 min A100 + 2k UltraFeedback → 3.2 → 4.1 helpfulness."
# > **You measured:** _<your win-rate from NB4>_.
# > **Why might they differ?** Dataset (English vs VN), base model (Llama-3.2-1B vs
# > deck's unspecified base), judge bias, sample size (8 prompts vs deck's full eval).
# > Đó chính là § 6 trong REFLECTION — what 1 change would close the gap.
