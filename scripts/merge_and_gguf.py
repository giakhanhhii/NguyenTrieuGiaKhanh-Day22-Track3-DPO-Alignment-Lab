#!/usr/bin/env python3
"""Merge SFT+DPO adapters and export GGUF.

This script intentionally avoids converting from a bitsandbytes 4-bit base.
llama.cpp's HF-to-GGUF converter cannot dequantize HF folders that still carry
`quantization_config: bitsandbytes`, which is the failure seen in Colab:

    NotImplementedError: Quant method is not yet supported: 'bitsandbytes'

Use a full/non-bnb base for GGUF export via `--base-model` or `GGUF_BASE_MODEL`.
If export still fails, the script writes a clear pending screenshot instead of
crashing the whole submission flow.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRAIN_BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
DEFAULT_GGUF_BASE = TRAIN_BASE_MODEL.removesuffix("-bnb-4bit")


def _find_converter() -> Path:
    candidates = [
        Path("/root/.unsloth/llama.cpp/convert_hf_to_gguf.py"),
        Path("/root/.unsloth/llama.cpp/unsloth_convert_hf_to_gguf.py"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "llama.cpp converter not found. Run the previous Unsloth GGUF cell once "
        "or install/build llama.cpp before running this script."
    )


def _find_quantize() -> Path:
    candidates = [
        Path("/root/.unsloth/llama.cpp/build/bin/llama-quantize"),
        Path("/root/.unsloth/llama.cpp/build/bin/quantize"),
        Path("/root/.unsloth/llama.cpp/llama-quantize"),
        Path("/root/.unsloth/llama.cpp/quantize"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("llama.cpp quantize binary not found.")


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _write_pending(repo: Path, reason: str) -> None:
    from scripts.lab_utils import render_text_card

    screenshot_dir = repo / "submission" / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    render_text_card(
        screenshot_dir / "06-gguf-smoke.png",
        "GGUF smoke test - pending",
        [
            "Status : GGUF conversion skipped / failed",
            f"Reason : {reason}",
            "",
            "Completed before this deploy step:",
            "- SFT adapter",
            "- Preference data",
            "- DPO adapter / reward analysis if available",
            "",
            "Next fix:",
            "Use a non-bnb base model, merge adapters, then convert to GGUF.",
        ],
        height=5.4,
    )
    eval_dir = repo / "data" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "deploy_meta.json").write_text(
        json.dumps(
            {
                "status": "pending",
                "reason": reason,
                "quantization": "q4_k_m",
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Wrote pending artifact to {screenshot_dir / '06-gguf-smoke.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=os.environ.get("GGUF_BASE_MODEL", DEFAULT_GGUF_BASE))
    parser.add_argument("--sft-path", default=str(REPO / "adapters" / "sft-mini"))
    parser.add_argument("--dpo-path", default=str(REPO / "adapters" / "dpo"))
    parser.add_argument("--merged-output", default=str(REPO / "adapters" / "merged-clean"))
    parser.add_argument("--gguf-output", default=str(REPO / "gguf"))
    parser.add_argument("--quant", default="Q4_K_M")
    parser.add_argument("--no-pending-card", action="store_true")
    args = parser.parse_args()

    repo = REPO
    base = args.base_model
    merged = Path(args.merged_output)
    gguf_dir = Path(args.gguf_output)
    sft_path = Path(args.sft_path)
    dpo_path = Path(args.dpo_path)
    merged.mkdir(parents=True, exist_ok=True)
    gguf_dir.mkdir(parents=True, exist_ok=True)

    print(f"GGUF base model: {base}")
    print(f"SFT adapter:     {sft_path}")
    print(f"DPO adapter:     {dpo_path}")
    print(f"Merged output:   {merged}")
    print(f"GGUF output:     {gguf_dir}")

    try:
        if "bnb" in base.lower() or "4bit" in base.lower():
            raise RuntimeError(
                f"Refusing GGUF export from quantized base {base!r}; set GGUF_BASE_MODEL "
                "to a full/non-bnb model such as meta-llama/Llama-3.2-1B-Instruct."
            )

        import gc

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        model = PeftModel.from_pretrained(model, str(sft_path))
        model = model.merge_and_unload()
        print("Merged SFT adapter")

        model = PeftModel.from_pretrained(model, str(dpo_path))
        model = model.merge_and_unload()
        print("Merged DPO adapter")

        model.save_pretrained(str(merged), safe_serialization=True)
        tokenizer.save_pretrained(str(merged))
        print(f"Saved clean merged HF model to {merged}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

        converter = _find_converter()
        quantize = _find_quantize()

        f16_gguf = gguf_dir / "merged-clean.F16.gguf"
        q_gguf = gguf_dir / f"merged-clean.{args.quant}.gguf"

        _run(
            [
                "python",
                str(converter),
                "--outfile",
                str(f16_gguf),
                "--outtype",
                "f16",
                str(merged),
            ]
        )
        _run([str(quantize), str(f16_gguf), str(q_gguf), args.quant])

        print(f"\nSaved GGUF: {q_gguf}")
        for path in sorted(gguf_dir.glob("*.gguf")):
            print(f"  {path.name:40s} {path.stat().st_size / 1e6:8.1f} MB")

    except Exception as exc:
        if args.no_pending_card:
            raise
        _write_pending(repo, str(exc))
        print(f"GGUF export pending: {exc}")


if __name__ == "__main__":
    main()
