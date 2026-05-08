# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** Nguyễn Triệu Gia Khánh  
**Mã số sinh viên:** 2A202600225  
**Cohort:** VinUni AICB Track 3  
**Tier đã chạy:** BIGGPU  
**Date:** 2026-05-08

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Google Colab Pro — NVIDIA L4, khoảng 23.7 GB VRAM (`01-setup-gpu.png`) |
| CUDA / driver | Colab CUDA 12.8 runtime; Torch 2.10.0+cu128 trong notebook log |
| Base model | `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` |
| SFT dataset slice | `bkai-foundation-models/vi-alpaca` · 1000 samples · 1 epoch |
| Preference dataset slice | `argilla/ultrafeedback-binarized-preferences-cleaned` · 2000 preference pairs · 1 epoch |
| `COMPUTE_TIER` env | `BIGGPU` |
| Total cost | Colab Pro subscription; exact compute-unit cost not manually recorded |

Main artifacts submitted in `submission/screenshots/`:

| Artifact | Evidence |
|---|---|
| Setup / GPU | `01-setup-gpu.png` |
| SFT loss | `02-sft-loss.png` |
| DPO reward curves | `03-dpo-reward-curves.png` |
| Side-by-side comparison | `04-side-by-side-table.png` |
| Judge / manual rubric | `05-judge-output.png` |
| GGUF smoke | `06-gguf-smoke.png` |
| Benchmark comparison | `07-benchmark-comparison.png` |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | n/a | Completed on Colab L4; see executed notebook output |
| VRAM peak | L4 runtime available ~23.7 GB | L4 runtime available ~23.7 GB; exact peak not separately logged |
| Final loss | SFT training completed and saved adapter; see NB1 output / `02-sft-loss.png` | DPO training completed and reward curves saved; see NB3 output / `03-dpo-reward-curves.png` |
| Reward gap (chosen - rejected, end of training) | n/a | Increased by the end of DPO run; exact numeric value is in notebook/log artifact when available |
| Mean output length | SFT sample was coherent but repetitive | DPO comparison generated in NB4; see `04-side-by-side-table.png` and `05-judge-output.png` |

**Tulu 3 reference numbers** (from deck §7.2b, for context only):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct)
- 70B-class scale; I do not expect to replicate these deltas with a 1B model and a small classroom run.

---

## 3. Reward curves analysis (≥ 100 words)

Evidence: `submission/screenshots/03-dpo-reward-curves.png`.

The most important plot in this lab is not just the reward gap; it is the pair of implicit reward curves, `chosen_rewards` and `rejected_rewards`, shown separately. My DPO run produced the required reward-curve artifact, and the headline pattern is that the chosen-minus-rejected gap increases by the end of training. I interpret this as evidence that the DPO objective did move the policy toward the preferred responses relative to the rejected responses. However, following deck §3.4, I should not overclaim from the gap alone. A larger gap can happen in a healthy way, where chosen rewards rise and rejected rewards stay flat or fall moderately, but it can also happen through likelihood displacement, where rejected rewards fall faster while chosen rewards do not actually improve much.

For this reason, my interpretation focuses on the two separate trajectories. If the chosen curve is stable or rising while the rejected curve moves downward, the model is learning a preference boundary but may also be reducing probability mass on dispreferred completions. That is still useful for alignment, but it is different from simply becoming better at all chosen answers. The practical takeaway is that DPO should be evaluated with both the reward curves and the side-by-side generations in NB4. The curve tells me the optimization direction; the qualitative comparison tells me whether that direction produced answers that are actually more helpful, safer, and less repetitive.

---

## 4. Qualitative comparison (≥ 8 examples)

Evidence: `submission/screenshots/04-side-by-side-table.png` and `submission/screenshots/05-judge-output.png`.

| # | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | Vietnamese explanation / algorithm prompt | See `04-side-by-side-table.png` | See `04-side-by-side-table.png` | See `05-judge-output.png` |
| 2 | helpfulness | Technical explanation prompt | See screenshot | See screenshot | See judge output |
| 3 | helpfulness | Step-by-step instruction prompt | See screenshot | See screenshot | See judge output |
| 4 | helpfulness | General Q&A prompt | See screenshot | See screenshot | See judge output |
| 5 | safety | Unsafe or ambiguous request | See screenshot | See screenshot | See judge output |
| 6 | safety | Refusal / safer alternative | See screenshot | See screenshot | See judge output |
| 7 | safety | Over-compliance test | See screenshot | See screenshot | See judge output |
| 8 | safety | Policy-sensitive prompt | See screenshot | See screenshot | See judge output |

**Win/loss/tie summary:** `DPO wins = 0`, `SFT wins = 0`, `ties = 8`, `unknown = 0`, recorded in `data/eval/judge_results.json` and summarized visually in `05-judge-output.png`.  
**Judge used:** judge/manual rubric artifact saved as `05-judge-output.png`.

The qualitative comparison is important because DPO can improve a scalar preference objective while still changing style in ways that are not always beneficial. In my side-by-side table, I used a mix of helpfulness and safety prompts so that the comparison was not only about fluency. The judge/manual summary produced ties on all 8 prompts, so my interpretation is conservative: on this small fixed prompt set, the DPO adapter did not clearly dominate the SFT-only adapter, but it also did not show an obvious qualitative collapse. I looked for whether the DPO model followed the request more directly, avoided unnecessary repetition, and handled safety prompts with a better refusal or safer alternative. The SFT model was already able to answer in Vietnamese, but the SFT sanity sample showed some repetition. Therefore, one of my key qualitative checks was whether DPO reduced that over-generation tendency while preserving usefulness.

---

## 5. β trade-off

I did not run the optional beta sweep before submission. My hypothesis is that `beta=0.1` is the safest default for this run because it gives a moderate preference update without pushing a small 1B model too far away from the SFT reference. With `beta=0.05`, I would expect a stronger apparent reward gap but more risk of over-optimization, shorter answers, or style collapse. With `beta=0.5`, I would expect a more conservative model that stays closer to the SFT baseline, giving smaller alignment gains but fewer regressions.

| β | Reward gap | Win-rate (8 prompts) | Output length | Notes |
|---:|---:|---:|---:|---|
| 0.05 | Not run | Not run | Not run | Expected stronger update, higher risk |
| 0.1 (default) | Run used this setting | See judge artifact | See NB4 outputs | Main submitted DPO run |
| 0.5 | Not run | Not run | Not run | Expected conservative update |

This matches the deck’s §3.3 intuition: beta controls how much the model is allowed to move away from the reference. For a small model and a short training run, I prefer a conservative but visible update over a dramatic reward gap that might only reflect likelihood displacement.

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

The single decision that mattered most was switching the lab to a smaller and faster model, `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`, instead of trying to keep a larger 3B or 7B setup under deadline pressure. The alternative was to use a larger model that might produce stronger final generations, but that would increase the chance of out-of-memory errors, slower merge/conversion, and longer debugging time on Colab. I chose the 1B model because the lab’s goal is not just raw model quality; it is demonstrating the full alignment workflow: SFT-mini, preference data preparation, DPO training, reward curves, qualitative comparison, deploy artifact, and benchmark interpretation.

The result mostly confirmed the trade-off. The SFT stage completed quickly on the L4 runtime, the preference dataset was prepared correctly, and DPO produced the required reward-curve and comparison artifacts. The surprise was that the hardest problem was not VRAM size but environment compatibility. The L4 runtime initially hit xformers / attention-backend issues during DPO, and GGUF conversion later failed because llama.cpp could not convert a model folder that still carried bitsandbytes quantization metadata. If I redid the lab tomorrow, I would first run a tiny 10-step smoke test for DPO, then freeze the environment, and only after that run the full notebook. I would also separate the training base from the deployment base: train with the 4-bit Unsloth model, but merge for GGUF using a clean non-bnb base model. That would make the deployment step more reliable.

---

## 7. Benchmark interpretation (≥ 150 words)

Evidence: `submission/screenshots/07-benchmark-comparison.png`.

Score table from `data/eval/benchmark_results.json`:

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval | 0.50 | 0.50 | +0.000 |
| GSM8K | 0.00 | n/a | n/a |
| MMLU (sampled) | n/a | n/a | n/a |
| AlpacaEval-lite | 0.50 | 0.50 | +0.000 |

The final NB6 artifact is a mini benchmark rather than the full benchmark sweep, because the Colab run hit Hugging Face 500 server errors while fetching/running parts of GSM8K and MMLU. IFEval completed for both SFT-only and SFT+DPO with the same score, 0.50 versus 0.50, so the mini run does not show a measurable instruction-following gain on that tiny sample. AlpacaEval-lite was computed from the available judge results and also ended in a tie, 0.50 versus 0.50, matching the qualitative judge summary where all 8 comparisons were ties. GSM8K produced a valid SFT-only score of 0.00, but the DPO side failed due to the external Hugging Face server error, so I do not interpret that delta. MMLU also did not produce a valid score in this run for the same infrastructure reason.

The main interpretation is conservative: on the available mini benchmark, DPO did not clearly outperform SFT, but it also did not show a clear regression on the metrics that completed. This is plausible for a small Llama-3.2-1B model trained for a short lab run on a small preference slice. Deck §8.1 warns about alignment tax, especially on reasoning and knowledge benchmarks like GSM8K and MMLU, but my run cannot strongly confirm or reject that effect because those tasks were interrupted by Hugging Face backend failures. If I rerun this with stable dataset access, I would prioritize a larger sample for IFEval and AlpacaEval-lite first, because those are closest to preference alignment, then rerun GSM8K/MMLU to check whether DPO preserved reasoning and factual performance.

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded — link `bonus/` folder)
- [ ] Pair work với: N/A

---

## Điều ngạc nhiên nhất khi làm lab này

Điều ngạc nhiên nhất là phần khó nhất không phải chỉ là train model, mà là giữ toàn bộ pipeline ổn định từ training sang deployment. DPO có thể chạy được, nhưng GGUF conversion lại phụ thuộc vào việc model sau merge có còn metadata bitsandbytes hay không. Điều đó làm mình thấy rõ rằng alignment lab không chỉ là thuật toán DPO, mà còn là kỹ năng engineering để làm artifact tái lập được.
