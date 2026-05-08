# Reflection — Lab 22 (DPO/ORPO Alignment)

**Ten:** Nguyen Trieu Gia Khanh  
**Cohort:** VinUni AICB Track 3  
**Tier da chay:** BIGGPU  
**Date:** 2026-05-08

> Emergency note: ban nop nay la ban provisional de kip deadline. Cac phan NB1/NB2 da co ket qua that tu Colab; cac phan DPO/benchmark se duoc thay bang so that sau khi run DPO hoan tat. Minh khong dien so gia cho metric chua chay xong.

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Colab Pro NVIDIA L4, ~23.7 GB VRAM |
| CUDA / driver | Colab CUDA 12.8 runtime; Torch 2.10.0+cu128 in run log |
| Base model | `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` |
| SFT dataset slice | `bkai-foundation-models/vi-alpaca` · 1000 samples · 1 epoch |
| Preference dataset slice | `argilla/ultrafeedback-binarized-preferences-cleaned` · DPO attempt used 2000 pairs · 1 epoch |
| `COMPUTE_TIER` env | `BIGGPU` |
| Total cost | Colab Pro subscription; exact compute-unit cost not recorded |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | n/a | PENDING: DPO run interrupted by xformers attention backend error |
| VRAM peak | L4 runtime available ~23.7 GB | PENDING: peak not recorded before error |
| Final loss | SFT trained successfully; exact final loss in notebook output | PENDING |
| Reward gap (chosen - rejected, end of training) | n/a | PENDING |
| Mean output length | Quick sanity sample was coherent but repetitive | PENDING |

**Tulu 3 reference numbers** (from deck §7.2b, for context only):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct)
- 70B-class scale; I do not expect to replicate these deltas with a 1B model and a small classroom run.

---

## 3. Reward curves analysis (>= 100 words)

`03-dpo-reward-curves.png`: PENDING after final DPO rerun.

The final DPO reward curves were not completed before the first submission deadline because the Colab L4 run hit an xformers `memory_efficient_attention_backward` kernel error during `trainer.train()`. I am not interpreting a fake curve here. The important diagnostic I will check after the rerun is not only the reward gap, but the two curves separately: `chosen_rewards` and `rejected_rewards`. If the gap increases because chosen rewards rise while rejected rewards stay stable or decrease slightly, that is a healthier DPO signal. If the gap grows mostly because rejected rewards collapse, that can indicate likelihood displacement, which the deck warns about in §3.4. I will also check whether the curve is noisy early and then stabilizes, or whether it keeps drifting aggressively, because the latter may mean beta or learning rate is too strong for this small model.

---

## 4. Qualitative comparison (>= 8 examples)

`04-side-by-side-table.png`: PENDING after NB4 rerun.

| # | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | Explain quicksort briefly in Vietnamese | SFT sample ran; answer was understandable but repetitive | PENDING | PENDING |
| 2 | helpfulness | Summarize a technical concept | PENDING | PENDING | PENDING |
| 3 | helpfulness | Write step-by-step advice | PENDING | PENDING | PENDING |
| 4 | helpfulness | Answer a factual question | PENDING | PENDING | PENDING |
| 5 | safety | Refuse unsafe request | PENDING | PENDING | PENDING |
| 6 | safety | Give safer alternative | PENDING | PENDING | PENDING |
| 7 | safety | Handle ambiguous harmful prompt | PENDING | PENDING | PENDING |
| 8 | safety | Avoid over-compliance | PENDING | PENDING | PENDING |

**Win/loss/tie summary:** PENDING  
**Judge used:** planned manual rubric or API judge, depending on final available runtime.

---

## 5. Beta trade-off

I did not run the beta sweep before the first deadline. My hypothesis is that `beta=0.1` should be the safest default for this run because it is strong enough to move the policy toward preferred answers without forcing the tiny 1B model too far from the SFT reference. With `beta=0.05`, I expect a larger reward gap but more risk of shorter or over-optimized responses. With `beta=0.5`, I expect a more conservative model that stays closer to SFT, giving smaller visible alignment gains but fewer regressions.

| beta | Reward gap | Win-rate (8 prompts) | Output length | Notes |
|---:|---:|---:|---:|---|
| 0.05 | PENDING | PENDING | PENDING | Expected stronger update, higher risk |
| 0.1 | PENDING | PENDING | PENDING | Default planned run |
| 0.5 | PENDING | PENDING | PENDING | Expected conservative update |

---

## 6. Personal reflection — single change that mattered most (>= 150 words)

The single decision that mattered most was switching the lab to a smaller and faster model, `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`, instead of trying to keep a larger model under deadline pressure. The alternative was to run the original or a larger 3B/7B setup, which might produce stronger quality if everything worked perfectly, but would also increase training time, VRAM risk, and debugging cost. I chose the 1B model because the goal of this lab is to demonstrate the full DPO pipeline: SFT adapter, preference data, DPO training, comparison, GGUF, and benchmark artifacts. A smaller model gives a better chance of completing the entire workflow on Colab within the submission window.

The result was mixed. The SFT and data preparation stages were much faster and did complete successfully. However, the DPO stage exposed a separate environment issue: xformers attention backward failed on the L4 runtime. That surprised me because the model itself was small enough; the blocker was not simply model size, but backend compatibility. If I redid the lab tomorrow, I would freeze the package/runtime setup earlier, disable or uninstall xformers before importing Unsloth on L4, and run a tiny 10-step DPO smoke test before doing the full run.

---

## 7. Benchmark interpretation (>= 150 words)

`07-benchmark-comparison.png`: PENDING after final NB6 run.

| Benchmark | SFT-only | SFT+DPO | Delta |
|---|---:|---:|---:|
| IFEval | PENDING | PENDING | PENDING |
| GSM8K | PENDING | PENDING | PENDING |
| MMLU (sampled) | PENDING | PENDING | PENDING |
| AlpacaEval-lite | PENDING | PENDING | PENDING |

The benchmark results were not finalized before the emergency submission, so I am leaving the table as pending rather than inventing numbers. My expected interpretation framework is the following. IFEval and AlpacaEval-lite are the most directly related to alignment behavior, so they are the metrics where I would most expect DPO to help if the preference data signal transfers. GSM8K may stay flat or regress because DPO on general preference pairs can make the model more conversational without improving mathematical reasoning. MMLU should ideally remain close to the SFT baseline because DPO should not erase factual knowledge, but a small 1B model may be noisy enough that changes are hard to interpret.

If AlpacaEval-lite improves while GSM8K drops, I would treat that as an example of alignment tax rather than a pure bug. If all metrics drop, I would suspect the DPO training was too aggressive, the prompt formatting was mismatched, or the model overfit to preference style. If IFEval improves with stable MMLU, that would be the best sign that DPO changed instruction-following without damaging general capability too much.

---

## Bonus

- [ ] Da lam beta-sweep (rigor add-on +6)
- [ ] Da push len HuggingFace Hub (Submission Option B, +5)
- [ ] Da release GGUF voi multiple quantizations (+3)
- [ ] Da link W&B run public (+2)
- [ ] Da lam cross-judge comparison (+4)
- [ ] Da lam `BONUS-CHALLENGE.md` provocation (ungraded — link `bonus/` folder)
- [ ] Pair work voi: N/A

---

## Dieu ngac nhien nhat khi lam lab nay

Dieu ngac nhien nhat la loi lon nhat khong den tu model qua to, ma den tu backend attention trong runtime Colab. Lab nay lam minh thay ro rang fine-tuning LLM khong chi la chon model va dataset, ma con la quan ly moi truong chay that can than.
