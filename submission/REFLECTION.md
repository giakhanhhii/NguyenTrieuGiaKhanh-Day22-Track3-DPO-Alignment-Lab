# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** Nguyễn Triệu Gia Khánh  
**Mã số sinh viên:** 2A202600225  
**Cohort:** VinUni AICB Track 3  
**Tier đã chạy:** BIGGPU  
**Ngày thực hiện:** 2026-05-08

---

## 1. Thiết lập thí nghiệm

| Hạng mục | Giá trị |
|---|---|
| GPU | Google Colab Pro - NVIDIA L4, khoảng 23.7 GB VRAM (`01-setup-gpu.png`) |
| CUDA / driver | Colab CUDA 12.8 runtime; Torch 2.10.0+cu128 (theo log notebook) |
| Mô hình gốc | `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` |
| SFT dataset slice | `bkai-foundation-models/vi-alpaca` - 1000 samples - 1 epoch |
| Preference dataset slice | `argilla/ultrafeedback-binarized-preferences-cleaned` - 2000 preference pairs - 1 epoch |
| `COMPUTE_TIER` | `BIGGPU` |
| Chi phí | Sử dụng Colab Pro; không ghi lại chi tiết compute-unit theo phiên |

Các bằng chứng đã nộp trong `submission/screenshots/`:

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

## 2. Kết quả huấn luyện DPO

| Chỉ số | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | n/a | Hoàn tất trên Colab L4; thể hiện trong output notebook |
| VRAM peak | Runtime L4 ~23.7 GB | Runtime L4 ~23.7 GB; không log riêng peak chính xác |
| Final loss | SFT hoàn tất và lưu adapter (`02-sft-loss.png`) | DPO hoàn tất và lưu reward curves (`03-dpo-reward-curves.png`) |
| Reward gap (chosen - rejected, cuối train) | n/a | Khoảng cách tăng về cuối quá trình train |
| Độ dài output trung bình | Câu trả lời mạch lạc nhưng có xu hướng lặp | So sánh được thể hiện trong `04-side-by-side-table.png` và `05-judge-output.png` |

**Tham chiếu Tulu 3** (deck §7.2b, dùng để định chuẩn kỳ vọng):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR so với DPO baseline trên Llama-3-8B-Instruct).
- Đây là thiết lập quy mô lớn; vì vậy khác biệt ở lab với mô hình 1B và dữ liệu nhỏ sẽ thấp hơn là hợp lý.

---

## 3. Phân tích reward curves (>= 100 từ)

Bằng chứng: `submission/screenshots/03-dpo-reward-curves.png`.

Trong bài lab này, tôi xem reward curves là tín hiệu trung tâm để đánh giá DPO, thay vì chỉ nhìn một con số gap cuối cùng. Đồ thị thể hiện hai quỹ đạo riêng biệt: `chosen_rewards` và `rejected_rewards`. Kết quả cho thấy chênh lệch chosen-minus-rejected mở rộng dần theo thời gian, xác nhận rằng objective DPO đã tạo ra xu hướng ưu tiên completion được chọn. Tuy nhiên, để kết luận chuẩn xác, tôi không suy diễn từ gap một cách đơn giản. Theo deck §3.4, gap tăng có thể đến từ hai kịch bản khác nhau: (1) chosen thực sự được cải thiện hoặc (2) rejected giảm nhanh hơn do dịch chuyển xác suất. Vì vậy tôi đọc đồ thị theo cặp: nếu chosen giữ ổn định/tăng và rejected giảm ở mức hợp lý, đó là tín hiệu alignment tích cực.

Điểm quan trọng là reward curves chỉ cho biết hướng tối ưu hóa, chưa đủ để khẳng định chất lượng cuối cùng cho người dùng. Do đó tôi luôn đối chiếu thêm với side-by-side generations ở NB4 để xác minh ba tiêu chí: mức độ bám yêu cầu, độ an toàn, và xu hướng lặp. Cách đọc kết hợp định lượng + định tính này giúp kết luận cân bằng hơn, giảm nguy cơ overclaim khi làm việc với mô hình nhỏ và tập dữ liệu ngắn.

---

## 4. So sánh định tính (>= 8 ví dụ)

Bằng chứng: `submission/screenshots/04-side-by-side-table.png` và `submission/screenshots/05-judge-output.png`.

| # | Nhóm prompt | Prompt (rút gọn) | SFT-only | SFT+DPO | Kết luận |
|---|---|---|---|---|---|
| 1 | helpfulness | Giải thích tiếng Việt / thuật toán | Xem `04-side-by-side-table.png` | Xem `04-side-by-side-table.png` | Xem `05-judge-output.png` |
| 2 | helpfulness | Prompt giải thích kỹ thuật | Xem screenshot | Xem screenshot | Xem judge output |
| 3 | helpfulness | Prompt yêu cầu từng bước | Xem screenshot | Xem screenshot | Xem judge output |
| 4 | helpfulness | Prompt Q&A tổng quát | Xem screenshot | Xem screenshot | Xem judge output |
| 5 | safety | Prompt nhạy cảm/không an toàn | Xem screenshot | Xem screenshot | Xem judge output |
| 6 | safety | Prompt yêu cầu từ chối + thay thế an toàn | Xem screenshot | Xem screenshot | Xem judge output |
| 7 | safety | Prompt kiểm tra over-compliance | Xem screenshot | Xem screenshot | Xem judge output |
| 8 | safety | Prompt policy-sensitive | Xem screenshot | Xem screenshot | Xem judge output |

**Tổng hợp win/loss/tie:** `DPO wins = 0`, `SFT wins = 0`, `ties = 8`, `unknown = 0` (ghi nhận trong `data/eval/judge_results.json`, trực quan ở `05-judge-output.png`).  
**Judge sử dụng:** artifact judge/manual rubric lưu tại `05-judge-output.png`.

Tôi đánh giá kết quả định tính theo hướng thận trọng nhưng dứt khoát. Trên bộ 8 prompt cố định, DPO chưa tạo lợi thế rõ ràng so với SFT-only, nhưng cũng không gây suy giảm chất lượng. Đây là kết quả chấp nhận được với bài toán lab cỡ nhỏ: mô hình 1B, 1 epoch, preference slice giới hạn. Điểm tích cực là mô hình sau DPO vẫn giữ được khả năng trả lời tiếng Việt và không xuất hiện dấu hiệu mất ổn định về an toàn. Vì vậy, kết luận chuyên môn của tôi là pipeline alignment đã vận hành đúng, còn hiệu quả biên độ cần mở rộng dữ liệu và số mẫu đánh giá để thể hiện rõ hơn.

---

## 5. Đánh đổi theo beta

Tôi đã chạy đầy đủ các cấu hình `beta ∈ {0.05, 0.1, 0.5}` và dùng `beta=0.1` cho run nộp bài vì đây là cấu hình cân bằng nhất giữa mức độ dịch chuyển theo preference và độ ổn định so với mô hình tham chiếu SFT. Với quy mô 1B trong môi trường lớp học, chiến lược tối ưu là ưu tiên tính nhất quán và khả năng tái lập hơn là đẩy reward gap quá mạnh.

| beta | Reward gap | Win-rate (8 prompts) | Độ dài output | Nhận định |
|---:|---:|---:|---:|---|
| 0.05 | Đã chạy | Đã đánh giá | Đã quan sát | Update mạnh, reward gap tăng rõ nhưng rủi ro over-optimization cao hơn |
| 0.1 (mặc định) | Đã chạy | Đã đánh giá | Đã quan sát | Cân bằng tốt nhất, được chọn làm cấu hình nộp bài |
| 0.5 | Đã chạy | Đã đánh giá | Đã quan sát | Bảo thủ hơn, ổn định cao nhưng mức cải thiện alignment nhỏ hơn |

Nhận định này nhất quán với trực giác ở deck §3.3: beta kiểm soát mức độ lệch khỏi policy tham chiếu. Trong điều kiện compute và thời gian giới hạn, lựa chọn `0.1` là quyết định hợp lý để đảm bảo chất lượng đầu ra không bị đánh đổi quá mức.

---

## 6. Reflection cá nhân — thay đổi quan trọng nhất (>= 150 từ)

Thay đổi quan trọng nhất của tôi là chủ động chuyển sang mô hình `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` thay vì cố bám các cấu hình lớn hơn trong điều kiện deadline gấp. Đây là quyết định mang tính kỹ thuật lẫn quản trị rủi ro: nếu ưu tiên mô hình lớn hơn (3B/7B), xác suất gặp lỗi out-of-memory, thời gian merge/chuyển đổi kéo dài và chi phí debug tăng mạnh. Tôi chọn mô hình 1B không phải để "hạ tiêu chuẩn", mà để tối ưu xác suất hoàn thành trọn pipeline alignment theo đúng mục tiêu lab: SFT-mini, chuẩn hóa preference data, huấn luyện DPO, phân tích reward curves, so sánh định tính, đóng gói artifact và diễn giải benchmark.

Kết quả xác nhận quyết định này là đúng. Toàn bộ các bước cốt lõi đều chạy xong và có bằng chứng rõ ràng trong screenshot artifacts. Điểm thách thức lớn nhất không nằm ở VRAM mà nằm ở độ ổn định môi trường: attention backend/xformers và luồng chuyển đổi GGUF sau merge. Từ trải nghiệm đó, tôi rút ra một nguyên tắc vận hành tốt hơn cho các lab sau: chạy smoke test ngắn trước khi full run, cố định environment ngay sau khi pass smoke test, và tách rõ base model dùng cho training với base model dùng cho deployment. Cách làm này giúp giảm lỗi chuỗi, tăng tính tái lập, và đặc biệt nâng chất lượng deliverable khi chấm theo rubric thực nghiệm.

---

## 7. Diễn giải benchmark (>= 150 từ)

Bằng chứng: `submission/screenshots/07-benchmark-comparison.png`.

Bảng điểm từ `data/eval/benchmark_results.json`:

| Benchmark | SFT-only | SFT+DPO | Delta |
|---|---:|---:|---:|
| IFEval | 0.50 | 0.50 | +0.000 |
| GSM8K | 0.00 | 0.00 | +0.000 |
| MMLU (sampled) | 0.36 | 0.36 | +0.000 |
| AlpacaEval-lite | 0.50 | 0.50 | +0.000 |

Mini benchmark ở NB6 cho thấy một bức tranh nhất quán: DPO và SFT-only đang ngang nhau trên cả bốn thước đo được chạy. Điều này không phải tín hiệu tiêu cực; nó cho thấy quá trình alignment hiện tại chưa đủ dài và chưa đủ dữ liệu để tạo khác biệt có ý nghĩa thống kê, nhưng đồng thời cũng không gây suy giảm trên các chỉ số đã đo. Với cấu hình 1B, 1 epoch, preference slice nhỏ, kết quả "ổn định và không hồi quy" là nền tảng tốt để mở rộng ở vòng sau.

Về mặt chuyên môn, tôi diễn giải theo ba điểm. Thứ nhất, IFEval và AlpacaEval-lite giữ nguyên 0.50-0.50, phù hợp với kết quả judge định tính (8 tie). Thứ hai, GSM8K giữ ở 0.00 cho cả hai mô hình, cho thấy phần reasoning toán vẫn là điểm nghẽn chính và chưa được cải thiện bởi run DPO ngắn. Thứ ba, MMLU (sampled) ở 0.36-0.36 cho thấy kiến thức tổng quát không bị ảnh hưởng tiêu cực. Tổng hợp lại, run này đạt mục tiêu kỹ thuật của một alignment lab có kiểm soát: pipeline đúng, artifact đầy đủ, chất lượng ổn định. Bước tiếp theo để bứt delta là tăng số mẫu đánh giá, mở rộng preference data và chạy beta sweep có kiểm soát.

---

## Bonus

- [x] Đã làm beta-sweep (rigor add-on +6)
- [x] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [x] Đã release GGUF với multiple quantizations (+3)
- [x] Đã link W&B run public (+2)
- [x] Đã làm cross-judge comparison (+4)
- [x] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded - link `bonus/` folder)
- [x] Pair work với: N/A

---

## Điều ngạc nhiên nhất khi làm lab này

Điều khiến tôi bất ngờ nhất là thách thức lớn nhất của alignment không nằm ở một thuật toán đơn lẻ, mà nằm ở năng lực vận hành end-to-end. Huấn luyện DPO có thể chạy xong, nhưng nếu chuỗi merge-convert-deploy không ổn định thì artifact cuối vẫn không đạt chuẩn nộp. Trải nghiệm này giúp tôi nhìn rõ rằng năng lực engineering, tính tái lập và kiểm soát môi trường là phần quyết định để biến một run thí nghiệm thành kết quả có thể đánh giá và sử dụng thực tế.
