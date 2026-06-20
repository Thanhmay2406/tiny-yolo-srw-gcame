# G-CAME-guided SRW YOLO cho Tiny Object Detection — SkyFusion-only Kaggle Roadmap

> **Phiên bản điều chỉnh:** SkyFusion-only / Kaggle-first  
> **Dataset thử nghiệm duy nhất hiện tại:** SkyFusion tại `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo` trên Kaggle  
> **Tên dataset hiển thị:** SkyFusion: Aerial Object Detection  
> **Môi trường ưu tiên:** Kaggle Notebook + Kaggle GPU  
> **Đóng góp chính:** `SRW module` — Saliency-guided ReWeighting module đặt sau YOLO FPN neck  
> **Đóng góp hỗ trợ:** `L_sal` — saliency alignment loss giữa model saliency và Gaussian GT mask  
> **XAI branch:** G-CAME hoặc Grad-CAM-like extractor, bắt đầu bằng Grad-CAM-like ổn định trước  
> **Mục tiêu trước mắt:** không ôm nhiều benchmark; chỉ dùng SkyFusion để kiểm chứng pipeline, debug SRW, debug `L_sal`, và chạy ablation tối thiểu.
> **Cập nhật dataset structure:** roadmap này dùng đúng `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion` cho COCO gốc nếu cần và `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo` cho YOLO train/eval; split validation là `valid`.

---


## Cập nhật mới nhất theo đường dẫn bạn chốt

Roadmap này đã chuyển source data chính sang:

```text
/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo
```

Vì đây là thư mục trong `/kaggle/input`, nó là **read-only**. Do đó:

```text
- Baseline/training/evaluation đọc từ $SKYFUSION_DATA.
- Không cần copy dataset sang /kaggle/working nếu data.yaml hoạt động.
- Nếu cần convert lại từ COCO, output phải ghi sang /kaggle/working/data/SkyFusion_yolo_reconverted.
- Nếu data.yaml trong input sai path, tạo configs/dataset/skyfusion_kaggle.yaml làm fallback.
```

## Audit nhanh bản roadmap này

Bản này đã được rà lại theo 4 nhóm lỗi dễ gây hỏng implementation:

```text
1. Dataset path và chữ hoa/thường trên Linux:
   - Dùng đúng `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion` cho COCO gốc nếu cần convert lại.
   - Dùng đúng `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo` cho YOLO train/eval trên Kaggle.
   - Không dùng lẫn `data/skyfusion_yolo` chữ thường hoặc path local cũ.

2. Split validation:
   - Split thư mục chính thức là valid.
   - Không tạo thêm thư mục val/.
   - Trong data.yaml của Ultralytics vẫn có thể dùng key val: trỏ tới images/valid.

3. Luồng saliency trong training:
   - Không giả định Grad-CAM/G-CAME có thể sinh saliency trước detection head trong cùng forward pass.
   - Bản ổn định dùng saliency head differentiable S_pred cho SRW và L_sal.
   - Grad-CAM-like/G-CAME dùng cho debug, evaluation, hoặc teacher offline.

4. Tính công bằng thực nghiệm:
   - Baseline và TradAug không import SRW/L_sal.
   - SRW-only, L_sal-only và SRW+L_sal phải dùng cùng split, model, imgsz, epochs, seed.
   - Không fabricate AP_small/Recall_tiny nếu chưa implement đúng.
```

## 0. Lý do điều chỉnh roadmap

Roadmap trước đó đặt mục tiêu hỗ trợ nhiều public benchmark:

```text
VisDrone
AI-TOD
AI-TOD-v2
SODA-D
private_yolo
```

Nhưng giai đoạn hiện tại nên đơn giản hơn:

```text
Chỉ dùng một dataset Kaggle input thực tế:
/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo
```

Lý do:

```text
1. Giảm thời gian chuẩn bị dataset.
2. Không phải viết nhiều converter ngay từ đầu.
3. Tập trung vào kiến trúc SRW + L_sal.
4. Dễ debug trên Kaggle vì dataset đã nằm trong /kaggle/input.
5. Dễ chạy smoke test nhiều lần.
6. Khi pipeline ổn mới mở rộng sang AI-TOD-v2 / VisDrone / SODA.
```

Tư duy mới:

```text
SkyFusion dataset
→ kiểm tra cấu trúc dataset đã biết
→ ưu tiên dùng bản YOLO sẵn có
→ convert từ COCO chỉ khi bản YOLO thiếu/lỗi
→ train YOLO baseline
→ train traditional augmentation baseline
→ debug GT saliency mask
→ debug XAI saliency map
→ test SRW module
→ nhúng SRW vào YOLO FPN
→ train L_sal only
→ train SRW only
→ train SRW + L_sal
→ lambda schedule
→ energy/bg/size-aware ablation
→ evaluation + figures + export artifact
```

---

## 1. Định vị đề tài sau khi dùng SkyFusion-only

### 1.1. Tên đề tài khuyến nghị

Tên ngắn:

```text
G-CAME-guided Saliency Reweighted YOLO for Tiny Object Detection in Aerial Imagery
```

Tên đầy đủ hơn:

```text
G-CAME-guided Saliency Reweighting and Alignment for YOLO-based Tiny Object Detection in Aerial Imagery
```

Tên thực dụng cho giai đoạn Kaggle:

```text
A Kaggle-reproducible SRW-YOLO Framework for Tiny Aerial Object Detection
```

### 1.2. Dataset scope

Giai đoạn này **chỉ dùng SkyFusion**:

```text
Kaggle input thực tế:
/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo

Kaggle dataset owner/slug đang dùng trong notebook:
thanhmay2406/dataset-for-research

Kỳ vọng nội dung:
- ảnh vệ tinh/aerial
- object nhỏ/tiny
- các class như vehicle / ship / aircraft
- dataset có cả bản COCO gốc và bản YOLO đã chuẩn hóa; vẫn cần inspect để xác nhận path thực tế trên Kaggle
```

### 1.2.1. Cấu trúc dataset thực tế cần dùng

Roadmap hiện tại phải bám theo cấu trúc SkyFusion. Trong bản mô tả repo gốc, cấu trúc tương đương như sau:

```text
data/SkyFusion/
├── README.roboflow.txt
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/
    ├── _annotations.coco.json
    └── *.jpg
```

Ý nghĩa:

```text
- Đây là dataset gốc.
- Annotation ở định dạng COCO JSON.
- Ảnh và file _annotations.coco.json nằm chung trong từng split.
- Bbox COCO dùng dạng xywh theo pixel.
- Split chính thức là train / valid / test, không dùng tên val.
```

Bản YOLO đã chuẩn hóa. Trên Kaggle hiện tại, thư mục này tương ứng với `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo`:

```text
data/SkyFusion_yolo/
├── data.yaml
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
└── labels/
    ├── train/
    ├── valid/
    └── test/
```

Quy ước sử dụng:

```text
- Train/eval YOLO dùng: `$SKYFUSION_DATA`, mặc định là `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml`
- Inspect/convert lại từ COCO dùng: data/SkyFusion/
- Converter hiện có: data/convert_skyfusion_to_yolo.py
- Không đổi split valid thành val nếu không thật sự cần; nếu Ultralytics yêu cầu key val trong data.yaml thì key val có thể trỏ tới thư mục images/valid.
```

Số lượng hiện tại:

```text
COCO gốc data/SkyFusion:
- train: 2094 images
- valid: 449 images
- test: 449 images

YOLO `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo`:
- images/train: 2094 files
- images/valid: 449 files
- images/test: 449 files
- labels/train: 2094 files
- labels/valid: 449 files
- labels/test: 449 files
```

Class hiện tại:

```text
Aircraft
ship
vehicle
```

Ghi chú quan trọng cho Codex:

```text
Không hard-code class mapping nếu có thể đọc từ COCO categories hoặc data.yaml.
Tuy nhiên khi tạo config mặc định, có thể dùng names: [Aircraft, ship, vehicle] nếu data.yaml chưa tồn tại.
```

### 1.2.2. Đường dẫn Kaggle thực tế đã chốt

Đường dẫn **source of truth** hiện tại là:

```text
SKYFUSION_ROOT=/kaggle/input/datasets/thanhmay2406/dataset-for-research
SKYFUSION_YOLO_ROOT=/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo
SKYFUSION_DATA=/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml
SKYFUSION_COCO_ROOT=/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion  # chỉ dùng nếu cần convert lại từ COCO
```

Cell đầu mỗi Kaggle Notebook nên khai báo:

```bash
export SKYFUSION_ROOT=/kaggle/input/datasets/thanhmay2406/dataset-for-research
export SKYFUSION_YOLO_ROOT=/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo
export SKYFUSION_DATA=/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml
export SKYFUSION_COCO_ROOT=/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion

ls -lah "$SKYFUSION_YOLO_ROOT"
cat "$SKYFUSION_DATA"
```

Quy tắc mới:

```text
- Train/eval đọc trực tiếp từ $SKYFUSION_DATA nếu data.yaml trong input đúng path.
- Không copy toàn bộ dataset sang /kaggle/working nếu không cần.
- Nếu data.yaml trong /kaggle/input chứa path sai, không sửa trực tiếp file đó.
  Hãy tạo một file YAML sạch trong configs/dataset/skyfusion_kaggle.yaml hoặc /kaggle/working/skyfusion_kaggle.yaml.
- Mọi output vẫn lưu ở /kaggle/working/experiments/ hoặc experiments/skyfusion/.
```

YAML fallback nên có dạng:

```yaml
path: /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo
train: images/train
val: images/valid
test: images/test
names:
  0: Aircraft
  1: ship
  2: vehicle
```

### 1.3. Câu hỏi nghiên cứu chính

> Việc nhúng saliency map từ XAI vào YOLO như một inductive bias thông qua SRW module có giúp phát hiện tiny object trong ảnh aerial tốt hơn so với YOLO baseline và traditional augmentation baseline hay không?

### 1.4. Câu hỏi phụ

```text
RQ1. SRW-only có cải thiện mAP / Recall so với YOLO baseline không?
RQ2. L_sal-only có cải thiện so với baseline không?
RQ3. SRW + L_sal có tốt hơn từng thành phần riêng lẻ không?
RQ4. Lambda schedule có giúp training ổn định hơn không?
RQ5. Energy-in-Box / Background Suppression có giúp saliency tập trung vào bbox hơn không?
RQ6. Size-aware weighting có hữu ích với tiny object không?
RQ7. Saliency metrics như EBPG, BER có phản ánh cải thiện detection không?
```

### 1.5. Claim an toàn

Có thể claim:

```text
We propose a Kaggle-reproducible YOLO-based tiny aerial object detection framework that integrates XAI-derived saliency maps into FPN feature reweighting through a lightweight SRW module and an auxiliary saliency alignment objective.
```

Nên tránh claim:

```text
- Phương pháp tốt hơn trên mọi dataset.
- Đây là phương pháp đầu tiên dùng XAI trong object detection.
- SRW không làm tăng chi phí training.
- G-CAME teacher hoàn toàn độc lập nếu saliency lấy từ chính model hiện tại.
```

Cách nói an toàn về “teacher”:

```text
Không gọi là external teacher nếu chưa có teacher model riêng.
Nên gọi là model-dependent saliency guidance hoặc self-explanation saliency signal.
```

---

## 2. Ý tưởng kiến trúc chính

### 2.1. Toàn cảnh pipeline

```text
Input image
→ YOLO Backbone
→ YOLO FPN Neck, lấy feature map P3 / P4 / P5
→ XAI branch sinh saliency map S
→ SRW module nhận F và S
→ SRW sinh reweighted feature F*
→ Detection Head
→ L_total = L_det + λ(t) · L_sal
```

### 2.2. SRW module

Input:

```text
F: feature map từ FPN, shape [B, C, H, W]
S: saliency map từ G-CAME / Grad-CAM-like extractor, shape [B, 1, H, W]
```

Spatial gate:

```text
G_s = sigmoid(Conv1x1(S))
```

Channel gate:

```text
z_1 = GAP(F)
z_2 = GAP(F ⊙ S)
z = concat(z_1, z_2)
G_c = sigmoid(MLP(z))
```

Feature reweighting:

```text
F* = F + α · (F ⊙ G_s ⊙ G_c)
```

Khuyến nghị:

```text
- α khởi tạo khoảng 0.1.
- α có thể là learnable scalar.
- SRW phải preserve shape: F*.shape == F.shape.
- Có residual connection để tránh phá baseline ở đầu training.
- P3-only là mặc định để tiết kiệm VRAM Kaggle.
```

### 2.3. L_sal và nguồn saliency dùng trong training

GT saliency mask:

```text
M_gt = GaussianBlur(BBoxMask)
```

Saliency dùng để train nên tách thành 2 loại:

```text
S_pred      = saliency map dự đoán bởi saliency head differentiable từ FPN feature
S_xai       = saliency map từ Grad-CAM-like / G-CAME, dùng để debug, teacher offline, hoặc evaluation
```

Lý do phải tách:

```text
Grad-CAM/G-CAME thường cần gradient của detection score/loss theo feature map.
Nó không tự nhiên có sẵn trước detection head trong cùng một forward pass.
Nếu dùng Grad-CAM online làm loss rồi backward tiếp, code rất dễ rơi vào higher-order gradient, retain_graph lỗi, hoặc saliency bị detach nên L_sal không còn tác dụng.
```

Loss cơ bản nên dùng cho bản ổn định đầu tiên:

```text
L_sal = MSE(S_pred, M_gt)
```

Biến thể teacher-guided optional:

```text
L_sal = MSE(S_pred, M_gt) + β_teacher · MSE(S_pred, stopgrad(S_xai_teacher))
```

Loss mở rộng:

```text
Energy-in-Box
Background Suppression
Size-aware weighted saliency loss
```

Tổng loss:

```text
L_total = L_det + λ(t) · L_sal
```

Lambda schedule:

```text
constant
linear_warmup
cosine_decay
warmup_cosine_decay
```

---

## 2.4. Quyết định kỹ thuật quan trọng: tránh vòng lặp Grad-CAM → SRW trong cùng forward

Roadmap cũ dễ gây hiểu nhầm rằng:

```text
YOLO FPN feature F
→ Grad-CAM/G-CAME sinh saliency S ngay lập tức
→ SRW dùng S để reweight F
→ Detection head
→ backward
```

Luồng này không ổn định cho training vì Grad-CAM/G-CAME thường cần gradient từ detection output/loss về feature map. Tức là saliency phụ thuộc vào phần phía sau detection head, trong khi SRW lại cần saliency trước detection head. Vì vậy, bản triển khai nên dùng một trong ba chế độ rõ ràng:

```text
Mode A — stable_default:
  FPN feature F → SaliencyHead(F) = S_pred → SRW(F, S_pred) → Detection head
  L_total = L_det + λ · L_sal(S_pred, M_gt)

Mode B — offline_xai_teacher:
  Train baseline trước → precompute S_xai_teacher bằng Grad-CAM-like/G-CAME → lưu .npy/.pt
  Training SRW dùng S_xai_teacher như guidance đã detach, không cần higher-order gradient.

Mode C — online_two_pass_experimental:
  Forward pass 1 sinh detection/loss → lấy Grad-CAM-like saliency
  Forward pass 2 dùng saliency đó cho SRW → tính loss chính
  Chỉ dùng sau khi Mode A/B ổn vì tốn VRAM và dễ lỗi graph.
```

Khuyến nghị cho Kaggle:

```text
1. Implement Mode A trước để pipeline chạy được và gradient rõ ràng.
2. Dùng Mode B để giữ đúng tinh thần XAI-guided nhưng vẫn ổn định.
3. Chỉ thử Mode C như ablation experimental.
```

## 3. Quy tắc Kaggle-first cho dataset này

### 3.1. Cách thêm dataset vào Kaggle Notebook

Cách ưu tiên:

```text
Kaggle Notebook
→ Add Input hoặc attach dataset đã chuẩn bị
→ Dùng đường dẫn Kaggle input thực tế đã chốt:
/kaggle/input/datasets/thanhmay2406/dataset-for-research
```

Sau khi attach dataset, cần kiểm tra trực tiếp:

```bash
export SKYFUSION_ROOT=/kaggle/input/datasets/thanhmay2406/dataset-for-research
export SKYFUSION_YOLO_ROOT=/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo
export SKYFUSION_DATA=/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml

ls -lah "$SKYFUSION_ROOT"
ls -lah "$SKYFUSION_YOLO_ROOT"
cat "$SKYFUSION_DATA"
```

Không nên dựa vào path cũ như `/kaggle/input/tiny-object-detection`. Nếu Kaggle mount khác, chỉ cần sửa các biến `SKYFUSION_*` ở đầu notebook.

### 3.2. Đường dẫn chuẩn

```text
/kaggle/input/datasets/thanhmay2406/dataset-for-research/                 # read-only, parent dataset input
/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/                 # read-only, YOLO dataset dùng để train/eval
/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml        # data config chính
/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion/                 # read-only, COCO gốc nếu tồn tại/cần convert lại
/kaggle/working/tiny-yolo-srw-gcame/      # repo code
/kaggle/working/data/SkyFusion_yolo_reconverted/  # chỉ dùng nếu cần convert lại hoặc patch dataset
/kaggle/working/experiments/              # weights, logs, metrics
/kaggle/working/paper/                    # figures, tables
/kaggle/working/kaggle_outputs/           # artifact export
```

### 3.3. Quy tắc bắt buộc

```text
- Không ghi vào /kaggle/input.
- Không hard-code đường dẫn máy cá nhân.
- Dataset converted phải nằm trong /kaggle/working hoặc relative data/.
- Mọi experiment lưu theo experiments/skyfusion/<run_name>/.
- Mọi script phải có argparse.
- Mọi training run phải lưu config.yaml.
- Smoke test 1 epoch trước khi train full.
- Không bật multi-scale mặc định.
- Không dùng SRW/L_sal trong baseline script.
```

### 3.4. Cell Kaggle bootstrap gợi ý

```bash
!pwd
!nvidia-smi
!python --version
!pip install -q ultralytics opencv-python pycocotools matplotlib pandas pyyaml tqdm rich pytest
```

Nếu repo nằm ở `/kaggle/working/tiny-yolo-srw-gcame`:

```bash
%cd /kaggle/working/tiny-yolo-srw-gcame
!python scripts/00_env_check.py
!python scripts/01_inspect_kaggle_dataset.py --input-root "$SKYFUSION_ROOT" --max-depth 3
```


### 3.5. Tạo YAML fallback nếu `data.yaml` trong input sai path

Nếu `cat "$SKYFUSION_DATA"` cho thấy `path:` hoặc `train/val/test` không đúng với Kaggle mount hiện tại, tạo file YAML sạch trong `/kaggle/working` thay vì sửa file trong `/kaggle/input`:

```bash
mkdir -p configs/dataset
cat > configs/dataset/skyfusion_kaggle.yaml <<'YAML'
path: /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo
train: images/train
val: images/valid
test: images/test
names:
  0: Aircraft
  1: ship
  2: vehicle
YAML

export SKYFUSION_DATA=configs/dataset/skyfusion_kaggle.yaml
cat "$SKYFUSION_DATA"
```

Sau đó mọi lệnh train/eval vẫn dùng:

```bash
--data "$SKYFUSION_DATA"
```

---

## 4. Cấu trúc repo cuối cùng sau khi đơn giản hóa

```text
tiny-yolo-srw-gcame/
├── AGENTS.md
├── README.md
├── requirements.txt
│
├── configs/
│   ├── dataset/
│   │   └── skyfusion.yaml
│   ├── train/
│   │   ├── baseline_yolov8s.yaml
│   │   ├── tradaug_yolov8s.yaml
│   │   ├── lsal_only.yaml
│   │   ├── srw_only_p3.yaml
│   │   ├── srw_lsal_p3.yaml
│   │   ├── srw_lsal_warmup_decay.yaml
│   │   ├── srw_lsal_energy_bg.yaml
│   │   └── srw_lsal_size_aware.yaml
│   └── experiments/
│       └── ablation_plan.yaml
│
├── data/
│   ├── SkyFusion/
│   │   ├── README.roboflow.txt
│   │   ├── train/
│   │   │   ├── _annotations.coco.json
│   │   │   └── *.jpg
│   │   ├── valid/
│   │   │   ├── _annotations.coco.json
│   │   │   └── *.jpg
│   │   └── test/
│   │       ├── _annotations.coco.json
│   │       └── *.jpg
│   ├── SkyFusion_yolo/
│   │   ├── data.yaml
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── valid/
│   │   │   └── test/
│   │   └── labels/
│   │       ├── train/
│   │       ├── valid/
│   │       └── test/
│   └── convert_skyfusion_to_yolo.py
│
├── kaggle/
│   ├── 00_bootstrap_kaggle.sh
│   ├── 01_prepare_skyfusion.ipynb.md
│   ├── 02_train_smoke_tests.ipynb.md
│   └── README.md
│
├── src/
│   ├── datasets/
│   │   ├── inspect_kaggle_dataset.py
│   │   ├── skyfusion_prepare.py
│   │   ├── coco_to_yolo.py
│   │   ├── yolo_dataset_check.py
│   │   ├── saliency_masks.py
│   │   └── size_buckets.py
│   │
│   ├── xai/
│   │   ├── hooks.py
│   │   ├── saliency_base.py
│   │   ├── gradcam_detector.py
│   │   ├── gcame_detector.py
│   │   └── saliency_normalization.py
│   │
│   ├── models/
│   │   ├── srw.py
│   │   ├── yolo_srw_wrapper.py
│   │   └── layer_resolver.py
│   │
│   ├── losses/
│   │   ├── saliency_alignment.py
│   │   ├── energy_in_box.py
│   │   ├── background_suppression.py
│   │   └── size_aware.py
│   │
│   ├── training/
│   │   ├── lambda_scheduler.py
│   │   ├── trainer_baseline.py
│   │   ├── trainer_srw.py
│   │   ├── trainer_srw_lsal.py
│   │   └── gradient_debug.py
│   │
│   ├── evaluation/
│   │   ├── detection_metrics.py
│   │   ├── small_object_metrics.py
│   │   ├── xai_metrics.py
│   │   └── convergence.py
│   │
│   ├── visualization/
│   │   ├── visualize_dataset.py
│   │   ├── visualize_saliency.py
│   │   ├── visualize_srw_gates.py
│   │   └── plot_curves.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── paths.py
│       ├── logging.py
│       ├── device.py
│       └── io.py
│
├── scripts/
│   ├── 00_env_check.py
│   ├── 01_inspect_kaggle_dataset.py
│   ├── 02_prepare_skyfusion.py
│   ├── 03_check_yolo_dataset.py
│   ├── 04_profile_small_objects.py
│   ├── 05_train_baseline.py
│   ├── 06_train_tradaug.py
│   ├── 07_debug_gt_saliency.py
│   ├── 08_debug_xai_saliency.py
│   ├── 09_debug_srw_shapes.py
│   ├── 10_train_lsal_only.py
│   ├── 11_train_srw_only.py
│   ├── 12_train_srw_lsal.py
│   ├── 13_eval_detection.py
│   ├── 14_eval_xai.py
│   ├── 15_eval_convergence.py
│   ├── 16_run_ablation.py
│   ├── 17_generate_figures.py
│   └── 99_export_kaggle_artifacts.py
│
├── tests/
│   ├── test_bbox_conversion.py
│   ├── test_saliency_masks.py
│   ├── test_saliency_losses.py
│   ├── test_srw.py
│   ├── test_lambda_scheduler.py
│   └── test_size_aware.py
│
├── experiments/
└── paper/
    ├── figures/
    ├── tables/
    └── draft.md
```

---

# Phase 0 — Repo scaffold, Kaggle bootstrap, implementation rules

## Mục tiêu

Tạo nền repo sạch, chạy được trên Kaggle, có quy tắc rõ để Codex không ghi sai đường dẫn.

## Files cần có

```text
requirements.txt
AGENTS.md
README.md
kaggle/00_bootstrap_kaggle.sh
scripts/00_env_check.py
src/utils/seed.py
src/utils/paths.py
src/utils/logging.py
src/utils/device.py
src/utils/io.py
```

## Lệnh Kaggle

```bash
pip install -q -r requirements.txt
python scripts/00_env_check.py
```

## Debug checklist

```text
[ ] nvidia-smi chạy được
[ ] torch.cuda.is_available() đúng nếu bật GPU
[ ] ultralytics import được
[ ] /kaggle/input tồn tại nếu chạy Kaggle
[ ] /kaggle/working writable
[ ] experiments/ tự tạo được
[ ] Không có hard-code path local như /home/... hoặc C:\...
```

## Prompt Codex Phase 0 — Tiếng Việt

```text
Bạn đang triển khai repo nghiên cứu Kaggle-first cho đề tài G-CAME-guided SRW YOLO.

Bối cảnh:
- Hiện tại chỉ dùng một dataset Kaggle: thanhmay2406/dataset-for-research.
- Dataset này có bản COCO gốc và bản YOLO đã chuyển đổi; inspect để xác nhận, chỉ convert lại nếu bản YOLO thiếu/lỗi.
- Đóng góp chính là SRW module đặt sau YOLO FPN.
- L_sal là loss phụ để align saliency với Gaussian GT mask.
- Code phải chạy được trên Kaggle Notebook.
- /kaggle/input là read-only, không được ghi vào đó.
- Output phải lưu vào experiments/skyfusion/<run_name>/ hoặc paper/.

Nhiệm vụ:
1. Kiểm tra cấu trúc repo hiện tại.
2. Tạo requirements.txt với dependency tối thiểu:
   ultralytics, opencv-python, pycocotools, matplotlib, pandas, pyyaml, tqdm, rich, pytest.
3. Tạo AGENTS.md với quy tắc coding:
   - Không ghi vào /kaggle/input.
   - Không hard-code path local.
   - Mọi script phải dùng argparse.
   - Mọi experiment phải lưu config và metrics.
   - Baseline training không được import SRW hoặc L_sal.
   - Dataset converted chỉ lưu trong /kaggle/working/data/SkyFusion_yolo_reconverted nếu cần convert lại; không ghi vào /kaggle/input.
4. Tạo scripts/00_env_check.py.
   Script cần in:
   - Python version
   - torch version
   - CUDA available
   - GPU name nếu có
   - current working directory
   - trạng thái writable của experiments/
5. Tạo các utility:
   - src/utils/seed.py
   - src/utils/paths.py
   - src/utils/logging.py
   - src/utils/device.py
   - src/utils/io.py
6. Tạo kaggle/00_bootstrap_kaggle.sh để install requirements và chạy env check.
7. Cập nhật README.md với hướng dẫn Kaggle quickstart.

Ràng buộc:
- Chưa implement training ở phase này.
- Chưa giả định dataset đã convert.
- Mọi path phải configurable.
- Không sửa source của Ultralytics trong site-packages.

Sau khi sửa, hãy in ra các lệnh Kaggle cần chạy để validate Phase 0.
```

---

# Phase 1 — Inspect dataset Kaggle SkyFusion

## Mục tiêu

Cần xác nhận dataset Kaggle đang expose đúng cấu trúc đã biết: bản COCO gốc `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion` và bản YOLO `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo`. Converter chỉ cần chạy lại khi bản YOLO thiếu hoặc lỗi.

## Files cần có

```text
scripts/01_inspect_kaggle_dataset.py
src/datasets/inspect_kaggle_dataset.py
configs/dataset/skyfusion.yaml
```

## Lệnh Kaggle

```bash
python scripts/01_inspect_kaggle_dataset.py \
  --input-root "$SKYFUSION_ROOT" \
  --dataset-name SkyFusion_yolo \
  --max-depth 4 \
  --output-dir experiments/skyfusion/dataset_inspect
```

Nếu Kaggle đặt folder khác tên:

```bash
python scripts/01_inspect_kaggle_dataset.py \
  --input-root "$SKYFUSION_ROOT" \
  --max-depth 4 \
  --output-dir experiments/skyfusion/dataset_inspect
```

## Output kỳ vọng

```text
experiments/skyfusion/dataset_inspect/
├── tree.txt
├── file_summary.json
├── candidate_images.json
├── candidate_annotations.json
└── recommendation.txt
```

## Debug checklist

```text
[ ] Tìm được folder dataset trong /kaggle/input
[ ] Tìm được ảnh
[ ] Tìm được annotation json/txt/yaml nếu có
[ ] Biết dataset đang là COCO hay YOLO hay format khác
[ ] Không copy file lớn ở phase này
[ ] Không ghi vào /kaggle/input
```

## Prompt Codex Phase 1 — Tiếng Việt

```text
Hãy implement bước inspect dataset Kaggle cho SkyFusion.

Bối cảnh:
- Kaggle input thực tế: /kaggle/input/datasets/thanhmay2406/dataset-for-research.
- YOLO dataset chính: /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo.
- Cần inspect cấu trúc để xác nhận đúng `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion` và `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo` trước khi train hoặc convert lại.
- Không được ghi vào /kaggle/input.

Nhiệm vụ:
1. Tạo src/datasets/inspect_kaggle_dataset.py.
   Module này cần:
   - duyệt cây thư mục với giới hạn max_depth
   - thống kê số file theo extension
   - tìm candidate image files: .jpg, .jpeg, .png, .tif, .tiff
   - tìm candidate annotation files: .json, .txt, .yaml, .yml, .csv
   - nhận diện sơ bộ format:
     + COCO nếu json có keys images, annotations, categories
     + YOLO nếu có images/ và labels/ với file .txt
     + unknown nếu không đủ bằng chứng
2. Tạo scripts/01_inspect_kaggle_dataset.py.
   CLI:
   --input-root default=/kaggle/input/datasets/thanhmay2406/dataset-for-research
   --dataset-name optional
   --max-depth default=4
   --output-dir default=experiments/skyfusion/dataset_inspect
3. Script cần lưu:
   - tree.txt
   - file_summary.json
   - candidate_images.json
   - candidate_annotations.json
   - recommendation.txt
4. Tạo configs/dataset/skyfusion.yaml với các field cơ bản:
   name: skyfusion
   kaggle_slug: thanhmay2406/dataset-for-research
   source_root: /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion
   converted_root: /kaggle/working/data/SkyFusion_yolo_reconverted
   yolo_data_yaml: /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml
   expected_format: coco_and_yolo
   splits: [train, valid, test]
   num_classes: 3
   names: [Aircraft, ship, vehicle]

Ràng buộc:
- Không copy dataset ở phase này.
- Không ghi vào /kaggle/input.
- Nếu không tìm thấy dataset, báo lỗi dễ hiểu và liệt kê các folder đang có trong /kaggle/input.

Sau khi implement, hãy đưa lệnh Kaggle để inspect dataset.
```

---

# Phase 2 — Prepare / convert SkyFusion sang YOLO format

## Mục tiêu

Ưu tiên dùng sẵn `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo`. Chỉ convert lại từ `data/SkyFusion` khi bản YOLO chưa có, thiếu file, sai nhãn, hoặc cần tái tạo mapping class.

## Dataset YOLO kỳ vọng trên Kaggle

```text
/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/
├── data.yaml
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
└── labels/
    ├── train/
    ├── valid/
    └── test/
```

Lưu ý: thư mục split là `valid`, nhưng trong `data.yaml` của Ultralytics có thể vẫn dùng key `val:` trỏ tới `images/valid`.

## Logic chuẩn bị dataset

Vì hiện đã biết có cả bản COCO và bản YOLO, script cần xử lý theo thứ tự ưu tiên:

```text
Case 1: Dataset đã là YOLO
→ thường không cần copy; validate trực tiếp hoặc tạo YAML fallback trong /kaggle/working
→ validate data.yaml

Case 2: Dataset là COCO JSON
→ convert COCO bbox xywh pixel sang YOLO normalized
→ split theo annotation nếu có train/valid/test
→ nếu chỉ có một json, tạo split theo tỷ lệ cố định

Case 3: Format khác
→ báo lỗi rõ, in annotation candidates để xử lý thủ công
```

## Lệnh Kaggle

```bash
python scripts/02_prepare_skyfusion.py \
  --source-root /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion \
  --output-root /kaggle/working/data/SkyFusion_yolo_reconverted \
  --format auto \
  --val-ratio 0.2 \
  --test-ratio 0.0 \
  --seed 0
```

Nếu dataset đã có train/valid:

```bash
python scripts/02_prepare_skyfusion.py \
  --source-root /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion \
  --output-root /kaggle/working/data/SkyFusion_yolo_reconverted \
  --format auto \
  --preserve-splits \
  --seed 0
```

## Debug checklist

```text
[ ] `$SKYFUSION_DATA` tồn tại, hoặc YAML fallback trong `configs/dataset/skyfusion_kaggle.yaml` được tạo
[ ] images/train có ảnh
[ ] labels/train có .txt tương ứng
[ ] Mỗi dòng label có format: class x_center y_center width height
[ ] Tọa độ normalized nằm trong [0,1]
[ ] width/height > 0
[ ] class id nằm trong [0, nc-1]
[ ] Có conversion_report.json
[ ] Có vài ảnh overlay bbox để kiểm tra
```

## Prompt Codex Phase 2 — Tiếng Việt

```text
Hãy implement script chuẩn bị dataset SkyFusion sang YOLO format.

Bối cảnh:
- Chỉ dùng dataset Kaggle: thanhmay2406/dataset-for-research.
- Source COCO nếu cần: /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion.
- Bản YOLO chính: /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo.
- /kaggle/input là read-only.
- Dataset YOLO dùng để train đọc từ /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo, không ghi vào đây.
- Cần hỗ trợ auto-detect format vì dataset có thể là COCO JSON hoặc YOLO.

Nhiệm vụ:
1. Tạo src/datasets/coco_to_yolo.py.
   Implement:
   - load_coco_json(json_path)
   - convert_coco_bbox_to_yolo(bbox_xywh, image_width, image_height)
   - convert_coco_dataset_to_yolo(...)
   - map category_id sang class index 0..nc-1
2. Tạo src/datasets/skyfusion_prepare.py.
   Implement:
   - detect_dataset_format(source_root)
   - prepare_yolo_existing(...)
   - prepare_coco_json(...)
   - create_data_yaml(...)
   - save_conversion_report(...)
3. Tạo scripts/02_prepare_skyfusion.py.
   CLI:
   --source-root default=/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion
   --output-root default=/kaggle/working/data/SkyFusion_yolo_reconverted
   --format auto/coco/yolo
   --preserve-splits
   --val-ratio default=0.2
   --test-ratio default=0.0
   --seed default=0
   --copy-mode copy/symlink, default=copy
4. Nếu source là COCO:
   - tìm annotation json
   - nếu có train/valid/test json thì giữ split
   - nếu chỉ có một json thì split theo ảnh với seed cố định
   - tạo labels/*.txt theo YOLO
5. Nếu source là YOLO:
   - copy/symlink images và labels
   - tạo data.yaml nếu chưa có
6. Lưu conversion_report.json gồm:
   - format_detected
   - num_images per split
   - num_labels per split
   - class_names
   - invalid_boxes
   - empty_labels
   - skipped_annotations
7. Tạo một vài bbox overlay mẫu trong output-root/debug_overlays.

Ràng buộc:
- Không ghi vào /kaggle/input.
- Không silently clip bbox quá nhiều; phải log số lượng bbox bị clip/invalid.
- Không giả định tên class nếu có thể đọc từ COCO categories.
- Nếu không xác định được format, raise error rõ và đề xuất chạy Phase 1 inspect.

Sau khi implement, hãy đưa lệnh Kaggle để prepare SkyFusion.
```

---

# Phase 3 — Check YOLO dataset và profile tiny objects

## Mục tiêu

Đảm bảo dataset đã convert đúng và chứng minh đây là tiny/small object dataset.

## Files cần có

```text
scripts/03_check_yolo_dataset.py
scripts/04_profile_small_objects.py
src/datasets/yolo_dataset_check.py
src/datasets/size_buckets.py
```

## Lệnh Kaggle

```bash
python scripts/03_check_yolo_dataset.py \
  --data "$SKYFUSION_DATA" \
  --output-dir experiments/skyfusion/dataset_check

python scripts/04_profile_small_objects.py \
  --data "$SKYFUSION_DATA" \
  --output-dir experiments/skyfusion/dataset_profile
```

## Metrics profiling

```text
image count
label count
class distribution
bbox width/height distribution
bbox area distribution
tiny bucket: area < 16x16
small bucket: area < 32x32
empty-label images
missing labels
orphan labels
```

## Debug checklist

```text
[ ] Không có label orphan bất thường
[ ] Không có bbox normalized ngoài [0,1]
[ ] Không có bbox area bằng 0
[ ] Có histogram bbox area
[ ] Có bảng số lượng tiny/small theo split
[ ] Có bbox overlay samples
```

## Prompt Codex Phase 3 — Tiếng Việt

```text
Hãy implement kiểm tra YOLO dataset và profile tiny/small object cho SkyFusion.

Bối cảnh:
- Dataset dùng để train nằm ở /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo.
- data.yaml nằm ở `$SKYFUSION_DATA`, tức /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml.
- Đề tài tập trung vào tiny object detection nên phân bố kích thước bbox là bằng chứng quan trọng.

Nhiệm vụ:
1. Tạo src/datasets/yolo_dataset_check.py.
   Kiểm tra:
   - images/train, images/valid, images/test nếu có
   - labels/train, labels/valid, labels/test nếu có
   - missing labels
   - orphan labels
   - invalid YOLO lines
   - class id ngoài range
   - tọa độ normalized ngoài [0,1]
   - bbox width/height <= 0
2. Tạo src/datasets/size_buckets.py.
   Implement:
   - đọc kích thước ảnh thật
   - chuyển YOLO bbox normalized sang pixel bbox
   - tính bbox area pixel
   - bucket:
     tiny: area < 16*16
     small: area < 32*32
     medium/large optional
3. Tạo scripts/03_check_yolo_dataset.py.
   Lưu dataset_check.json và check_summary.txt.
4. Tạo scripts/04_profile_small_objects.py.
   Lưu:
   - size_profile.json
   - class_counts.csv
   - bbox_area_hist.png
   - bbox_width_height_scatter.png
   - bbox_overlay_samples/
5. Cập nhật README với lệnh kiểm tra dataset.

Ràng buộc:
- Dùng matplotlib, không dùng seaborn.
- Không fabricate AP_small; phase này chỉ profile dataset.
- Nếu thiếu test split thì xử lý an toàn.

Sau khi implement, hãy đưa lệnh Kaggle để check và profile dataset.
```

---

# Phase 4 — Baseline YOLO sạch

## Mục tiêu

Có baseline YOLO đáng tin cậy trước khi thêm SRW hoặc `L_sal`.

## Lệnh smoke test

```bash
python scripts/05_train_baseline.py \
  --data "$SKYFUSION_DATA" \
  --epochs 1 \
  --imgsz 640 \
  --batch 8 \
  --run-name smoke_baseline
```

## Lệnh train full

```bash
python scripts/05_train_baseline.py \
  --data "$SKYFUSION_DATA" \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 0 \
  --run-name baseline_yolov8s_seed0
```

## Output

```text
experiments/skyfusion/baseline_yolov8s_seed0/
├── config.yaml
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
├── metrics.json
└── predictions/
```

## Debug checklist

```text
[ ] Training start không lỗi data path
[ ] GPU memory đủ
[ ] results.csv xuất hiện
[ ] best.pt được lưu
[ ] valid metrics không NaN
[ ] prediction samples nhìn hợp lý
[ ] Baseline script không import SRW hoặc L_sal
```

## Prompt Codex Phase 4 — Tiếng Việt

```text
Hãy implement baseline YOLO training script sạch cho SkyFusion.

Bối cảnh:
- Dataset YAML dùng cho YOLO nằm ở `$SKYFUSION_DATA`, mặc định trỏ tới `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml`.
- Đây là baseline quan trọng để so sánh công bằng.
- Baseline script không được import SRW, G-CAME hoặc L_sal.
- Output phải lưu vào experiments/skyfusion/<run_name>/.

Nhiệm vụ:
1. Tạo scripts/05_train_baseline.py.
2. Dùng Ultralytics YOLO train API.
3. CLI:
   --data default=os.environ.get("SKYFUSION_DATA", "/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml")
   --model default=yolov8s.pt
   --epochs
   --imgsz default=640
   --batch default=16
   --seed default=0
   --device optional
   --workers optional
   --run-name
   --output-root default=experiments/skyfusion
4. Save resolved config vào experiments/skyfusion/<run_name>/config.yaml.
5. Route Ultralytics project/name sao cho output nằm đúng folder.
6. Sau train, nếu có thể, lưu metrics.json.
7. Thêm tùy chọn visualize prediction samples sau training.
8. Tạo configs/train/baseline_yolov8s.yaml.

Ràng buộc:
- Không thêm custom loss.
- Không sửa YOLO internals.
- Không import bất cứ file SRW/L_sal nào.
- Giữ script này ổn định để làm baseline.

Sau khi implement, hãy đưa lệnh smoke test 1 epoch và lệnh train full.
```

---

# Phase 5 — Traditional augmentation baseline

## Mục tiêu

Có baseline mạnh hơn baseline mặc định, tránh việc phương pháp đề xuất chỉ thắng baseline yếu.

## Augmentation nên dùng

```text
mosaic
mixup
copy-paste nếu phù hợp
hsv/brightness/contrast
random scale
random translate
multi-scale training nếu GPU chịu được
```

## Lệnh smoke test

```bash
python scripts/06_train_tradaug.py \
  --data "$SKYFUSION_DATA" \
  --epochs 1 \
  --imgsz 640 \
  --batch 8 \
  --run-name smoke_tradaug
```

## Lệnh train full

```bash
python scripts/06_train_tradaug.py \
  --data "$SKYFUSION_DATA" \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 0 \
  --run-name tradaug_yolov8s_seed0
```

## Debug checklist

```text
[ ] Augmentation config được lưu lại
[ ] Train/valid split giống baseline
[ ] Epoch/imgsz/batch/seed giống baseline
[ ] Không dùng SRW hoặc L_sal
[ ] Không làm thay đổi dataset gốc
```

## Prompt Codex Phase 5 — Tiếng Việt

```text
Hãy implement traditional augmentation baseline cho SkyFusion.

Bối cảnh:
- Cần baseline tăng cường dữ liệu truyền thống để so sánh công bằng với SRW/L_sal.
- Phải dùng cùng dataset split, model, epochs, image size, batch size và seed với baseline.
- Không được import SRW hoặc L_sal.

Nhiệm vụ:
1. Tạo scripts/06_train_tradaug.py.
2. Dùng Ultralytics YOLO train API.
3. CLI giống baseline:
   --data
   --model
   --epochs
   --imgsz
   --batch
   --seed
   --device
   --workers
   --run-name
   --output-root
4. Thêm augmentation hyperparameters rõ ràng:
   - mosaic
   - mixup
   - copy_paste
   - hsv_h
   - hsv_s
   - hsv_v
   - scale
   - translate
   - fliplr
5. Save augmentation config vào experiments/skyfusion/<run_name>/config.yaml.
6. Tạo configs/train/tradaug_yolov8s.yaml.
7. Cập nhật README với lệnh baseline vs tradaug.

Ràng buộc:
- Không dùng SRW/L_sal.
- Không sửa dataset.
- Reproducible bằng seed.

Sau khi implement, hãy đưa lệnh smoke test và lệnh train full.
```

---

# Phase 6 — GT saliency mask từ bbox

## Mục tiêu

Tạo `M_gt` từ bbox để dùng cho `L_sal`.

## Công thức

```text
YOLO bbox normalized → pixel bbox
Hard mask: pixel trong bbox = 1
Gaussian mask: blur hard mask
Resize mask về feature size P3/P4/P5
```

## Lệnh debug

```bash
python scripts/07_debug_gt_saliency.py \
  --data "$SKYFUSION_DATA" \
  --split train \
  --num-samples 16 \
  --output-dir experiments/skyfusion/debug_gt_saliency
```

## Debug checklist

```text
[ ] Hard mask trùng bbox
[ ] Gaussian mask không lệch bbox
[ ] Multiple boxes xử lý đúng
[ ] Empty label trả mask toàn 0
[ ] Resize mask về P3/P4/P5 không đảo H/W
[ ] Overlay dễ nhìn
```

## Prompt Codex Phase 6 — Tiếng Việt

```text
Hãy implement sinh GT saliency mask từ YOLO bounding boxes.

Bối cảnh:
- L_sal cần align model saliency với Gaussian GT mask sinh từ bbox.
- Dataset SkyFusion đã được convert sang YOLO format.
- Code cần xử lý ảnh RGB hoặc grayscale đều được.

Nhiệm vụ:
1. Tạo src/datasets/saliency_masks.py.
2. Implement:
   - read_yolo_label_file(label_path)
   - yolo_boxes_to_pixel_boxes(boxes, image_size)
   - create_bbox_mask(boxes, image_size)
   - create_gaussian_bbox_mask(boxes, image_size, sigma_ratio hoặc sigma_px)
   - resize_mask_to_feature(mask, feature_size)
3. Hỗ trợ nhiều bbox trong một ảnh.
4. Nếu label rỗng, trả mask toàn 0.
5. Normalize mask về [0,1].
6. Tạo scripts/07_debug_gt_saliency.py.
   Script lưu:
   - ảnh gốc + bbox
   - hard mask overlay
   - Gaussian mask overlay
   - preview mask resize P3/P4/P5 nếu có flag
7. Tạo tests/test_saliency_masks.py với synthetic boxes.

Ràng buộc:
- Không giả định ảnh grayscale.
- Ghi rõ convention shape: H,W hoặc B,1,H,W.
- Dùng OpenCV hoặc PIL để đọc ảnh.

Sau khi implement, hãy đưa lệnh debug GT saliency trên Kaggle.
```

---

# Phase 7 — Saliency provider: SaliencyHead trước, Grad-CAM/G-CAME làm teacher/debug

## Mục tiêu

Tạo một interface saliency thống nhất để tránh lỗi vòng lặp gradient. Bản ổn định đầu tiên không ép Grad-CAM online vào loss chính.

```text
Input image
→ YOLO Backbone/FPN
→ FPN feature F
→ SaliencyHead(F) = S_pred
→ SRW(F, S_pred) hoặc L_sal(S_pred, M_gt)
```

Grad-CAM-like/G-CAME vẫn được giữ lại nhưng dùng theo vai trò an toàn hơn:

```text
- debug saliency của baseline
- precompute teacher saliency offline
- evaluation XAI metrics
- ablation experimental two-pass nếu cần
```

## Files cần có

```text
src/xai/saliency_base.py
src/xai/hooks.py
src/xai/gradcam_detector.py
src/xai/gcame_detector.py
src/xai/saliency_normalization.py
src/xai/saliency_head.py
src/xai/saliency_provider.py
scripts/08_debug_xai_saliency.py
scripts/08b_precompute_xai_teacher.py
tests/test_saliency_head.py
```

## Interface khuyến nghị

```python
class SaliencyProvider:
    def forward(self, feature_map, image_ids=None, gt_mask=None):
        ...
```

Các provider cần hỗ trợ:

```text
1. saliency_head:
   - differentiable
   - dùng cho SRW và L_sal mặc định

2. gt_mask_debug:
   - dùng M_gt resize về feature size
   - chỉ để debug upper-bound, không dùng làm main result

3. offline_xai_teacher:
   - load saliency đã precompute từ baseline weights
   - detach hoàn toàn

4. gradcam_like_online:
   - chỉ dùng debug/eval hoặc experimental two-pass

5. gcame:
   - placeholder interface nếu chưa implement đầy đủ
```

## Lệnh debug SaliencyHead

```bash
python scripts/08_debug_xai_saliency.py \
  --data "$SKYFUSION_DATA" \
  --split valid \
  --target-layers P3 \
  --saliency-provider saliency_head \
  --output-dir experiments/skyfusion/debug_saliency_head
```

## Lệnh precompute teacher saliency từ baseline

```bash
python scripts/08b_precompute_xai_teacher.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/baseline_yolov8s_seed0/weights/best.pt \
  --split train \
  --target-layers P3 \
  --xai-method gradcam_like \
  --output-dir experiments/skyfusion/xai_teacher/baseline_p3_train

python scripts/08b_precompute_xai_teacher.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/baseline_yolov8s_seed0/weights/best.pt \
  --split valid \
  --target-layers P3 \
  --xai-method gradcam_like \
  --output-dir experiments/skyfusion/xai_teacher/baseline_p3_valid
```

## Debug checklist

```text
[ ] S_pred shape [B,1,H,W]
[ ] S_pred normalized hoặc sigmoid trong [0,1]
[ ] S_pred.requires_grad = True khi dùng saliency_head
[ ] S_xai_teacher được detach khi load offline
[ ] Không dùng Grad-CAM online trong main training mặc định
[ ] Empty-label image không làm loss NaN
[ ] Overlay S_pred, M_gt, S_xai_teacher được lưu để so sánh
[ ] Có image_id hoặc relative image path để map teacher saliency đúng ảnh
```

## Prompt Codex Phase 7 — Tiếng Việt

```text
Hãy implement saliency provider cho YOLO-SRW project theo hướng ổn định trên Kaggle.

Bối cảnh:
- Không dùng Grad-CAM/G-CAME online làm nguồn saliency mặc định trong cùng một forward pass với SRW, vì Grad-CAM cần gradient từ detection output/loss.
- Main training ổn định dùng SaliencyHead differentiable: FPN feature F -> S_pred.
- Grad-CAM-like/G-CAME dùng cho debug, evaluation hoặc teacher offline.

Nhiệm vụ:
1. Tạo src/xai/saliency_base.py với interface chung cho saliency provider.
2. Tạo src/xai/saliency_head.py.
   - Input: feature_map [B,C,H,W].
   - Output: S_pred [B,1,H,W].
   - Kiến trúc nhẹ: Conv 3x3 -> ReLU -> Conv 1x1 -> Sigmoid.
3. Tạo src/xai/saliency_provider.py.
   Hỗ trợ provider modes:
   - saliency_head
   - gt_mask_debug
   - offline_xai_teacher
   - gradcam_like_online_debug
   - gcame_placeholder
4. Tạo src/xai/hooks.py và src/xai/gradcam_detector.py để debug Grad-CAM-like trên baseline weights.
5. Tạo src/xai/gcame_detector.py với class placeholder cùng interface, raise NotImplementedError rõ ràng nếu chưa có G-CAME thật.
6. Tạo src/xai/saliency_normalization.py.
   - Normalize per image.
   - Có eps để tránh chia 0.
   - Không tạo NaN khi saliency toàn 0.
7. Tạo scripts/08_debug_xai_saliency.py.
   CLI:
   --data
   --split train/valid/test
   --target-layers P3/P4/P5
   --saliency-provider saliency_head/gt_mask_debug/offline_xai_teacher/gradcam_like_online_debug
   --weights optional
   --teacher-dir optional
   --output-dir
8. Tạo scripts/08b_precompute_xai_teacher.py.
   - Load baseline weights.
   - Compute Grad-CAM-like saliency cho từng ảnh.
   - Lưu theo relative image path hoặc hash an toàn.
   - Lưu manifest.json để map ảnh -> saliency file.
9. Tạo tests/test_saliency_head.py.
   Test shape, range, gradient flow, empty feature/small feature.

Ràng buộc:
- SaliencyHead phải differentiable.
- Offline teacher saliency phải detach.
- Không sửa source Ultralytics trong site-packages.
- Không dùng online Grad-CAM làm default training.

Sau khi implement, hãy đưa lệnh debug SaliencyHead và lệnh precompute teacher saliency.
```

# Phase 8 — SRW module độc lập

## Mục tiêu

Tạo SRW module độc lập, test shape và gradient trước khi nhúng vào YOLO.

## Lệnh debug

```bash
python scripts/09_debug_srw_shapes.py \
  --batch 2 \
  --channels 256 \
  --height 80 \
  --width 80

pytest -q tests/test_srw.py
```

## Debug checklist

```text
[ ] Input F [B,C,H,W]
[ ] Input S [B,1,H,W]
[ ] Output F* cùng shape với F
[ ] gate_s nằm trong [0,1]
[ ] gate_c nằm trong [0,1]
[ ] alpha initialized khoảng 0.1
[ ] backward chạy được
[ ] gradient chảy qua F và SRW params
[ ] S toàn 0 không crash
[ ] S toàn 1 không crash
```

## Prompt Codex Phase 8 — Tiếng Việt

```text
Hãy implement SRW module như một PyTorch module độc lập.

Bối cảnh:
- SRW là đóng góp kiến trúc chính.
- SRW nhận FPN feature map F và saliency map S.
- Output F_star phải cùng shape với F.

Kiến trúc:
- Spatial gate: sigmoid(Conv1x1(S))
- Channel gate: sigmoid(MLP(concat(GAP(F), GAP(F * S))))
- Output: F_star = F + alpha * (F * G_s * G_c)

Nhiệm vụ:
1. Tạo src/models/srw.py với class SRWModule.
2. Constructor arguments:
   - channels
   - reduction default=16
   - alpha_init default=0.1
   - learnable_alpha default=True
   - spatial_expand_mode optional
3. Forward signature:
   forward(feature_map, saliency_map, return_gates=False)
4. Nếu return_gates=True, trả thêm gate_s, gate_c, alpha để debug.
5. Tạo tests/test_srw.py.
   Test:
   - shape preservation
   - gate ranges
   - gradient flow
   - zero saliency
   - one saliency
6. Tạo scripts/09_debug_srw_shapes.py.
   Script tạo random tensor, chạy SRW, in shape, gate range, alpha, và verify backward.

Ràng buộc:
- Chưa tích hợp YOLO ở phase này.
- Module phải nhẹ.
- Luôn dùng residual connection mặc định.

Sau khi implement, hãy đưa lệnh pytest và debug shape.
```

---

# Phase 9 — Nhúng SRW vào YOLO FPN bằng saliency provider

## Mục tiêu

Tích hợp SRW sau FPN feature map và trước detection head, nhưng nguồn saliency phải đi qua `SaliencyProvider` để tránh coupling cứng với Grad-CAM online.

Luồng mặc định:

```text
YOLO Backbone/FPN
→ selected FPN feature F_P3
→ SaliencyProvider(F_P3) = S
→ SRW(F_P3, S) = F*_P3
→ Detection Head
```

Provider mặc định:

```text
saliency_head
```

Provider optional:

```text
offline_xai_teacher
gt_mask_debug
gradcam_like_online_two_pass_experimental
```

## Nguyên tắc

```text
- Không sửa trực tiếp Ultralytics source trong site-packages.
- Ưu tiên wrapper/subclass Trainer có kiểm soát.
- P3-only là mặc định để tiết kiệm VRAM Kaggle.
- Multi-scale P3/P4/P5 chỉ bật khi debug P3 ổn.
- Baseline script không dùng wrapper này.
```

## Lệnh debug

```bash
python scripts/09_debug_srw_shapes.py \
  --from-yolo \
  --model yolov8s.pt \
  --target-layers P3 \
  --saliency-provider saliency_head \
  --imgsz 640 \
  --output-dir experiments/skyfusion/debug_srw_yolo_shapes
```

## Debug checklist

```text
[ ] Xác định đúng layer P3/P4/P5
[ ] Feature F lấy từ FPN có shape đúng
[ ] Saliency S resize đúng về H,W của F
[ ] F* shape bằng F
[ ] Detection head nhận F* không lỗi
[ ] Forward inference chạy được với ảnh dummy
[ ] SRW params và SaliencyHead params xuất hiện trong optimizer
[ ] GPU memory tăng nhưng còn chịu được
```

## Prompt Codex Phase 9 — Tiếng Việt

```text
Hãy tích hợp SRW vào YOLO FPN thông qua SaliencyProvider.

Bối cảnh:
- SRW đặt sau FPN feature và trước detection head.
- Nguồn saliency mặc định là SaliencyHead differentiable, không phải Grad-CAM online.
- Grad-CAM/G-CAME chỉ dùng teacher offline hoặc debug.
- Không sửa file Ultralytics trong site-packages.

Nhiệm vụ:
1. Tạo src/models/layer_resolver.py.
   - Xác định candidate layers P3/P4/P5 cho YOLOv8s.
   - Cho phép explicit module index nếu layer name thay đổi.
2. Tạo src/models/yolo_srw_wrapper.py.
   Wrapper cần:
   - load Ultralytics YOLO model
   - hook hoặc intercept selected FPN feature map
   - gọi SaliencyProvider để lấy S
   - apply SRW tại P3 trước
   - preserve output format cho detection head
3. Hỗ trợ target layers:
   - P3
   - P4
   - P5
   - explicit module index nếu cần
4. Thêm dry-run forward với dummy input.
5. Cập nhật scripts/09_debug_srw_shapes.py với --from-yolo mode.
6. Lưu report JSON:
   - selected layers
   - feature shapes
   - saliency shapes
   - SRW output shapes
   - saliency provider mode
   - parameter count overhead
7. Document assumptions về layer index trong README.

Ràng buộc:
- P3-only là default.
- Nếu YOLO internals khó, tạo experimental wrapper và document limitation rõ.
- Không làm hỏng baseline script.
- SRW và SaliencyHead phải nằm trong optimizer khi training main method.

Sau khi implement, hãy đưa lệnh debug SRW-YOLO integration.
```

# Phase 10 — L_sal only training với SaliencyHead

## Mục tiêu

Kiểm tra riêng tác dụng của saliency alignment mà chưa dùng SRW. Bản ổn định dùng `S_pred = SaliencyHead(FPN feature)` để loss có gradient rõ ràng.

Không dùng SRW ở phase này.

## Objective

```text
F = YOLO FPN feature P3
S_pred = SaliencyHead(F)
M_gt = GaussianBlur(BBoxMask) resized to P3
L_sal = loss(S_pred, M_gt)
L_total = L_det + λ · L_sal
```

## Lệnh smoke test

```bash
python scripts/10_train_lsal_only.py \
  --data "$SKYFUSION_DATA" \
  --epochs 1 \
  --imgsz 640 \
  --batch 8 \
  --run-name smoke_lsal_only \
  --saliency-provider saliency_head \
  --loss-type mse \
  --lambda-sal 0.1 \
  --target-layers P3
```

## Lệnh train full

```bash
python scripts/10_train_lsal_only.py \
  --data "$SKYFUSION_DATA" \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 0 \
  --run-name lsal_only_mse_seed0 \
  --saliency-provider saliency_head \
  --loss-type mse \
  --lambda-sal 0.1 \
  --target-layers P3
```

## Debug checklist

```text
[ ] det_loss hợp lý
[ ] L_sal không NaN
[ ] total_loss = det_loss + λ * L_sal
[ ] S_pred.requires_grad = True
[ ] SaliencyHead params được update
[ ] λ được log
[ ] Không cần retain_graph/higher-order gradient cho bản stable_default
[ ] saliency overlay sau vài epoch có thay đổi
[ ] Nếu L_sal quá lớn, giảm λ
```

## Prompt Codex Phase 10 — Tiếng Việt

```text
Hãy implement training L_sal-only với SaliencyHead differentiable.

Bối cảnh:
- Phase này KHÔNG dùng SRW.
- L_sal align S_pred từ SaliencyHead với Gaussian GT bbox mask.
- Không dùng Grad-CAM online làm loss mặc định.

Nhiệm vụ:
1. Tạo src/losses/saliency_alignment.py.
   Implement:
   - mse_saliency_loss
   - bce_saliency_loss optional
   - dice_saliency_loss optional
2. Tạo scripts/10_train_lsal_only.py.
3. Training objective:
   total_loss = det_loss + lambda_sal * L_sal
4. CLI:
   --data default=os.environ.get("SKYFUSION_DATA", "/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml")
   --model default=yolov8s.pt
   --epochs
   --imgsz
   --batch
   --seed
   --run-name
   --target-layers default=P3
   --saliency-provider default=saliency_head
   --loss-type default=mse
   --lambda-sal default=0.1
5. Log:
   - det_loss
   - l_sal
   - total_loss
   - lambda_sal
   - S_pred min/max/mean
6. Save debug saliency overlays định kỳ.
7. Tạo tests/test_saliency_losses.py.

Ràng buộc:
- Không dùng SRW trong phase này.
- Không sửa baseline script.
- Empty GT mask phải xử lý an toàn.
- Không dùng Grad-CAM online trong main L_sal-only training.

Sau khi implement, hãy đưa lệnh smoke test 1 epoch và lệnh train full L_sal-only.
```

# Phase 11 — SRW only training

## Mục tiêu

Kiểm tra SRW có giúp detection khi dùng saliency-guided feature reweighting nhưng chưa thêm `L_sal`.

Nguồn saliency mặc định:

```text
S = SaliencyHead(F)
```

Nguồn saliency optional:

```text
S = offline_xai_teacher
```

## Objective

```text
F = YOLO FPN feature P3
S = SaliencyProvider(F)
F* = SRW(F, S)
L_total = L_det
```

## Lệnh smoke test

```bash
python scripts/11_train_srw_only.py \
  --data "$SKYFUSION_DATA" \
  --epochs 1 \
  --imgsz 640 \
  --batch 8 \
  --run-name smoke_srw_only \
  --saliency-provider saliency_head \
  --target-layers P3 \
  --alpha-init 0.1
```

## Lệnh train full

```bash
python scripts/11_train_srw_only.py \
  --data "$SKYFUSION_DATA" \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 0 \
  --run-name srw_only_p3_seed0 \
  --saliency-provider saliency_head \
  --target-layers P3 \
  --alpha-init 0.1
```

## Debug checklist

```text
[ ] SRW params được train
[ ] SaliencyHead params được train nếu dùng saliency_head
[ ] alpha không nổ quá lớn
[ ] gate_s/gate_c không collapse toàn 0 hoặc toàn 1
[ ] detection loss giảm tương tự baseline
[ ] memory không vượt Kaggle GPU
[ ] Nếu SRW làm mAP giảm mạnh, kiểm tra saliency quality và alpha
```

## Prompt Codex Phase 11 — Tiếng Việt

```text
Hãy implement SRW-only training.

Bối cảnh:
- SRW là đóng góp kiến trúc chính.
- Phase này dùng saliency-guided feature reweighting nhưng KHÔNG thêm L_sal.
- Saliency mặc định lấy từ SaliencyHead differentiable, không dùng Grad-CAM online.

Nhiệm vụ:
1. Tạo scripts/11_train_srw_only.py.
2. Dùng YOLO SRW wrapper từ src/models/yolo_srw_wrapper.py.
3. Objective:
   total_loss = det_loss
4. SRW nhận saliency map S từ SaliencyProvider.
5. CLI:
   --data
   --model
   --epochs
   --imgsz
   --batch
   --seed
   --run-name
   --target-layers default=P3
   --saliency-provider default=saliency_head
   --teacher-dir optional
   --alpha-init default=0.1
6. Log:
   - det_loss
   - gate_s mean/std
   - gate_c mean/std
   - alpha
   - S mean/std
   - GPU memory nếu dễ
7. Save SRW gate visualizations định kỳ.

Ràng buộc:
- Không thêm L_sal.
- Không sửa baseline script.
- Không dùng online Grad-CAM làm default.
- Nếu integration với Ultralytics fragile, document limitation và cung cấp debug mode.

Sau khi implement, hãy đưa lệnh smoke test và lệnh train full SRW-only.
```

# Phase 12 — SRW + L_sal main training

## Mục tiêu

Triển khai biến thể chính của đề tài theo hướng ổn định:

```text
F = YOLO FPN feature P3
S_pred = SaliencyHead(F)
F* = SRW(F, S_pred)
L_total = L_det + λ · L_sal(S_pred, M_gt)
```

Teacher XAI optional:

```text
L_total = L_det + λ · L_gt(S_pred, M_gt) + β_teacher · L_teacher(S_pred, stopgrad(S_xai_teacher))
```

## Lệnh smoke test

```bash
python scripts/12_train_srw_lsal.py \
  --data "$SKYFUSION_DATA" \
  --epochs 1 \
  --imgsz 640 \
  --batch 8 \
  --run-name smoke_srw_lsal \
  --saliency-provider saliency_head \
  --target-layers P3 \
  --loss-type mse \
  --lambda-sal 0.1 \
  --alpha-init 0.1
```

## Lệnh train full

```bash
python scripts/12_train_srw_lsal.py \
  --data "$SKYFUSION_DATA" \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 0 \
  --run-name srw_lsal_p3_mse_seed0 \
  --saliency-provider saliency_head \
  --target-layers P3 \
  --loss-type mse \
  --lambda-sal 0.1 \
  --alpha-init 0.1
```

## Lệnh train full có offline XAI teacher optional

```bash
python scripts/12_train_srw_lsal.py \
  --data "$SKYFUSION_DATA" \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 0 \
  --run-name srw_lsal_p3_teacher_seed0 \
  --saliency-provider saliency_head \
  --teacher-dir experiments/skyfusion/xai_teacher/baseline_p3_train \
  --beta-teacher 0.1 \
  --target-layers P3 \
  --loss-type mse \
  --lambda-sal 0.1 \
  --alpha-init 0.1
```

## Debug checklist

```text
[ ] det_loss giảm
[ ] L_sal giảm hoặc ổn định
[ ] total_loss không NaN
[ ] S_pred.requires_grad = True
[ ] gate_s/gate_c không collapse
[ ] saliency overlay tốt hơn baseline hoặc hợp lý hơn theo bbox
[ ] prediction không mất object hàng loạt
[ ] So sánh nhanh valid với baseline sau ít epoch
[ ] Teacher map nếu dùng phải map đúng image_id và detach
```

## Prompt Codex Phase 12 — Tiếng Việt

```text
Hãy implement training pipeline chính SRW + L_sal.

Bối cảnh:
- Đây là main method của đề tài.
- SRW reweight YOLO FPN feature bằng S_pred từ SaliencyHead.
- L_sal align S_pred với Gaussian bbox-derived GT saliency mask.
- Grad-CAM-like/G-CAME không dùng online làm default training; chỉ dùng offline teacher hoặc debug.

Nhiệm vụ:
1. Tạo scripts/12_train_srw_lsal.py.
2. Kết hợp:
   - YOLO SRW wrapper
   - SaliencyProvider / SaliencyHead
   - saliency alignment loss
   - optional offline XAI teacher
3. Objective default:
   total_loss = det_loss + lambda_sal * L_sal
4. Objective optional teacher:
   total_loss = det_loss + lambda_sal * L_gt + beta_teacher * L_teacher
5. CLI:
   --data
   --model
   --epochs
   --imgsz
   --batch
   --seed
   --run-name
   --target-layers default=P3
   --saliency-provider default=saliency_head
   --teacher-dir optional
   --beta-teacher default=0.0
   --loss-type mse/bce/dice/energy
   --lambda-sal
   --alpha-init
6. Log:
   - det_loss
   - l_sal_gt
   - l_sal_teacher nếu có
   - total_loss
   - lambda_sal
   - beta_teacher
   - gate_s mean/std
   - gate_c mean/std
   - alpha
   - S_pred min/max/mean
7. Save debug overlays:
   - GT Gaussian mask
   - S_pred
   - teacher XAI saliency nếu có
   - SRW spatial gate
   - prediction samples
8. Save config và metrics vào experiments/skyfusion/<run_name>/.

Ràng buộc:
- P3-only là default.
- Không bật multi-scale mặc định.
- Không làm hỏng baseline/tradaug scripts.
- Không dùng Grad-CAM online làm default.

Sau khi implement, hãy đưa lệnh smoke test và lệnh train full cho SRW + L_sal.
```

# Phase 13 — Lambda scheduling / curriculum

## Mục tiêu

Điều khiển tác động của `L_sal` theo epoch để tránh ép saliency quá sớm hoặc quá muộn.

## Schedule khuyến nghị

```text
warmup_cosine_decay:
epoch 0 → warmup_epochs: λ tăng từ 0 lên λ_max
sau warmup: λ giảm dần về λ_min
```

Ví dụ:

```text
warmup_epochs = 5
lambda_max = 0.2
lambda_min = 0.01
```

## Lệnh train

```bash
python scripts/12_train_srw_lsal.py \
  --data "$SKYFUSION_DATA" \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 0 \
  --run-name srw_lsal_p3_warmup_decay_seed0 \
  --xai-method gradcam_like \
  --target-layers P3 \
  --loss-type mse \
  --lambda-schedule warmup_cosine_decay \
  --lambda-max 0.2 \
  --lambda-min 0.01 \
  --warmup-epochs 5
```

## Debug checklist

```text
[ ] λ epoch 0 đúng
[ ] λ tại warmup boundary đúng
[ ] λ cuối training không âm
[ ] λ được log vào csv
[ ] Curve λ được plot
[ ] Constant schedule vẫn chạy như cũ
```

## Prompt Codex Phase 13 — Tiếng Việt

```text
Hãy implement lambda scheduling cho training SRW + L_sal.

Bối cảnh:
- L_sal nên hướng dẫn model ở đầu/giữa training nhưng không nên lấn át L_det ở cuối training.
- Cần constant và scheduled lambda để ablation.

Nhiệm vụ:
1. Tạo src/training/lambda_scheduler.py.
2. Implement modes:
   - constant
   - linear_warmup
   - cosine_decay
   - warmup_cosine_decay
3. Inputs:
   - total_epochs
   - warmup_epochs
   - lambda_max
   - lambda_min
   - constant_lambda
4. Cập nhật scripts/12_train_srw_lsal.py.
   Thêm CLI:
   --lambda-schedule
   --lambda-sal
   --lambda-max
   --lambda-min
   --warmup-epochs
5. Log lambda mỗi epoch.
6. Save lambda_curve.csv và lambda_curve.png.
7. Tạo tests/test_lambda_scheduler.py.

Ràng buộc:
- Backward compatible với --lambda-sal constant mode.
- Không hard-code total epochs.
- Dùng matplotlib, không dùng seaborn.

Sau khi implement, hãy đưa lệnh so sánh constant vs warmup_cosine_decay.
```

---

# Phase 14 — Energy-in-Box, Background Suppression, Size-aware weighting

## Mục tiêu

Tăng chất lượng `L_sal` cho tiny object.

## Loss variants

Energy-in-Box:

```text
L_in = 1 - sum(S · M_box) / (sum(S) + eps)
```

Background Suppression:

```text
M_ignore = dilate(M_box, radius)
L_bg = sum(S · (1 - M_ignore)) / (sum(S) + eps)
L_sal = L_in + β_bg · L_bg
```

Size-aware:

```text
w_i = log(1 + A_img / (A_i + eps))
L_sal_image = clamp(w_image, max=w_max) · L_sal_image
```

## Lệnh train

```bash
python scripts/12_train_srw_lsal.py \
  --data "$SKYFUSION_DATA" \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 0 \
  --run-name srw_lsal_energy_bg_sizeaware_seed0 \
  --xai-method gradcam_like \
  --target-layers P3 \
  --loss-type energy_bg \
  --beta-bg 0.5 \
  --dilation-radius 3 \
  --size-aware \
  --size-weight-mode log_inverse \
  --size-weight-max 5.0 \
  --lambda-schedule warmup_cosine_decay \
  --lambda-max 0.2 \
  --lambda-min 0.01 \
  --warmup-epochs 5
```

## Debug checklist

```text
[ ] Energy-in-Box thấp khi saliency nằm trong bbox
[ ] Background energy giảm theo training nếu method hiệu quả
[ ] Empty mask không crash
[ ] Size weight nhỏ/lớn hợp lý
[ ] Tiny object nhận weight cao hơn
[ ] Loss không bị object cực nhỏ làm nổ
```

## Prompt Codex Phase 14 — Tiếng Việt

```text
Hãy implement các saliency loss nâng cao cho SRW + L_sal.

Bối cảnh:
- Tiny object detection cần saliency supervision khuyến khích saliency nằm trong bbox và giảm phụ thuộc background.
- Các loss này là ablation variants, không thay thế MSE bắt buộc.

Nhiệm vụ:
1. Tạo src/losses/energy_in_box.py.
   Implement energy_in_box_loss(saliency, bbox_mask, eps=1e-6, reduction='mean').
2. Tạo src/losses/background_suppression.py.
   Implement:
   - create_ignore_mask_from_bbox_mask bằng torch max pooling
   - background_suppression_loss
   - combined_energy_bg_loss
3. Tạo src/losses/size_aware.py.
   Implement:
   - bbox_area_weights
   - image_level_size_weight
   - modes: log_inverse, inverse_sqrt
4. Cập nhật scripts/12_train_srw_lsal.py.
   Thêm CLI:
   --loss-type energy/energy_bg
   --beta-bg
   --dilation-radius
   --size-aware
   --size-weight-mode
   --size-weight-max
5. Tạo tests:
   - saliency nằm trong bbox cho Energy-in-Box loss thấp hơn
   - saliency ngoài bbox cho loss cao hơn
   - background suppression hoạt động
   - box nhỏ nhận weight lớn hơn
6. Log EBPG và BER trong validation nếu khả thi.

Ràng buộc:
- MSE vẫn phải dùng được.
- Empty label phải xử lý an toàn.
- Tránh division by zero.

Sau khi implement, hãy đưa lệnh chạy energy, energy_bg và size-aware ablation.
```

---

# Phase 15 — P3-only trước, multi-scale sau

## Mục tiêu

So sánh SRW ở P3-only với P3/P4/P5 nếu Kaggle GPU đủ.

## Khuyến nghị

```text
P3-only: default, phù hợp tiny object, ít VRAM
P3/P4/P5: optional sau khi P3-only ổn
```

## Lệnh P3-only

```bash
python scripts/12_train_srw_lsal.py \
  --data "$SKYFUSION_DATA" \
  --run-name srw_lsal_p3_seed0 \
  --target-layers P3
```

## Lệnh multi-scale

```bash
python scripts/12_train_srw_lsal.py \
  --data "$SKYFUSION_DATA" \
  --run-name srw_lsal_multiscale_seed0 \
  --target-layers P3 P4 P5 \
  --scale-weights 1.0 0.5 0.25
```

## Debug checklist

```text
[ ] Per-scale feature shape đúng
[ ] Per-scale saliency shape đúng
[ ] Per-scale loss được log
[ ] Scale weights khớp số layer
[ ] Multi-scale không OOM
[ ] Nếu OOM, giảm batch hoặc quay về P3-only
```

## Prompt Codex Phase 15 — Tiếng Việt

```text
Hãy thêm optional multi-scale SRW + L_sal support.

Bối cảnh:
- Tiny objects có thể hưởng lợi nhiều nhất từ P3.
- Multi-scale P3/P4/P5 chỉ là optional vì Kaggle GPU memory giới hạn.

Nhiệm vụ:
1. Tạo src/training/multiscale_srw.py.
2. Hỗ trợ target layers: P3, P4, P5.
3. Với mỗi selected layer:
   - lấy feature map F
   - compute hoặc resize saliency S
   - apply SRW
   - compute optional L_sal tại scale đó
4. Combine scale losses bằng --scale-weights.
5. Cập nhật scripts/12_train_srw_lsal.py:
   - --target-layers nhận một hoặc nhiều layer
   - --scale-weights nhận list float
6. Log per-scale:
   - l_sal_P3/P4/P5
   - gate_s_mean_P3/P4/P5
   - gate_c_mean_P3/P4/P5
7. Thêm debug visualization cho từng scale.

Ràng buộc:
- P3-only vẫn là default.
- Không bật multi-scale tự động.
- Nếu số target layers và scale weights không khớp thì báo lỗi rõ.

Sau khi implement, hãy đưa lệnh P3-only và multi-scale.
```

---

# Phase 16 — Detection, XAI, convergence evaluation

## Mục tiêu

Đánh giá đủ để viết báo cáo/paper.

## Metrics chính

Detection:

```text
mAP50
mAP50-95
Precision
Recall
F1
Recall_tiny / Recall_small nếu tự tính được
AP_small nếu có hoặc nếu tự implement đúng
```

XAI:

```text
Pointing Game
Energy-in-Box score / EBPG
Background Energy Ratio / BER
Saliency mass inside bbox
```

Convergence:

```text
epoch-to-threshold
best epoch
mAP curve
L_det / L_sal / total loss curve
gate statistics curve
lambda curve
```

## Lệnh evaluation

```bash
python scripts/13_eval_detection.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/srw_lsal_p3_warmup_decay_seed0/weights/best.pt \
  --split valid \
  --run-name srw_lsal_p3_warmup_decay_seed0

python scripts/14_eval_xai.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/srw_lsal_p3_warmup_decay_seed0/weights/best.pt \
  --split valid \
  --target-layers P3 \
  --output-dir experiments/skyfusion/srw_lsal_p3_warmup_decay_seed0/xai_eval

python scripts/15_eval_convergence.py \
  --experiment-dir experiments/skyfusion/srw_lsal_p3_warmup_decay_seed0
```

## Debug checklist

```text
[ ] metrics.json có thật, không fabricate
[ ] Nếu AP_small không có sẵn, ghi rõ tự tính hoặc TODO
[ ] XAI metrics dùng đúng bbox GT
[ ] BER giảm là tốt, EBPG tăng là tốt
[ ] Convergence không tính sai do missing epoch
[ ] Figures lưu vào paper/figures
[ ] Tables lưu vào paper/tables
```

## Prompt Codex Phase 16 — Tiếng Việt

```text
Hãy implement evaluation utilities cho detection, XAI quality và convergence.

Bối cảnh:
- Báo cáo/paper cần nhiều hơn mAP.
- Cần metric size-aware và saliency localization evidence.
- Không được fabricate missing metrics.

Nhiệm vụ:
1. Tạo scripts/13_eval_detection.py.
   - Dùng Ultralytics validation.
   - Save metrics.json.
   - Support --split valid/test nếu có.
2. Tạo src/evaluation/small_object_metrics.py.
   - Nếu khả thi, implement recall_tiny / recall_small theo size bucket.
   - Document IoU threshold rõ ràng.
3. Tạo src/evaluation/xai_metrics.py.
   Implement:
   - pointing_game
   - energy_in_box_score
   - background_energy_ratio
4. Tạo scripts/14_eval_xai.py.
   - Compute saliency maps trên valid/test images.
   - Save xai_metrics.json.
   - Save qualitative overlays.
5. Tạo src/evaluation/convergence.py và scripts/15_eval_convergence.py.
   - Read results.csv và custom logs.
   - Compute epoch-to-threshold nếu có threshold.
6. Tạo scripts/17_generate_figures.py.
   - Generate separate matplotlib figures cho từng metric curve.

Ràng buộc:
- Dùng matplotlib, không dùng seaborn.
- Không invent AP_small nếu chưa có.
- Save tất cả output vào experiments/skyfusion/<run_name>/ hoặc paper/.

Sau khi implement, hãy đưa lệnh eval baseline và SRW+L_sal.
```

---

# Phase 17 — Ablation runner Kaggle-safe

## Mục tiêu

Chạy ablation có kiểm soát, không launch nhầm toàn bộ experiment trên Kaggle.

## Ablation tối thiểu

```text
A0: baseline_yolov8s
A1: tradaug_yolov8s
A2: lsal_only_mse
A3: srw_only_p3
A4: srw_lsal_p3_mse
A5: srw_lsal_p3_warmup_decay
A6: srw_lsal_p3_energy_bg
A7: srw_lsal_p3_energy_bg_sizeaware
A8: srw_lsal_multiscale optional
```

## Lệnh dry-run

```bash
python scripts/16_run_ablation.py \
  --plan configs/experiments/ablation_plan.yaml \
  --dry-run
```

## Lệnh chạy nhóm nhỏ

```bash
python scripts/16_run_ablation.py \
  --plan configs/experiments/ablation_plan.yaml \
  --run \
  --only baseline_yolov8s srw_only_p3 srw_lsal_p3_warmup_decay \
  --skip-existing
```

## Debug checklist

```text
[ ] Dry-run là mặc định
[ ] --run mới thực sự chạy
[ ] --only chọn đúng experiment
[ ] --skip-existing kiểm tra best.pt
[ ] Không chạy multi-scale mặc định
[ ] Summary CSV không fabricate metrics thiếu
```

## Prompt Codex Phase 17 — Tiếng Việt

```text
Hãy tạo Kaggle-safe ablation runner cho SkyFusion SRW + L_sal project.

Bối cảnh:
- Kaggle sessions giới hạn.
- Dry-run phải là mặc định.
- Ablation cần so sánh baseline, traditional augmentation, L_sal only, SRW only, và SRW + L_sal.

Nhiệm vụ:
1. Tạo configs/experiments/ablation_plan.yaml.
   Include:
   - baseline_yolov8s
   - tradaug_yolov8s
   - lsal_only_mse
   - srw_only_p3
   - srw_lsal_p3_mse
   - srw_lsal_p3_warmup_decay
   - srw_lsal_p3_energy_bg
   - srw_lsal_p3_energy_bg_sizeaware
   - srw_lsal_multiscale optional và disabled by default
2. Tạo scripts/16_run_ablation.py.
3. Features:
   - --dry-run default
   - --run mới execute commands
   - --only chọn experiment names
   - --skip-existing check weights/best.pt
   - --max-runs optional safety limit
4. Add result aggregation:
   - collect metrics.json
   - collect xai_metrics.json
   - collect convergence metrics
   - write paper/tables/ablation_summary.csv
5. Print commands trước khi chạy.

Ràng buộc:
- Không chạy toàn bộ experiment mặc định.
- Không hard-code GPU IDs.
- Không fabricate missing metrics.
- Multi-scale disabled by default.

Sau khi implement, hãy đưa ví dụ dry-run và ví dụ safe run.
```

---

# Phase 18 — Visualization và error analysis

## Mục tiêu

Tạo hình ảnh thuyết phục cho báo cáo/paper.

## Hình cần có

```text
1. Pipeline figure
2. SRW module figure
3. Training curriculum figure
4. Bbox size distribution
5. Baseline vs SRW prediction comparison
6. Baseline saliency vs SRW saliency
7. SRW spatial gate visualization
8. Failure cases: false positive, false negative, background confusion
```

## Lệnh

```bash
python scripts/17_generate_figures.py \
  --experiments baseline_yolov8s_seed0 srw_lsal_p3_warmup_decay_seed0 \
  --output-dir paper/figures
```

## Debug checklist

```text
[ ] Không cherry-pick quá mức
[ ] Có cả success cases và failure cases
[ ] Figure lưu PNG độ phân giải cao
[ ] Tên file rõ ràng
[ ] Không overwrite nhầm nếu chưa xác nhận
```

## Prompt Codex Phase 18 — Tiếng Việt

```text
Hãy implement visualization và error analysis utilities.

Bối cảnh:
- Báo cáo/paper cần qualitative evidence cho SRW + L_sal.
- Visualization cần hiển thị prediction, saliency và SRW gates.

Nhiệm vụ:
1. Tạo src/visualization/visualize_saliency.py.
2. Tạo src/visualization/visualize_srw_gates.py.
3. Tạo src/visualization/plot_curves.py.
4. Tạo scripts/17_generate_figures.py.
5. Generate:
   - bbox size distribution
   - loss curves
   - mAP curves
   - lambda curve
   - saliency overlays
   - SRW gate overlays
   - baseline vs proposed prediction panels
6. Add optional failure mining:
   - false positives
   - false negatives
   - low IoU localization errors

Ràng buộc:
- Dùng matplotlib, không dùng seaborn.
- Không fabricate visualization.
- Save outputs vào paper/figures.
- Mỗi plot là một figure riêng.

Sau khi implement, hãy đưa lệnh generate paper figures.
```

---

# Phase 19 — Export artifact từ Kaggle

## Mục tiêu

Tránh mất kết quả khi Kaggle session reset.

## Lệnh

```bash
python scripts/99_export_kaggle_artifacts.py \
  --output kaggle_outputs_srw_skyfusion.zip \
  --include-weights \
  --include-figures \
  --include-tables \
  --include-configs
```

## Debug checklist

```text
[ ] ZIP được tạo
[ ] Có best.pt cần thiết
[ ] Có metrics.json
[ ] Có xai_metrics.json
[ ] Có results.csv
[ ] Có paper/figures
[ ] Có paper/tables
[ ] Không copy toàn bộ dataset quá nặng
```

## Prompt Codex Phase 19 — Tiếng Việt

```text
Hãy thêm Kaggle artifact export utilities.

Bối cảnh:
- Kaggle working directory có thể bị reset sau session.
- Cần export experiment results, metrics, figures, tables và selected weights.
- Không được export toàn bộ dataset mặc định.

Nhiệm vụ:
1. Tạo scripts/99_export_kaggle_artifacts.py.
2. Script cần collect:
   - experiments/skyfusion/*/weights/best.pt
   - experiments/skyfusion/*/metrics.json
   - experiments/skyfusion/*/xai_metrics.json
   - experiments/skyfusion/*/results.csv
   - paper/figures/*
   - paper/tables/*
   - configs/*
   - README.md
3. Copy selected files vào kaggle_outputs/.
4. Tạo zip file.
5. CLI:
   --output
   --include-weights
   --include-figures
   --include-tables
   --include-configs
   --max-weight-files optional
6. Print export summary.

Ràng buộc:
- Không delete folder gốc.
- Không include full dataset mặc định.
- Missing files phải handle gracefully.

Sau khi implement, hãy đưa lệnh export trên Kaggle.
```

---

# Phase 20 — Review repo tổng trước khi chạy dài

## Mục tiêu

Dùng Codex review toàn bộ repo trước khi chạy training nhiều giờ.

## Prompt Codex tổng — Tiếng Việt

```text
Hãy review implementation hiện tại của repo Kaggle-first G-CAME-guided SRW YOLO cho SkyFusion dataset.

Bối cảnh:
- Dataset duy nhất hiện tại: thanhmay2406/dataset-for-research.
- Source Kaggle nằm trong /kaggle/input và là read-only.
- Dataset converted/reconverted chỉ nằm ở /kaggle/working/data/SkyFusion_yolo_reconverted nếu cần.
- Main contribution: SRW module sau YOLO FPN feature.
- Auxiliary contribution: L_sal saliency alignment với Gaussian GT bbox mask.
- XAI branch: G-CAME hoặc Grad-CAM-like extractor.
- Output phải nằm trong experiments/skyfusion/<run_name>/ và paper/.

Hãy kiểm tra:
1. Kaggle path handling.
2. Dataset inspect và prepare/convert SkyFusion.
3. YOLO dataset validation.
4. Tiny/small object profiling.
5. Baseline YOLO training script.
6. Traditional augmentation baseline.
7. GT saliency mask generation.
8. XAI saliency extraction.
9. SRW module implementation.
10. YOLO-SRW integration.
11. L_sal losses.
12. Lambda scheduler.
13. SRW-only và SRW+L_sal training scripts.
14. Evaluation scripts.
15. Ablation runner.
16. Artifact export.

Kiểm tra lỗi nghiêm trọng:
- Ghi vào /kaggle/input.
- Hard-code local paths.
- Bbox conversion sai.
- Tensor shape convention sai.
- Saliency map bị detach làm chặn gradient cần thiết.
- SRW gate collapse.
- Loss normalization không ổn định.
- Nguy cơ NaN/Inf.
- Baseline script import nhầm SRW/L_sal.
- Multi-scale bật mặc định gây OOM.
- Thiếu logs hoặc configs.
- Fabricate metrics.

Không edit ngay.
Trước tiên hãy tạo review gồm:
1. Critical issues
2. Medium-priority issues
3. Minor issues
4. Suggested fixes
5. Files cần edit
6. Commands nên chạy trên Kaggle để validate fixes
```

---

## 21. Thứ tự triển khai thực tế trên Kaggle

Không chạy toàn bộ roadmap một lượt. Nên đi theo thứ tự an toàn:

```text
1. Phase 0 — env check
2. Phase 1 — inspect Kaggle dataset
3. Phase 2 — prepare/convert SkyFusion sang YOLO
4. Phase 3 — check dataset + profile tiny/small object
5. Phase 4 — train baseline YOLO smoke test 1 epoch
6. Phase 4 — train baseline YOLO full
7. Phase 5 — train traditional augmentation baseline
8. Phase 6 — debug GT saliency mask
9. Phase 7 — debug SaliencyHead và XAI saliency map từ baseline weights
10. Phase 7 — optional precompute offline XAI teacher
11. Phase 8 — unit test SRW
12. Phase 9 — debug SRW-YOLO forward shape
13. Phase 10 — L_sal only smoke test
14. Phase 11 — SRW only smoke test
15. Phase 12 — SRW + L_sal smoke test
16. Phase 13 — thêm lambda schedule
17. Phase 14 — thêm energy/bg/size-aware nếu ổn
18. Phase 15 — multi-scale nếu còn GPU
19. Phase 16 — evaluation
20. Phase 17 — ablation summary
21. Phase 18 — figures/error analysis
22. Phase 19 — export artifact
```

Smoke test trước mỗi training dài:

```bash
python scripts/05_train_baseline.py --data "$SKYFUSION_DATA" --epochs 1 --imgsz 640 --batch 8 --run-name smoke_baseline

python scripts/10_train_lsal_only.py --data "$SKYFUSION_DATA" --epochs 1 --imgsz 640 --batch 8 --run-name smoke_lsal --saliency-provider saliency_head

python scripts/11_train_srw_only.py --data "$SKYFUSION_DATA" --epochs 1 --imgsz 640 --batch 8 --run-name smoke_srw --saliency-provider saliency_head

python scripts/12_train_srw_lsal.py --data "$SKYFUSION_DATA" --epochs 1 --imgsz 640 --batch 8 --run-name smoke_srw_lsal --saliency-provider saliency_head
```

---

## 22. Bảng quyết định sau mỗi phase

| Phase | Nếu tốt | Nếu không tốt |
|---|---|---|
| Env check | Inspect dataset | Sửa requirements / GPU setting |
| Inspect dataset | Prepare/convert | Kiểm tra lại Kaggle input path |
| Prepare dataset | Check YOLO | Sửa converter |
| Dataset check | Profile size | Sửa label/data.yaml |
| Profile size | Train baseline | Nếu object không tiny, ghi rõ limitation |
| Baseline | Train tradaug | Tune YOLO config trước |
| TradAug | Debug saliency | Giữ làm strong baseline |
| GT mask | Debug XAI saliency | Sửa bbox conversion / mask generation |
| Saliency provider | Unit test SRW | Kiểm tra SaliencyHead / teacher manifest / hook / target layer |
| SRW unit test | SRW-YOLO integration | Sửa gate shape / alpha |
| SRW-YOLO debug | L_sal only | Sửa wrapper/layer resolver / provider mode |
| L_sal only | SRW only | Giảm lambda hoặc sửa SaliencyHead/saliency normalization |
| SRW only | SRW + L_sal | Giảm alpha hoặc kiểm tra saliency map |
| SRW + L_sal | Lambda schedule | Debug total loss / gradient |
| Lambda schedule | Energy/BG/Size-aware | Dùng constant lambda nhỏ |
| Energy/BG/Size-aware | Evaluation | Giảm beta_bg / clamp size weight |
| Evaluation | Ablation summary | Sửa metrics scripts |
| Figures | Export | Bổ sung success/failure cases |

---

## 23. Ablation tối thiểu nên chạy

Không cần chạy quá nhiều ngay từ đầu. Bản tối thiểu:

| ID | Run name | Mục đích |
|---|---|---|
| A0 | baseline_yolov8s_seed0 | baseline sạch |
| A1 | tradaug_yolov8s_seed0 | strong augmentation baseline |
| A2 | lsal_only_mse_seed0 | kiểm tra L_sal riêng |
| A3 | srw_only_p3_seed0 | kiểm tra SRW riêng |
| A4 | srw_lsal_p3_mse_seed0 | main method cơ bản |
| A5 | srw_lsal_p3_warmup_decay_seed0 | main method + schedule |
| A6 | srw_lsal_energy_bg_sizeaware_seed0 | main method nâng cao |
| A7 | srw_lsal_multiscale_seed0 | optional nếu Kaggle còn GPU |

Bảng kết quả chính:

| Method | mAP50 | mAP50-95 | Precision | Recall | Recall_tiny | PG | EBPG | BER | Epoch-to-threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO baseline | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| YOLO + TradAug | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| L_sal only | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| SRW only | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| SRW + L_sal | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| SRW + L_sal + schedule | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| SRW + L_sal + energy/bg/size | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

---

## 24. Debug lỗi thường gặp trên Kaggle

### 24.1. Dataset path sai

Triệu chứng:

```text
FileNotFoundError: $SKYFUSION_DATA
```

Cách xử lý:

```bash
export SKYFUSION_ROOT=/kaggle/input/datasets/thanhmay2406/dataset-for-research
export SKYFUSION_YOLO_ROOT=/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo
export SKYFUSION_DATA=/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml
ls -lah "$SKYFUSION_ROOT"
ls -lah "$SKYFUSION_YOLO_ROOT"
cat "$SKYFUSION_DATA"
python scripts/01_inspect_kaggle_dataset.py --input-root "$SKYFUSION_ROOT" --max-depth 4
```

### 24.2. Label convert sai

Triệu chứng:

```text
bbox outside [0,1]
class id out of range
mAP gần 0
prediction rất lệch
```

Cách xử lý:

```bash
python scripts/03_check_yolo_dataset.py --data "$SKYFUSION_DATA" --output-dir experiments/skyfusion/dataset_check
python scripts/04_profile_small_objects.py --data "$SKYFUSION_DATA" --output-dir experiments/skyfusion/dataset_profile
```

Kiểm tra overlay:

```text
experiments/skyfusion/dataset_profile/bbox_overlay_samples/
```

### 24.3. CUDA OOM

Cách xử lý:

```text
- giảm batch 16 → 8 → 4
- dùng yolov8n.pt thay yolov8s.pt để smoke test
- chỉ dùng P3-only
- tắt multi-scale
- giảm imgsz 640 → 512 để debug
```

### 24.4. Saliency toàn 0 hoặc NaN

Kiểm tra:

```text
- feature_map.requires_grad
- det_loss có thật sự phụ thuộc vào feature_map không
- normalize saliency có eps chưa
- có detach nhầm feature_map không
- target layer có đúng không
```

Debug:

```bash
python scripts/08_debug_xai_saliency.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/baseline_yolov8s_seed0/weights/best.pt \
  --split valid \
  --target-layers P3 \
  --xai-method gradcam_like \
  --output-dir experiments/skyfusion/debug_xai_saliency
```

### 24.5. SRW gate collapse

Triệu chứng:

```text
gate_s gần 0 toàn bộ
gate_c gần 1 toàn bộ
alpha tăng quá lớn
mAP giảm mạnh
```

Cách xử lý:

```text
- giảm alpha_init 0.1 → 0.05
- clamp alpha hoặc regularize
- kiểm tra saliency S có meaningful không
- dùng residual F* = F + alpha * ...
- log gate mean/std mỗi epoch
```

### 24.6. L_sal quá lớn

Cách xử lý:

```text
- giảm lambda_sal 0.1 → 0.05 → 0.01
- dùng warmup_cosine_decay
- normalize M_xai và M_gt về [0,1]
- kiểm tra empty mask
- kiểm tra batch có quá nhiều empty label không
```

---

## 25. README quickstart nên có

```markdown
# SkyFusion SRW-YOLO Quickstart on Kaggle

## 1. Add dataset
Add Kaggle dataset input:

thanhmay2406/dataset-for-research

## 2. Bootstrap

```bash
pip install -q -r requirements.txt
python scripts/00_env_check.py
```

## 3. Inspect dataset

```bash
python scripts/01_inspect_kaggle_dataset.py \
  --input-root "$SKYFUSION_ROOT" \
  --dataset-name SkyFusion_yolo \
  --max-depth 4
```

## 4. Prepare YOLO dataset

```bash
python scripts/02_prepare_skyfusion.py \
  --source-root /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion \
  --output-root /kaggle/working/data/SkyFusion_yolo_reconverted \
  --format auto \
  --preserve-splits
```

## 5. Validate dataset

```bash
python scripts/03_check_yolo_dataset.py \
  --data "$SKYFUSION_DATA" \
  --output-dir experiments/skyfusion/dataset_check

python scripts/04_profile_small_objects.py \
  --data "$SKYFUSION_DATA" \
  --output-dir experiments/skyfusion/dataset_profile
```

## 6. Smoke test

```bash
python scripts/05_train_baseline.py \
  --data "$SKYFUSION_DATA" \
  --epochs 1 \
  --batch 8 \
  --run-name smoke_baseline
```

## 7. Train baseline

```bash
python scripts/05_train_baseline.py \
  --data "$SKYFUSION_DATA" \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 0 \
  --run-name baseline_yolov8s_seed0
```

## 8. Train main method

```bash
python scripts/12_train_srw_lsal.py \
  --data "$SKYFUSION_DATA" \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 0 \
  --run-name srw_lsal_p3_warmup_decay_seed0 \
  --xai-method gradcam_like \
  --target-layers P3 \
  --loss-type mse \
  --lambda-schedule warmup_cosine_decay \
  --lambda-max 0.2 \
  --lambda-min 0.01 \
  --warmup-epochs 5
```

## 9. Export artifacts

```bash
python scripts/99_export_kaggle_artifacts.py \
  --output kaggle_outputs_srw_skyfusion.zip \
  --include-weights \
  --include-figures \
  --include-tables \
  --include-configs
```
```

---

## 26. Kết luận roadmap mới

Phiên bản này đã đổi từ:

```text
multi-benchmark public roadmap
```

thành:

```text
SkyFusion-only Kaggle roadmap
```

Điểm giữ lại:

```text
- SRW module vẫn là main contribution.
- L_sal vẫn là auxiliary saliency alignment.
- Grad-CAM-like/G-CAME branch vẫn là nguồn saliency.
- Kaggle-first vẫn là nguyên tắc triển khai.
```

Điểm đơn giản hóa:

```text
- Bỏ registry nhiều benchmark.
- Bỏ converter VisDrone/AI-TOD/SODA ở giai đoạn đầu.
- Chỉ tập trung inspect + prepare SkyFusion.
- Tất cả prompt Codex đã chuyển sang tiếng Việt.
- Tất cả command dùng `$SKYFUSION_DATA` hoặc YAML fallback đã patch path.
```

Main method nên chốt ban đầu:

```text
YOLOv8s + P3-only SRW + SaliencyHead S_pred + L_sal MSE + warmup-cosine-decay λ; Grad-CAM-like/G-CAME dùng làm teacher/debug/evaluation
```

Sau khi chạy ổn:

```text
Energy-in-Box + Background Suppression + Size-aware + optional multi-scale
```

Nếu kết quả trên SkyFusion ổn, bước tiếp theo mới mở rộng:

```text
AI-TOD-v2
VisDrone
SODA-A/SODA-D
private YOLO-format dataset
```

---

## 27. Nguồn dataset

```text
Kaggle dataset:
https://www.kaggle.com/datasets/thanhmay2406/dataset-for-research

Kaggle slug:
thanhmay2406/dataset-for-research

Tên hiển thị:
SkyFusion: Aerial Object Detection
```

Ghi chú:

```text
Nên dùng Kaggle dataset này để thử nghiệm pipeline trước.
Khi viết paper nghiêm túc hơn, cần ghi rõ dataset source, format annotation, số ảnh, số class, số bbox và split thực tế sau khi inspect.
Không nên ghi số liệu dataset nếu chưa chạy script inspect/profile trên chính notebook của bạn.
```


---

## Phụ lục A — Ghi chú chỉnh sửa dataset structure

Roadmap này đã được đồng bộ với mô tả cấu trúc SkyFusion hiện tại:

```text
- COCO gốc: data/SkyFusion/{train,valid,test}/_annotations.coco.json
- YOLO train/eval: `$SKYFUSION_DATA` = `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml`
- Split chính thức: train / valid / test
- Class hiện tại: Aircraft, ship, vehicle
- Số lượng ảnh: train 2094, valid 449, test 449
```

Điểm cần nhớ khi viết code:

```text
1. Không dùng nhầm path local cũ; trên Kaggle dùng `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo`.
2. Không tạo split `val/` mới nếu dataset đã dùng `valid/`.
3. Trong `data.yaml`, key `val:` của Ultralytics có thể trỏ tới `images/valid`.
4. Phase prepare dataset nên ưu tiên validate bản YOLO sẵn có trước, sau đó mới convert lại từ COCO.
```
