# SkyFusion Dataset Structure

## Tong quan

Repo hien co 2 thu muc dataset lien quan den SkyFusion:

- `data/SkyFusion`: dataset goc, annotation dang COCO JSON
- `data/SkyFusion_yolo`: dataset da duoc chuyen sang dinh dang YOLO detection

## 1. Cau truc `data/SkyFusion`

```text
data/SkyFusion/
|-- README.roboflow.txt
|-- train/
|   |-- _annotations.coco.json
|   |-- *.jpg
|-- valid/
|   |-- _annotations.coco.json
|   |-- *.jpg
|-- test/
|   |-- _annotations.coco.json
|   |-- *.jpg
```

Y nghia:

- `README.roboflow.txt`: metadata export tu Roboflow.
- `train/`, `valid/`, `test/`: 3 split chinh cua dataset.
- `_annotations.coco.json`: file annotation cho tung split theo dinh dang COCO.
- `*.jpg`: anh dau vao trong moi split.

Luu y:

- Anh va annotation dang nam chung trong cung thu muc split.
- Annotation su dung bbox theo kieu COCO `xywh` tinh theo pixel.
- Class hien tai cua dataset:
  - `Aircraft`
  - `ship`
  - `vehicle`

## 2. Cau truc `data/SkyFusion_yolo`

```text
data/SkyFusion_yolo/
|-- data.yaml
|-- images/
|   |-- train/
|   |   |-- *.jpg
|   |-- valid/
|   |   |-- *.jpg
|   |-- test/
|   |   |-- *.jpg
|-- labels/
|   |-- train/
|   |   |-- *.txt
|   |-- valid/
|   |   |-- *.txt
|   |-- test/
|       |-- *.txt
```

Y nghia:

- `data.yaml`: file cau hinh dataset cho YOLO/Ultralytics.
- `images/{train,valid,test}`: anh dau vao sau khi sap xep theo cau truc YOLO.
- `labels/{train,valid,test}`: nhan YOLO tuong ung voi tung anh.
- Moi file `*.txt` trong `labels/` co ten trung voi anh, chi khac phan mo rong.

## 3. Quan he giua dataset goc va dataset YOLO

- `data/SkyFusion` la nguon goc de chuyen doi.
- `data/SkyFusion_yolo` la ban da duoc chuan hoa de train bang YOLO.
- Script su dung de chuyen doi:

```text
data/convert_skyfusion_to_yolo.py
```

## 4. So luong hien tai

`data/SkyFusion`:

- `train`: 2094 images
- `valid`: 449 images
- `test`: 449 images

`data/SkyFusion_yolo`:

- `images/train`: 2094 files
- `images/valid`: 449 files
- `images/test`: 449 files
- `labels/train`: 2094 files
- `labels/valid`: 449 files
- `labels/test`: 449 files

## 5. Muc dich su dung

- Dung `data/SkyFusion` khi can inspect annotation goc COCO.
- Dung `data/SkyFusion_yolo/data.yaml` khi can train/eval bang Ultralytics YOLO.
