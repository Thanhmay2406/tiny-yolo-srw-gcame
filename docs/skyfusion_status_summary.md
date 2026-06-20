# SkyFusion Status Summary

## Hien trang dataset nguon

- Nguon du lieu hien tai nam o `data/SkyFusion`.
- Cau truc split gom `train/`, `valid/`, `test/`.
- Annotation cua moi split dang o dinh dang COCO JSON:
  - `data/SkyFusion/train/_annotations.coco.json`
  - `data/SkyFusion/valid/_annotations.coco.json`
  - `data/SkyFusion/test/_annotations.coco.json`
- Class hien co:
  - `Aircraft`
  - `ship`
  - `vehicle`

## Script chuyen doi da tao

- Da them script: `data/convert_skyfusion_to_yolo.py`
- Muc dich:
  - Doc annotation COCO cua SkyFusion
  - Chuyen bbox `xywh` pixel sang YOLO normalized format
  - Tao cau truc YOLO detection standard
  - Sinh file `data.yaml`
- Lenh chay:

```bash
.venv/bin/python data/convert_skyfusion_to_yolo.py --clean
```

- Tuy chon bo sung:
  - `--source`: doi thu muc dataset nguon
  - `--output`: doi thu muc dataset dau ra
  - `--image-mode copy|hardlink|symlink`: cach dat anh vao dataset YOLO

## Dataset YOLO da duoc tao

- Thu muc dau ra: `data/SkyFusion_yolo`
- Cau truc da tao:
  - `data/SkyFusion_yolo/images/train`
  - `data/SkyFusion_yolo/images/valid`
  - `data/SkyFusion_yolo/images/test`
  - `data/SkyFusion_yolo/labels/train`
  - `data/SkyFusion_yolo/labels/valid`
  - `data/SkyFusion_yolo/labels/test`
  - `data/SkyFusion_yolo/data.yaml`

## So lieu sau khi convert

- `train`: 2094 images, 43575 objects, 0 empty label files
- `valid`: 449 images, 8387 objects, 0 empty label files
- `test`: 449 images, 11751 objects, 0 empty label files

## Noi dung `data.yaml`

- `path`: tro toi `data/SkyFusion_yolo`
- `train`: `images/train`
- `val`: `images/valid`
- `test`: `images/test`
- `nc`: `3`
- `names`:
  - `0: Aircraft`
  - `1: ship`
  - `2: vehicle`

## Ghi chu

- Script hien tai duoc viet rieng cho SkyFusion COCO export dang co trong repo.
- Script khong can cai them thu vien ngoai; co the chay bang Python co san trong `.venv` hoac he thong.
- Dataset YOLO da san sang de dua vao pipeline train Ultralytics/YOLO.
