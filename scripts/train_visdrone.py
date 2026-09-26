"""在 VisDrone 训练集上微调 YOLO11（切片辅助微调：整图 + 原分辨率裁块，同 SAHI 论文的 SF 做法）。

目的：去掉"零训练 COCO 检测器"这个前提，看两条结论还在不在——
  (1) Glance-SAHI 相对 SAHI 少切片、精度几乎不变；
  (2) 实验 A 的"高分辨率整图推理支配 SAHI"。
预注册见 docs/PREREG-2026-09-26-ft.md（先提交，后训练）。

prepare  流式读 VisDrone2019-DET-train.zip（HTTP Range，不落 zip，约 1.55GB 流量），每张图生成：
           (a) 整图缩到长边 640          —— 扫视 / 整图推理的尺度
           (b) 1 块原分辨率 640×640 裁块  —— SAHI 切片（512→640）的尺度；80% 概率以某个目标为中心
         类别并成 2 类，与评测口径一致：0 person ← pedestrian, people；1 vehicle ← car, van, truck, bus。
         忽略区域（ignored region、others、score=0）先用灰色 114 抹掉，避免把没标注的目标当成背景学。
         按文件名哈希留 5% 训练图做 mini-val（只用来选 best.pt）；VisDrone val 的 548 张不参与训练和选模型。
train    YOLO(yolo11s.pt).train(...)，按时间预算结束（--hours），best.pt 复制到 weights/yolo11s-visdrone-ft.pt。

  python scripts/train_visdrone.py prepare
  python scripts/train_visdrone.py train --hours 1.2
显存不够时（如 4GB 卡）：--batch 8 或 4（Ultralytics 会按名义 batch 64 自动做梯度累积），
再不行换 --model yolo11n.pt，或 --imgsz 512（精度会降）。
评测（与实验 A 同一套脚本，数据集名 visdrone_ft，结果写 results/visdrone_ft/）：
  python scripts/run_eval.py cache --dataset visdrone_ft --weights weights/yolo11s-visdrone-ft.pt
  python scripts/holdout_theta.py --dataset visdrone_ft
  python scripts/strong_baselines.py run  --dataset visdrone_ft --weights weights/yolo11s-visdrone-ft.pt
  python scripts/strong_baselines.py eval --dataset visdrone_ft
"""

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip"
OUT = ROOT / "datasets" / "VisDrone-ft"
TO_CLASS = {1: 0, 2: 0, 4: 1, 5: 1, 6: 1, 9: 1}  # VisDrone 类别 -> 训练类别
IGNORE = {0, 11}
SIZE = 640
GRAY = 114


def parse_annotation(text):
    """返回 (目标框 [x, y, w, h, cls] 列表, 忽略区域 [x, y, w, h] 列表)。"""
    boxes, ignore = [], []
    for line in text.splitlines():
        v = [int(t) for t in line.strip().strip(",").split(",")[:6] if t.strip()]
        if len(v) < 6 or v[2] <= 0 or v[3] <= 0:
            continue
        x, y, w, h, score, cat = v
        if cat in IGNORE or score == 0:
            ignore.append((x, y, w, h))
        elif cat in TO_CLASS:
            boxes.append((x, y, w, h, TO_CLASS[cat]))
    return boxes, ignore


def yolo_lines(boxes, x0, y0, cw, ch, scale):
    """把原图坐标的框裁到窗口 [x0, x0+cw)×[y0, y0+ch) 内、再乘 scale，输出 YOLO 行。

    裁完面积不到原来 40%、或边长不到 2 像素（输出尺度）的框丢掉。
    """
    out = []
    ow, oh = cw * scale, ch * scale
    for x, y, w, h, c in boxes:
        a1, b1 = max(x, x0), max(y, y0)
        a2, b2 = min(x + w, x0 + cw), min(y + h, y0 + ch)
        if a2 <= a1 or b2 <= b1 or (a2 - a1) * (b2 - b1) < 0.4 * w * h:
            continue
        bw, bh = (a2 - a1) * scale, (b2 - b1) * scale
        if bw < 2 or bh < 2:
            continue
        cx, cy = ((a1 + a2) / 2 - x0) * scale, ((b1 + b2) / 2 - y0) * scale
        out.append(f"{c} {cx / ow:.6f} {cy / oh:.6f} {bw / ow:.6f} {bh / oh:.6f}")
    return out


def make_samples(img, boxes, ignore, rng):
    """返回 [(后缀, 图像, YOLO 行)]：整图缩放一份 + 原分辨率裁块一份。"""
    img = img.copy()
    for x, y, w, h in ignore:
        img[y:y + h, x:x + w] = GRAY
    h, w = img.shape[:2]
    s = SIZE / max(h, w)
    full = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    samples = [("full", full, yolo_lines(boxes, 0, 0, w, h, s))]

    cw, ch = min(SIZE, w), min(SIZE, h)
    if boxes and rng.random() < 0.8:
        x, y, bw, bh, _ = boxes[rng.integers(len(boxes))]
        cx, cy = x + bw / 2 + rng.uniform(-0.3, 0.3) * cw, y + bh / 2 + rng.uniform(-0.3, 0.3) * ch
        x0, y0 = int(np.clip(cx - cw / 2, 0, w - cw)), int(np.clip(cy - ch / 2, 0, h - ch))
    else:
        x0, y0 = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
    samples.append(("crop", img[y0:y0 + ch, x0:x0 + cw], yolo_lines(boxes, x0, y0, cw, ch, 1.0)))
    return samples


def is_val(stem):
    return int(hashlib.md5(stem.encode()).hexdigest(), 16) % 20 == 0  # 5%


def write_yaml():
    path = OUT / "data.yaml"
    path.write_text(f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\n"
                    "names:\n  0: person\n  1: vehicle\n", encoding="utf-8")
    return path


def cmd_prepare(args):
    from glance_sahi.data.remote_zip import HttpRangeFile

    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    src = open(args.zip, "rb") if args.zip else HttpRangeFile(URL)
    with zipfile.ZipFile(src) as zf:
        infos = sorted(zf.infolist(), key=lambda i: i.header_offset)
        anns = {Path(i.filename).stem: zf.read(i).decode() for i in infos
                if "/annotations/" in i.filename and i.filename.endswith(".txt")}
        images = [i for i in infos if "/images/" in i.filename and i.filename.endswith(".jpg")]
        if args.limit:
            images = images[: args.limit]
        print(f"{len(images)} images, {len(anns)} annotation files")
        n_box = {"train": 0, "val": 0}
        for info in tqdm(images, desc="prepare"):
            stem = Path(info.filename).stem
            split = "val" if is_val(stem) else "train"
            if (OUT / "labels" / split / f"{stem}_crop.txt").exists():  # 断点续跑
                continue
            img = cv2.imdecode(np.frombuffer(zf.read(info), np.uint8), cv2.IMREAD_COLOR)
            boxes, ignore = parse_annotation(anns.get(stem, ""))
            for suffix, im, lines in make_samples(img, boxes, ignore, rng):
                cv2.imwrite(str(OUT / "images" / split / f"{stem}_{suffix}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (OUT / "labels" / split / f"{stem}_{suffix}.txt").write_text("\n".join(lines))
                n_box[split] += len(lines)
    if hasattr(src, "fetched"):
        print(f"downloaded {src.fetched / 2**20:.0f} MB")
    print(f"boxes written: {n_box}; yaml: {write_yaml()}")


def cmd_train(args):
    from ultralytics import YOLO

    data = write_yaml()
    model = YOLO(args.model)
    model.train(data=str(data), imgsz=args.imgsz, epochs=args.epochs, time=args.hours, batch=args.batch,
                workers=args.workers, device=args.device, seed=0, project=str(ROOT / "runs_ft"),
                name=args.name, exist_ok=True, plots=True, amp=True, cache=False, close_mosaic=3)
    best = ROOT / "runs_ft" / args.name / "weights" / "best.pt"
    dst = ROOT / "weights" / args.out
    dst.parent.mkdir(exist_ok=True)
    shutil.copy2(best, dst)
    print(f"best -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "train"])
    ap.add_argument("--zip", default=None, help="prepare：已下载的本地 zip（默认流式读远程）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=str(ROOT / "yolo11s.pt"))
    ap.add_argument("--imgsz", type=int, default=SIZE)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hours", type=float, default=1.2, help="训练时间预算（小时），到时即停并按剩余时间调学习率")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="visdrone_ft")
    ap.add_argument("--out", default="yolo11s-visdrone-ft.pt")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "train": cmd_train}[a.cmd](a)
