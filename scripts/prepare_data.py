"""准备评测数据：下载（若不存在）并转成 COCO json。

  python scripts/prepare_data.py                   # VisDrone2019-DET-val（548 张）
  python scripts/prepare_data.py --dataset dota    # DOTA-v1.0 val（458 张，800~4000+ 像素大图）
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glance_sahi.data import dota, visdrone  # noqa: E402

DOTA_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip"
# (目录名, 转换函数, 下载地址, 真值文件名)
SOURCES = {
    "visdrone": ("VisDrone2019-DET-val", visdrone.convert,
                 "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
                 "coco_eval.json"),
    # 2GB 的完整包，只需要 val：可手动 `tar -xf DOTAv1.zip DOTAv1/images/val DOTAv1/labels/val`
    "dota": ("DOTAv1", dota.convert, DOTA_URL, "coco_eval.json"),
    # 同一份 DOTA val，评全部 15 类（配 OBB 检测器）；图像与 dota 共用，只多一个真值文件
    "dota15": ("DOTAv1", dota.convert_all, DOTA_URL, "coco_eval_dota15.json"),
    # 受控实验画布：不是下载来的，用 `scripts/sparsity_sweep.py --save` 生成
    "sparse4k": ("VisDrone-Sparse4K", visdrone.convert, None, "coco_eval.json"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="visdrone", choices=list(SOURCES))
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    name, convert, url, gt_name = SOURCES[args.dataset]
    root = Path(args.root)
    data_dir = root / name
    if not (data_dir / "images").exists():
        if url is None:
            raise SystemExit(f"{data_dir} 不存在：先跑 `python scripts/sparsity_sweep.py --save` 生成画布")
        root.mkdir(parents=True, exist_ok=True)
        zip_path = root / f"{name}.zip"
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, zip_path)
        zipfile.ZipFile(zip_path).extractall(root)

    coco = convert(data_dir, data_dir / gt_name, args.limit)
    n_obj = sum(1 for a in coco["annotations"] if not a["iscrowd"])
    n_small = sum(1 for a in coco["annotations"] if not a["iscrowd"] and a["area"] < 32 * 32)
    sizes = [im["width"] * im["height"] for im in coco["images"]]
    print(f"images={len(coco['images'])} objects={n_obj} small(<32²)={n_small} ({n_small / n_obj:.1%}) "
          f"mean_megapixels={sum(sizes) / len(sizes) / 1e6:.2f}")
