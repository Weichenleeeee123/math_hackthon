"""强基线对照：Glance-SAHI 在不在"只调参数就能得到"的速度-精度前沿上。

评委最可能问的一句话："把 SAHI 的切片调大，或者直接用高分辨率整图推理，不也能省时间？"
本脚本在同一次运行、同一检测器、同一后处理下实测这些点：
  full@S    整图推理，检测器输入长边 S（S 越大越清楚、越慢）
  sahi@S    官方 get_sliced_prediction，切片边长 S、重叠 0.2（S 越大切片越少、每片放大越少）
  glance@θ  Glance-SAHI（det+edge，切片网格 = --slice-size）
每张图上方法顺序按图序号轮换，逐图保存耗时与检测结果；AP 差与耗时比用按图配对的 bootstrap 给
95% 区间（glance_sahi/bootstrap.py）。判定规则见预注册 docs/PREREG-2026-09-26.md，先提交后运行。

  python scripts/strong_baselines.py run  --dataset visdrone
  python scripts/strong_baselines.py eval --dataset visdrone
  # DOTA，换成在 DOTA 上训练过的 OBB 检测器（15 类，评水平框）
  python scripts/strong_baselines.py run  --dataset dota15 --weights yolo11s-obb.pt --imgsz 1024 \
      --slice-size 1024 --full-sizes 1024,2048,3072 --sahi-sizes 1024,1536,2048 --glance-ths 0.5,0.9
  python scripts/strong_baselines.py eval --dataset dota15 --op 0.9

输出（results/<dataset>/）：strong<tag>.pkl（逐图检测与耗时，已忽略）、strong<tag>.csv、
figures/fig_strong_pareto<tag>.png。
"""

import argparse
import contextlib
import io
import json
import pickle
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_eval as R  # noqa: E402
from glance_sahi.config import GlanceConfig  # noqa: E402

# 非劣效界：比 Glance-SAHI 显著更快、且 AP 与 AP_small 都不比它差超过这个界（AP 点），就算"支配"它
NONINFERIORITY = 0.5


def _ints(s):
    return [int(v) for v in s.split(",") if v.strip()]


def _floats(s):
    return [float(v) for v in s.split(",") if v.strip()]


def method_names(args):
    return ([f"full@{s}" for s in _ints(args.full_sizes)] + [f"sahi@{s}" for s in _ints(args.sahi_sizes)]
            + [f"glance@{t:g}" for t in _floats(args.glance_ths)])


def env_info():
    import sahi
    import torch
    import ultralytics

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = "", None
    return {"commit": commit, "git_dirty": dirty, "python": platform.python_version(), "torch": torch.__version__,
            "sahi": sahi.__version__, "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def letterbox_mpx(h, w, imgsz, stride=32):
    """检测器实际处理的像素数（百万）：长边缩放到 imgsz（ultralytics 会放大小图），短边补到 stride 的倍数。

    与硬件无关的算量代理：卷积网络的 FLOPs 近似与输入像素数成正比。
    """
    r = imgsz / max(h, w)
    nh, nw = round(h * r), round(w * r)
    return (int(np.ceil(nh / stride)) * stride) * (int(np.ceil(nw / stride)) * stride) / 1e6


def slices_mpx(slices, idx, imgsz):
    return sum(letterbox_mpx(slices[k][3] - slices[k][1], slices[k][2] - slices[k][0], imgsz) for k in idx)


def run_one(name, img, model, cfg, base_imgsz, exclude):
    """跑一个方法，返回 (ObjectPrediction 列表, 秒数, 跑了几个切片, 送进检测器的百万像素)。计时不含读图。"""
    from sahi.predict import get_prediction
    from sahi.slicing import get_slice_bboxes

    from glance_sahi.predict import glance_sliced_prediction, sahi_uniform_prediction

    h, w = img.shape[:2]
    kind, val = name.split("@")
    if kind == "full":
        model.image_size = int(val)
        try:
            t0 = time.perf_counter()
            r = get_prediction(img, model, exclude_classes_by_id=exclude)
            dt = time.perf_counter() - t0
        finally:
            model.image_size = base_imgsz
        return r.object_prediction_list, dt, 0, letterbox_mpx(h, w, int(val))
    glance_mpx = letterbox_mpx(h, w, base_imgsz)  # SAHI 的 standard pred / Glance 的扫视，都是基准尺寸整图一次
    if kind == "sahi":
        c = replace(cfg, slice_size=int(val))
        r, dt, n = sahi_uniform_prediction(img, model, c, exclude)
        slices = get_slice_bboxes(h, w, c.slice_size, c.slice_size, False, c.overlap_ratio, c.overlap_ratio)
        return r.object_prediction_list, dt, n, glance_mpx + slices_mpx(slices, range(len(slices)), base_imgsz)
    t0 = time.perf_counter()
    r, st = glance_sliced_prediction(img, model, replace(cfg, threshold=float(val)), exclude)
    dt = time.perf_counter() - t0
    return r.object_prediction_list, dt, st.n_slices_run, glance_mpx + slices_mpx(st.slices, st.selected, base_imgsz)


def cmd_run(args):
    from glance_sahi.detector import build_model

    names = method_names(args)
    cfg = GlanceConfig(slice_size=args.slice_size, img_prior="edge", img_weight=args.img_weight)
    coco = json.loads(R.GT.read_text())
    images = coco["images"][: args.limit] if args.limit else coco["images"]
    model = build_model(args.weights, conf=cfg.output_conf, device=args.device, image_size=args.imgsz)

    # 预热：每个方法在前 3 张图上各跑一遍（每种新输入尺寸第一次都有 CUDA/cuDNN 一次性开销），不计时
    for im in images[:3]:
        img = R.load_rgb(R.IMAGES / im["file_name"])
        for name in names:
            run_one(name, img, model, cfg, args.imgsz, R.EXCLUDE)

    out = {"methods": names, "args": vars(args), "env": env_info(), "images": [],
           "dets": {n: [] for n in names}, "times": {n: [] for n in names}, "n_run": {n: [] for n in names},
           "mpx": {n: [] for n in names}}
    for i, im in enumerate(tqdm(images, desc="strong")):
        img = R.load_rgb(R.IMAGES / im["file_name"])
        out["images"].append({"id": im["id"], "hw": img.shape[:2]})
        order = names[i % len(names):] + names[: i % len(names)]  # 轮换顺序，抵消"排第几个跑"的系统偏差
        for name in order:
            preds, dt, n, mpx = run_one(name, img, model, cfg, args.imgsz, R.EXCLUDE)
            out["dets"][name].append(R.preds_to_np(preds))
            out["times"][name].append(dt)
            out["n_run"][name].append(n)
            out["mpx"][name].append(mpx)
    R.RES.mkdir(parents=True, exist_ok=True)
    path = R.RES / f"strong{args.tag}.pkl"
    path.write_bytes(pickle.dumps(out))
    print(f"wrote {path}  ({len(images)} images x {len(names)} methods)")


def dominated_by(df, target, cost="time"):
    """按预注册规则，列出支配 target 的方法：ΔAP、ΔAP_small 的 95% 下界都 > −界，且
    cost="time"：耗时比 95% 上界 < 1（显著更快）；cost="mpx"：平均送检像素更少（算量，确定量，不需要区间）。"""
    rows = df[(df.ref == target) & (df.method != target)]
    ok = (rows.dAP_lo > -NONINFERIORITY) & (rows.dAPs_lo > -NONINFERIORITY)
    if cost == "time":
        cheaper = rows.time_ratio_hi < 1
    else:
        own = df.loc[(df.ref == target) & (df.method == target), "mpx_per_img"].iloc[0]
        cheaper = rows.mpx_per_img < own
    return list(rows[ok & cheaper].method)


def cmd_eval(args):
    from pycocotools.coco import COCO

    from glance_sahi.bootstrap import PreparedEval, paired_bootstrap

    data = pickle.loads((R.RES / f"strong{args.tag}.pkl").read_bytes())
    names, ids = data["methods"], [im["id"] for im in data["images"]]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(R.GT))
    prepared = {}
    for n in names:
        dets = [d for img_id, arr in zip(ids, data["dets"][n]) for d in R.to_coco_dets(img_id, arr)]
        prepared[n] = PreparedEval(gt, dets, ids, R.DS["max_dets"])
    times = {n: np.asarray(data["times"][n]) for n in names}
    target = f"glance@{args.op:g}"
    base = f"sahi@{data['args']['slice_size']}"
    refs = [r for r in (target, base) if r in names]
    df = pd.DataFrame(paired_bootstrap(prepared, times, refs, args.boot, args.seed))
    df["slices_per_img"] = df.method.map({n: float(np.mean(data["n_run"][n])) for n in names})
    df["mpx_per_img"] = df.method.map({n: float(np.mean(data["mpx"][n])) for n in names})

    # 前沿：没有别的方法同时"更快且 AP 不低"（点估计）
    pts = df[df.ref == refs[0]].set_index("method")
    for n in names:
        faster = pts.ms_per_img < pts.at[n, "ms_per_img"]
        better = pts.AP >= pts.at[n, "AP"]
        df.loc[df.method == n, "on_frontier"] = not bool((faster & better).any())
    out = R.RES / f"strong{args.tag}.csv"
    df.to_csv(out, index=False)

    cols = ["method", "slices_per_img", "mpx_per_img", "ms_per_img", "AP", "APs", "dAP", "dAP_lo", "dAP_hi",
            "dAPs", "dAPs_lo", "dAPs_hi", "time_ratio", "time_ratio_lo", "time_ratio_hi", "on_frontier"]
    pd.set_option("display.width", 200)
    for ref in refs:
        print(f"\n== 相对 {ref}（ΔAP 为 AP 点；耗时比 = 方法 / {ref}；{args.boot} 次按图配对 bootstrap）")
        print(df[df.ref == ref][cols].round(3).to_string(index=False))
    print(f"\nenv: {data['env']}")
    if target in names:
        for cost, label in (("time", "墙钟"), ("mpx", "算量（送检像素）")):
            dom = dominated_by(df, target, cost)
            print(f"\n预注册判定·{label}（非劣效界 {NONINFERIORITY} AP 点）：{target} "
                  + (f"被支配，支配者：{dom}" if dom else "未被任何强基线支配"))
    plot(df[df.ref == refs[0]], data, args)
    print(f"wrote {out}")


def plot(df, data, args):
    import make_figures as MF

    plt = MF.plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    style = {"full": ("#2a78d6", "o", "整图推理（输入边长）"), "sahi": ("#eb6834", "s", "SAHI（切片边长）"),
             "glance": ("#1baf7a", "D", "Glance-SAHI（θ）")}
    panels = ((axes[0], "ms_per_img", "AP", "每图耗时（ms，同一次运行）"),
              (axes[1], "ms_per_img", "APs", "每图耗时（ms，同一次运行）"),
              (axes[2], "mpx_per_img", "APs", "每图送进检测器的像素（百万，≈算量）"))
    for ax, x, metric, xlabel in panels:
        for kind, (color, marker, legend) in style.items():
            part = df[df.method.str.startswith(kind + "@")].sort_values(x)
            if part.empty:
                continue
            ax.plot(part[x], part[metric], marker=marker, color=color, label=legend, lw=1.2, ms=6)
            for _, r in part.iterrows():
                ax.annotate(r.method.split("@")[1], (r[x], r[metric]), textcoords="offset points",
                            xytext=(4, 4), fontsize=8, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("AP" if metric == "AP" else "AP_small")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    gpu = data["env"].get("gpu", "")
    fig.suptitle(f"{args.dataset}：强基线速度-精度前沿（{gpu}，{len(data['images'])} 张）", fontsize=11)
    fig.tight_layout()
    fig_dir = R.RES / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"fig_strong_pareto{args.tag}.png"
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "eval"])
    ap.add_argument("--dataset", default="visdrone")
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--imgsz", type=int, default=640, help="检测器的基准输入尺寸（扫视与切片都用它）")
    ap.add_argument("--slice-size", type=int, default=512, help="SAHI 基线与 Glance-SAHI 共用的切片网格")
    ap.add_argument("--full-sizes", default="640,960,1280,1600,1920,2560")
    ap.add_argument("--sahi-sizes", default="512,640,768,1024")
    ap.add_argument("--glance-ths", default="0.9,0.99")
    ap.add_argument("--img-weight", type=float, default=0.3)
    ap.add_argument("--op", type=float, default=0.9, help="eval：要判定的 Glance-SAHI 工作点")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    R.set_dataset(a.dataset)
    {"run": cmd_run, "eval": cmd_eval}[a.cmd](a)
