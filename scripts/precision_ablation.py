"""FP16 + 切片批推理 vs FP32 逐片（REPORT 3.18）：同一块 GPU、同一检测器，逐方法比 AP 与耗时。

读两次 strong_baselines.py run 的逐图结果，按图配对 bootstrap（同一组重采样下标）：

  & $py scripts/precision_ablation.py --dataset visdrone_ft --a _fp32b1 --b _fp16b16

输出 results/<dataset>/precision_ablation.csv
"""

import argparse
import contextlib
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from glance_sahi.bootstrap import PreparedEval  # noqa: E402
from glance_sahi.resroute import MixedEval  # noqa: E402

import run_eval as R  # noqa: E402


def main(a):
    from pycocotools.coco import COCO

    R.set_dataset(a.dataset)
    A = pickle.loads((R.RES / f"strong{a.a}.pkl").read_bytes())
    B = pickle.loads((R.RES / f"strong{a.b}.pkl").read_bytes())
    ids = [im["id"] for im in A["images"]]
    assert ids == [im["id"] for im in B["images"]], "两次运行的图序列必须相同"
    common = [m for m in A["methods"] if m in B["methods"]]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(R.GT))
    prep, ms = {}, {}
    for tag, D in ((a.a, A), (a.b, B)):
        for m in common:
            dets = [d for i, arr in zip(ids, D["dets"][m]) for d in R.to_coco_dets(i, arr)]
            prep[f"{m}{tag}"] = PreparedEval(gt, dets, ids, R.DS["max_dets"])
            ms[f"{m}{tag}"] = 1000 * np.asarray(D["times"][m])
    mixed = MixedEval(prep)
    rng = np.random.default_rng(a.seed)
    arr = np.array(ids)
    boots = [rng.integers(0, len(ids), len(ids)) for _ in range(a.boot)]
    rows = []
    for m in common:
        ka, kb = f"{m}{a.a}", f"{m}{a.b}"
        pa, pb = mixed.ap({i: ka for i in ids}), mixed.ap({i: kb for i in ids})
        d_ap, d_aps, ratio = [], [], []
        for b in boots:
            x, y = mixed.ap({i: ka for i in ids}, arr[b]), mixed.ap({i: kb for i in ids}, arr[b])
            d_ap.append(100 * (y["AP"] - x["AP"]))
            d_aps.append(100 * (y["APs"] - x["APs"]))
            ratio.append(ms[kb][b].mean() / ms[ka][b].mean())
        rows.append(dict(method=m, AP_a=100 * pa["AP"], AP_b=100 * pb["AP"], dAP=100 * (pb["AP"] - pa["AP"]),
                         dAP_lo=np.percentile(d_ap, 2.5), dAP_hi=np.percentile(d_ap, 97.5),
                         dAPs=100 * (pb["APs"] - pa["APs"]), dAPs_lo=np.percentile(d_aps, 2.5),
                         dAPs_hi=np.percentile(d_aps, 97.5), ms_a=ms[ka].mean(), ms_b=ms[kb].mean(),
                         speedup=ms[ka].mean() / ms[kb].mean(), speedup_lo=1 / np.percentile(ratio, 97.5),
                         speedup_hi=1 / np.percentile(ratio, 2.5)))
    df = pd.DataFrame(rows)
    df.to_csv(R.RES / "precision_ablation.csv", index=False)
    print(f"a = {a.a}（{A['env'].get('gpu')}），b = {a.b}；Δ = b − a，{a.boot} 次按图配对 bootstrap")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="visdrone_ft")
    ap.add_argument("--a", default="_fp32b1")
    ap.add_argument("--b", default="_fp16b16")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
