"""10/24 오후 일지 그림: ① np2→np3 학습에서 바뀐 가중치 비율 (bfloat16 반올림) ② 혼자 과제별 np3 vs np5f ③ 실패 종류.
   python make_fig_np5f.py → outputs/media/np5f_bf16.png"""
import glob
import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
from safetensors import safe_open

font_manager.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID, C1, C2 = "#1f1f1f", "#6b6b6b", "#e6e6e6", "#9aa7b8", "#2e6fbf"

a = safe_open("outputs/piper_np2_model/merged/model.safetensors", "pt")
b = safe_open("outputs/piper_np3_model/merged/model.safetensors", "pt")
bins = [0, 1e-3, 3e-3, 1e-2, 3e-2, 1e9]
cnt = np.zeros((len(bins) - 1, 2))
for k in b.keys():
    if "lm_expert" not in k or b.get_slice(k).get_dtype() != "BF16":
        continue
    x, y = a.get_tensor(k).float().flatten(), b.get_tensor(k).float().flatten()
    ax_, ch = x.abs(), (x != y)
    for i in range(len(bins) - 1):
        m = (ax_ >= bins[i]) & (ax_ < bins[i + 1])
        cnt[i] += [ch[m].sum().item(), m.sum().item()]
frac = 100 * cnt[:, 0] / cnt[:, 1]
share = 100 * cnt[:, 1] / cnt[:, 1].sum()


def per_task(base):
    out = []
    for t in range(10):
        f = f"outputs/{base}_forget/A{t}_Bnone_grasp3_none_lat9.json"
        out.append(sum(e["a_success"] for e in json.load(open(f))["episodes"]))
    return np.array(out)


s3, s5 = per_task("piper_np3mch"), per_task("piper_np5fh")
ORDER = ["판 안(3cm 밖)", "판 밖에 놓음", "쥐었다 빠짐", "블록 위에서 못 쥠", "블록까지 못 감"]


def fails(base):
    txt = subprocess.run([".venv/bin/python", "fail_classify.py", f"outputs/{base}"], capture_output=True, text=True).stdout
    row = next(l for l in txt.splitlines() if l.startswith("| 혼자 | **합계**"))
    head = next(l for l in txt.splitlines() if l.startswith("| 구분"))
    cols = [c.strip() for c in head.split("|")[3:-1]]
    vals = [int(v.strip() or 0) for v in row.split("|")[3:-1]]
    d = dict(zip(cols, vals))
    return [d.get(k, 0) for k in ORDER]


f3, f5 = fails("piper_np3mch"), fails("piper_np5fh")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1.4, 1.2]})
lab = ["<0.001", "0.001~0.003", "0.003~0.01", "0.01~0.03", "≥0.03"]
ax[0].bar(range(5), frac, color=C2, width=0.6)
for i, (v, s) in enumerate(zip(frac, share)):
    ax[0].text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=10, color=INK)
    ax[0].text(i, -12, f"(값의 {s:.0f}%)", ha="center", fontsize=8, color=MUTED)
ax[0].set_xticks(range(5), lab, fontsize=8)
ax[0].set_ylim(-16, 110)
ax[0].set_ylabel("np2→np3 1만 스텝에 바뀐 값 (%)", color=MUTED)
ax[0].set_xlabel("가중치 크기 |w|", color=MUTED)
ax[0].set_title("① 동작 전문가 bfloat16 가중치: 큰 값은 안 바뀜", loc="left", fontsize=11)
x = np.arange(10)
ax[1].bar(x - 0.2, s3, 0.38, color=C1, label=f"np3 (전) {s3.sum()}/200")
ax[1].bar(x + 0.2, s5, 0.38, color=C2, label=f"np5f (float32 원본) {s5.sum()}/200")
ax[1].set_xticks(x, [f"T{t}" for t in range(10)])
ax[1].set_ylabel("성공 (20장면 중, 3cm)", color=MUTED)
ax[1].set_ylim(0, 14.5)
ax[1].legend(frameon=False, fontsize=9, loc="upper left")
ax[1].set_title("② VLA 단독 혼자 과제 (보조 없음, 지연 9)", loc="left", fontsize=11)
y = np.arange(len(ORDER))
ax[2].barh(y + 0.2, f3, 0.38, color=C1, label="np3")
ax[2].barh(y - 0.2, f5, 0.38, color=C2, label="np5f")
ax[2].set_yticks(y, ORDER, fontsize=9)
ax[2].invert_yaxis()
ax[2].set_xlabel("실패 장면 수 (200장면 중)", color=MUTED)
ax[2].legend(frameon=False, fontsize=9, loc="lower right")
ax[2].set_title("③ 실패 종류: 쥐기는 줄고 놓기는 그대로", loc="left", fontsize=11)
for a_ in ax:
    a_.grid(axis="y" if a_ is not ax[2] else "x", color=GRID, lw=0.8)
    a_.set_axisbelow(True)
    for s in ("top", "right"):
        a_.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("outputs/media/np5f_bf16.png", dpi=140)
print("outputs/media/np5f_bf16.png", frac.round(1), s3.sum(), s5.sum(), f3, f5)
