#!/usr/bin/env python
"""실물 위 카메라 사진으로 색 분포를 다시 맞춘다 (world_detect 와 같은 꼴, 2026-11-04).

시뮬 색 분포(piper_sim/worlds/detect_default.json)는 시뮬 그림에서 배운 것이라 실물 색과 다르다. 실물 물체·판을 책상에 놓고
사진을 몇 장(조명·자리를 바꿔 5장 이상) 찍으며, 사진마다 물체·판·책상·로봇을 클릭하고 이름을 고른다.
클릭한 자리에서 비슷한 색으로 퍼진 영역(반지름 60px 안)을 그 이름의 픽셀로 쓴다.
이름: obj:<색> (물체), pad:<색> (판), table, robot, floor — 색 이름은 설정 파일(piper_sim/worlds/default.json)의 영어 이름.
계산은 world_detect.fit 과 같다: 64³ 색 칸 → 사진마다 책상 밝기를 기준(ref_Y)으로 맞춘 Lab → 이름마다 평균·공분산(+4·I).
모양 분포(shapes)는 시뮬 것을 그대로 둔다. 결과 → real_piper/detect_real.json, config_real.json 의 detect_model.
  python real_piper/fit_colors.py --device 0
  (사진 창: 왼쪽 클릭 = 그 자리 이름 넣기 (터미널), n = 다음 사진, q = 끝)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--radius", type=int, default=60)
    p.add_argument("--tol", type=int, default=14)
    p.add_argument("--out", default=str(HERE / "detect_real.json"))
    a = p.parse_args()
    import os
    import cv2
    os.environ.setdefault("PIPER_NOPED", "1")
    os.environ.setdefault("PIPER_WORLD", str(ROOT / "piper_sim/worlds/default.json"))
    import piper_sim.world_detect as WD
    from real_piper import cameras
    cfg = json.loads((HERE / "config_real.json").read_text())
    cam = cameras.make(cfg["cameras"]["top"], "위")
    sim_model = json.loads(WD.MODEL.read_text())
    names = sorted(sim_model["classes"])
    print("이름 목록:", " ".join(f"{i}:{n}" for i, n in enumerate(names)))
    shots = []                                     # (사진 (모델 꼴), {이름: 픽셀 마스크})
    clicks = []

    def on_click(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
    while True:
        img = cam.read()                           # 모델 꼴 (위아래 뒤집힘)
        view = np.ascontiguousarray(img[::-1])     # 사람이 보는 방향
        masks = {}
        cv2.namedWindow("fit colors")
        cv2.setMouseCallback("fit colors", on_click)
        while True:
            show = cv2.cvtColor(view, cv2.COLOR_RGB2BGR).copy()
            for n, m in masks.items():
                show[m[::-1]] = (0.5 * show[m[::-1]] + [0, 128, 0]).astype(np.uint8)
            cv2.imshow("fit colors", show)
            k = cv2.waitKey(30) & 0xFF
            if clicks:
                x, y = clicks.pop(0)
                s = input(f"  ({x},{y}) 이름 번호 또는 이름: ").strip()
                name = names[int(s)] if s.isdigit() and int(s) < len(names) else s
                if name not in names:
                    print("  모르는 이름 — 건너뜀")
                    continue
                ff = np.zeros((view.shape[0] + 2, view.shape[1] + 2), np.uint8)
                cv2.floodFill(view.copy(), ff, (x, y), 0, (a.tol,) * 3, (a.tol,) * 3, cv2.FLOODFILL_MASK_ONLY | 4 | (255 << 8))
                m = ff[1:-1, 1:-1] > 0
                yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
                m &= (yy - y) ** 2 + (xx - x) ** 2 <= a.radius ** 2
                m_model = m[::-1]                  # 모델 꼴 방향으로
                masks[name] = masks.get(name, np.zeros_like(m_model)) | m_model
                print(f"  {name}: {int(m_model.sum())} 픽셀")
            if k in (ord("n"), ord("q")):
                break
        if masks:
            shots.append((img, masks))
        if k == ord("q"):
            break
    cv2.destroyAllWindows()
    if not shots or not any("table" in m for _, m in shots):
        sys.exit("책상(table)을 클릭한 사진이 없다 — 밝기 기준을 잴 수 없음")
    Yt = []
    for img, m in shots:
        code = WD._code(img)
        Yt.append(WD._table_Y(code, m["table"]) if "table" in m else np.nan)
    ref = float(np.nanmedian(Yt))
    model = {k: sim_model[k] for k in ("camera", "size", "flip", "shapes", "shape_norm") if k in sim_model}
    model.update({"ref_Y": ref, "classes": {}, "classes_raw": {}, "source": "real_piper/fit_colors.py (실물 사진)"})
    for space in ("classes", "classes_raw"):
        samples = {}
        for (img, m), y in zip(shots, Yt):
            code = WD._code(img)
            gain = np.clip(ref / y, 0.3, 3.0) if space == "classes" and np.isfinite(y) else 1.0
            for k, mk in m.items():
                c = code[mk]
                if len(c):
                    samples.setdefault(k, []).append(WD.lin2lab(WD.CUBE_LIN[c] * gain))
        for k, vs in samples.items():
            v = np.concatenate(vs)
            if len(v) < 50:
                print(f"  {k}: 픽셀 {len(v)}개뿐 — 더 클릭할 것")
            model[space][k] = {"mu": v.mean(0).tolist(), "cov": (np.cov(v.T) + np.eye(3) * 4.0).tolist(), "n": int(len(v))}
    Path(a.out).write_text(json.dumps(model, indent=1))
    d = json.loads((HERE / "config_real.json").read_text())
    d["detect_model"] = str(Path(a.out).resolve().relative_to(ROOT)) if Path(a.out).resolve().is_relative_to(ROOT) else a.out
    (HERE / "config_real.json").write_text(json.dumps(d, ensure_ascii=False, indent=2))
    print(f"이름 {len(model['classes'])}개 저장 (기준 책상 밝기 {ref:.3f}): {a.out}")
    missing = [n for n in names if n.startswith(("obj:", "pad:")) and n not in model["classes"]]
    if missing:
        print("  실물 사진에 없던 이름 (그 색 물체·판은 못 찾음):", " ".join(missing))


if __name__ == "__main__":
    main()
