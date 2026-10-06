"""
색·모양으로 블록과 판 찾기 (2026-10-06) — VLA 입력에 넣을 물체 위치

왜
  학생 모델(SmolVLA)은 사진만 보고 블록·판 위치를 1cm 단위로 찾지 못하고 '평소 자리'로 갔다
  (쥔 손 위치가 실제 블록 위치를 따라가는 기울기 b1c 0.1~0.2, ba1 0.5~0.8). 블록 장면은 물체마다 색이 다르므로,
  카메라 사진에서 색과 모양으로 위치를 계산해 모델에 같이 준다. 입력은 여전히 카메라 2대 + 로봇 상태 + 지시문 뿐이라
  실물에서도 같은 방법(색 기준만 다시 맞춤)으로 쓸 수 있다. 어느 블록을 어디에 놓을지는 모델이 지시문을 읽고 고른다.

무엇을 내나 (24개, 화면 좌표 0~1, 못 찾으면 -1)
  정면 카메라: 블록 4 + 판 4 의 무게중심 (u, v)  → 16
  손목 카메라: 블록 4 의 무게중심 (u, v)          → 8
  순서: OBJECTS (빨강·초록·파랑·노랑 블록, 보라·회색·주황·흰색 판)

어떻게
  HSV 색 범위로 가린 뒤 붙어 있는 덩어리 중 크기·모양(판은 정사각형에 가깝고 꽉 찬 것)이 맞는 가장 큰 것.
  회색 판은 로봇 팔(회색), 흰색 판은 밝은 책상과 헷갈리므로 모양 조건을 쓴다.
"""
import numpy as np
from scipy import ndimage

OBJECTS = ["red", "green", "blue", "yellow", "purple", "gray", "orange", "white"]
BLOCKS = OBJECTS[:4]
PADS = OBJECTS[4:]
N_FEAT = 2 * len(OBJECTS) + 2 * len(BLOCKS)       # 24


def hsv(img):
    """uint8 RGB (H,W,3) → H(0~360), S(0~1), V(0~1). PIL 의 C 변환을 써서 빠르게 (numpy 로 하면 사진 한 장 10ms)."""
    from PIL import Image
    x = np.asarray(Image.fromarray(np.ascontiguousarray(img)).convert("HSV"), dtype=np.float32)
    return x[..., 0] * (360.0 / 255.0), x[..., 1] / 255.0, x[..., 2] / 255.0


def masks(img):
    h, s, v = hsv(img)
    return {
        "red": ((h < 12) | (h > 345)) & (s > 0.55) & (v > 0.2),
        "green": (h > 95) & (h < 165) & (s > 0.45) & (v > 0.15),
        "blue": (h > 205) & (h < 250) & (s > 0.5) & (v > 0.15),
        "yellow": (h > 42) & (h < 65) & (s > 0.5) & (v > 0.35),
        "orange": (h > 15) & (h < 35) & (s > 0.6) & (v > 0.4),
        "purple": (h > 262) & (h < 305) & (s > 0.35) & (v > 0.2),
        "gray": (s < 0.08) & (v > 0.25) & (v < 0.55),     # 정면 사진에서 (87,85,86) V 0.34
        "white": (s < 0.06) & (v > 0.65),                 # 정면 사진에서 (193,193,193) V 0.76, 책상은 S 0.17~0.20
    }


def _pick(mask, square=False, min_area=4, max_area=None, max_v=None):
    """가장 큰 덩어리(판이면 정사각형에 가깝고 꽉 찬 것만)의 무게중심 (u, v). 없으면 None.
    max_area·max_v: 정면 사진 아래쪽의 넓은 회색 바닥(약 9000픽셀)을 판으로 잡지 않게 (판은 약 780픽셀, 화면 위쪽)."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    H, W = mask.shape
    best, best_a = None, 0
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        comp = lab[sl] == i
        a = int(comp.sum())
        if a < min_area or a <= best_a or (max_area is not None and a > max_area):
            continue
        if max_v is not None and (sl[0].start + sl[0].stop) / 2.0 / H > max_v:
            continue
        if square:
            hh, ww = comp.shape
            if not (0.4 < hh / max(ww, 1) < 2.5) or a / float(hh * ww) < 0.55:
                continue
        best, best_a = (sl, i), a
    if best is None:
        return None
    ys, xs = np.nonzero(lab == best[1])
    return (float(xs.mean()) / W, float(ys.mean()) / H)


def detect(agent, wrist):
    """정면·손목 사진 (uint8 RGB) → 24개 (못 찾은 것은 -1)."""
    out = np.full(N_FEAT, -1.0, dtype=np.float32)
    ma = masks(agent)
    for k, name in enumerate(OBJECTS):
        if name in PADS:
            p = _pick(ma[name], square=name in ("gray", "white"), min_area=6, max_area=2500, max_v=0.7)
        else:
            p = _pick(ma[name], min_area=3)
        if p is not None:
            out[2 * k:2 * k + 2] = p
    mw = masks(wrist)
    base = 2 * len(OBJECTS)
    for k, name in enumerate(BLOCKS):
        p = _pick(mw[name], min_area=20)
        if p is not None:
            out[base + 2 * k:base + 2 * k + 2] = p
    return out
