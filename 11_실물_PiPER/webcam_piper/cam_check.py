"""카메라 자리 맞추기 미리보기 (2026-11-18, ABKO APC930 두 대).

연결된 카메라를 나란히 띄우고, A4 종이를 찾으면 파란 테두리와 카메라 자리(종이 + 기준 앞·왼쪽·위 cm)를 보여 준다.
화면을 보면서 카메라를 숙여 종이가 화면 가운데 아래쪽에 오게 맞추면 된다. 끝은 Q.

  .venv/bin/python cam_check.py              # 연결된 카메라를 모두 (리눅스: /dev/video0, 2, ...)
  .venv/bin/python cam_check.py --cams 0 2
"""
import argparse
import glob
import os
import threading
import time

import cv2
import numpy as np

from geometry import CORNERS_TABLE
from paper import find_paper


def capture_nodes():
    """영상을 내는 /dev/video 번호만 (카메라 하나에 노드가 둘씩 생기는데 둘째는 메타데이터)."""
    out = []
    for d in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            if open(os.path.join(d, "index")).read().strip() == "0":
                out.append(int(os.path.basename(d)[5:]))
        except OSError:
            pass
    return out


class Cam:
    def __init__(self, idx, w, h):
        self.idx = idx
        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if os.name != "nt" else cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.frame, self.n, self.fps = None, 0, 0.0
        self.paper, self.lock, self.run = None, threading.Lock(), self.cap.isOpened()
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._find, daemon=True).start()

    def _read(self):
        t0, k = time.monotonic(), 0
        while self.run:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame, self.n = f, self.n + 1
            k += 1
            if time.monotonic() - t0 > 1.0:
                self.fps, t0, k = k / (time.monotonic() - t0), time.monotonic(), 0

    def _find(self):
        while self.run:
            with self.lock:
                f = None if self.frame is None else self.frame.copy()
            self.paper = None if f is None else find_paper(f)
            time.sleep(0.2)

    def close(self):
        self.run = False
        time.sleep(0.1)
        self.cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=int, nargs="*", default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    a = ap.parse_args()
    idxs = a.cams if a.cams else capture_nodes()
    if not idxs:
        raise SystemExit("카메라가 없습니다 (USB 를 확인하세요)")
    cams = [Cam(i, a.width, a.height) for i in idxs]
    print("카메라:", ", ".join(f"{c.idx}{'' if c.run else ' (못 엶)'}" for c in cams), " — 창에서 Q 로 끝")
    cv2.namedWindow("cam_check")
    last, t_scan = {}, time.monotonic()
    while True:
        if not a.cams and time.monotonic() - t_scan > 2.0:          # 새로 꽂은 카메라를 2초마다 찾아 더함
            t_scan = time.monotonic()
            have = {c.idx for c in cams}
            for i in capture_nodes():
                if i not in have:
                    cams.append(Cam(i, a.width, a.height))
                    print(f"[cam {i}] 새로 연결됨")
            for c in [c for c in cams if not os.path.exists(f"/dev/video{c.idx}")]:
                print(f"[cam {c.idx}] 연결 끊김")
                c.close()
                cams.remove(c)
            if not cams:
                time.sleep(0.5)
                continue
        tiles = []
        for c in cams:
            with c.lock:
                f = None if c.frame is None else c.frame.copy()
            if f is None:
                f = np.zeros((a.height, a.width, 3), np.uint8)
            h, w = f.shape[:2]
            cv2.line(f, (w // 2, 0), (w // 2, h), (80, 80, 80), 1)
            cv2.line(f, (0, h // 2), (w, h // 2), (80, 80, 80), 1)
            p = c.paper
            if p is not None:
                corners, cal = p
                cv2.polylines(f, [np.round(corners).astype(np.int32)], True, (255, 200, 0), 5)
                cv2.drawMarker(f, tuple(cal.project([[0, 0, 0]])[0].astype(int)), (0, 200, 255), cv2.MARKER_CROSS, 50, 5)
                pos = np.round(cal.cam_pos * 100).astype(int)
                txt = [f"cam {c.idx}: PAPER OK", f"camera at fwd {pos[0]} left {pos[1]} up {pos[2]} cm, f {cal.f:.0f}px"]
                col = (0, 255, 0)
                if last.get(c.idx) is None:
                    print(f"[cam {c.idx}] 종이 찾음: 카메라 자리 앞 {pos[0]} 왼쪽 {pos[1]} 위 {pos[2]} cm, 초점 {cal.f:.0f}px")
            else:
                txt = [f"cam {c.idx}: no paper - tilt the camera so the whole A4 is in view",
                       "keep hands and things off the paper edges"]
                col = (0, 0, 255)
                if last.get(c.idx) is not None:
                    print(f"[cam {c.idx}] 종이를 놓침")
            last[c.idx] = p
            tile = cv2.resize(f, (640, int(640 * h / w)))
            for i, s in enumerate(txt + [f"{c.fps:.0f} fps, {w}x{h}"]):
                cv2.putText(tile, s, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(tile, s, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col if i == 0 else (255, 255, 255), 1)
            tiles.append(tile)
        hh = max(t.shape[0] for t in tiles)
        tiles = [np.vstack([t, np.zeros((hh - t.shape[0], t.shape[1], 3), np.uint8)]) for t in tiles]
        cv2.imshow("cam_check", np.hstack(tiles))
        if (cv2.waitKey(30) & 0xFF) in (ord("q"), 27):
            break
    for c in cams:
        c.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
