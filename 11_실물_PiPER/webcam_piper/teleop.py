"""웹캠 하나로 PiPER 원격조종 + 시범 기록 (2026-11-15).

My_portfolio 2026/05 웹캠 조종(SO-ARM101, 팔 관절 각도를 그대로 옮김)을 PiPER 용으로 바꾼 것.
PiPER 는 팔 길이가 사람과 달라 관절 각도를 옮기면 집게가 엉뚱한 곳에 간다. 그래서 여기서는
**사람 엄지·검지 끝 가운데(집기점)가 움직인 만큼 집게 끝을 똑같이 움직인다** (mapping.py). 관절은 역기구학이 푼다.

실행
  python teleop.py                     # 팔 없이 화면 미리보기 (처음엔 이것부터)
  python teleop.py --robot agxbus      # 실물 PiPER 가 이 컴퓨터에 붙어 있을 때 (agx-bus, CAN can0)
  python teleop.py --send 192.168.0.10 # 팔은 연구실 PC 에: 손 측정만 보내고, 연구실 PC 에서 robot_server.py 가 팔을 움직임
화면 글자는 영어다 (OpenCV 가 한글을 못 그림). 키:
  SPACE 따라가기 켜기/끄기   R 기록 시작/끝   X 마지막 기록 지우기   G 집게 따라가기   Y 손 돌림 따라가기
  [ 집게 약하게 ] 세게 (5 mm)   --grip pinch 면 엄지·검지로 집음 (기본은 손 전체)   C 종이 모서리 다시   S 손 시작 자세 다시   Q 또는 ESC 끝
  P 위치 확인: 다섯 손가락 끝을 모아 종이 모서리나 + 에 대고 누르면, 가장 가까운 점과 몇 cm 다른지 띄우고 checks.jsonl 에 남김
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np

from geometry import CORNER_NAMES, CORNERS_TABLE, A4_D, A4_W, HandMeasure, TableCalib, flat_fit, scale_for_flat
from hand import HandTracker, OneEuro, draw_hand
from mapping import Mapper
from net import PORT, Link
from panel import DISP_H, robot_panel
from piper_kin import HOME_Q, PiperKin
from recorder import CamGrabber, Recorder, jpeg, open_camera
from robots import make_arm

HERE = os.path.dirname(os.path.abspath(__file__))
BACKSPACE = (8, 127)                                       # 127 = 맥의 delete 키
# P 로 위치를 확인할 점: 종이 네 모서리와 + (사람 책상 좌표)
CHECK_PTS = [(n.split(" ", 1)[1], CORNERS_TABLE[i]) for i, n in enumerate(CORNER_NAMES)] + [("center", np.zeros(3))]


def check_point(meas, scale, path):
    """P (11/16): 다섯 손가락 끝을 모아 종이 모서리나 + 에 댄 채 누른다. 가장 가까운 점과의 차이를 돌려주고 path 에 한 줄 더한다.
    손가락 끝을 모으면 집기점이 그 끝에 오므로, 차이가 곧 위치 오차다 (높이는 손가락 끝 살 두께만큼, 1 cm 안팎이 맞음)."""
    if not meas:
        return "P: hand not found"
    p = np.asarray(meas["pinch"], float)
    name, ref = min(CHECK_PTS, key=lambda c: np.linalg.norm(p[:2] - c[1][:2]))
    d = (p - ref) * 100
    row = {"t": round(time.time(), 1), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "point": name, "scale": round(scale, 4),
           "hand_cm": np.round(p * 100, 2).tolist(), "off_cm": np.round(d, 2).tolist(), "gap_mm": round(meas["gap"] * 1000, 1)}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[확인] {name}: x {d[0]:+.1f} y {d[1]:+.1f} cm 차이, 높이 {p[2] * 100:.1f} cm, 손가락 사이 {meas['gap'] * 1000:.0f} mm")
    return f"P: {name} off by x {d[0]:+.1f} y {d[1]:+.1f} cm, z {p[2] * 100:.1f} cm (saved)"


def _load_guide(width=400):
    """보정 2/2 화면에 띄울 시작 자세 그림 (img/guide_start.png). 없으면 글만."""
    g = cv2.imread(os.path.join(HERE, "img", "guide_start.png"))
    if g is None:
        return None
    return cv2.resize(g, (width, int(g.shape[0] * width / g.shape[1])), interpolation=cv2.INTER_AREA)


GUIDE = _load_guide()


def parse():
    ap = argparse.ArgumentParser(description="웹캠 PiPER 원격조종")
    ap.add_argument("--cam", type=int, default=0, help="웹캠 번호 (노트북 내장 = 0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hfov", type=float, default=None, help="카메라 가로 시야각(도). 모르면 비워 두면 종이로 추정")
    ap.add_argument("--hand", choices=["right", "left"], default="right")
    ap.add_argument("--robot", choices=["preview", "agxbus", "sdk"], default="preview")
    ap.add_argument("--send", default=None, metavar="주소[:포트]",
                    help=f"팔이 붙은 연구실 PC 주소. 주면 손 측정만 보내고 팔은 그쪽 robot_server.py 가 움직임 (포트 기본 {PORT})")
    ap.add_argument("--can", default="can0")
    ap.add_argument("--speed", type=int, default=30, help="실물 팔 속도 %%")
    ap.add_argument("--x0", type=float, default=0.30, help="로봇 시작 표시: 밑동에서 앞으로 (m)")
    ap.add_argument("--y0", type=float, default=0.0, help="로봇 시작 표시: 밑동에서 왼쪽으로 (m)")
    ap.add_argument("--gain", type=float, default=1.0, help="사람 1 cm → 로봇 몇 cm (같은 물체면 1.0)")
    ap.add_argument("--max-speed", type=float, default=0.12, help="집게 끝 최고 속도 (m/s)")
    ap.add_argument("--grip", choices=["whole", "pinch"], default="whole",
                    help="whole: 엄지 ↔ 네 손가락으로 손 전체로 쥠 (기본), pinch: 엄지 ↔ 검지로 집음")
    ap.add_argument("--squeeze", type=float, default=None,
                    help="집게 폭 = 엄지와 맞은편 손가락 사이 − 이 값 (m). 기본 whole 0.025, pinch 0.02")
    ap.add_argument("--calib", default=os.path.join(HERE, "calib.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--checks", default=os.path.join(HERE, "checks.jsonl"), help="P 로 잰 위치 확인을 남길 파일")
    ap.add_argument("--wrist-cam", type=int, default=None, help="기록할 로봇 손목 카메라 번호 (선택)")
    ap.add_argument("--front-cam", type=int, default=None, help="기록할 로봇 앞/위 카메라 번호 (선택)")
    return ap.parse_args()


class Smoother:
    """손 측정 거르기. 11/16 검토에서 고침: 손을 다시 찾은 첫 장면이 엉뚱하면 그 값에 계속 묶여 있던 문제.
    - 새로 잡을 때: 3장이 서로 3 cm 안에서 맞아야 받아들인다 (그 전에는 '손 없음').
    - 잡은 뒤 한 장면에 15 cm 넘게 튀면 버린다. 다만 튄 쪽 값이 5장 연달아 서로 3 cm 안이면 손이 정말 옮겨 간 것으로 보고 다시 잡는다."""
    ACQUIRE_N, REJUMP_N, AGREE = 3, 5, 0.03

    def __init__(self):
        self.pinch, self.yaw, self.gap = OneEuro(1.5, 0.5), OneEuro(1.0, 0.3), OneEuro(1.5, 0.3)
        self.last = None
        self.yaw_unwrapped = None
        self.lost_since = None
        self.pending = []

    def reset(self):
        for f in (self.pinch, self.yaw, self.gap):
            f.reset()
        self.last = self.yaw_unwrapped = None
        self.pending = []

    def _agree(self, n):
        if len(self.pending) < n:
            return False
        ref = self.pending[-1]["pinch"]
        return all(np.linalg.norm(p["pinch"] - ref) < self.AGREE for p in self.pending[-n:])

    def __call__(self, t, m):
        if m is None:
            if self.lost_since is None:
                self.lost_since = t
            if t - self.lost_since > 0.5:
                self.reset()
            self.pending = []
            return None
        self.lost_since = None
        if self.last is None:
            self.pending = (self.pending + [m])[-self.ACQUIRE_N:]
            if not self._agree(self.ACQUIRE_N):
                return None
            self.pending = []
        elif np.linalg.norm(m["pinch"] - self.last["pinch"]) > 0.15:
            self.pending = (self.pending + [m])[-self.REJUMP_N:]
            if not self._agree(self.REJUMP_N):
                return self.last                               # 한 장면에 15 cm 넘게 튀면 잘못 찾은 것으로 본다
            self.reset()                                       # 손이 정말 옮겨 감: 새로 잡는다
        else:
            self.pending = []
        y = m["yaw"]
        if self.yaw_unwrapped is not None:
            y = self.yaw_unwrapped + math.remainder(y - self.yaw_unwrapped, 2 * math.pi)
        self.yaw_unwrapped = y
        out = {"pinch": self.pinch(t, m["pinch"]), "yaw": float(self.yaw(t, y)), "gap": float(self.gap(t, m["gap"]))}
        self.last = out
        return out


def main():
    a = parse()
    kin = PiperKin()
    link = Link(a.send) if a.send else None
    arm = make_arm("preview" if link else a.robot, a.can, a.speed)
    if link:
        print(f"[연결] 손 측정을 {link.addr[0]}:{link.addr[1]} 로 보냄. 연구실 PC 에서 robot_server.py 를 켜 두세요.")
    if arm.is_real:
        print("\n[안전] 실물 팔입니다. 팔 둘레를 비우고, 전원 스위치 옆에 사람이 있어야 합니다.")
        print("       SPACE 를 눌러야 따라가기 시작하고, 손을 1초 넘게 놓치면 멈춥니다. 속도 %d%%" % a.speed)
        input("       준비됐으면 Enter (취소 Ctrl+C) ")
    if a.squeeze is None:
        a.squeeze = 0.025 if a.grip == "whole" else 0.02
    mapper = Mapper(kin, a.x0, a.y0, a.gain, max_speed=a.max_speed, squeeze=a.squeeze,
                    yaw_gap_min=0.05 if a.grip == "whole" else 0.025,
                    yaw_pref=math.radians(-60 if a.hand == "right" else 60))
    mapper.sync_to(arm.joints())

    cap, frame = open_camera(a.cam, a.width, a.height)
    if cap is None:
        raise SystemExit(f"웹캠 {a.cam} 을 못 엶 (다른 프로그램이 쓰는 중인지, 카메라 권한, --cam 1 을 보세요)")
    H, W = frame.shape[:2]
    extra = {} if link else {k: CamGrabber(i) for k, i in (("wrist", a.wrist_cam), ("front", a.front_cam))
                             if i is not None}

    tracker = HandTracker(a.hand)
    smooth = Smoother()
    calib, measure, mode, clicks = None, None, "corners", []
    if os.path.exists(a.calib):
        with open(a.calib, encoding="utf-8") as f:
            saved = json.load(f)
        if saved["table"]["w"] == W and saved["table"]["h"] == H:
            calib = TableCalib.from_dict(saved["table"])
            measure = HandMeasure(calib, saved["scale"], a.grip, saved.get("mirror", False))
            if saved.get("label") and saved.get("hand", "right") == a.hand:
                tracker.want_label = saved["label"]
            mode = "run"
            print(f"[보정] {a.calib} 불러옴 (거리 배율 {saved['scale']:.3f})")
    rec = None if link else Recorder(a.out, {
        "args": vars(a), "robot_start": mapper.robot_start.tolist(),
        "note": "사람 책상 좌표: 원점 A4 가운데, x 앞(노트북 쪽), y 사람 왼쪽, z 위. 로봇 좌표: 밑동 (0,0,0.90)"})
    # 연구실 PC 로 보낼 때: 기록은 그쪽에서 한다. 여기서는 바라는 상태만 보내고, 그쪽이 알려 준 실제 상태를 그린다.
    rec_req, disc_n, seen_msg_n = False, 0, None
    mismatch = {"follow": None, "rec": None}
    scale_disp = DISP_H / H
    win = "webcam_piper"
    cv2.namedWindow(win)

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and mode == "corners" and len(clicks) < 4 and x < W * scale_disp:
            clicks.append((W - 1 - x / scale_disp, y / scale_disp))      # 화면은 거울로 보여 주므로 되돌린다
    cv2.setMouseCallback(win, on_mouse)

    follow, collecting, samples, msg, msg_t = False, False, [], "", 0.0
    rec_on = (lambda: rec.on) if rec else (lambda: bool(link.status and link.status.get("rec")))
    t_prev = time.monotonic()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.monotonic()
        dt, t_prev = t - t_prev, t
        fps = 0.9 * fps + 0.1 / max(dt, 1e-3)
        det = tracker.detect(frame, t * 1000)
        raw = measure(det["world"], det["img_px"]) if (measure and det) else None
        meas = smooth(t, raw)
        view = frame.copy()

        # 화면에 그릴 것 (원본 좌표에 먼저 그린 뒤 거울로 뒤집는다)
        if det:
            draw_hand(view, det["img_px"])
        if calib:
            grid = calib.project(np.array([[x, y, 0.0] for x in np.linspace(-A4_D / 2, A4_D / 2, 4)
                                           for y in np.linspace(-A4_W / 2, A4_W / 2, 5)]))
            for p in grid:
                cv2.circle(view, (int(p[0]), int(p[1])), 3, (255, 200, 0), -1)
            c = calib.project(CORNERS_TABLE).astype(int)
            cv2.polylines(view, [c], True, (255, 200, 0), 2)
            cv2.drawMarker(view, tuple(calib.project([[0, 0, 0]])[0].astype(int)), (0, 200, 255),
                           cv2.MARKER_CROSS, 24, 2)
        for i, p in enumerate(clicks):
            cv2.circle(view, (int(p[0]), int(p[1])), 7, (0, 0, 255), -1)
        disp = cv2.resize(cv2.flip(view, 1), (int(W * scale_disp), DISP_H))

        lines = []
        if mode == "corners":
            if len(clicks) < 4:
                lines = ["CALIBRATION 1/2: click the paper corners (seen from YOUR seat)",
                         f"next: {CORNER_NAMES[len(clicks)]}   (BACKSPACE: undo)"]
            else:
                try:
                    calib = TableCalib(clicks, W, H, a.hfov)
                    mode = "corners_check"
                except RuntimeError as e:
                    print(e)
                    clicks.clear()
        if mode == "corners_check":
            lines = ["Check: blue dots must sit on the paper.  ENTER = ok,  BACKSPACE = redo",
                     f"focal {calib.f:.0f}px ({calib.focal_source}), corner error {calib.reproj_px:.1f}px"]
        if mode == "hand":
            lines = [f"CALIBRATION 2/2: lay your {a.hand} hand FLAT on the paper,",
                     "palm center on the cross, fingers toward the laptop. Hold still, press S",
                     "(keep the other hand out of view)"]
            if collecting:
                lines.append(f"measuring... {len(samples)}/20  (keep still)")
                if det is not None:
                    samples.append(det)
                if len(samples) >= 20:
                    collecting = False
                    # MediaPipe 손 3D 가 화면과 같은 손인지 거울상인지: 펴서 올려놓은 21점이 종이 높이에 더 잘 맞는 쪽 (다수결)
                    votes = []
                    for d in samples:
                        r0, r1 = HandMeasure(calib, 1.0, a.grip, False)(d["world"], d["img_px"]), \
                            HandMeasure(calib, 1.0, a.grip, True)(d["world"], d["img_px"])
                        if r0 and r1:
                            votes.append(flat_fit(calib, r1["lm_cam_raw"])[1] < flat_fit(calib, r0["lm_cam_raw"])[1])
                    mirror = bool(votes) and sum(votes) > 0.7 * len(votes)
                    # 이 손을 MediaPipe 가 뭐라고 부르는지 (두 손이 다 보일 때 이 표시로 고름)
                    labels = [d["label"] for d in samples]
                    label = max(set(labels), key=labels.count)
                    if labels.count(label) >= 0.8 * len(labels):
                        tracker.want_label = label
                    m1 = HandMeasure(calib, 1.0, a.grip, mirror)
                    raws = [m1(d["world"], d["img_px"]) for d in samples]
                    raws = [r for r in raws if r]
                    s = scale_for_flat(calib, np.median([r["lm_cam_raw"] for r in raws], axis=0)) if raws else None
                    if s is None:
                        msg, msg_t = "hand calibration failed - try again (S)", t
                    else:
                        measure = HandMeasure(calib, s, a.grip, mirror)
                        rs = [r for r in (measure(d["world"], d["img_px"]) for d in samples) if r]
                        palm = np.median([r["palm"] for r in rs], axis=0)       # 손바닥 가운데가 + 에서 얼마나
                        with open(a.calib, "w", encoding="utf-8") as f:
                            json.dump({"table": calib.to_dict(), "scale": s, "pose": "flat hand", "mirror": mirror,
                                       "mirror_votes": [int(sum(votes)), len(votes)],
                                       "palm_off_m": palm.tolist(), "hand": a.hand, "label": tracker.want_label,
                                       "label_votes": [labels.count(label), len(labels), label]}, f, indent=1)
                        msg = (f"saved. scale {s:.3f}, palm center off the cross by x {palm[0]*100:+.1f}"
                               f" y {palm[1]*100:+.1f} cm (near 0 is good), 3D {'MIRRORED' if mirror else 'ok'}"
                               f" {len(votes) - sum(votes)}/{len(votes)}")
                        print(f"[보정] 배율 {s:.3f}, 손 3D {'거울상으로 판단해 뒤집어 씀' if mirror else '화면과 같은 손'}"
                              f" (거울상 표 {sum(votes)}/{len(votes)}), MediaPipe 표시 '{label}' {labels.count(label)}/{len(labels)}")
                        msg_t = t
                        smooth.reset()
                        mode = "run"
        if mode == "run":
            if not link:
                q_cmd, width = mapper.step(meas if follow else None, dt, follow)
            if follow:
                if smooth.lost_since is not None and t - smooth.lost_since > 1.0:
                    follow = False
                    arm.hold()
                    msg, msg_t = "hand lost > 1 s: follow stopped (SPACE to resume)", t
                elif not link:
                    arm.send(q_cmd, width)
            if meas:
                p = meas["pinch"]
                lines = [f"hand x {p[0]*100:+5.1f}  y {p[1]*100:+5.1f}  z {(p[2])*100:4.1f} cm   "
                         f"grip {meas['gap']*1000:3.0f} mm   turn {math.degrees(math.remainder(meas['yaw'] + math.pi / 2, math.pi)):+4.0f} deg"]
            else:
                lines = ["hand not found"]
            lines.append(f"{fps:4.1f} fps   SPACE follow  R rec  G grip:{'on' if mapper.follow_grip else 'off'}"
                         f"  Y yaw:{'on' if mapper.follow_yaw else 'off'}  [ ] squeeze {mapper.squeeze*1000:.0f}mm")
        if mode == "hand" and GUIDE is not None and GUIDE.shape[0] + 16 <= disp.shape[0] \
                and GUIDE.shape[1] + 16 <= disp.shape[1]:          # 시작 자세 그림을 왼쪽 아래에 (화면이 작으면 생략)
            gh, gw = GUIDE.shape[:2]
            disp[DISP_H - gh - 8:DISP_H - 8, 8:8 + gw] = GUIDE
        if msg and t - msg_t < 6:
            lines.append(msg)
        for i, s_ in enumerate(lines):
            cv2.putText(disp, s_, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(disp, s_, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if link:
            # 연구실 PC 로 손 측정 보내기 (보정 중에는 손 없음·따라가기 꺼짐으로 보냄)
            run = mode == "run"
            hand_out = ([round(float(v), 5) for v in meas["pinch"]] + [round(meas["yaw"], 4),
                                                                       round(meas["gap"], 4)]) if (run and meas) else None
            link.send(hand=hand_out, follow=int(run and follow), rec=int(run and rec_req),
                      grip=int(mapper.follow_grip), yawf=int(mapper.follow_yaw), sq=round(mapper.squeeze, 4),
                      ygm=mapper.yaw_gap_min, ypref=round(mapper.yaw_pref, 4), disc=disc_n, bye=0)
            st = link.poll()
            if st and st.get("msg_n") != seen_msg_n:
                if seen_msg_n is not None and st.get("msg"):
                    msg, msg_t = "robot PC: " + st["msg"], t
                seen_msg_n = st.get("msg_n")
            # 연구실 PC 가 따라가기·기록을 껐는데 여기는 켜져 있으면 (통신 끊김·멈춤) 여기도 맞춰 끈다
            if link.connected and st:
                for k, want in (("follow", follow), ("rec", rec_req)):
                    if bool(st.get(k)) != want:
                        mismatch[k] = mismatch[k] or t
                        if t - mismatch[k] > (0.7 if k == "follow" else 1.5):
                            if k == "follow":
                                follow = False
                            else:
                                rec_req = False
                            mismatch[k] = None
                    else:
                        mismatch[k] = None
            if st:
                q_now = np.array(st["q"], float)
                w_now = float(st["w"]) if st.get("w") is not None else float("nan")
                tgt, fol, x0, y0 = np.array(st["tgt"], float), bool(st["follow"]), st["x0"], st["y0"]
            else:
                q_now, w_now, tgt, fol, x0, y0 = HOME_Q, float("nan"), None, False, a.x0, a.y0
            if link.connected:
                extra_txt = [f"robot PC {link.addr[0]}", f"ping {link.rtt*1000:.0f} ms" if link.rtt else "",
                             ("!REAL ARM" if st.get("real") else "preview arm") + (" PAUSED" if st.get("paused") else "")]
            else:
                extra_txt = [f"!no answer from", f"!{link.addr[0]}:{link.addr[1]}", "!start robot_server.py"]
            panel = robot_panel(kin, q_now, tgt, x0, y0, w_now, fol, rec_on(), extra_txt)
        else:
            q_now = arm.joints()
            w_now = arm.gripper()
            panel = robot_panel(kin, q_now, mapper.cur, a.x0, a.y0, w_now if np.isfinite(w_now) else mapper.width,
                                follow, rec.on)
        cv2.imshow(win, np.hstack([disp, panel]))

        if rec and rec.on:
            imgs = {k: c.latest() for k, c in extra.items()}
            rec.add(t=t, q=q_now, grip=w_now, q_cmd=mapper.q_cmd.copy(), grip_cmd=mapper.width,
                    hand=(meas["pinch"] if meas else np.full(3, np.nan)),
                    hand_yaw=(meas["yaw"] if meas else np.nan),
                    hand_gap=(meas["gap"] if meas else np.nan), target=mapper.cur.copy(), tracked=meas is not None,
                    follow=follow, human_jpg=jpeg(frame, 480),
                    **{f"{k}_jpg": jpeg(v, 640) for k, v in imgs.items()})

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if mode == "corners" and key in BACKSPACE and clicks:
            clicks.pop()
        elif mode == "corners_check":
            if key in (13, 10):
                mode, collecting, samples = "hand", False, []
                measure = HandMeasure(calib, 1.0, a.grip)
            elif key in BACKSPACE:
                clicks.clear()
                calib, mode = None, "corners"
        elif mode == "hand" and key == ord("s") and not collecting:
            collecting, samples = True, []
        elif mode == "run":
            if key == ord(" "):
                follow = not follow
                if follow:
                    if not link:
                        mapper.sync_to(arm.joints())
                    smooth.reset()
                else:
                    arm.hold()
            elif key == ord("r"):
                if link:
                    rec_req = not rec_req                    # 연구실 PC 가 기록 시작/저장
                elif rec.on:
                    r = rec.stop()
                    msg, msg_t = (f"saved {os.path.basename(r[0])} ({r[1]} frames)" if r else "nothing recorded"), t
                else:
                    rec.start()
            elif key == ord("x"):
                if link:
                    disc_n += 1
                else:
                    p = rec.discard_last()
                    msg, msg_t = (f"deleted {os.path.basename(p)}" if p else "no episode to delete"), t
            elif key == ord("g"):
                mapper.follow_grip = not mapper.follow_grip
            elif key == ord("y"):
                mapper.follow_yaw = not mapper.follow_yaw
            elif key == ord("["):
                mapper.squeeze = max(0.0, mapper.squeeze - 0.005)
            elif key == ord("]"):
                mapper.squeeze = min(0.05, mapper.squeeze + 0.005)
            elif key == ord("c"):
                follow = False
                arm.hold()
                clicks.clear()
                calib, measure, mode = None, None, "corners"
            elif key == ord("s"):
                follow = False
                arm.hold()
                mode, collecting, samples = "hand", False, []
                measure = HandMeasure(calib, 1.0, a.grip)
            elif key == ord("p"):
                msg, msg_t = check_point(meas, measure.scale, a.checks), t

    if link:
        for _ in range(5):                                   # 연구실 PC 에 "끝" 을 알림 (팔 멈춤·기록 저장)
            link.send(hand=None, follow=0, rec=0, grip=int(mapper.follow_grip), yawf=int(mapper.follow_yaw),
                      sq=round(mapper.squeeze, 4), disc=disc_n, bye=1)
            time.sleep(0.02)
        link.close()
    if rec and rec.on:
        rec.stop()
    arm.hold()
    arm.close()
    tracker.close()
    cap.release()
    for c in extra.values():
        c.close()
    cv2.destroyAllWindows()
    if rec:
        print(f"[끝] 기록 {len(rec.saved)}개" + (f" → {rec.dir}" if rec.saved else ""))
    else:
        print("[끝] 기록은 연구실 PC 의 data/ 폴더에 있습니다")


if __name__ == "__main__":
    main()
