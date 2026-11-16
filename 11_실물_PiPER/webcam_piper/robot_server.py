"""연구실 PC 쪽: 노트북이 보낸 손 위치로 PiPER 를 움직이고 기록한다 (2026-11-15).

노트북은 카메라로 손만 재서 숫자(집기점 위치·손 돌림·손가락 사이)를 보낸다 (teleop.py --send).
이 프로그램은 그 숫자를 mapping.py 로 집게 끝 목표로 바꾸고, 역기구학으로 관절을 풀어 팔에 보낸다.
mediapipe·mujoco 가 필요 없다 (numpy, opencv + 실물이면 agx-bus 나 piper_sdk). 찬혁님 piper-teleop .venv 로 돈다.

실행
  python robot_server.py                          # 팔 없이 가상 팔로 연결만 시험
  python robot_server.py --robot agxbus --speed 20  # 실물 PiPER

안전
  - 실물이면 시작할 때 터미널에서 Enter 를 받는다.
  - 노트북에서 따라가기를 새로 켜야 움직인다 (이 프로그램을 켜기 전에 켜져 있던 것은 무시, 노트북 쪽이 저절로 꺼짐).
  - 노트북 패킷이 0.5초 넘게 안 오면 그 자리에 멈추고 따라가기를 끈다. 다시 움직이려면 노트북에서 SPACE.
  - 집게 끝 속도·작업 상자·관절 속도 제한은 노트북 혼자 쓸 때와 같다 (mapping.py).
  - 창이 있으면 이 PC 에서도 SPACE 로 멈춤/풀기, Q 로 끝.
"""
import argparse
import os
import time

import cv2
import numpy as np

from mapping import Mapper
from net import PORT, Server, my_ips
from panel import robot_panel
from piper_kin import PiperKin
from recorder import CamGrabber, Recorder, jpeg
from robots import make_arm

HERE = os.path.dirname(os.path.abspath(__file__))


def parse():
    ap = argparse.ArgumentParser(description="노트북 손 측정 → PiPER")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--robot", choices=["preview", "agxbus", "sdk"], default="preview")
    ap.add_argument("--can", default="can0")
    ap.add_argument("--speed", type=int, default=30, help="실물 팔 속도 %%")
    ap.add_argument("--x0", type=float, default=0.30, help="로봇 종이 가운데: 밑동에서 앞으로 (m)")
    ap.add_argument("--y0", type=float, default=0.0, help="로봇 종이 가운데: 밑동에서 왼쪽으로 (m)")
    ap.add_argument("--gain", type=float, default=1.0, help="사람 1 cm → 로봇 몇 cm")
    ap.add_argument("--max-speed", type=float, default=0.12, help="집게 끝 최고 속도 (m/s)")
    ap.add_argument("--min-z", type=float, default=0.006,
                    help="집게 끝을 책상 위 이 높이(m) 아래로 내리지 않음. 실물 첫 시험은 0.10 처럼 높게 (11/16)")
    ap.add_argument("--hz", type=float, default=30.0, help="팔에 명령 보내고 기록하는 박자")
    ap.add_argument("--timeout", type=float, default=0.5, help="노트북 패킷이 이만큼(초) 안 오면 멈춤")
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--wrist-cam", type=int, default=None, help="기록할 손목 카메라 번호 (선택)")
    ap.add_argument("--front-cam", type=int, default=None, help="기록할 앞/위 카메라 번호 (선택)")
    ap.add_argument("--headless", action="store_true", help="창 없이 (끝낼 때 Ctrl+C)")
    ap.add_argument("--run-seconds", type=float, default=None, help=argparse.SUPPRESS)   # 시험용
    return ap.parse_args()


def main():
    a = parse()
    kin = PiperKin()
    arm = make_arm(a.robot, a.can, a.speed)
    if arm.is_real:
        print("\n[안전] 실물 팔입니다. 팔 둘레를 비우고, 전원 스위치 옆에 사람이 있어야 합니다.")
        print("       노트북에서 SPACE 를 눌러야 따라가고, 통신이 %.1f초 끊기면 멈춥니다. 속도 %d%%" % (a.timeout, a.speed))
        print("       집게 끝 최고 속도 %.0f cm/s, 책상 위 %.0f cm 아래로는 내려가지 않음" % (a.max_speed * 100, a.min_z * 100))
        input("       준비됐으면 Enter (취소 Ctrl+C) ")
    mapper = Mapper(kin, a.x0, a.y0, a.gain, max_speed=a.max_speed, min_z=a.min_z)
    mapper.yaw0 = 0.0                                   # 노트북이 집게 방향(yaw)을 그대로 보냄
    mapper.sync_to(arm.joints())
    srv = Server(a.port)
    cams = {k: CamGrabber(i) for k, i in (("wrist", a.wrist_cam), ("front", a.front_cam)) if i is not None}
    for k, c in cams.items():
        if not c.ok:
            print(f"[카메라] {k} 카메라를 못 엶 — 사진 없이 기록")
    rec = Recorder(a.out, {"args": vars(a), "robot_start": mapper.robot_start.tolist(), "source": "laptop --send",
                           "note": "사람 책상 좌표: 원점 A4 가운데, x 앞(노트북 쪽), y 사람 왼쪽, z 위. 로봇 좌표: 밑동 (0,0,0.90)"})
    ips = ", ".join(my_ips()) or "?"
    print(f"[대기] 노트북에서:  python teleop.py --send <이 PC 주소>:{a.port}    이 PC 주소: {ips}")
    win = None if a.headless else "robot_server"
    if win:
        cv2.namedWindow(win)

    follow, paused = False, False
    prev_req = None                    # 노트북이 바로 전에 보낸 follow (None = 아직 못 받음 → 껐다 켜야 움직임)
    rec_req_prev, disc_prev = False, None
    stop_n, msg, msg_n = 0, "", 0
    hand, last = None, None
    period = 1.0 / a.hz
    t_start = t_prev = time.monotonic()
    seen_peer = None

    def say(text):
        nonlocal msg, msg_n
        msg, msg_n = text, msg_n + 1
        print("[알림]", text)

    def force_stop(reason):
        nonlocal follow, stop_n
        if follow:
            follow = False
            arm.hold()
            stop_n += 1
            say(reason)

    try:
        while True:
            t = time.monotonic()
            dt, t_prev = t - t_prev, t
            m = srv.poll()
            if srv.new_peer:
                srv.new_peer = False
                force_stop("new laptop connected: follow stopped")
                prev_req, rec_req_prev, disc_prev = None, False, None
                seen_peer = srv.peer[0]
                say(f"laptop {seen_peer[0]} connected")
            if m is not None:
                last = m
                h = m.get("hand")
                hand = {"pinch": np.array(h[:3], float), "yaw": float(h[3]), "gap": float(h[4])} if h else None
                mapper.follow_grip = bool(m.get("grip", 1))
                mapper.follow_yaw = bool(m.get("yawf", 1))
                mapper.squeeze = float(np.clip(m.get("sq", 0.025), 0.0, 0.05))
                mapper.yaw_gap_min = float(np.clip(m.get("ygm", 0.05), 0.0, 0.1))
                mapper.yaw_pref = float(np.clip(m.get("ypref", -1.047), -3.2, 3.2))
                req = bool(m.get("follow", 0))
                if m.get("bye"):
                    force_stop("laptop closed")
                    req = False
                if prev_req is None:
                    if req:
                        say("follow was ON before connecting: press SPACE again")
                elif req and not prev_req:
                    if paused:
                        say("paused on robot PC (press SPACE there)")
                    else:
                        mapper.sync_to(arm.joints())
                        follow = True
                        say("follow started")
                elif not req and prev_req:
                    if follow:
                        follow = False
                        arm.hold()
                prev_req = req
                # 기록: 노트북 R 이 켜면 시작, 끄면 저장
                rq = bool(m.get("rec", 0)) and not m.get("bye")
                if rq and not rec_req_prev and not rec.on:
                    rec.start()
                    say(f"recording ep_{len(rec.saved):03d}")
                elif not rq and rec_req_prev and rec.on:
                    r = rec.stop()
                    say(f"saved {os.path.basename(r[0])} ({r[1]} frames)" if r else "nothing recorded")
                rec_req_prev = rq
                d = int(m.get("disc", 0))
                if disc_prev is not None and d != disc_prev:
                    p = rec.discard_last()
                    say(f"deleted {os.path.basename(p)}" if p else "no episode to delete")
                disc_prev = d

            age = srv.age()
            if follow and age > a.timeout:
                force_stop(f"no packet for {age:.1f} s: follow stopped (SPACE on laptop)")
            if rec.on and age > 3.0:
                r = rec.stop()
                rec_req_prev = False
                say(f"link lost, saved {os.path.basename(r[0])} ({r[1]} frames)" if r else "link lost")
            fresh = age < 0.25
            q_cmd, width = mapper.step(hand if (follow and fresh) else None, dt, follow)
            if follow:
                arm.send(q_cmd, width)
            q_now = arm.joints()
            w_now = arm.gripper()

            if rec.on:
                imgs = {k: c.latest() for k, c in cams.items()}
                lh = last.get("hand") if last else None
                rec.add(t=t, seq=int(last["seq"]) if last else -1, q=q_now, grip=w_now, q_cmd=mapper.q_cmd.copy(),
                        grip_cmd=mapper.width, hand=(np.array(lh[:3], float) if lh else np.full(3, np.nan)),
                        hand_yaw=(float(lh[3]) if lh else np.nan), hand_gap=(float(lh[4]) if lh else np.nan),
                        target=mapper.cur.copy(), tracked=lh is not None and fresh, follow=follow, link_age=age,
                        **{f"{k}_jpg": jpeg(v, 640) for k, v in imgs.items()})

            srv.reply(q=[round(float(v), 5) for v in q_now], w=(round(float(w_now), 4) if np.isfinite(w_now) else None),
                      tgt=[round(float(v), 4) for v in mapper.cur], follow=int(follow), rec=int(rec.on),
                      real=int(arm.is_real), x0=a.x0, y0=a.y0, stop=stop_n, msg=msg, msg_n=msg_n,
                      paused=int(paused), eps=len(rec.saved))

            if win:
                link = (f"laptop {seen_peer[0]}" if seen_peer else f"waiting :{a.port}")
                extra = [link, (f"!no packet {age:.1f}s" if age > a.timeout and seen_peer else
                                (f"packet {age*1000:.0f} ms ago" if seen_peer else "")),
                         "!PAUSED (SPACE)" if paused else ("REAL ARM" if arm.is_real else "preview arm")]
                panel = robot_panel(kin, q_now, mapper.cur, a.x0, a.y0, w_now if np.isfinite(w_now) else mapper.width,
                                    follow, rec.on, extra)
                if msg:
                    cv2.putText(panel, msg[:48], (8, 24 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1)
                cv2.imshow(win, panel)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                    if paused:
                        force_stop("paused on robot PC")
                        if not follow:
                            arm.hold()
                    else:
                        say("unpaused: press SPACE on laptop to follow")
                        prev_req = None
            if a.run_seconds is not None and t - t_start > a.run_seconds:
                break
            spare = period - (time.monotonic() - t)
            if spare > 0:
                time.sleep(spare)
    except KeyboardInterrupt:
        pass
    finally:
        if rec.on:
            r = rec.stop()
            if r:
                print(f"[기록] {r[0]} ({r[1]} 장면)")
        arm.hold()
        arm.close()
        for c in cams.values():
            c.close()
        srv.close()
        if win:
            cv2.destroyAllWindows()
        print(f"[끝] 기록 {len(rec.saved)}개" + (f" → {rec.dir}" if rec.saved else ""))


if __name__ == "__main__":
    main()
