#!/usr/bin/env python
"""실물 PiPER 에서 VLA 돌리기 (2026-11-04) — 시뮬 평가와 같은 입력·지연·동작 변환, 관절 각도로 내보냄.

  # 1) 연습 (팔에 아무것도 안 보냄, 카메라·모델·계산만):
  python real_piper/run_real.py --backend dry --task "put the red cube on the gray square"
  # 2) 실물 (처음에는 느리게):
  python real_piper/run_real.py --backend sdk --speed 20 --task "빨간 정육면체를 회색 판에 놓아줘"
  # 3) 시험 (실물 대신 시뮬레이터 팔·카메라로 같은 코드 길을 끝까지):
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python real_piper/run_real.py --backend sim --seed 1000

도는 중 키보드: 새 지시문(한국어·영어)을 치고 Enter → 그 지시로 바뀜 (끼어들기는 "잠깐, … 먼저"), "이어서" → 쌓아 둔 일로,
  "멈춰" → 제자리, Ctrl-C → 멈추고 끝. 기록은 --log 폴더 (사진·관절·동작, 학습 데이터와 같은 꼴).
"""
import argparse
import io
import json
import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from real_piper.kin import Kin  # noqa: E402
from real_piper.policy import VLA, Planner, make_obs, to_delta  # noqa: E402
from real_piper import robot_io  # noqa: E402

DT = 0.05            # 20 Hz (시뮬 control_freq 와 같음)


def load_cfg(path):
    cfg = json.loads(Path(path).read_text())
    return cfg


class Safety:
    """목표 손끝을 작업 공간 안으로 자르고, 한 스텝 관절 이동을 묶는다 (시뮬 범위 안에서는 손대지 않는 값)."""

    def __init__(self, cfg, kin):
        s = cfg["safety"]
        self.base = kin.base
        self.r_min, self.r_max = s["radius_m"]
        self.z_min, self.z_max = self.base[2] + s["z_above_table_m"][0], self.base[2] + s["z_above_table_m"][1]
        self.max_step = np.radians(s["max_joint_step_deg"])
        self.n_clip = 0

    def goal(self, p):
        p = np.array(p, dtype=np.float64, copy=True)
        v = p[:2] - self.base[:2]
        r = float(np.linalg.norm(v))
        rc = float(np.clip(r, self.r_min, self.r_max))
        clipped = False
        if r > 1e-6 and abs(rc - r) > 1e-9:
            p[:2] = self.base[:2] + v * rc / r
            clipped = True
        z = float(np.clip(p[2], self.z_min, self.z_max))
        clipped |= abs(z - p[2]) > 1e-9
        p[2] = z
        self.n_clip += int(clipped)
        return p

    def joints(self, q_goal, q_now):
        return q_now + np.clip(q_goal - q_now, -self.max_step, self.max_step)


class Commands:
    """키보드 지시문 → 지금 할 일 (world_cmd.Controller: 끼어들기는 쌓고, '이어서' 면 꺼냄). 영어 지시문은 그대로."""

    def __init__(self, first):
        import world_cmd
        self.wc = world_cmd
        self.ctl = world_cmd.Controller()
        self.q = queue.Queue()
        self.say = print
        if first:
            self.apply(first)

    def start(self):
        """키보드 지시 읽기 (시작 확인 Enter 를 가로채지 않게 확인이 끝난 뒤에 켠다)."""
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in sys.stdin:
            self.q.put(line.strip())

    def apply(self, text):
        if not text:
            return
        if re.search(r"[가-힣]", text):
            cmds = self.ctl.command(text)
            for c in cmds:
                self.say(f"  → {c.get('say', '')}  ({c.get('instr', c['type'])})")
        else:
            self.ctl.current, self.ctl.queue, self.ctl.stack = text, [], []
            self.say(f"  → {text}")

    def poll(self):
        changed = False
        while not self.q.empty():
            self.apply(self.q.get())
            changed = True
        return changed

    @property
    def current(self):
        return self.ctl.current


class Log:
    """학습 데이터(world_collect SegRec)와 같은 꼴 npz: img·wrist (JPEG), eef_pos, eef_quat, grip, action, joints, task."""

    def __init__(self, folder, every=1):
        self.dir = Path(folder) if folder else None
        self.every = max(1, int(every))
        self.rows = []

    def add(self, t, obs, act, task, extra):
        if self.dir is None:
            return
        r = {"t": t, "task": task, "eef_pos": obs["robot_state"]["eef"]["pos"].astype(np.float32),
             "eef_quat": obs["robot_state"]["eef"]["quat"].astype(np.float32),
             "grip": obs["robot_state"]["gripper"]["qpos"].astype(np.float32),
             "joints": obs["robot_state"]["joints"]["pos"].astype(np.float32), "action": np.asarray(act, np.float32), **extra}
        if t % self.every == 0:
            r["img"], r["wrist"] = _jpeg(obs["pixels"]["image"]), _jpeg(obs["pixels"]["image2"])
        self.rows.append(r)

    def save(self, meta):
        if self.dir is None or not self.rows:
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        f = self.dir / f"real_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        keys = ["eef_pos", "eef_quat", "grip", "joints", "action", "delta", "q_goal"]
        out = {k: np.stack([r[k] for r in self.rows]) for k in keys if all(k in r for r in self.rows)}
        out["t"] = np.array([r["t"] for r in self.rows])
        out["task"] = np.array([r["task"] or "" for r in self.rows])
        out["img"] = np.array([r.get("img", b"") for r in self.rows], dtype=object)
        out["wrist"] = np.array([r.get("wrist", b"") for r in self.rows], dtype=object)
        out["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
        np.savez(f, **out)
        return f


class _Still:
    def __init__(self, img):
        self.img = img

    def read(self):
        return self.img


def _jpeg(img):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(b, format="JPEG", quality=90)
    return b.getvalue()


def go_home(robot, kin, q_home, seconds=5.0, real_time=True):
    """처음 자세(시뮬 init_qpos)로 관절을 고르게 나눠 천천히 (시뮬은 늘 이 자세에서 시작)."""
    q0 = robot.joints()
    n = max(1, int(seconds / DT))
    for i in range(1, n + 1):
        robot.send_joints(q0 + (q_home - q0) * i / n)
        if real_time:
            time.sleep(DT)
        elif hasattr(robot, "tick"):
            robot.tick()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(HERE / "config_real.json"))
    p.add_argument("--backend", choices=["sdk", "dry", "sim"], default="dry")
    p.add_argument("--task", default=None, help="처음 지시문 (영어 틀 또는 한국어). sim 이면 장면의 지시문")
    p.add_argument("--model", default=None, help="모델 폴더 (없으면 설정 model)")
    p.add_argument("--speed", type=int, default=None, help="실물 관절 속도 % (없으면 설정)")
    p.add_argument("--steps", type=int, default=None, help="최대 스텝 (sim 기본 600)")
    p.add_argument("--seed", type=int, default=1000, help="sim: world_eval 과 같은 장면 번호")
    p.add_argument("--latency", type=int, default=None, help="지연 스텝 (sim: 장면 동작 변수, 실물: 설정)")
    p.add_argument("--log", default=None, help="기록 폴더 (없으면 설정 log_dir)")
    p.add_argument("--yes", action="store_true", help="시작 확인을 묻지 않음 (sim·dry 만)")
    p.add_argument("--fake-cams", action="store_true", help="dry 시험: 카메라 대신 reference/*_model.png (카메라 없는 PC 에서 실제 시간 길 시험)")
    a = p.parse_args()
    cfg = load_cfg(a.config)
    kin = Kin()
    sim = a.backend == "sim"
    L = a.latency if a.latency is not None else int(cfg["latency_steps"])
    vla = VLA(ROOT / (a.model or cfg["model"]) if not Path(a.model or cfg["model"]).is_absolute() else (a.model or cfg["model"]),
              latency_steps=L, cuda_graph=cfg.get("cuda_graph", True), world_cfg=cfg["world"],
              detect_model=None if sim else (ROOT / cfg["detect_model"] if cfg.get("detect_model") else None),
              camera_calib=None if sim else cfg.get("top_camera_calib"))
    if sim:
        import piper_sim.world as W
        robot = robot_io.SimRobot(a.seed, vla.r)
        if a.latency is None:
            mv = W.sample_motion(a.seed, 1)
            if mv is not None and mv.get("latency") is not None:
                L = int(mv["latency"])
        task = a.task or robot.instruction
        cams = None
    else:
        if a.backend == "sdk":
            robot = robot_io.PiperSDK(cfg["can"], speed_pct=a.speed or cfg["speed_pct"], grip_open_m=cfg["gripper"]["open_m"],
                                      grip_effort_nm=cfg["gripper"]["effort_nm"], interface=cfg.get("sdk_interface", "v2"))
        else:
            robot = robot_io.DryRun(np.array(cfg["home_qpos"]), grip_open_m=cfg["gripper"]["open_m"])
        from real_piper import cameras
        if a.fake_cams:
            from PIL import Image
            cams = {k: _Still(np.asarray(Image.open(HERE / "reference" / f"{n}_model.png").convert("RGB")))
                    for k, n in (("image", "top"), ("image2", "wrist"))}
        else:
            cams = {"image": cameras.make(cfg["cameras"]["top"], "위"), "image2": cameras.make(cfg["cameras"]["wrist"], "손목")}
        task = a.task
    planner = Planner(vla, latency_steps=L, mode="steps" if sim else "thread")
    safety = Safety(cfg, kin)
    cmds = Commands(task)
    log = Log(a.log or (None if sim else cfg.get("log_dir")), every=cfg.get("log_every", 1))
    max_steps = a.steps or (600 if sim else int(cfg.get("max_steps", 2400)))
    q_off = 0.0 if sim else float(cfg["gripper"]["q_offset_m"])
    print(f"백엔드 {a.backend}, 지연 {L}스텝 ({'시뮬 스텝' if sim else '실제 시간'}), 모델 {a.model or cfg['model']}, 지시 '{cmds.current}'")
    if not sim:                                            # 예열: CUDA 그래프 녹화 (첫 추론 수 초) + 실제 추론 시간 재기
        o0 = make_obs(cams["image"].read(), cams["image2"].read(), kin, robot.joints(), robot.gripper(), q_off)
        ts = []
        for _ in range(4):                                 # 추론 스레드에서 (CUDA 그래프를 쓸 스레드에서 녹화)
            t0 = time.time()
            planner.pool.submit(vla.infer, o0, cmds.current or "put the red cube on the gray square").result()
            ts.append(time.time() - t0)
        vla.reset()
        ms = 1000 * float(np.median(ts[1:]))
        print(f"추론 한 번 {ms:.0f} ms → 지연 {ms / 1000 / DT:.1f}스텝 (설정 {L}). 많이 다르면 config latency_steps 를 {int(np.ceil(ms / 1000 / DT))} 로")
        if a.backend == "sdk" or not a.yes:
            input("팔 주변을 비우고 비상 정지 스위치를 손에 쥔 뒤 Enter (처음 자세로 천천히 감) ")
        go_home(robot, kin, np.array(cfg["home_qpos"]), seconds=cfg.get("home_seconds", 5.0))
        if a.backend == "sdk" or not a.yes:
            input("시작하려면 Enter (도는 중 Ctrl-C 로 멈춤) ")
    kin.reset()
    cmds.start()
    grip_close = False
    t, ok = 0, False
    t_start = time.time()
    try:
        while t < max_steps:
            tick0 = time.time()
            if cmds.poll():
                print(f"[{t}] 지시: {cmds.current}")
            instr = cmds.current
            q = robot.joints()
            g = robot.gripper()
            if sim:
                px = robot.obs()["pixels"]
                top, wrist = px["image"], px["image2"]
            else:
                top, wrist = cams["image"].read(), cams["image2"].read()
            obs = make_obs(top, wrist, kin, q, g, q_off)
            eef = obs["robot_state"]["eef"]["pos"]
            if instr is None:                              # 할 일 없음 → 제자리
                robot.hold()
                act = np.r_[eef, np.zeros(3), [-1.0]].astype(np.float32)
                delta = to_delta(act, eef, vla.abs_pos)
                q_goal = q
            else:
                act = planner.next(obs, instr, eef)
                delta = to_delta(act, eef, vla.abs_pos)
                gp, gM = kin.goal(q, delta)
                gp = safety.goal(gp)
                q_goal = safety.joints(kin.ik(gp, gM, q), q)
                robot.send_joints(q_goal)
                if delta[6] > 0.1:                         # 시뮬 그리퍼: + 닫기, − 열기 (작은 떨림은 무시)
                    grip_close = True
                elif delta[6] < -0.1:
                    grip_close = False
                robot.send_gripper(grip_close)
            log.add(t, obs, act, instr, {"delta": delta.astype(np.float32), "q_goal": np.asarray(q_goal, np.float32)})
            if sim:
                robot.tick()
                if robot.success():
                    ok = True
                    break
            else:
                time.sleep(max(0.0, DT - (time.time() - tick0)))
            t += 1
            if t % 100 == 0:
                print(f"[{t}] 손끝 {np.round(eef, 3)} 그리퍼 {'닫힘' if grip_close else '열림'} 추론 {planner.n_infer}번, 작업공간 자름 {safety.n_clip}번"
                      f" ({(time.time() - t_start) / t * 1000:.0f} ms/스텝)")
    except KeyboardInterrupt:
        print("Ctrl-C — 멈춤")
    finally:
        robot.hold()
        f = log.save({"backend": a.backend, "model": str(a.model or cfg["model"]), "latency": L, "seed": a.seed if sim else None,
                      "success": ok if sim else None, "steps": t})
        if f:
            print("기록:", f)
        if sim:
            print(f"SIM seed {a.seed}: {'성공' if ok else '실패'} ({t}스텝) 지시 '{instr}'")
        robot.close()


if __name__ == "__main__":
    main()
