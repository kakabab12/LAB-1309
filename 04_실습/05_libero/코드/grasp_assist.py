"""
정렬 보조 (2026-10-15) — VLA 가 블록을 쥐러 내려갈 때 앞뒤좌우만 블록 가운데로 맞춰 준다

왜
  al1 실패의 절반이 '쥐었다 빠짐'이고, 그중 대부분은 쥠이 1스텝만 잡혔다 — 손가락이 블록 모서리를 스치며 오므려
  블록이 튕겨 나갔다. 영상을 보면 학생은 손목 화면에서 블록을 가운데로 못 맞춘 채 내려간다 (10/14).
  시범은 블록 가운데에서 ±0.2mm 로 쥔다. 학생이 0.45초 늦게 보고 1~2cm 어긋나는 것을 카메라 계산으로 메운다.

무엇을 쓰나 (실물에도 있는 것만)
  정면 카메라 사진 → 색 검출(objdet) → 2차 변환식(front_calib.json, 평가 장면에서 평균 0.09cm·최대 0.24cm) → 블록의 책상 위 x, y
  로봇 손 위치 (로봇 상태). 손이 블록 가까이 오면 정면 카메라에서 블록이 손에 가릴 수 있어, 손이 8cm 넘게 떨어져 있을 때 잰 값을 기억해 쓴다.
  어느 블록인지는 지시문의 색 낱말 ("put the red block ...") 로 정한다.

무엇을 하나 (그리퍼가 열려 있고, 손이 그 블록에서 옆으로 4cm·위로 10cm 안일 때만)
  ① 앞뒤좌우 = 블록 가운데로 (위치 명령의 x, y 만 바꿈. 내려가기·손목·그리퍼는 VLA 그대로)
  ② 0.6cm 넘게 어긋나 있으면 쥐는 높이 3cm 아래에서는 조금씩 들어 올린다 (윗면에 걸린 손가락을 빼고 옆으로 맞추게)
  ③ 0.8cm 넘게 어긋나 있으면 오므리기를 미룬다 (맞춰지면 VLA 명령대로)

놓기 보조 (10/16, place=True, 평가 --place-assist): 지시문의 블록을 쥐고 있을 때만
  ④ 놓을 곳(판 가운데, 쌓기면 아래 블록 가운데)에서 옆으로 5cm 안이면 앞뒤좌우를 그 가운데로
  ⑤ 놓을 곳에서 3cm 넘게 떨어져 있으면 손을 벌리지 않는다 (al1 실패의 20% 가 판 밖에 놓기 — 대부분 판까지 가기 전에 놓음)
  일 바꾸기 때 쥐고 있던 다른 블록(새 지시의 대상이 아님)을 아무 데나 내려놓는 것은 막지 않는다.
  어느 블록을 쥐었나: 오므려 무언가를 쥔 순간(손가락 0.005~0.028) 손에서 4cm 안의 가장 가까운 블록. 판은 장면 처음에 잰 값.

쥐기 마무리 (10/16, complete=True, 평가 --grasp-complete):
  ⑥ 손이 열린 채 지시문의 블록 위 1cm 안·높이 10cm 안에서 20스텝 동안 1cm 도 안 내려가면 (맴돌기, 9/18 policy idling)
     그 자리에서 똑바로 내려가 쥐고 5cm 들어 올린 뒤 VLA 에 돌려준다.
     일 바꾸기 뒤 B 블록 바로 위에서 300스텝 넘게 맴돌다 시간이 다 된 장면이 있었다 (al1 + 놓기 보조, A8→B5 장면 1003).

내려가기 재촉 (10/17, descend=True, 평가 --descend-assist):
  ⑦ 손이 열린 채 지시문의 블록 위 0.8cm 안으로 맞춰졌는데 20스텝 동안 1cm 도 안 내려가면, 쥐는 높이까지 내려가게 한다
     (그동안 오므리지 않음). 바닥에서 10스텝을 기다려도 VLA 가 오므리지 않으면 오므린다. 들어 올리기·옮기기는 VLA.
     als2 실패 나누기: '블록 위에서 못 쥠' 은 손이 블록 중심 4~7cm 위에서 멈춘다 (⑥ 쥐기 마무리는 1cm 위 조건이라 거의 안 켜졌고,
     켜졌을 때는 들어 올린 뒤 VLA 가 옮기기를 시작하지 못했다).
다가가기 (10/17, approach=True, 평가 --approach-assist):
  ⑧ 빈손인데 ⓐ 지시문의 놓을 곳 6cm 안에 있거나 ⓑ 60스텝(3초) 동안 지시문의 블록에 1cm 도 다가가지 않으면 (그 블록에서 6cm 넘게),
     높이 13cm 로 올라 그 블록 위로 가서 블록 위 8cm 에서 VLA 에 돌려준다. 돌아오기 실패의 대부분이 원래 블록을 집지 않고
     빈손으로 놓을 판 위를 서성였다 (als1 + 보조, als2: 마지막 10초에 손이 70~110cm 움직임).
"""
import json
import re
from pathlib import Path

import numpy as np

import objdet

HERE = Path(__file__).resolve().parent
NAMES = {"red": "akita_black_bowl_1", "green": "cream_cheese_1", "blue": "blue_block_1", "yellow": "wine_bottle_1"}
TABLE_Z = 0.90          # 책상 높이 (실물: 로봇 밑동 기준으로 한 번 잼)
GRASP_H = {"red": 0.0225, "green": 0.0225, "blue": 0.0225, "yellow": 0.020}   # 쥐는 높이 = 블록 높이 절반 (시범과 같음)


def _feats(u, v):
    return np.array([u, v, 1.0, u * u, u * v, v * v])


class GraspAssist:
    def __init__(self, calib=HERE / "piper_sim" / "front_calib.json", radius=0.04, height=0.10,
                 stop_err=0.006, close_err=0.008, vmax=0.4, place=False, place_radius=0.05, release_err=0.03,
                 complete=False, idle_steps=20, descend=False, approach=False):
        self.C = {k: np.array(v) for k, v in json.loads(Path(calib).read_text()).items()}
        self.radius, self.height, self.stop_err, self.close_err, self.vmax = radius, height, stop_err, close_err, vmax
        self.place, self.place_radius, self.release_err = place, place_radius, release_err
        self.n_place = 0
        self.n_hold_release = 0
        self.complete, self.idle_steps = complete, idle_steps
        self.n_complete = 0
        self.ep = None
        self.n_used = 0
        self.descend, self.approach = descend, approach
        self.stat = {}

    def stats(self):
        """이번 장면에서 보조가 손댄 스텝 수 (평가 기록용)."""
        return dict(self.stat)

    def _target(self, instruction):
        m = re.search(r"put the (red|green|blue|yellow) block", instruction.lower())
        return m.group(1) if m else None

    def _dest(self, instruction):
        """놓을 곳 색 (판이면 보라·회색·주황·흰색, 쌓기면 아래 블록 색)."""
        m = re.search(r" on the (red|green|blue|yellow|purple|gray|orange|white) (square|block)", instruction.lower())
        return m.group(1) if m else None

    def _update(self, ep):
        """손이 8cm 넘게 떨어진 블록만 위치를 새로 잰다 (가까우면 손에 가려 중심이 틀어질 수 있다)."""
        f = objdet.detect(np.asarray(ep.obs["pixels"]["image"]), np.asarray(ep.obs["pixels"]["image2"]))
        hand = np.asarray(ep.obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        for k, c in enumerate(objdet.OBJECTS):
            if f[2 * k] < 0 or c not in self.C:
                continue
            xy = self.C[c] @ _feats(f[2 * k], f[2 * k + 1])
            if c in objdet.PADS:                # 판은 움직이지 않는다 — 처음 값 (블록을 올리면 가려져 틀어진다)
                self.pos.setdefault(c, xy)
            elif np.linalg.norm(xy - hand[:2]) > 0.08 or c not in self.pos:
                self.pos[c] = xy

    def __call__(self, ep, act, instruction):
        if ep is not self.ep:              # 새 장면
            self.ep, self.pos, self.held = ep, {}, None
            self.prim, self.zhist = None, []
            self.dmode, self.dbottom, self.amode, self.dhist, self.dhist_a = False, 0, False, [], []
            self.stat = {"align": 0, "place": 0, "descend": 0, "close": 0, "approach": 0}
        self._update(ep)
        q = float(ep.obs["robot_state"]["gripper"]["qpos"][0])
        hand = np.asarray(ep.obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        if q > 0.028 or q < 0.005:         # 벌렸거나 빈손으로 끝까지 닫힘 → 쥔 것 없음
            self.held = None
        elif self.held is None:            # 방금 무언가를 쥠 → 손에 가장 가까운 블록
            near = [(np.linalg.norm(self.pos[b] - hand[:2]), b) for b in objdet.BLOCKS if b in self.pos]
            if near and min(near)[0] < 0.04:
                self.held = min(near)[1]
        c = self._target(instruction)
        if self.prim is not None:          # ⑥ 쥐기 마무리 중
            return self._primitive(act, hand, q)
        if c is None or c not in self.pos:
            return act
        if q <= 0.028:
            return self._place(act, instruction, c, hand) if self.place else act
        e = self.pos[c] - hand[:2]
        d = float(np.linalg.norm(e))
        dz = hand[2] - (TABLE_Z + GRASP_H[c])           # 쥐는 높이 위로 얼마나
        if self.approach and self.held is None:
            a2 = self._approach(act, instruction, c, hand, d)
            if a2 is not None:
                return a2
        if d > self.radius or dz > self.height or dz < -0.02:
            self.dmode, self.dhist = False, []
            return act
        act = np.array(act, dtype=np.float32, copy=True)
        act[:2] = np.clip(e / 0.05, -self.vmax, self.vmax)          # ① 앞뒤좌우 = 블록 가운데
        if d > self.stop_err and dz < 0.03:
            act[2] = max(act[2], 0.15)                              # ② 많이 어긋나면 조금 들어 올린다 (손가락이 윗면에 걸려 옆으로 못 움직였다)
        if d > self.close_err and act[6] > 0:
            act[6] = -1.0                                           # ③ 오므리기를 미룬다
        self.n_used += 1
        self.stat["align"] = self.stat.get("align", 0) + 1
        if self.descend:
            act = self._descend(act, hand, d, dz)
        if self.complete:
            self.zhist.append(hand[2])
            if d < 0.01 and 0.005 < dz and len(self.zhist) >= self.idle_steps \
                    and self.zhist[-self.idle_steps] - hand[2] < 0.01:     # ⑥ 블록 위에서 내려가지 않고 맴돎
                self.prim = {"c": c, "phase": "down", "n": 0, "xy": self.pos[c].copy(),
                             "z": TABLE_Z + GRASP_H[c]}
                self.n_complete += 1
                self.zhist = []
                return self._primitive(act, hand, q)
        return act

    def _descend(self, act, hand, d, dz):
        """⑦ 맞춰졌는데 내려가지 않으면 쥐는 높이까지 내려가게, 바닥에서 오래 안 쥐면 오므림."""
        self.dhist.append(hand[2])
        if not self.dmode and d < self.close_err and dz > 0.015 and len(self.dhist) >= self.idle_steps \
                and self.dhist[-self.idle_steps] - hand[2] < 0.01:
            self.dmode, self.dbottom = True, 0
        if not self.dmode:
            return act
        if d > 0.015:                      # 많이 어긋나면 멈추고 VLA·정렬에 맡김
            self.dmode = False
            return act
        if dz > 0.008:
            act[2] = float(np.clip(-dz / 0.05, -0.3, -0.05))
            act[6] = -1.0
            self.stat["descend"] += 1
            return act
        self.dbottom += 1
        if act[6] <= 0 and self.dbottom > 10:
            act[6] = 1.0
            self.stat["close"] += 1
        if act[6] > 0:
            self.dmode, self.dhist = False, []
        return act

    def _approach(self, act, instruction, c, hand, d):
        """⑧ 빈손으로 놓을 곳 근처에 있거나 블록에 다가가지 않으면 블록 위로 데려간다. 안 하면 None."""
        self.dhist_a.append(d)
        dst = self._dest(instruction)
        at_dst = dst is not None and dst in self.pos and np.linalg.norm(self.pos[dst] - hand[:2]) < 0.06
        stuck = len(self.dhist_a) > 60 and self.dhist_a[-61] - d < 0.01
        if not self.amode and d > 0.06 and (at_dst or stuck):
            self.amode = True
        if not self.amode:
            return None
        if d < 0.02:                       # 블록 위에 왔다 → VLA·정렬 보조에 돌려줌
            self.amode, self.dhist_a = False, []
            return None
        z = TABLE_Z + (0.13 if d > 0.05 else GRASP_H[c] + 0.08)
        tgt = np.array([self.pos[c][0], self.pos[c][1], z])
        if hand[2] < TABLE_Z + 0.11 and d > 0.05:      # 낮으면 먼저 올라감 (쌓은 블록 9cm 위로)
            tgt[:2] = hand[:2]
        a = np.array(act, dtype=np.float32, copy=True)
        a[:3] = np.clip((tgt - hand) / 0.05, -self.vmax, self.vmax)
        a[6] = -1.0
        self.stat["approach"] += 1
        return a

    def _primitive(self, act, hand, q):
        """⑥ 쥐기 마무리: 똑바로 내려가기 → 오므리기 10스텝 → 5cm 들어 올리기 → VLA 로 돌려줌."""
        pr = self.prim
        pr["n"] += 1
        a = np.zeros(7, dtype=np.float32)
        a[3:6] = act[3:6]                   # 손목 방향은 VLA 그대로
        tgt = np.array([pr["xy"][0], pr["xy"][1], pr["z"]])
        if pr["phase"] == "down":
            a[:3] = np.clip((tgt - hand) / 0.05, -0.3, 0.3)
            a[6] = -1.0
            if abs(hand[2] - pr["z"]) < 0.004 or pr["n"] > 40:
                pr["phase"], pr["n"] = "close", 0
        elif pr["phase"] == "close":
            a[6] = 1.0
            if pr["n"] >= 10:
                pr["phase"], pr["n"], pr["z_up"] = "lift", 0, hand[2] + 0.05
        else:
            a[:3] = np.clip((np.array([hand[0], hand[1], pr["z_up"]]) - hand) / 0.05, -0.3, 0.3)
            a[6] = 1.0
            if hand[2] > pr["z_up"] - 0.005 or pr["n"] > 20:
                self.prim = None
        return a

    def _place(self, act, instruction, c, hand):
        dst = self._dest(instruction)
        if self.held != c or dst is None or dst not in self.pos:
            return act                      # 지시문의 블록을 쥔 게 아니면 (일 바꾸기 때 내려놓기 등) 손대지 않는다
        e = self.pos[dst] - hand[:2]
        d = float(np.linalg.norm(e))
        act = np.array(act, dtype=np.float32, copy=True)
        if d < self.place_radius:
            act[:2] = np.clip(e / 0.05, -self.vmax, self.vmax)        # ④ 놓을 곳 가운데로
            self.n_place += 1
            self.stat["place"] = self.stat.get("place", 0) + 1
        if d > self.release_err and act[6] < 0:
            act[6] = 1.0                                              # ⑤ 놓을 곳에서 멀면 벌리지 않는다
            self.n_hold_release += 1
        return act
