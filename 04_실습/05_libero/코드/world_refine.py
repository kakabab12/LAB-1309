"""
VLA 의도를 따라 마지막 몇 cm 만 다듬기 (2026-10-26) — 사용자 "논문에서 좋게 볼수있게 해줘"

  world_assist(블록 장면 보조를 옮긴 것)는 지시문을 스스로 읽어 옮길 물체·놓을 곳을 알았다. 그러면 '무엇을 어디로'까지 보조가
  하는 셈이라, VLA + 보조 점수가 VLA 덕인지 보조 덕인지 가를 수 없다. 여기서는 지시문을 읽지 않는다.
    옮길 물체 = 빈손일 때 손 바로 아래(가장 가까운) 물체, 손이 그 물체 윗면 + engage_h 안으로 내려왔을 때만
    놓을 곳   = 들고 있을 때 손 아래 가장 가까운 판(또는 쌓을 수 있는 물체), 손이 그 윗면 + 든 물체 높이 + engage_h 안으로 내려왔을 때만
  그 위에서 하는 일은 world_assist(GraspAssist) 의 다듬기 그대로: 물체 가운데·손목 방향 맞추기, 내려가 쥐기 마무리,
  판 가운데로 맞추기, 판에서 먼 곳에서는 놓지 않기. 지시문이 있어야 하는 것(멀리서 데려가기, 엉뚱한 물체 막기·내려놓기,
  데려다 놓기, 쌓은 것 피하기)은 쓰지 않는다.
  숫자: 물체·판 크기는 설정(world.json), 그리퍼는 모델 파일(grasp_assist._from_model), 위치는 위 카메라(world_detect).
  engage_h(10cm)·r_margin(5cm) 은 모든 장면·설정에서 같은 값.

  기록 (논문 표): n_steps 중 n_changed 스텝에서 VLA 동작을 바꿈 (개입률), first_intent = VLA 가 처음 쥐러 내려간 물체.
  parts: "grasp", "place" 를 골라 켤 수 있다 (빼보기).
  "objc" (10/29, 기본 꺼짐): 놓기 다듬기가 손이 아니라 든 물체 가운데를 판 가운데로 맞춘다. 쥔 순간 위 카메라 기억의 물체 자리 −
    손 자리를 손 방향 기준으로 적어 두고, 놓을 때 지금 손 방향으로 돌려 더한다 (5cm 넘으면 기억이 틀린 것으로 보고 안 씀).
    w8 + 다듬기의 '판 안, 3cm 밖' 8장면은 물체가 판 가운데에서 3.1~6.9cm — 막대를 한쪽으로 치우쳐 쥐면 손을 맞춰도 물체는 비껴 놓인다 (가설).
  "gplan" (10/29, 기본 꺼짐): 쥐는 자세 계산 (world_gplan.GraspPlan) — VLA 가 빈손으로 고른 물체 3cm 안까지 내려오면 위 카메라 자리·방향과
    역기구학으로 쥐는 자세를 정해 쥐고 조금 든 뒤 돌려준다. 원래 쥐기 다듬기(grasp)는 그대로 켜 두고 그 위에 얹는다.
  "fmask" (10/29 밤, 기본 꺼짐): 손목 맞추기가 방향을 잴 때 손목 사진에서 그리퍼 손가락 자리(world_detect.WristFingerMask, 그리퍼 값 →
    손가락 메시 투영)를 지우고, 손가락 끝 사이 기대 자리(손 축 위 물체 윗면 높이의 투영)에 가장 가까운 그 색 덩어리로 잰다.
    w8 + 다듬기 기본 1004 (검은 막대): 막대가 검은 손가락과 한 덩어리로 붙어 −90° 로 읽고 손목을 반대로 돌리며 내려가기를 막았다.
    저장한 손목 사진 24장에서 예전 −90° 18장 → 새로 재면 실제와 2~4° 안.
  "settle" (10/29 밤, 기본 꺼짐): 빈손으로 쥘 물체 바로 위(1.5cm 안)·쥐는 높이 1.5cm 안에서 오므리기 시작하면 그리퍼가 다 닫힐 때까지
    (최대 10스텝, 닫히는 게 멈추면 끝) 손을 그 자리(쥐는 높이까지는 내림)에 둔다. 기본 1024: 막대 바로 위·방향 0.8° 로 내려와
    VLA 가 오므리며 같은 스텝에 손을 들어 (그리퍼는 다 닫히는 데 5~6스텝) 손가락이 막대 위 허공에서 닫혔다.
  "wxy" (10/29 밤, 기본 꺼짐): 빈손으로 물체 위(옆 4cm·윗면 위 2.5~12cm)에 오면 손목 사진의 그 물체 덩어리를 윗면 높이 평면으로
    되쏘아(손목 카메라 자리 = 로봇 상태 손 자세 + 그리퍼 파일) 기억 자리를 반씩 고친다. 덩어리가 사진 가장자리에 닿으면 막대는 긴 축 선으로
    짧은 축 방향만. 위 카메라 기억이 팔에 가려 3.3~4.3cm 틀린 장면이 막대 실패 추적 9개 중 4개. VLA 없는 시험 (기억 3.5cm 틀리게):
    막대 쥠 켬 12/12, 끔 7/12 (기본·흔든 각 6).
  "pfine" (10/29 밤, 기본 꺼짐): 놓기 마무리 — 든 채 판 위에서 VLA 가 손을 펴려 하면 든 물체 가운데(objc 같이 켜면)·손을 판 가운데
    0.8cm 안·판 윗면 0.8cm 안까지 맞춘 뒤 편다. w8 + 다듬기 '판 안 3cm 밖' 추적: 막대 끝을 쥐어 손은 판에서 2cm·막대는 5.9cm (기본 1039),
    판 위 0.9cm 에서 놓아 막대가 1.3cm 밀림 (기본 1009, 끝 3.1cm). 지금 놓기 다듬기는 VLA 가 펴는 순간에야 붙고 손이 3cm 안이면 그대로 편다.
"""
import numpy as np

import piper_sim.world as W
import piper_sim.world_assist as WA


class WorldRefine(WA.WorldAssist):
    def __init__(self, parts=("grasp", "place"), engage_h=0.10, r_margin=0.05, hold_max=30, stack_r=0.02, **kw):
        pad = W.CFG["pad"]["size"]
        self.r_dest = float(np.hypot(pad[0], pad[1]) / 2 + r_margin)     # 판 반대각선 + 여유 안에서 내리면 그 판에 놓으려는 것
        kw.setdefault("place_radius", self.r_dest)
        super().__init__(**kw)
        self.parts, self.engage_h, self.r_margin = set(parts), engage_h, r_margin
        self.hold_max, self.stack_r = hold_max, stack_r       # 손 펴기를 늦추는 최대 스텝 (1.5초), 물체 위 쌓기로 볼 손-물체 거리
        self.lock, self.lock_n, self._act = None, 0, None
        self.first_intent, self.first_dest = None, None
        self._hand = np.zeros(3)
        self.obj_center = "objc" in self.parts
        self.ymem, self._k_now = {}, None
        self.gplan = None
        if "gplan" in self.parts:
            import piper_sim.world_gplan as WG
            self.gplan = WG.GraspPlan()
        self.fmask = None                                     # 손목 카메라 모델 (손가락 자리·투영) — fmask·wxy 가 같이 쓴다
        self.use_fmask, self.use_wxy = "fmask" in self.parts, "wxy" in self.parts
        if self.use_fmask or self.use_wxy:
            import piper_sim.world_detect as WD
            self.fmask = WD.WristFingerMask()
        self.settle, self.settle_max, self._st_n, self._st_q = "settle" in self.parts, 10, 0, []
        self.pfine, self._pf = "pfine" in self.parts, None
        self._off_key, self._off_h = None, None

    def __call__(self, ep, act, instruction):
        if ep is not getattr(self, "ep", None):
            self.first_intent, self.first_dest = None, None
            self.lock, self.lock_n = None, 0
            self._off_key, self._off_h = None, None
            self.ymem, self._k_now = {}, None
            self._st_n, self._st_q, self._pf = 0, [], None
            if self.gplan is not None:
                self.gplan.reset()
        if getattr(self, "held", None) is None:              # 앞 스텝에 빈손이면 다음에 쥘 때 물체 − 손 을 새로 잰다
            self._off_key = None
        self._hand = np.asarray(ep.obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        self._act = np.asarray(act, dtype=np.float64)
        out = super().__call__(ep, act, instruction)          # 개입률은 WorldAssist 가 센다
        if self.gplan is not None:
            out = self._gplan_step(ep, act, out)
        if self.settle and (self.gplan is None or self.gplan.plan is None):
            out = self._settle_step(ep, act, out)
        if self.pfine:
            out = self._pfine_step(ep, act, out)
        return out

    def _pfine_step(self, ep, act, out):
        """pfine: 든 채 판 위에서 VLA 가 손을 펴려 하면 (놓을 곳 lock) 든 물체 가운데(objc 면)·손을 판 가운데 0.8cm 안으로 맞추고
        물체 바닥을 판 윗면 0.8cm 안까지 내린 뒤 편다 (6스텝), 1.5cm 들고 돌려준다. 맞추기는 최대 30스텝."""
        P = self._pf
        if P is None:
            if self.lock is None or getattr(self, "held", None) is None or act[6] >= 0 or self.lock not in self.pos:
                return out
            ga = getattr(self, "grasp_at", None)
            P = self._pf = {"phase": "center", "n": 0, "dst": self.lock, "off_z": float(ga[0][2] - W.TABLE_Z) if ga is not None else 0.03}
            self.stat["pfine_n"] = self.stat.get("pfine_n", 0) + 1
        P["n"] += 1
        hand = self._hand
        new = np.zeros(7, dtype=np.float32)
        if P["phase"] == "center":
            cxy = (self._held_center(hand) if self.obj_center else hand)[:2]
            e = self.pos[P["dst"]] - cxy
            above = float(hand[2] - P["off_z"] - self._top(P["dst"]))
            new[:2] = np.clip(e / 0.05, -self.vmax, self.vmax)
            new[2] = float(np.clip(-(above - 0.004) / 0.05, -0.2, 0.1))
            new[6] = 1.0
            if (np.linalg.norm(e) < 0.008 and above < 0.008) or P["n"] > 30:
                P["phase"], P["n"] = "open", 0
        elif P["phase"] == "open":
            new[6] = -1.0
            if P["n"] >= 6:
                P["phase"], P["n"] = "up", 0
        else:
            new[2], new[6] = 0.3, -1.0
            if P["n"] >= 5:
                self._pf = None
        self.stat["pfine"] = self.stat.get("pfine", 0) + 1
        self._changed(act, out, new)
        return new

    def _changed(self, act, out, new):
        """원래 다듬기가 안 바꾼 스텝을 새 부품이 바꿨으면 개입으로 센다."""
        same = not np.any(np.abs(np.asarray(out, dtype=np.float64) - np.asarray(act, dtype=np.float64)) > 1e-6)
        if same and np.any(np.abs(np.asarray(new, dtype=np.float64) - np.asarray(act, dtype=np.float64)) > 1e-6):
            self.n_changed = getattr(self, "n_changed", 0) + 1

    def _settle_step(self, ep, act, out):
        q = float(ep.obs["robot_state"]["gripper"]["qpos"][0])
        k = self._k_now
        name = WA._name(self.ep.scene, k) if k is not None else None
        dz = None
        if name is not None and k in self.pos:
            import piper_sim.world_expert as WE
            dz = float(self._hand[2] - (W.TABLE_Z + WE.grasp_z(name)))
        if self._st_n > 0:                                      # 붙잡는 중: 닫히는 게 멈추거나 10스텝이면 끝
            self._st_n += 1
            self._st_q.append(q)
            moving = len(self._st_q) < 3 or self._st_q[-3] - self._st_q[-1] > 0.0005
            if self._st_n > self.settle_max or not moving:
                self._st_n, self._st_q = 0, []
                return out
        else:
            if getattr(self, "held", None) is not None or out[6] <= 0 or q < WA.GA.Q_WIDE or dz is None:
                return out
            if np.linalg.norm(self._hand[:2] - self.pos[k]) > 0.015 or not (-0.005 < dz < 0.015):
                return out
            self._st_n, self._st_q = 1, [q]
            self.stat["settle_n"] = self.stat.get("settle_n", 0) + 1
        new = np.array(out, dtype=np.float32, copy=True)
        new[:6] = 0.0
        if dz is not None and dz > 0.003:
            new[2] = float(np.clip(-dz / 0.05, -0.2, 0.0))
        new[6] = 1.0
        self.stat["settle"] = self.stat.get("settle", 0) + 1
        self._changed(act, out, new)
        return new

    def _cam(self, ep):
        return self.fmask.cam_pose(np.asarray(ep.obs["robot_state"]["eef"]["mat"], dtype=np.float64).reshape(3, 3), self._hand)

    def _wrist_blob(self, ep, c):
        """손목 사진에서 물체 c 색 덩어리 (손가락 자리 지움) 중 기억 자리(윗면 높이)를 투영한 곳에 가장 가까운 것. 없으면 None."""
        from scipy import ndimage
        img = np.asarray(ep.obs["pixels"]["image2"])
        lab = self.wrist_det.classify(img, own_table=False)
        ck = f"obj:{W.parse(c)[2]}"
        i = self.wrist_det.keys.index(ck) if ck in self.wrist_det.keys else None
        if i is None:
            return None
        h, w = lab.shape
        q = float(ep.obs["robot_state"]["gripper"]["qpos"][0])
        m = (lab == i) & ~self.fmask.mask(q, h, w)
        cc, n = ndimage.label(m)
        if n == 0:
            return None
        rc = self.fmask.project([*self.pos[c], self._top(c)], self._cam(ep), h, w) if c in self.pos else None
        if rc is None:
            rc = self.fmask.expected_pixel(max(float(self._hand[2] - self._top(c)), 0.0), h, w)
        sizes = ndimage.sum(m, cc, range(1, n + 1))
        best, bd = None, np.inf
        for k in range(1, n + 1):
            if sizes[k - 1] < 1500 * h * w / (256 * 256):
                continue
            ys, xs = np.nonzero(cc == k)
            dd = float(np.min((ys - rc[0]) ** 2 + (xs - rc[1]) ** 2))
            if dd < bd:
                best, bd = k, dd
        return None if best is None else cc == best

    def _wxy(self, ep):
        """wxy: 빈손으로 물체 위(옆 4cm·윗면 위 2.5~12cm)에 오면 손목 사진 덩어리를 윗면 높이로 되쏘아 기억 자리를 반씩 고친다.
        덩어리가 사진 가장자리에 닿으면 긴 물체(막대)는 긴 축 선만 믿고 짧은 축 방향만, 다른 모양은 고치지 않는다."""
        import cv2
        if getattr(self, "held", None) is not None:
            return
        hand = self._hand
        objs = [k for k in self.pos if k in WA._Shim.BLOCKS]
        if not objs:
            return
        k = min(objs, key=lambda b: float(np.linalg.norm(self.pos[b] - hand[:2])))
        top = self._top(k)
        if np.linalg.norm(self.pos[k] - hand[:2]) > 0.04 or not (0.025 < hand[2] - top < 0.12):
            return
        blob = self._wrist_blob(ep, k)
        if blob is None:
            return
        h, w = blob.shape
        cam = self._cam(ep)
        ys, xs = np.nonzero(blob)
        edge = bool(blob[0].any() or blob[-1].any() or blob[:, 0].any() or blob[:, -1].any())
        name = WA._name(self.ep.scene, k)
        per = WA.yaw_period(name) if name else None
        if not edge:
            est = self.fmask.backproject(ys.mean(), xs.mean(), top, cam, h, w)[:2]
        elif per is not None and per >= 180:
            (cx, cy), (bw, bh), ang = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
            t = np.radians(ang if bw >= bh else ang + 90.0)
            L = max(bw, bh) / 2
            p1 = self.fmask.backproject(cy - L * np.sin(t), cx - L * np.cos(t), top, cam, h, w)[:2]
            p2 = self.fmask.backproject(cy + L * np.sin(t), cx + L * np.cos(t), top, cam, h, w)[:2]
            u = p2 - p1
            if np.linalg.norm(u) < 1e-3:
                return
            u /= np.linalg.norm(u)
            est = p1 + u * float(np.dot(self.pos[k] - p1, u))
        else:
            return
        d = est - self.pos[k]
        if np.linalg.norm(d) > 0.05:
            return
        self.pos[k] = self.pos[k] + 0.5 * d
        self.stat["wxy"] = self.stat.get("wxy", 0) + 1

    def _wrist_angle(self, img, c):
        """fmask 면 손가락 자리를 지우고 기대 자리에 가장 가까운 덩어리로. 아니면 WorldAssist 그대로."""
        if not self.use_fmask:
            return super()._wrist_angle(img, c)
        import cv2
        name = WA._name(self.ep.scene, c)
        per = WA.yaw_period(name) if name else 90.0
        if per is None:
            return 0.0
        blob = self._wrist_blob(self.ep, c)
        if blob is None:
            return None
        cnts, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        if per >= 90:                       # 상자: 최소 넓이 사각형 (WorldAssist 와 같음)
            (_, _), (bw, bh), ang = cv2.minAreaRect(cnt.astype(np.float32))
            if per >= 180:
                longang = ang if bw >= bh else ang + 90
                return ((longang - 90 + 90) % 180) - 90
            return ((ang + 45) % 90) - 45
        poly = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True).reshape(-1, 2).astype(float)
        if len(poly) < 3:
            return None
        e = np.roll(poly, -1, 0) - poly
        j = int(np.argmax(np.linalg.norm(e, axis=1)))
        ang = np.degrees(np.arctan2(e[j, 1], e[j, 0]))
        return ((ang - 90 + per / 2) % per) - per / 2

    def _gplan_step(self, ep, act, out):
        q = float(ep.obs["robot_state"]["gripper"]["qpos"][0])
        if self.gplan.plan is None and not self.gplan.maybe_start(self, ep, act, self._k_now, self._hand, q, WA.GA.Q_WIDE):
            return out
        a = self.gplan.step(self, ep, self._hand)
        if a is None:
            return out
        same = not np.any(np.abs(np.asarray(out, dtype=np.float64) - np.asarray(act, dtype=np.float64)) > 1e-6)
        if same and np.any(np.abs(a.astype(np.float64) - np.asarray(act, dtype=np.float64)) > 1e-6):
            self.n_changed = getattr(self, "n_changed", 0) + 1     # 원래 다듬기가 안 바꾼 스텝을 쥐는 자세 계산이 바꿈
        return a

    def _update(self, ep):
        super()._update(ep)
        hand = self._hand
        for c, xy in self.fresh.items():                  # 위 카메라 긴 축 방향도 기억 (손이 8cm 넘게 멀 때만 — 가까우면 손에 가려 틀림)
            if c in WA._Shim.BLOCKS and c in self.det.yaw and np.linalg.norm(xy - hand[:2]) > 0.08:
                self.ymem[c] = float(self.det.yaw[c][0])
        if self.use_wxy:
            self._wxy(ep)

    def _top(self, key):
        """그 물체·판 윗면 높이 (책상 기준 아님, 세계 z)."""
        name = WA._name(self.ep.scene, key)
        return W.TABLE_Z + (W.height(name) if name else 0.0)

    def _target(self, instruction):
        self._k_now = None
        k = self._target_inner(instruction)
        if self.held is None:
            self._k_now = k
        return k

    def _target_inner(self, instruction):
        if self.held is not None:                 # 들고 있으면 그 물체 (놓기 다듬기가 씀)
            return self.held if "place" in self.parts else None
        if "grasp" not in self.parts:
            return None
        hand = self._hand
        objs = [k for k in self.pos if k in WA._Shim.BLOCKS]
        if not objs:
            return None
        k = min(objs, key=lambda b: float(np.linalg.norm(self.pos[b] - hand[:2])))
        name = WA._name(self.ep.scene, k)
        reach = 0.5 * (W.max_extent(name) if name else 0.05) + self.r_margin
        if np.linalg.norm(self.pos[k] - hand[:2]) > reach or hand[2] > self._top(k) + self.engage_h:
            return None
        if self.first_intent is None:
            self.first_intent = k
        return k

    def _dest(self, instruction):
        """놓을 곳 = VLA 가 손을 펴려 할 때(동작 그리퍼 < 0) 손 아래의 판 (판 반대각선 + 여유 안), 판이 없으면 손 바로 아래(stack_r 안)의
        쌓을 수 있는 물체. 한 번 정하면 놓을 때까지 그대로 (lock). 10/27: 들고 가는 도중 손이 다른 물체 위를 낮게 지나가면
        그 물체를 놓을 곳으로 잡아 쌓아 버리거나 수백 스텝 들고 있었다 (w3 + 다듬기, 물체 많음 개입 46%)."""
        if self.held is None or "place" not in self.parts:
            self.lock, self.lock_n = None, 0
            return None
        if self.lock is not None:
            return self.lock
        if self._act is None or self._act[6] >= 0:            # 아직 놓으려 하지 않음 → 손대지 않는다
            return None
        hand = self._hand
        held = WA._name(self.ep.scene, self.held)
        h_held = W.height(held) if held else 0.05
        pads = [k for k in self.pos if k in WA._Shim.PADS and np.linalg.norm(self.pos[k] - hand[:2]) <= self.r_dest
                and hand[2] <= self._top(k) + h_held + self.engage_h]
        if pads:
            k = min(pads, key=lambda b: float(np.linalg.norm(self.pos[b] - hand[:2])))
        else:
            stackable = lambda b: bool(W.CFG["shapes"][W.parse(b)[1]].get("stack_on"))   # noqa: E731
            objs = [b for b in self.pos if b != self.held and b in WA._Shim.BLOCKS and stackable(b)
                    and np.linalg.norm(self.pos[b] - hand[:2]) <= self.stack_r and hand[2] <= self._top(b) + h_held + self.engage_h]
            if not objs:
                return None
            k = min(objs, key=lambda b: float(np.linalg.norm(self.pos[b] - hand[:2])))
        self.lock, self.lock_n = k, 0
        if self.first_dest is None:
            self.first_dest = k
        return k

    def _place(self, act, instruction, c, hand):
        """GraspAssist 놓기 다듬기 (가운데로 맞추기, 먼 곳에서 놓지 않기) — 정한 뒤 hold_max 스텝까지만. 그 뒤로는 VLA 그대로."""
        if self.lock is not None:
            self.lock_n += 1
            if self.lock_n > self.hold_max:
                return act
        if self.obj_center:
            hand = self._held_center(hand)
        return super()._place(act, instruction, c, hand)

    def _hand_yaw(self):
        M = np.asarray(self.ep.obs["robot_state"]["eef"]["mat"], dtype=np.float64).reshape(3, 3)
        return float(np.arctan2(M[1, 0], M[0, 0]))          # 손가락이 벌어지는 축의 책상 위 방향

    def _held_center(self, hand):
        """objc: 손 대신 든 물체 가운데 (x, y) 를 돌려준다 (높이는 손 그대로)."""
        if self.held is None or self.held not in self.pos:
            return hand
        if self._off_key != self.held:                      # 새로 쥠 → 쥔 순간의 물체 − 손 을 손 방향 기준으로
            ga = getattr(self, "grasp_at", None)
            h0 = ga[0][:2] if ga is not None else hand[:2]
            off = np.asarray(self.pos[self.held], dtype=np.float64) - h0
            y = self._hand_yaw()
            c, s = np.cos(-y), np.sin(-y)
            self._off_key = self.held
            self._off_h = np.array([c * off[0] - s * off[1], s * off[0] + c * off[1]]) if np.linalg.norm(off) <= 0.05 else None
        if self._off_h is None:
            return hand
        y = self._hand_yaw()
        c, s = np.cos(y), np.sin(y)
        out = np.array(hand, dtype=np.float64, copy=True)
        out[:2] += [c * self._off_h[0] - s * self._off_h[1], s * self._off_h[0] + c * self._off_h[1]]
        return out


def make(parts=("grasp", "place")):
    """world_eval --refine 에서: world_assist 켜기 옵션 중 지시문이 필요 없는 다듬기만."""
    return WorldRefine(parts=parts, place="place" in parts, descend="grasp" in parts, yaw="grasp" in parts,
                       unjam="grasp" in parts, fresh_held=True, mem_refresh=True, stuck_grasp=40 if "grasp" in parts else 0,
                       yaw_none_reset=10, mem_tol=0.012, open_empty="grasp" in parts)
