#!/usr/bin/env python
"""
스크립트 전문가 — **어디서 시작하든** 물체로 가서 집어 옮긴다

왜 필요한가 (2026-09-28)
  LoRA 1~4차가 전부 실패했다. 네 번 다 **정책이 스스로 성공한 궤적을 따라하게** 했는데,
  정책은 교란된 자세에서 물체를 다시 집는 데 거의 성공하지 못한다 → 따라할 좋은 예가 없다.
  다시 뽑기로 억지로 모은 성공은 **운 좋은 노이즈 덕**이라, 따라하면 오히려 망가진다
  (4차: 학습한 태스크 T8 이 교란 없는 상태에서도 90% → 10%).

      정책이 못하는 것은 정책의 데이터로 가르칠 수 없다. 더 나은 선생이 필요하다.

  시뮬레이터는 **물체의 정확한 위치를 안다.** 그걸로 어디서 시작하든 집는 스크립트를 만들고,
  그 시범을 정책에 가르친다. 교란된 상태 문제(covariate shift)를 푸는 표준 방법이다.
  📚 DART (2017), DAgger (2011) — 전문가가 **학습자가 실제로 가는 상태**에서 시범을 보인다

집는 자세는 추측하지 않는다
  원래 정책이 **성공할 때** 그릇을 집는 자세를 재서 그대로 쓴다 (`measure_grasp_pose.py`):
      그릇 중심 기준 손 위치 [0.6, 4.0, 4.6]cm  (테두리)
      퍼짐 x·y 0.3cm 이하 — 매우 일정하다

제약을 지키는가
  · 초기 자세로 돌아가지 않는다 — **물체로 직행**한다
  · 스크립트 동작은 **학습 데이터로만** 쓴다. 실제로 움직이는 것은 학습된 정책이다
  · 움직임이 부드럽도록 속도를 제한하고 목표에 가까울수록 느려지게 했다
"""
import numpy as np
from scipy.spatial.transform import Rotation

import switch_experiment as sx

# ── DART (Laskey et al., 2017): 시범 중 **실행하는 동작에만** 잡음을 섞는다 ─────────
#   기록하는 동작은 전문가의 원래 의도 그대로다. 그러면 로봇이 조금 틀어진 상태가 데이터에 들어가고,
#   그 상태에서 **전문가가 어떻게 바로잡는지**가 함께 기록된다 → 학습된 정책이 작은 실수에서 복구한다.
#   0 이면 끔. 그리퍼(7번째)에는 섞지 않는다.
DART_SIGMA = 0.0
_rng = np.random.default_rng(0)


# 정밀 구간 표시 (2026-10-01): 이 값이 True 인 동안은 DART 잡음을 넣지 않는다.
#   서랍 손잡이 걸기·쥐기·누르며 끌기처럼 1cm 단위 동작에 잡음을 넣으니 시범이 무너졌다
#   (잡음 0.1: 가운데 서랍 107번 중 11번, 위 서랍+그릇 109번 중 2번 / 잡음 0.03 에서도 8/10).
#   잡음은 이동 구간(목표 허용 오차 1.5cm 이상)에만 넣어, 틀어졌다 바로잡는 동작을 배우게 한다.
PRECISE = False


def _exec(a):
    """실제로 실행할 동작 = 전문가 동작 + DART 잡음 (위치·회전만, 이동 구간에서만)."""
    if DART_SIGMA <= 0 or PRECISE:
        return a
    b = a.copy()
    b[:6] = np.clip(b[:6] + _rng.normal(0, DART_SIGMA, 6), -1, 1)
    return b

# 원래 정책이 성공할 때 그릇을 집는 자세 (2026-09-28 실측, n=8)
BOWL_GRASP_OFFSET = np.array([0.006, 0.040, 0.046])

# ⚠️ 물체의 **정지 방향**. 그릇은 바로 서 있을 때 이미 z축으로 90도 돌아가 있다 (실측, 모든 에피소드 동일).
#    위 오프셋은 이 정지 상태에서 잰 것이므로, 기울어진 그릇에 맞추려면
#    **정지 방향 대비 상대 회전**만 적용해야 한다. 절대 방향을 쓰면 90도 어긋난다.
REST_ROT = {"akita_black_bowl_1": Rotation.from_rotvec([0, 0, np.pi / 2])}

# 태스크별: (집을 물체, 놓을 곳, 물체 기준 집기 오프셋)
TASKS = {
    8: ("akita_black_bowl_1", "plate_1", BOWL_GRASP_OFFSET),     # 그릇을 접시에
}


def servo(ep, target_pos, target_mat, grip, max_steps=120, tol=0.008, vmax=0.5, record=None, phase="E"):
    """목표 자세까지 부드럽게 움직인다. 도달하면 True.

    vmax: 한 스텝 최대 동작 크기 (1.0 = 5cm). 0.5 면 2.5cm/스텝 — 사람 팔 속도에 가깝다.
    목표에 가까울수록 비례해서 느려진다 (P 제어). 급정거·급출발이 없다.
    """
    global PRECISE
    prev, PRECISE = PRECISE, tol < 0.015
    try:
        avoid = tol >= 0.015 and HOME_AVOID > 0 and \
            np.linalg.norm(np.asarray(target_pos) - ep.home[0]) >= HOME_AVOID
        if avoid:                                      # 이동 구간: 직선이 초기 자세 근처를 지나면 돌아간다
            w = _home_detour(ep, np.asarray(target_pos, dtype=float))
            if w is not None:
                _servo_loop(ep, w, target_mat, grip, max_steps // 2, 0.03, vmax, record, phase, avoid=True)
        return _servo_loop(ep, target_pos, target_mat, grip, max_steps, tol, vmax, record, phase, avoid=avoid)
    finally:
        PRECISE = prev


# 초기 자세 피해 가기 (2026-10-02)
#   재개 A4→B9 에서 와인 선반 → 그릇으로 곧장 가는 길이 초기 자세 5.6~6.9cm 안을 지나 7cm 제약에 걸렸다
#   (평가 장면 20~49 중 7번, A1→B7 재개 2번). 이동 구간(허용 오차 1.5cm 이상)에서만, 직선이 초기 자세
#   HOME_AVOID 안을 지나면 그 바깥 한 점을 거쳐 간다. 잡기·놓기 같은 정밀 구간에는 쓰지 않는다.
HOME_AVOID = 0.085   # 10/3: 0.12 는 너무 넓어 와인병(처음 자세 바로 아래)을 들어 올리다 막혔다 (새 배치 T2 90/100)


def _keep_off_home(ep, pos, v, margin=0.02):
    """손이 초기 자세 HOME_AVOID+margin 안이면 그쪽으로 다가가는 성분을 빼고 살짝 밀어낸다 (10/2).
    팔은 축마다 따로 속도가 잘려 직선으로 가지 않아, 직선만 보고 돌아가는 방식으로는 A1→B7 재개에서
    그릇을 들고 스토브로 가다 6.5cm 까지 다가간 것을 막지 못했다."""
    h = np.asarray(ep.home[0], dtype=float)
    r = np.asarray(pos, dtype=float) - h
    dist = np.linalg.norm(r)
    if dist >= HOME_AVOID + margin or dist < 1e-6:
        return v
    n = r / dist
    v0 = np.linalg.norm(v)
    radial = float(v @ n)
    if radial < 0:
        v = v - radial * n                              # 안쪽으로 들어가는 성분 제거 → 바깥을 따라 미끄러짐
        if np.linalg.norm(v) < 0.3 * v0:               # 목표가 처음 자세 뒤쪽이라 거의 멈추면: 옆·위로 비켜 간다 (10/3)
            t = np.cross(n, [0.0, 0.0, 1.0])
            if np.linalg.norm(t) < 1e-3:
                t = np.array([1.0, 0.0, 0.0])
            t = t / np.linalg.norm(t)
            up = np.array([0.0, 0.0, 1.0]) - n[2] * n    # 구면을 따라 위로
            esc = t + (up / (np.linalg.norm(up) + 1e-6))
            v = v + esc / (np.linalg.norm(esc) + 1e-6) * 0.5 * v0
    push = (HOME_AVOID + margin - dist) / sx.POS_SCALE  # 안쪽에 있으면 바깥으로
    return np.clip(v + n * min(push, 0.3), -1, 1)


def _home_detour(ep, target):
    h = np.asarray(ep.home[0], dtype=float)
    p = np.asarray(sx.eef_pos(ep.obs), dtype=float)
    d = target - p
    L2 = float(d @ d)
    if L2 < 0.05 ** 2 or np.linalg.norm(p - h) < HOME_AVOID or np.linalg.norm(target - h) < HOME_AVOID:
        return None
    t = float(np.clip((h - p) @ d / L2, 0.0, 1.0))
    q = p + t * d
    if np.linalg.norm(q - h) >= HOME_AVOID:
        return None
    away = q - h
    if np.linalg.norm(away) < 1e-3:                     # 정확히 지나가면 수평으로 직각 방향
        away = np.array([-d[1], d[0], 0.0])
    away[2] = min(away[2], 0.0) if q[2] < h[2] else away[2]
    return h + away / np.linalg.norm(away) * (HOME_AVOID + 0.03)


STALL_STEPS = 8        # PiPER: 이만큼 동안 목표에 1mm·0.6° 도 가까워지지 않으면 막힌 것으로 보고 끝낸다


def _servo_loop(ep, target_pos, target_mat, grip, max_steps, tol, vmax, record, phase, avoid=False):
    # 10/3 PiPER: 손잡이·스토브 손잡이에 손가락이 닿아 목표 1cm 앞에서 막히면 max_steps(60~80) 동안 제자리에 있었다.
    #   이런 정지 구간이 시범 1,408개 중 350개에 있었다 — 학생 모델이 '맴돌기'를 배울 수 있어 일찍 끝낸다
    hist = [] if _is_piper(ep) else None
    for _ in range(max_steps):
        pos = sx.eef_pos(ep.obs)
        dp = target_pos - pos
        rerr = Rotation.from_matrix(target_mat @ ep.obs["robot_state"]["eef"]["mat"].T).as_rotvec()
        if np.linalg.norm(dp) < tol and np.linalg.norm(rerr) < 0.08:
            return True
        if hist is not None:
            hist.append((np.linalg.norm(dp), np.linalg.norm(rerr)))
            if len(hist) > STALL_STEPS and hist[-1 - STALL_STEPS][0] - hist[-1][0] < 0.001 \
                    and hist[-1 - STALL_STEPS][1] - hist[-1][1] < 0.01:
                return False
        v = np.clip(dp / sx.POS_SCALE, -vmax, vmax)
        if avoid:
            v = _keep_off_home(ep, pos, v)
        a = np.concatenate([v, np.clip(rerr / sx.ROT_SCALE, -vmax, vmax), [grip]]).astype(np.float32)
        if record is not None:
            record(ep.obs, a)
        ep.step(_exec(a), phase)
    return False


def hold(ep, grip, n, record=None, phase="E"):
    """제자리에서 그리퍼만 움직인다 (닫기·열기에 몇 스텝이 걸린다). 잡음 없음."""
    global PRECISE
    prev, PRECISE = PRECISE, True
    try:
        _hold_loop(ep, grip, n, record, phase)
    finally:
        PRECISE = prev


def _hold_loop(ep, grip, n, record=None, phase="E"):
    for _ in range(n):
        a = np.array([0, 0, 0, 0, 0, 0, grip], dtype=np.float32)
        if record is not None:
            record(ep.obs, a)
        ep.step(_exec(a), phase)


def obj_rot(ep, name):
    """물체의 현재 방향 (Rotation)."""
    q = ep.inner.sim.data.body_xquat[ep.inner.obj_body_id[name]]   # w, x, y, z
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def grasp_candidates(ep, obj, off, down):
    """테두리를 **어느 쪽에서** 집을지 후보를 만든다.

    왜 (2026-09-28 측정)
      전환 뒤 떨어진 그릇은 8개 중 7개가 **12~35도 기울어져** 있었다.
      원래 정책이 집는 한 점([0.6, 4.0, 4.6]cm)은 **바로 선 그릇** 기준이라 어긋난다.
      그런데 테두리는 원이다 — **어느 쪽이든 집을 수 있다.**

    ⚠️ 두 가지를 같이 맞춰야 한다 (첫 구현에서 둘 다 틀려 0/10 이 나왔다):
      ① **정지 방향 대비 상대 회전**만 쓴다 — 그릇은 바로 서 있어도 90도 돌아가 있다
      ② 테두리의 다른 점을 집으려면 **그리퍼도 같이 돌아야** 한다.
         평행 그리퍼는 테두리 벽을 가로질러 쥐어야 한다. 위치만 돌리고 손을 그대로 두면 헛잡는다.

    후보: 원래 집던 쪽(0도), 반대쪽(180도), 옆(±90도). (위치, 그리퍼 방향)을 함께 돌린다.
    **가장 높은 곳**(위에서 접근하기 가장 쉬운 곳)부터 시도한다.
    """
    R = obj_rot(ep, obj) * REST_ROT.get(obj, Rotation.identity()).inv()   # 정지 대비 기울기
    o = ep.obj_pos(obj)
    D = Rotation.from_matrix(down)
    cands = []
    for th in np.radians([0, 180, 90, -90]):
        Q = R * Rotation.from_rotvec([0, 0, th])
        cands.append((o + Q.apply(off), (Q * D).as_matrix()))
    return sorted(cands, key=lambda c: -c[0][2])


def pick_place(ep, task, record=None, lift=0.10, above=0.08):
    """task 의 물체를 집어 목표 위에 놓는다. (성공, 사용 스텝) 반환.

    record(obs, action) 을 주면 매 스텝 (관측, 동작)을 넘긴다 → 학습 데이터로 모은다.
    """
    obj, dst, off = TASKS[task]
    chk = sx.GoalChecker(ep.r.suite, task)
    n0 = len(ep.log["pos"])
    # 그리퍼 방향: 원래 정책이 집을 때와 같은 '위에서 아래로' 방향 (초기 방향 기준, 평균 10도 안쪽)
    down = ep.home[1].copy()

    hold(ep, -1.0, 6, record)                                   # ① 편다
    got = False
    for g, gmat in grasp_candidates(ep, obj, off, down)[:3]:    # 가장 높은 테두리부터 세 곳까지
        # ② 물체 위로 — 경유점이라 2.5cm 안이면 충분하다 (정밀도는 ③에서)
        if not servo(ep, g + [0, 0, above], gmat, -1.0, tol=0.025, record=record):
            continue
        if not servo(ep, g, gmat, -1.0, tol=0.008, vmax=0.3, record=record):   # ③ 천천히 내려간다
            servo(ep, g + [0, 0, above], gmat, -1.0, tol=0.03, max_steps=30, record=record)
            continue
        got = True
        break
    if not got:
        return False, len(ep.log["pos"]) - n0
    hold(ep, 1.0, 12, record)                                   # ④ 쥔다
    servo(ep, g + [0, 0, lift], down, 1.0, record=record)       # ⑤ 든다
    d = ep.obj_pos(dst)
    tgt = d + off + [0, 0, 0.03]                                # 물체가 목표 바로 위에 오도록
    servo(ep, tgt + [0, 0, above], down, 1.0, record=record)    # ⑥ 목표 위로
    servo(ep, tgt, down, 1.0, vmax=0.3, record=record)          # ⑦ 내려놓을 높이로
    hold(ep, -1.0, 10, record)                                  # ⑧ 놓는다
    servo(ep, sx.eef_pos(ep.obs) + [0, 0, 0.06], down, -1.0, max_steps=20, record=record)  # ⑨ 손을 뺀다
    for _ in range(10):                                         # 물체가 안정될 시간
        if chk(ep.env):
            break
        ep.step(sx.get_libero_dummy_action(), "E")
    return bool(chk(ep.env)), len(ep.log["pos"]) - n0


# ───────────────────────────────────────────────────────────────────────────
# 방향 틀기 (redirect) — 그릇을 **쥔 채** 새 목적지로
# ───────────────────────────────────────────────────────────────────────────
# 원래 정책이 **성공할 때** 그릇이 놓인 위치 (2026-09-28 실측, 태스크당 6회, 퍼짐 1~2cm)
BOWL_PLACE = {
    1: np.array([-0.228, 0.204, 0.933]),   # 스토브 위
    4: np.array([-0.003, -0.222, 1.130]),  # 찬장 위
    8: None,                               # 접시 위 — 접시가 물체라 매번 위치를 읽는다
}


def bowl_place_target(ep, task):
    if task == 8:
        p = ep.obj_pos("plate_1")
        return np.array([p[0], p[1], 0.911])   # 접시 위 그릇 높이 (실측 91.1cm)
    if task == 4 and _is_piper(ep):
        # PiPER 장면은 캐비닛을 옮겼다 (10/3). 고정 좌표(Panda 장면)면 앞 모서리에 놓여 떨어졌다 (0/10) →
        # 캐비닛 윗면 영역에서 읽고, 가운데보다 로봇 쪽 4cm (손이 덜 뻗게)
        m, d = ep.inner.sim.model, ep.inner.sim.data
        c = np.array(d.site_xpos[m.site_name2id("wooden_cabinet_1_top_side")])
        return np.array([c[0] - 0.04, c[1], c[2] + 0.003])
    return BOWL_PLACE[task]


def _is_piper(ep):
    try:
        ep.inner.sim.model.joint_name2id("robot0_joint7")
        return False
    except Exception:
        return True


def grasp_obj(ep, obj, off, record=None, above=0.08):
    """물체가 어디에 어떻게 놓여 있든 집어 든다. 성공하면 True."""
    down = ep.home[1].copy()
    hold(ep, -1.0, 6, record)
    for g, gmat in grasp_candidates(ep, obj, off, down)[:3]:
        if not servo(ep, g + [0, 0, above], gmat, -1.0, tol=0.025, record=record):
            continue
        if not servo(ep, g, gmat, -1.0, tol=0.008, vmax=0.3, record=record):
            servo(ep, g + [0, 0, above], gmat, -1.0, tol=0.03, max_steps=30, record=record)
            continue
        hold(ep, 1.0, 12, record)
        z0 = ep.obj_pos(obj)[2]
        servo(ep, g + [0, 0, 0.10], gmat, 1.0, tol=0.02, record=record)
        if ep.obj_pos(obj)[2] > z0 + 0.04:
            return True
    return False


def secure_hold(ep, obj, record=None):
    """넘겨받은 순간 **정말 쥐고 있는지** 확인한다.

    왜 (2026-09-28 측정)
      전환 시점(쥔 지 3스텝)에 넘겨받아 바로 옮겼더니, 실패의 대부분이
      **그릇이 탁자 높이에 기울지 않은 채 원래 자리 근처**에 있었다 = 옮겨지지 않았다.
      3스텝이면 아직 쥐는 중이라, 빠르게 들면 빠져나간다.
    → 꽉 쥐고 **천천히** 4cm 들어 본다. 물체가 손을 따라 올라오면 쥔 것이다.

    ⚠️ 판정 기준: 손을 4cm 올렸을 때 물체가 **1.5cm 이상** 따라 올라오면 쥔 것으로 본다.
       처음에 2.5cm 로 했다가, 실제로 쥐고 있는데도(2.2~2.4cm 따라옴) 놓친 것으로 판정해
       멀쩡한 그릇을 놓고 다시 집으려다 A1→B8 이 100% → 0% 가 됐다 (2026-09-28).
    """
    hold(ep, 1.0, 5, record)
    z0 = ep.obj_pos(obj)[2]
    mat = ep.obs["robot_state"]["eef"]["mat"].copy()
    servo(ep, sx.eef_pos(ep.obs) + [0, 0, 0.04], mat, 1.0, tol=0.01, vmax=0.2, max_steps=25, record=record)
    return ep.obj_pos(obj)[2] > z0 + 0.015


def carry_place(ep, P, record=None, above=0.10, obj="akita_black_bowl_1", z_safe=None, keepout=()):
    """쥔 그릇을 P 에 놓는다. 들어 올리며 목적지로 향하고, 위에서 천천히 내려 놓는다.

    ⚠️ 손−그릇 관계를 **지금 실제로 잰 값**으로 쓴다.
       정책이 쥔 순간의 관계는 [−2.6, 4.8, 1.5]cm 쯤인데, 들어 올린 뒤 잰 집기 오프셋은
       [0.6, 4.0, 4.6]cm 라 **3cm 씩 다르다** (2026-09-28). 가정한 값을 쓰면 놓는 자리가 어긋난다.
    """
    hand0 = sx.eef_pos(ep.obs)
    mat = ep.obs["robot_state"]["eef"]["mat"].copy()   # 쥔 방향 그대로 (그릇이 기울지 않게)
    rel = hand0 - ep.obj_pos(obj)                     # 지금 실제 손−그릇 관계
    tgt = P + rel + [0, 0, 0.015]
    if _is_piper(ep):
        # PiPER (10/3): 밑동 가까이에서 손을 똑바로 높이 올리면 팔이 접히는 한계(3번 관절)에 걸려 손이 30cm
        # 앞으로 튀었다 (캐비닛 위 0/10). 목표 쪽으로 나아가며 닿는 높이로 올라가 위에서 내려놓는다
        import piper_sim.ik as pik
        # 그릇은 수평만 유지되면 되므로 그릇 중심을 지나는 수직축으로 손을 돌린 자세(30° 간격)도 후보.
        # 쥔 손이 로봇 쪽으로 18° 기울어 있으면 캐비닛 위에서 그 방향으로는 손이 닿지 않았다
        ab = min(above, 0.06)
        best = None
        for psi in range(0, 360, 30):
            Rz = Rotation.from_euler("z", psi, degrees=True).as_matrix()
            t, M = P + Rz @ rel + [0, 0, 0.015], Rz @ mat
            # 지나가는 높이: 놓는 손 높이 + 6cm. +2cm 로는 손 아래 8cm 에 매달린 그릇 바닥이 캐비닛 윗면 모서리에
            # 걸려 떨어졌다 (재개 A4→B9→A4 장면 1004 등 4/10 실패)
            zs = max(hand0[2], t[2] + 0.06, z_safe or 0.0)
            b = pik.plan_branch(ep, [np.array([t[0], t[1], max(zs, t[2] + ab)]), t + [0, 0, ab], t], [M], n_seeds=4)
            if b is not None and (best is None or b[0] > best[0] + np.radians(2)):
                best = (b[0], t, M, zs)
        if best is not None and pik.transit(ep, best[1], best[2], 1.0, best[3], record=record, above=ab, tol=0.008,
                                            keepout=keepout):
            hold(ep, -1.0, 10, record)
            servo(ep, sx.eef_pos(ep.obs) + [0, 0, 0.06], best[2], -1.0, max_steps=20, record=record)
            return
    # ① 들어 올리며 목적지 쪽으로. 높은 목적지(찬장 위 113cm)는 모서리에 걸리지 않게 먼저 충분히 올린다
    lift_z = max(hand0[2], tgt[2]) + above
    servo(ep, np.array([hand0[0], hand0[1], lift_z]), mat, 1.0, tol=0.03, record=record)
    servo(ep, tgt + [0, 0, above], mat, 1.0, tol=0.02, record=record)   # ② 목적지 위
    servo(ep, tgt, mat, 1.0, tol=0.008, vmax=0.25, record=record)       # ③ 천천히 내려놓는 높이로
    hold(ep, -1.0, 10, record)                                          # ④ 놓는다
    servo(ep, sx.eef_pos(ep.obs) + [0, 0, 0.06], mat, -1.0, max_steps=20, record=record)  # ⑤ 손을 뺀다


def redirect(ep, b_task, record=None, obj="akita_black_bowl_1"):
    """⭐ 그릇을 **쥔 채** 새 목적지로 방향을 틀어 놓는다.

    "아, 그거 접시 말고 스토브에 놔줘" — 사람이라면 **놓지 않고** 그냥 방향을 튼다.
    그런데 정책은 '지시가 바뀌면 일단 놓는' 반사가 있다 (필요한 물체도 10/10 놓음).
    이 시범은 **놓지 않고 이어서 가는** 행동을 보여 준다.

    넘겨받은 순간 제대로 쥐고 있지 않으면(아직 쥐는 중) 바로 선 그릇을 다시 집는다.

    제약: 초기 자세로 돌아가지 않는다. 멈춤 없이 들어 올리며 바로 목적지로 향한다.
    """
    chk = sx.GoalChecker(ep.r.suite, b_task)
    n0 = len(ep.log["pos"])
    if not secure_hold(ep, obj, record):
        if _is_piper(ep):
            # PiPER: 전환 순간 미끄러진 그릇은 PiPER 집기(놓을 자리까지 보고 쥐는 방향 고르기)로 다시 집는다.
            #   옛 grasp_obj 로는 다시 집지 못해 A4→B1 장면 1001 실패
            import task_experts as te
            if not te.grasp_bowl_safe(ep, obj, record=record, dest=bowl_place_target(ep, b_task)):
                return False, len(ep.log["pos"]) - n0
        elif not grasp_obj(ep, obj, BOWL_GRASP_OFFSET, record):
            return False, len(ep.log["pos"]) - n0
    carry_place(ep, bowl_place_target(ep, b_task), record)
    for _ in range(10):
        if chk(ep.env):
            break
        ep.step(sx.get_libero_dummy_action(), "E")
    return bool(chk(ep.env)), len(ep.log["pos"]) - n0


# ───────────────────────────────────────────────────────────────────────────
# 내려놓기 (put down) — 쥔 물체를 **바로 세워** 근처에 둔다
# ───────────────────────────────────────────────────────────────────────────
TABLE_Z = 0.898   # 탁자 위 그릇 높이 (정지 상태 실측 89.8cm)


def put_down(ep, obj="akita_black_bowl_1", record=None, away=None):
    """쥔 물체를 **바로 세워** 탁자에 내려놓고 손을 뺀다.

    왜 (2026-09-28)
      정책은 **상황에 따라** 놓을지 판단한다 — 같은 그릇이 필요하면 쥐고(90~100% 성공),
      손이 필요하면 놓는다. 실패는 **들고 있던 것을 내려놓고 다른 조작을 해야 할 때** 난다
      (접시 밀기 0%, 와인병 0%, 서랍+그릇 0%).
      정책이 스스로 놓으면 그릇이 **12~35도 기울어진 채** 엉뚱한 곳(접시 모서리 등)에 떨어진다.

    이 시범: 제자리 근처에서 **탁자 높이까지 내려가 수평으로** 놓는다. 사람이 컵을 내려놓듯.
    '아무 데나 놔도 된다'는 허용 범위 안이고, 초기 자세로 가지 않는다.

    away: (선택) 다음 작업을 방해하지 않게 피할 방향 (xy, m). 주면 그쪽 반대로 조금 옮겨 놓는다.
    """
    if not secure_hold(ep, obj, record):
        return False
    hand = sx.eef_pos(ep.obs)
    mat = ep.obs["robot_state"]["eef"]["mat"].copy()
    rel = hand - ep.obj_pos(obj)
    spot = ep.obj_pos(obj).copy()
    if away is not None:
        d = spot[:2] - np.asarray(away)[:2]
        if np.linalg.norm(d) > 1e-6:
            spot[:2] += 0.06 * d / np.linalg.norm(d)
    spot[2] = TABLE_Z
    tgt = spot + rel + [0, 0, 0.012]
    servo(ep, tgt + [0, 0, 0.05], mat, 1.0, tol=0.02, record=record)
    servo(ep, tgt, mat, 1.0, tol=0.008, vmax=0.2, record=record)        # 천천히 닿게
    hold(ep, -1.0, 10, record)                                          # 놓는다
    servo(ep, sx.eef_pos(ep.obs) + [0, 0, 0.08], mat, -1.0, max_steps=25, record=record)  # 손을 뺀다
    R = obj_rot(ep, obj) * REST_ROT.get(obj, Rotation.identity()).inv()
    tilt = np.degrees(np.arccos(np.clip(R.apply([0, 0, 1])[2], -1, 1)))
    return tilt < 10


# ───────────────────────────────────────────────────────────────────────────
# 접근 (approach) — 작업 대상 근처의 '정책에게 익숙한 자세'까지 데려간다
# ───────────────────────────────────────────────────────────────────────────
_APPROACH = None


def approach_pose(task, path="outputs/expert/approach.json"):
    """정책이 성공할 때 작업 대상 근처에 도착한 순간의 자세 (measure_approach.py 로 잰 값)."""
    global _APPROACH
    if _APPROACH is None:
        import json
        _APPROACH = {int(k): v for k, v in json.load(open(path)).items()}
    v = _APPROACH[task]
    return np.array(v["pos"]), Rotation.from_rotvec(v["rotvec"]).as_matrix()


def approach(ep, b_task, record=None, clear=0.06):
    """⭐ 손을 B 의 작업 대상 근처, **정책이 원래 성공하던 궤적 위의 한 점**으로 데려간다.

    왜 (2026-09-28)
      정책은 교란된 자세에서 무너진다. 그런데 자기 궤적 위에 있으면 잘 한다.
      전문가가 **그 궤적 위의 한 점**까지 부드럽게 데려가 주면, 거기서부터는 정책이 마무리한다.

    제약
      목표 자세는 **초기 자세가 아니라 작업 대상 근처**다 (정책 궤적의 중간 지점).
      높이를 조금 올려 물체를 피하며, 멈춤 없이 한 번에 간다.
    """
    pos, mat = approach_pose(b_task)
    here = sx.eef_pos(ep.obs)
    mid = (here + pos) / 2
    mid[2] = max(here[2], pos[2]) + clear            # 넘어가는 길에 물체를 치지 않게
    servo(ep, mid, mat, -1.0, tol=0.04, record=record)
    return servo(ep, pos, mat, -1.0, tol=0.015, vmax=0.4, record=record)


# ───────────────────────────────────────────────────────────────────────────
# ⭐ 궤적 재생 전문가 — 정책이 **원래 성공한 조작**을 어디서든 재생한다
# ───────────────────────────────────────────────────────────────────────────
_DEMOS = None


def load_demos(path="outputs/expert/demos.pkl"):
    global _DEMOS
    if _DEMOS is None:
        import pickle
        _DEMOS = pickle.load(open(path, "rb"))
    return _DEMOS


def demo_structure(d):
    """녹화 궤적의 구조를 읽는다.

    obj  : 가장 많이 움직인 물체 (조작 대상). 거의 안 움직이면 None → 가구 조작 (서랍·스토브)
    kind : 'fixture'(가구) · 'push'(밀기, 들리지 않음) · 'pick'(집어 옮기기, 들림)
    g    : 'pick' 이면 물체가 처음 들린 스텝
    c    : 첫 **접촉** 스텝 — 그리퍼가 처음 닫히거나 물체가 처음 움직인 때 중 이른 것
    """
    moves = {k: float(np.linalg.norm(v[-1] - v[0])) for k, v in d["objs"].items()}
    obj = max(moves, key=moves.get)
    grip = d["action"][:, 6]
    closes = np.nonzero((grip[:-1] <= 0) & (grip[1:] > 0))[0]
    first_close = int(closes[0] + 1) if len(closes) else len(grip) - 1
    if moves[obj] < 0.02:
        return {"obj": None, "kind": "fixture", "g": None, "c": first_close}
    o = d["objs"][obj]
    moved = np.nonzero(np.linalg.norm(o - o[0], axis=1) > 0.01)[0]
    first_move = int(moved[0]) if len(moved) else len(o) - 1
    lifted = np.nonzero(o[:, 2] > o[0, 2] + 0.02)[0]
    if len(lifted):
        return {"obj": obj, "kind": "pick", "g": int(lifted[0]), "c": min(first_close, first_move)}
    return {"obj": obj, "kind": "push", "g": None, "c": min(first_close, first_move)}


def pick_demo(ep, task):
    """지금 장면과 가장 비슷한 녹화 — 조작 대상의 시작 위치가 가장 가까운 것 (옮길 거리가 최소)."""
    ds = load_demos()[task]
    best, bd = None, 1e9
    for d in ds:
        s = demo_structure(d)
        if s["obj"] is None:
            return d, s                          # 가구 조작은 아무 녹화나 같다
        dist = float(np.linalg.norm(ep.obj_pos(s["obj"]) - d["objs"][s["obj"]][0]))
        if dist < bd:
            best, bd = (d, s), dist
    return best


def replay_offsets(d, s, delta):
    """스텝마다 녹화 궤적을 얼마나 옮길지.

    'push'    : 전체를 물체 기준으로 옮긴다 (밀어낸 뒤 위치도 물체 기준)
    'pick'    : 집기 전 — 손이 물체에 가까울수록 물체 기준 (멀면 0: 그 전에 하는 가구 조작은 그대로)
                집은 뒤 — 목적지에 가까울수록 0 으로 (목적지는 고정이므로)
    'fixture' : 옮기지 않는다
    """
    n = len(d["pos"])
    off = np.zeros((n, 3))
    if s["kind"] == "fixture":
        return off
    if s["kind"] == "push":
        off[:] = delta
        return off
    o = d["objs"][s["obj"]]
    end = d["pos"][-1]
    for t in range(n):
        if t < s["g"]:
            dn = np.linalg.norm(d["pos"][t] - o[t])
            w = np.clip(1 - (dn - 0.05) / 0.15, 0, 1)
        else:
            de = np.linalg.norm(d["pos"][t] - end)
            w = np.clip((de - 0.05) / 0.15, 0, 1)
        off[t] = w * delta
    return off


MIN_HOME = 0.10   # ⛔ 시작점은 초기 자세에서 10cm 이상 (9/18: 7cm 안이면 리셋과 구분 안 됨)


def replay_plan(ep, task, radius=0.15):
    """재생할 궤적(물체 위치에 맞게 옮긴 것)과 시작 스텝 st.

    st = 첫 접촉점 반경 안에 처음 들어오면서 **초기 자세에서 MIN_HOME 이상** 떨어진 스텝.
    ⚠️ 와인병(T9)은 초기 자세에서 17~19cm 밖에 안 떨어져 있어, 반경 조건만 쓰면
       초기 자세 4~7cm 안의 점을 고른다 (= 리셋, 제약 위반). 2026-09-28 에 잡았다
    """
    d, s = pick_demo(ep, task)
    delta = np.zeros(3) if s["obj"] is None else ep.obj_pos(s["obj"]) - d["objs"][s["obj"]][0]
    P = d["pos"] + replay_offsets(d, s, delta)
    R = [Rotation.from_rotvec(r).as_matrix() for r in d["rotvec"]]
    grip = d["action"][:, 6]
    c = s["c"]
    near = np.linalg.norm(P - P[c], axis=1) < radius
    far = np.linalg.norm(P - ep.home[0], axis=1) >= MIN_HOME
    ok = np.nonzero((near & far)[:c + 1])[0]
    st = int(ok[0]) if len(ok) else c
    return P, R, grip, st


def _go_to_start(ep, pos, mat, record=None, clear=0.05):
    here = sx.eef_pos(ep.obs)
    mid = (here + pos) / 2
    mid[2] = max(here[2], pos[2]) + clear
    servo(ep, mid, mat, -1.0, tol=0.04, record=record)
    return servo(ep, pos, mat, -1.0, tol=0.015, vmax=0.4, record=record)


def demo_approach(ep, task, record=None, radius=0.15):
    """재생 전문가의 **앞부분만** — 녹화 궤적 위, 첫 접촉점 근처까지 데려가고 멈춘다.
    그 뒤는 정책이 한다. 고정 접근 자세(approach.json)와 달리 **물체가 있는 곳 기준**이라
    집어 옮기는 태스크(와인병)에도 쓸 수 있다."""
    P, R, grip, st = replay_plan(ep, task, radius)
    return _go_to_start(ep, P[st], R[st], record)


def replay(ep, task, record=None, lead=3, lag_tol=0.03, max_steps=400):
    """⭐ 정책이 원래 성공한 조작을 **지금 자리에서** 재생한다.

    ① 녹화 중 지금 장면과 가장 비슷한 것을 고른다
    ② 첫 접촉점 근처(15cm 안)에 처음 들어온 스텝 s 를 찾는다
    ③ 지금 손에서 그 점까지 **들어 올려 넘어가며** 부드럽게 이동 (초기 자세를 거치지 않는다)
    ④ 거기서부터 녹화 궤적을 **닫힌 고리로 따라간다** (lead 스텝 앞을 목표로 P 제어).
       팔이 3cm 넘게 뒤처지면 기다린다 → 그리퍼 명령이 팔 위치와 어긋나지 않는다
    """
    chk = sx.GoalChecker(ep.r.suite, task)
    P, R, grip, st = replay_plan(ep, task)
    n = len(P)

    # ③ 시작점까지 — 물체를 치지 않게 한 번 올라갔다 내려온다
    _go_to_start(ep, P[st], R[st], record)

    # ④ 따라간다
    t, used = st, 0
    while t < n and used < max_steps:
        k = min(t + lead, n - 1)
        cur = sx.eef_pos(ep.obs)
        rerr = Rotation.from_matrix(R[k] @ ep.obs["robot_state"]["eef"]["mat"].T).as_rotvec()
        a = np.concatenate([np.clip((P[k] - cur) / sx.POS_SCALE, -1, 1),
                            np.clip(rerr / sx.ROT_SCALE, -1, 1), [grip[t]]]).astype(np.float32)
        if record is not None:
            record(ep.obs, a)
        ep.step(_exec(a), "E")
        used += 1
        if chk(ep.env):
            return True, used
        if np.linalg.norm(P[t] - sx.eef_pos(ep.obs)) < lag_tol:
            t += 1
    for _ in range(20):
        if chk(ep.env):
            return True, used
        ep.step(np.array([0, 0, 0, 0, 0, 0, grip[-1]], dtype=np.float32), "E")
    return bool(chk(ep.env)), used


def min_home_dist(ep, since=0):
    """since 스텝 이후 손이 초기 자세에 가장 가까이 간 거리 (m). ⛔ 제약 점검용 — 7cm 안이면 리셋과 구분 안 됨."""
    P = np.array(ep.log["pos"][since:])
    return float(np.linalg.norm(P - ep.home[0], axis=1).min()) if len(P) else float("nan")
