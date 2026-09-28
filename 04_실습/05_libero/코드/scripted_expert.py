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
    for _ in range(max_steps):
        pos = sx.eef_pos(ep.obs)
        dp = target_pos - pos
        rerr = Rotation.from_matrix(target_mat @ ep.obs["robot_state"]["eef"]["mat"].T).as_rotvec()
        if np.linalg.norm(dp) < tol and np.linalg.norm(rerr) < 0.08:
            return True
        a = np.concatenate([np.clip(dp / sx.POS_SCALE, -vmax, vmax),
                            np.clip(rerr / sx.ROT_SCALE, -vmax, vmax), [grip]]).astype(np.float32)
        if record is not None:
            record(ep.obs, a)
        ep.step(a, phase)
    return False


def hold(ep, grip, n, record=None, phase="E"):
    """제자리에서 그리퍼만 움직인다 (닫기·열기에 몇 스텝이 걸린다)."""
    for _ in range(n):
        a = np.array([0, 0, 0, 0, 0, 0, grip], dtype=np.float32)
        if record is not None:
            record(ep.obs, a)
        ep.step(a, phase)


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
    return BOWL_PLACE[task]


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


def carry_place(ep, P, record=None, above=0.10, obj="akita_black_bowl_1"):
    """쥔 그릇을 P 에 놓는다. 들어 올리며 목적지로 향하고, 위에서 천천히 내려 놓는다.

    ⚠️ 손−그릇 관계를 **지금 실제로 잰 값**으로 쓴다.
       정책이 쥔 순간의 관계는 [−2.6, 4.8, 1.5]cm 쯤인데, 들어 올린 뒤 잰 집기 오프셋은
       [0.6, 4.0, 4.6]cm 라 **3cm 씩 다르다** (2026-09-28). 가정한 값을 쓰면 놓는 자리가 어긋난다.
    """
    hand0 = sx.eef_pos(ep.obs)
    mat = ep.obs["robot_state"]["eef"]["mat"].copy()   # 쥔 방향 그대로 (그릇이 기울지 않게)
    rel = hand0 - ep.obj_pos(obj)                     # 지금 실제 손−그릇 관계
    tgt = P + rel + [0, 0, 0.015]
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
        if not grasp_obj(ep, obj, BOWL_GRASP_OFFSET, record):
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
