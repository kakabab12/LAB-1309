"""
태스크별 스크립트 전문가 — 원래 모델이 혼자서도 약한 태스크를 거의 100% 해내는 시범 만들기 (2026-09-28~)

왜
  목표는 모든 조건 85% 이상이다. 그런데 원래 SmolVLA는 전환 없이 혼자 할 때도
  서랍 열기 60%, 서랍+그릇 50%, 치즈 50%, 와인병 50%다 (커뮤니티 재현도 LIBERO-Goal 81%).
  기본 실력을 먼저 올려야 하므로, 이 태스크들을 거의 100% 해내는 전문가를 만들어 시범으로 쓴다.

⚠️ scripted_expert.py 는 돌고 있는 LoRA 5차 파이프라인이 쓰므로 고치지 않는다. 여기서 가져다 쓴다.

서랍 (T0 가운데, T3 위)
  미닫이 관절 wooden_cabinet_1_{middle,top}_level, 열면 +y, 최대 0.16m.
  손잡이는 세로 막대 (가운데 g29, 위 g18). 원래 모델의 성공 궤적을 따라 한다:
  그리퍼를 벌린 채 뒤 손가락을 손잡이 뒤에 걸고 +y 로 당긴다.
   - 위 서랍: 손 방향은 처음 자세 그대로, 손잡이 기준 [0.003, +0.05, +0.018] 에서 당김
   - 가운데 서랍: 위 서랍 손잡이를 피하려고 손목을 기울인다 (euler xyz [-128, 0, -177]),
     손잡이 기준 [0.005, -0.013, +0.018] 에서 당김
"""
import numpy as np
from scipy.spatial.transform import Rotation

import scripted_expert as se
import switch_experiment as sx


# ───────────────────────────────────────────────────────────────────────────
# 2026-10-01: 손목 방향을 정할 때 평행 그리퍼의 대칭을 쓴다
#   평행 그리퍼는 손끝 축으로 180도 돌려도 같은 모양이다. 목표 방향 M 과 M·Rz(180) 중
#   **지금 손목에 가까운 쪽**을 고른다. 전환 상황에서 그릇을 쥔 뒤 손목이 목표와 180도 반대로
#   시작하면, 먼 쪽으로 돌다가 관절 한계에 걸려 손이 손잡이까지 못 갔다 (A8→B0 0/10)
# ───────────────────────────────────────────────────────────────────────────
_RZ180 = Rotation.from_euler("z", 180, degrees=True).as_matrix()


J7_LIMIT = 2.90        # 7번 관절(손을 빙글 돌리는 마지막 관절) 한계, 라디안
J7_MARGIN = 0.90       # 손을 기울이는 회전도 7번 관절이 나눠 맡아서, 예측보다 더 돈다 (0.3 으로는 2.21 예측이 실제 2.90)


# 로봇별 "손을 빙글 돌리는 마지막 관절" (10/3 PiPER 추가). 손을 접근축으로 θ 돌리면 이 관절이 같은 부호로 돈다
#   Panda: joint7, 한계 ±2.90 / PiPER: joint6, 한계 ±2.094(±120°, 공식 URDF), 실측 0.86θ
LAST_JOINT = {"panda": ("robot0_joint7", 2.90, 0.90), "piper": ("robot0_joint6", 2.094, 0.45)}


def _robot(ep):
    m = ep.inner.sim.model
    try:
        m.joint_name2id("robot0_joint7")
        return "panda"
    except Exception:
        return "piper"


def _j7(ep):
    m, d = ep.inner.sim.model, ep.inner.sim.data
    return float(d.qpos[m.jnt_qposadr[m.joint_name2id(LAST_JOINT[_robot(ep)][0])]])


def _jlimit(ep):
    """마지막 관절이 쓸 수 있는 범위 (한계 − 여유)."""
    _, lim, margin = LAST_JOINT[_robot(ep)]
    return lim - margin


def _aim(ep, mat):
    """목표 방향 mat 과 180도 돌린 것 중 고른다.
    ① 7번 관절이 한계(±2.9)에서 0.3 이상 여유가 남는 쪽 ② 그중 덜 도는 쪽.
    손을 자기 축으로 돌린 만큼 7번 관절도 거의 같이 변한다 (+30도 → +27도, 실측).
    처음엔 ②만 봤더니, 그릇을 집느라 이미 2.18 까지 돈 손목이 서랍을 열려고 더 돌다
    한계 2.90 에 걸려 손잡이 1cm 앞에서 멈췄다 (A8→B0 0/10)."""
    cur = ep.obs["robot_state"]["eef"]["mat"]
    q7 = _j7(ep)
    cands = []
    for cand in (mat, mat @ _RZ180):
        rel = Rotation.from_matrix(cur.T @ cand)
        pred = q7 + rel.as_rotvec()[2]
        ok = abs(pred) < _jlimit(ep)
        cands.append((not ok, rel.magnitude(), abs(pred), cand))
    cands.sort(key=lambda c: (c[0], c[1] if not c[0] else c[2]))
    return cands[0][3]

DRAWER = {
    "middle": {"handle": "wooden_cabinet_1_g29", "joint": "wooden_cabinet_1_middle_level",
               "euler": None, "euler_deg": [-128, 0, -177],
               "path": [[0.0, 0.07, 0.055], [-0.005, -0.013, 0.045]], "grip": [0.005, -0.013, 0.018]},
    "top": {"handle": "wooden_cabinet_1_g18", "joint": "wooden_cabinet_1_top_level",
            "euler_deg": None,
            "path": [[0.003, 0.055, 0.08]], "grip": [0.003, 0.05, 0.018]},
}
PULL = 0.15


def handle_pos(ep, which):
    m, d = ep.inner.sim.model, ep.inner.sim.data
    return np.array(d.geom_xpos[m.geom_name2id(DRAWER[which]["handle"])])


def drawer_qpos(ep, which):
    m, d = ep.inner.sim.model, ep.inner.sim.data
    return float(d.qpos[m.jnt_qposadr[m.joint_name2id(DRAWER[which]["joint"])]])


def open_drawer(ep, which, record=None, pull=PULL):
    """서랍을 연다. 연 거리(m)를 돌려준다 (관절 값의 부호를 바꾼 것, 0.16 이 끝)."""
    cfg = DRAWER[which]
    mat = (Rotation.from_euler("xyz", cfg["euler_deg"], degrees=True).as_matrix()
           if cfg["euler_deg"] is not None else ep.home[1].copy())
    h = handle_pos(ep, which)
    se.hold(ep, -1.0, 4, record)
    here = sx.eef_pos(ep.obs)
    first = h + cfg["path"][0]
    mid = (here + first) / 2
    mid[2] = max(here[2], first[2]) + 0.03
    se.servo(ep, mid, mat, -1.0, tol=0.04, record=record)
    for p in cfg["path"]:
        se.servo(ep, h + p, mat, -1.0, tol=0.012, vmax=0.4, record=record)
    g = h + cfg["grip"]
    se.servo(ep, g, mat, -1.0, tol=0.008, vmax=0.25, record=record)
    # 당긴다 — 손잡이를 따라가도록 조금씩
    se.servo(ep, g + [0, pull, 0], mat, -1.0, tol=0.01, vmax=0.25, max_steps=150, record=record)
    # 손을 뺀다: 위로 조금
    se.servo(ep, sx.eef_pos(ep.obs) + [0, 0.03, 0.06], mat, -1.0, tol=0.02, max_steps=30, record=record)
    return -drawer_qpos(ep, which)


# ───────────────────────────────────────────────────────────────────────────
# 2026-09-28 고침: 손잡이는 세로가 아니라 x 방향 가로 막대다 (반길이 4.4cm, 굵기 1.6cm).
#   막대와 서랍 앞판 사이 틈은 1.6cm, 양끝에 받침이 있다. 손가락은 y 방향으로 벌어진다.
#   → 손가락을 반쯤 오므린 채(한쪽 1.8cm) 위에서 내려가 막대를 앞뒤로 감싸고, 쥐고, 당긴다.
#   가운데 서랍도 같다 — 내려가는 길에 두 손가락이 위 서랍 손잡이를 앞뒤로 비껴 지나간다.
# 첫 시도(모델 궤적 흉내, 벌린 채 걸기)는 위·가운데 모두 0/10 이었다.
# ───────────────────────────────────────────────────────────────────────────
PARTIAL = 0.2        # 그리퍼 반쯤: 손가락 관절 ≈ 0.010 → 손끝이 가운데서 약 1.8cm
GRIP_Z = 0.012       # 쥐는 높이: grip site 가 막대 중심보다 1.2cm 위 (손끝이 막대 높이)


def open_drawer_pinch(ep, which, record=None, pull=PULL, above=0.10):
    """서랍 손잡이를 위에서 감싸 쥐고 당긴다. 연 거리(m)를 돌려준다."""
    mat = ep.home[1].copy()
    h = handle_pos(ep, which)
    se.hold(ep, -1.0, 3, record)
    here = sx.eef_pos(ep.obs)
    top = h + [0, 0, above]
    mid = (here + top) / 2
    mid[2] = max(here[2], top[2]) + 0.02
    se.servo(ep, mid, mat, -1.0, tol=0.04, record=record)
    se.servo(ep, top, mat, PARTIAL, tol=0.01, vmax=0.4, record=record)
    se.hold(ep, PARTIAL, 6, record)
    se.servo(ep, h + [0, 0, GRIP_Z], mat, PARTIAL, tol=0.006, vmax=0.2, max_steps=80, record=record)
    se.hold(ep, 1.0, 8, record)
    g = sx.eef_pos(ep.obs)
    se.servo(ep, g + [0, pull, 0], mat, 1.0, tol=0.01, vmax=0.2, max_steps=150, record=record)
    se.hold(ep, -1.0, 6, record)
    se.servo(ep, sx.eef_pos(ep.obs) + [0, 0.02, 0.08], mat, -1.0, tol=0.02, max_steps=30, record=record)
    return -drawer_qpos(ep, which)


# 2026-09-28 고침 2: 그리퍼 값은 폭 목표가 아니다 (0.2 를 주면 결국 완전히 닫힌다).
#   → 손가락 관절 값을 되먹임으로 맞춘다. 완전히 닫아도 손끝 간격이 1.8cm 라 막대(1.6cm)를
#     꽉 쥐지는 못하고 감싸기만 한다 → 뒤 손가락이 막대 뒤에 걸려 당겨진다.
#   손끝은 grip site 보다 1.2cm 아래. 처음엔 손끝이 막대 윗면에 겨우 닿아 미끄러졌다.
TIP_BELOW = 0.012


def grip_toward(ep, q_target, k=250.0):
    q = float(ep.obs["robot_state"]["gripper"]["qpos"][0])
    return float(np.clip((q - q_target) * k, -1.0, 1.0))   # + 닫기, − 열기


def servo_q(ep, target_pos, target_mat, q_target, max_steps=120, tol=0.008, vmax=0.5, record=None, phase="E"):
    """servo 와 같지만 그리퍼는 손가락 관절 목표값으로 맞춘다."""
    for _ in range(max_steps):
        pos = sx.eef_pos(ep.obs)
        dp = target_pos - pos
        rerr = Rotation.from_matrix(target_mat @ ep.obs["robot_state"]["eef"]["mat"].T).as_rotvec()
        g = grip_toward(ep, q_target)
        if np.linalg.norm(dp) < tol and np.linalg.norm(rerr) < 0.08 and abs(g) < 0.3:
            return True
        a = np.concatenate([np.clip(dp / sx.POS_SCALE, -vmax, vmax),
                            np.clip(rerr / sx.ROT_SCALE, -vmax, vmax), [g]]).astype(np.float32)
        if record is not None:
            record(ep.obs, a)
        ep.step(se._exec(a), phase)
    return False


def open_drawer_hook(ep, which, record=None, pull=PULL, above=0.10, q_open=0.008, tip_depth=0.006):
    """서랍 손잡이를 손가락으로 감싸 걸고 당긴다. 연 거리(m)를 돌려준다.

    ① 손잡이 위 10cm 로 (손가락 반쯤: 관절 0.008 → 손끝이 가운데서 ±1.7cm)
    ② 손끝이 막대 가운데보다 tip_depth 아래가 될 때까지 천천히 내려간다
    ③ 닫는다 (손끝 ±0.9cm → 막대를 앞뒤로 감쌈)
    ④ +y 로 당긴다 → ⑤ 놓고 위로
    """
    mat = ep.home[1].copy()
    h = handle_pos(ep, which)
    here = sx.eef_pos(ep.obs)
    top = h + [0, 0, above]
    mid = (here + top) / 2
    mid[2] = max(here[2], top[2]) + 0.02
    se.servo(ep, mid, mat, -1.0, tol=0.04, record=record)
    servo_q(ep, top, mat, q_open, tol=0.008, vmax=0.4, record=record)
    servo_q(ep, h + [0, 0, TIP_BELOW - tip_depth], mat, q_open, tol=0.005, vmax=0.15, max_steps=80, record=record)
    se.hold(ep, 1.0, 6, record)
    g = sx.eef_pos(ep.obs)
    se.servo(ep, g + [0, pull, 0.005], mat, 1.0, tol=0.01, vmax=0.2, max_steps=150, record=record)
    se.hold(ep, -1.0, 6, record)
    se.servo(ep, sx.eef_pos(ep.obs) + [0, 0.02, 0.08], mat, -1.0, tol=0.02, max_steps=30, record=record)
    return -drawer_qpos(ep, which)


# 2026-09-28 고침 3: 원래 모델이 실제로 여는 방법 (diag_drawer_policy.py, T3 ep100)
#   그리퍼를 **완전히 벌린 채**, 손은 막대보다 4.7cm 앞·1.7cm 위. 그러면 뒤 손가락 끝이
#   막대 **윗면 바로 위**(y 오차 0.1cm, 높이 +0.6cm)에 온다. 손가락 끝으로 윗면을 누르며 +y 로 당기면
#   마찰로 서랍이 손과 함께 나온다 (7스텝에 15cm). 걸거나 쥐는 것이 아니었다.
PRESS = {"top": [0.007, 0.047, 0.017]}


def open_drawer_press(ep, which, record=None, pull=0.16, press=0.006, above=0.07):
    mat = ep.home[1].copy()
    h = handle_pos(ep, which)
    rel = np.array(PRESS[which])
    se.hold(ep, -1.0, 3, record)
    here = sx.eef_pos(ep.obs)
    pre = h + rel + [0, 0, above]
    mid = (here + pre) / 2
    mid[2] = max(here[2], pre[2]) + 0.02
    se.servo(ep, mid, mat, -1.0, tol=0.04, record=record)
    se.servo(ep, pre, mat, -1.0, tol=0.01, vmax=0.4, record=record)
    se.servo(ep, h + rel, mat, -1.0, tol=0.006, vmax=0.2, max_steps=60, record=record)
    # 윗면을 누르며 당긴다: 목표 높이를 press 만큼 낮춰 둔다
    g = h + rel - [0, 0, press]
    for k in range(1, 9):
        se.servo(ep, g + [0, pull * k / 8, 0], mat, -1.0, tol=0.006, vmax=0.3, max_steps=12, record=record)
    se.servo(ep, sx.eef_pos(ep.obs) + [0, 0.02, 0.07], mat, -1.0, tol=0.02, max_steps=30, record=record)
    return -drawer_qpos(ep, which)


# 가운데 서랍 (diag_drawer_policy.py, T0 ep102):
#   손목을 90도 돌려 손가락을 x 방향으로 벌리고(±4.5cm, 막대 양끝 받침 바깥),
#   두 손가락 끝을 막대 **뒤쪽 틈**(손잡이보다 1~2cm 안쪽)에 넣어 걸고 당긴다.
#   손 위치: 손잡이 기준 [-0.005, -0.011, +0.015]. 손 방향은 성공 궤적(T0 ep102, 104스텝)에서 그대로 가져온다.
def _demo_mat(task, ep_idx, step):
    D = se.load_demos()
    d = next(x for x in D[task] if x["episode"] == ep_idx)
    return Rotation.from_rotvec(d["rotvec"][step]).as_matrix()


def rotate_staged(ep, target, record=None, n=4, lift=0.0):
    """손 방향을 target 까지 n 단계로 나눠 돌린다 (그 자리에서, 필요하면 살짝 들고).
    한 번에 150도 넘게 돌리면 로봇이 짧은 쪽으로 돌다 관절 한계에 걸릴 수 있다."""
    cur = ep.obs["robot_state"]["eef"]["mat"].copy()
    rel = Rotation.from_matrix(cur.T @ target).as_rotvec()
    pos = sx.eef_pos(ep.obs) + [0, 0, lift]
    for k in range(1, n + 1):
        mk = cur @ Rotation.from_rotvec(rel * k / n).as_matrix()
        se.servo(ep, pos, mk, -1.0, tol=0.02, vmax=0.5, max_steps=25, record=record)


def open_middle_hook(ep, record=None, pull=0.15):
    mat = _aim(ep, _demo_mat(0, 102, 104))
    h = handle_pos(ep, "middle")
    se.hold(ep, -1.0, 3, record)
    if Rotation.from_matrix(ep.obs["robot_state"]["eef"]["mat"].T @ mat).magnitude() > np.radians(90):
        rotate_staged(ep, mat, record, lift=0.04)     # 전환 뒤처럼 손목이 많이 돌아가 있을 때
    here = sx.eef_pos(ep.obs)
    p1 = h + [-0.006, 0.05, 0.06]
    mid = (here + p1) / 2
    mid[2] = max(here[2], p1[2]) + 0.02
    se.servo(ep, mid, mat, -1.0, tol=0.04, record=record)
    se.servo(ep, p1, mat, -1.0, tol=0.01, vmax=0.4, record=record)
    # 이 지점은 위 서랍 손잡이 받침에 걸려 y -0.002 까지만 간다 (처음엔 -0.012 로 잡아 120스텝을 헛썼다)
    se.servo(ep, h + [-0.005, -0.002, 0.042], mat, -1.0, tol=0.008, vmax=0.25, max_steps=40, record=record)
    se.servo(ep, h + [-0.005, -0.011, 0.015], mat, -1.0, tol=0.006, vmax=0.15, max_steps=60, record=record)
    g = h + [-0.005, -0.011, 0.013]
    for k in range(1, 9):
        se.servo(ep, g + [0, pull * k / 8, 0], mat, -1.0, tol=0.006, vmax=0.3, max_steps=12, record=record)
    se.servo(ep, sx.eef_pos(ep.obs) + [0, 0.03, 0.07], mat, -1.0, tol=0.02, max_steps=30, record=record)
    return -drawer_qpos(ep, "middle")


# PiPER 서랍 열기 (2026-10-03): 그리퍼를 아래로 세우고 벌린 채, 손가락 하나만 손잡이 막대와 서랍 앞판 사이
# 틈(앞뒤 1.56cm, 좌우 5.5cm)에 내려 넣어 막대 뒤를 걸고 앞(+y)으로 당긴다.
#   Panda 의 기울인 손목 걸기(사람 시범에서 따온 자세)는 PiPER 팔 길이·관절로 손잡이 위치에서 만들기 어렵다.
#   손가락을 오므려 넣으면 틈 여유가 0.3cm 라 빡빡하고, 벌린 손가락 하나(두께 0.5cm)는 여유 0.5cm.
FINGER_OFF = 0.0374       # 벌렸을 때 손가락 중심이 손끝 기준점에서 옆으로 떨어진 거리 (측정)
FINGER_TIP = 0.015        # 손가락 끝이 손끝 기준점보다 아래로 나온 길이 (측정)
BAR_HALF_Y, BAR_HALF_Z, GAP_Y = 0.0077, 0.0082, 0.0156   # 손잡이 막대 반두께, 반높이, 막대-앞판 틈 (측정)


def open_drawer_finger(ep, which="middle", pull=0.15, record=None, depth=0.010, steps=10):
    mat = ep.home[1].copy()                       # 그리퍼 아래, 손가락은 y 로 벌어짐
    if Rotation.from_matrix(ep.obs["robot_state"]["eef"]["mat"].T @ mat).magnitude() > np.radians(90):
        rotate_staged(ep, mat, record, lift=0.04)
    h = handle_pos(ep, which)
    y_gap = h[1] - BAR_HALF_Y - GAP_Y / 2           # 틈 가운데
    cy = y_gap + FINGER_OFF                         # 손가락 하나(−y 쪽)가 틈에 오도록 손 중심은 앞쪽에
    z_hook = h[2] - BAR_HALF_Z - depth + FINGER_TIP # 손가락 끝이 막대 아래로 depth 만큼
    se.hold(ep, -1.0, 4, record)
    here = sx.eef_pos(ep.obs)
    # 옆으로 움직일 때 손가락 끝이 위 서랍 손잡이(가운데 서랍이면 바로 위)보다 2cm 이상 높게.
    # 처음엔 h+10cm 로 잡았다가 끝이 위 손잡이보다 0.3cm 높아 막대에 걸리고 기둥에 부딪혔다 (10/3)
    top_bar = handle_pos(ep, "top")[2] + BAR_HALF_Z
    above = np.array([h[0], cy, max(h[2] + BAR_HALF_Z, top_bar) + 0.02 + FINGER_TIP + 0.01])
    mid = (here + above) / 2
    mid[2] = max(here[2], above[2]) + 0.02
    se.servo(ep, mid, mat, -1.0, tol=0.04, record=record)
    se.servo(ep, above, mat, -1.0, tol=0.006, vmax=0.4, record=record)
    se.servo(ep, np.array([h[0], cy, z_hook]), mat, -1.0, tol=0.004, vmax=0.2, max_steps=80, record=record)
    for k in range(1, steps + 1):
        se.servo(ep, np.array([h[0], cy + pull * k / steps, z_hook]), mat, -1.0, tol=0.006, vmax=0.3,
                 max_steps=12, record=record)
    se.servo(ep, sx.eef_pos(ep.obs) + [0, 0.02, 0.08], mat, -1.0, tol=0.02, max_steps=30, record=record)
    return -drawer_qpos(ep, which)


# PiPER 서랍 열기 2차 (2026-10-03): 캐비닛을 돌려 서랍이 로봇(−x) 쪽을 보게 한 장면 (piper_sim/make_bddl.py).
#   손잡이 막대는 이제 y 방향 가로 막대이고 서랍은 −x 로 열린다. 손을 앞에서 비스듬히 아래로(수평에서 30°)
#   향하게 해서 손가락 두 개로 막대 위아래를 감싸 쥐고 −x 로 당긴다.
#   역기구학 계산: 수평(0°)은 쥐는 자리에서 4~9cm 모자라고, 30° 숙이면 쥐는 자리·15cm 당긴 끝 모두 오차 0.
#   위에서 손가락 하나를 틈에 넣는 방식(open_drawer_finger)은 손 몸통이 캐비닛 윗판 모서리에 부딪혔다.
GRASP_PITCH = 45


GRASP_YAW = 0


def side_grasp_mat(pitch=GRASP_PITCH, yaw=GRASP_YAW):
    """접근 방향(z_E) = +x 를 yaw 만큼 돌리고 pitch 만큼 아래로, 손가락 벌어지는 방향(x_E) = 그에 수직인 위쪽."""
    zE = Rotation.from_euler("zy", [yaw, pitch], degrees=True).apply([1.0, 0, 0])
    xE = np.array([0, 0, 1.0]) - zE * zE[2]
    xE /= np.linalg.norm(xE)
    return np.column_stack([xE, np.cross(zE, xE), zE])


# 서랍별 손 방향 후보 (pitch, yaw). 가운데 서랍은 60° 로 숙이면 다가가다 벌린 손가락이 위 서랍 손잡이에 걸린다
DRAWER_AIMS = {"middle": [(45, 0), (45, -15), (45, 15)],
               "top": [(45, 0), (45, -15)]}           # 60°·30° 는 다가가다 캐비닛 윗판·앞판에 걸림 (10/3)


# 위 서랍: 손가락을 반쯤 오므려(한쪽 2cm) 다가간다. 다 벌리면 위쪽 손가락이 캐비닛 윗판 앞 모서리에 걸렸다 (10/3).
#   손끝 기준점도 막대보다 1cm 앞에 둔다 (손가락 끝이 서랍 앞판에 닿지 않게)
DRAWER_PARTIAL = {"middle": 0, "top": 4}      # 오므리는 스텝 수 (한 스텝에 손가락마다 약 0.44cm)
DRAWER_LEAD = {"middle": 0.004, "top": 0.010}


def open_drawer_grasp(ep, which="middle", pull=0.158, record=None, back=0.05, lead=None, steps=10,
                      pitch=None, yaw=None, far_back=0.15, partial=None):
    import piper_sim.ik as pik
    lead = DRAWER_LEAD[which] if lead is None else lead
    partial = DRAWER_PARTIAL[which] if partial is None else partial
    se.hold(ep, -1.0, 4, record)
    h = handle_pos(ep, which)
    # 손을 크게 눕혀야 해서 손끝 변화량만으로는 손목이 특이 자세에 빠진다 → 관절 길을 미리 계산해 따라간다.
    # 손잡이 앞 멀리(far) → 쥐는 자리 → 끝까지 당긴 자리까지 한 가지(branch)로 이어지는 관절 해 중
    # 관절 한계 여유가 가장 큰 방향을 고른다. 가까이서 옆으로 움직이면 벌린 손가락이 위 서랍 손잡이 끝에 걸려서
    # (pitch 45, 손이 손잡이 5cm 앞에서 멈춤) 먼저 멀리 앞으로 간 뒤 접근 방향을 따라 들어간다
    aims = [(pitch, yaw)] if pitch is not None else DRAWER_AIMS[which]
    best = None
    for pt, yw in aims:
        M0 = side_grasp_mat(pt, yw)
        for M in (M0, M0 @ _RZ180):
            ap = M[:, 2]
            g = h - ap * lead
            pts = [g - ap * far_back, g - ap * back, g] + [g + [-pull * k / 6, 0, 0] for k in range(1, 7)]
            b = pik.plan_branch(ep, pts, [M])
            if b is not None and (best is None or b[0] > best[0]):
                best = (b[0], M, b[2], g)
    if best is None:
        return 0.0
    _, mat, q_far, g = best
    ap = mat[:, 2]
    pik.servo_to_q(ep, q_far, -1.0, record=record)
    grip = -1.0
    if partial:
        se.hold(ep, 1.0, partial, record)
        grip = 0.0                                   # 0 이면 그리퍼가 지금 벌린 정도를 그대로 둔다
    se.servo(ep, g - ap * back, mat, grip, tol=0.006, vmax=0.3, record=record)
    se.servo(ep, g, mat, grip, tol=0.004, vmax=0.2, max_steps=80, record=record)
    se.hold(ep, 1.0, 14, record)
    # 서랍이 열린 만큼 보면서 그보다 3cm 앞을 목표로 당긴다. 정해진 횟수만 당기면 손이 뒤처져 9.7cm 에서 끝났다
    se.PRECISE = True
    for _ in range(160):
        opened = -drawer_qpos(ep, which)
        if opened >= pull:
            break
        dp = g + [-(opened + 0.03), 0, 0] - sx.eef_pos(ep.obs)
        rerr = Rotation.from_matrix(mat @ ep.obs["robot_state"]["eef"]["mat"].T).as_rotvec()
        a = np.concatenate([np.clip(dp / sx.POS_SCALE, -0.3, 0.3), np.clip(rerr / sx.ROT_SCALE, -0.3, 0.3),
                            [1.0]]).astype(np.float32)
        if record is not None:
            record(ep.obs, a)
        ep.step(se._exec(a), "E")
    se.PRECISE = False
    se.hold(ep, -1.0, 10, record)
    se.servo(ep, sx.eef_pos(ep.obs) - ap * 0.06 + [0, 0, 0.03], mat, -1.0, tol=0.02, max_steps=30, record=record)
    return -drawer_qpos(ep, which)


# PiPER T3 (10/3): 위 서랍을 끝까지 열고(15.8cm), 그릇을 캐비닛에서 먼 쪽(−x) 테두리로 집어 서랍 안 앞쪽에 놓는다.
#   열린 서랍에서 캐비닛 윗판 밖으로 나온 부분은 앞판 안쪽 ~ 캐비닛 앞면 사이 약 13cm, 그릇 지름 11.1cm.
#   그릇 중심을 그 사이 가운데에 둔다. 손이 +x 테두리를 쥐면 손가락이 캐비닛 윗판 모서리에 닿을 수 있어 −x 쪽을 먼저 고른다
BOWL_R = 0.056            # 그릇 반지름 (실측 지름 11.1cm)
PLATE_IN = 0.029          # 손잡이 막대 중심 → 서랍 앞판 안쪽 면 (실측)
CAB_FRONT = 0.023         # 닫힌 서랍 손잡이 중심 → 캐비닛 윗판 앞 모서리 (실측)


def drawer_keepout(ep, opened):
    """열린 위 서랍 + 캐비닛 앞쪽: 그릇을 가지러 가는 관절 길이 들어가면 안 되는 상자 (서랍 앞판 위 1.12m + 여유)."""
    h = handle_pos(ep, "top")
    c = site_pos(ep, "wooden_cabinet_1_top_side")
    return [(np.array([h[0] - 0.03, c[1] - 0.15, 0.85]), np.array([c[0] + 0.15, c[1] + 0.15, 1.16]))]


def carry_place_piper(ep, P, record=None, above=0.12, obj="akita_black_bowl_1", keepout=()):
    """쥔 물체를 P 에 놓는다 (PiPER). 손 방향은 쥔 그대로, 목적지 위 → 놓는 높이까지 한 가지 관절 해로 이어지는지
    역기구학으로 보고 관절 길로 간다. 손끝 변화량만으로 가면 손이 목적지 5cm 앞에서 관절 한계에 걸려
    그릇을 캐비닛 모서리에 놓았다 (T3 ep2002)."""
    import piper_sim.ik as pik
    mat = ep.obs["robot_state"]["eef"]["mat"].copy()
    rel = sx.eef_pos(ep.obs) - ep.obj_pos(obj)
    tgt = P + rel + [0, 0, 0.015]
    b = pik.plan_branch(ep, [tgt + [0, 0, above], tgt + [0, 0, above / 2], tgt], [mat])
    if b is None:
        se.carry_place(ep, P, record=record, above=above, obj=obj)
        return
    pik.servo_to_q(ep, b[2], 1.0, record=record, tol=0.012, keepout=keepout, lift_z=max(1.20, tgt[2] + above))
    se.servo(ep, tgt + [0, 0, above / 2], mat, 1.0, tol=0.01, vmax=0.3, record=record)
    se.servo(ep, tgt, mat, 1.0, tol=0.008, vmax=0.25, record=record)
    se.hold(ep, -1.0, 10, record)
    se.servo(ep, sx.eef_pos(ep.obs) + [0, 0, 0.06], mat, -1.0, max_steps=20, record=record)


READY_DZ = 0.09


def ready_pose(ep, record=None, keepout=(), grip=-1.0):
    """PiPER 준비 자세: 처음 자세 READY_DZ 위, 손은 아래를 봄. 관절 길로 간다."""
    import piper_sim.ik as pik
    M = ep.home[1].copy()
    b = pik.plan_branch(ep, [ep.home[0] + [0, 0, READY_DZ]], [M, M @ _RZ180])
    if b is None:
        return False
    return pik.servo_to_q(ep, b[2], grip, record=record, tol=0.02, keepout=keepout, lift_z=1.20)


def drawer_bowl_piper(ep, record=None):
    opened = open_drawer_grasp(ep, "top", record=record)
    if opened < 0.13:
        return False
    # 서랍 앞판에서 손을 떼고 위로 (앞판 윗면 1.12m 위)
    mat = ep.obs["robot_state"]["eef"]["mat"].copy()
    here = sx.eef_pos(ep.obs)
    se.servo(ep, np.array([here[0] - 0.03, here[1], max(here[2], 1.18)]), mat, -1.0, tol=0.02, record=record)
    keep = drawer_keepout(ep, opened)
    # 꺾인 서랍 자세에서 그릇으로 곧장 가는 관절 길은 열린 서랍을 지나간다 → 처음 자세 9cm 위(준비 자세)를 거친다.
    # 처음 자세에서 7cm 이상 떨어져 있어 전환 규칙(처음 자세 복귀 금지)에도 걸리지 않는다
    ready_pose(ep, record, keepout=keep)
    if not grasp_bowl_safe(ep, record=record, prefer=lambda d: d[0], keepout=keep):
        return False
    opened = -drawer_qpos(ep, "top")
    hx = handle_pos(ep, "top")[0]
    lo = hx + PLATE_IN + BOWL_R + 0.005
    hi = hx + opened + CAB_FRONT - BOWL_R - 0.010
    c = region_pos(ep, "wooden_cabinet_1_top_region")
    P = np.array([(lo + hi) / 2, c[1], TOP_FLOOR_Z + BOWL_REST_DZ])
    # 그릇을 든 채로는 그릇(반지름 5.6cm, 손보다 5cm 아래)까지 피해야 한다: 상자를 +y·위로 8cm 넓힌다
    #   (넓히기 전: 든 그릇이 서랍 옆벽에 부딪혀 서랍을 16→11.7cm 닫음, ep2005)
    keep_b = [(lo_, hi_ + [0, 0.08, 0.08]) for lo_, hi_ in drawer_keepout(ep, opened)]
    carry_place_piper(ep, P, record=record, above=0.12, keepout=keep_b)
    return True


# T3 "위 서랍을 열고 그릇을 넣어라": 위 서랍 열기 → 그릇 집기 → 서랍 안(앞쪽, 캐비닛 윗판 밖)에 놓기
TOP_FLOOR_Z = 1.064      # 위 서랍 바닥 윗면 (geom g16: 중심 1.060, 두께 반 0.004)
BOWL_REST_DZ = -0.002    # 탁자 위 그릇 몸체 높이 0.898 − 탁자면 0.900


def region_pos(ep, site):
    m, d = ep.inner.sim.model, ep.inner.sim.data
    return np.array(d.site_xpos[m.site_name2id(site)])


def drawer_bowl(ep, record=None, y_front=0.05, pull=0.10, clear_y=-1.0, move_y=0.10, carry_above=0.12,
                pre_clear=0.128):
    """pull: 서랍을 얼마나 열까. 끝까지(16cm) 열면 앞판이 그릇 위로 와서 손이 내려가지 못한다 (0/10).
    pre_clear (10/2): 그릇이 손잡이에서 이만큼(y) 안쪽이면 서랍을 열기 **전에** 그릇을 앞으로 옮긴다.
      평가 장면 20~49 실패 3번(24, 32, 36)이 모두 그릇이 손잡이 앞 11.5~12.3cm 였고, 서랍이 그릇에 걸려
      2~4cm 만 열린 채 포기했다. clear_y 조건은 절대 좌표 −1m 라 한 번도 실행되지 않고 있었다."""
    # 그릇이 캐비닛 쪽에 가까이 있으면 먼저 앞으로 옮겨 둔다.
    # 그대로 두면 열린 서랍 앞판에 손이 걸리거나(5회), 그릇이 서랍 길을 막는다(3회) → 30회 중 22회
    b0 = ep.obj_pos("akita_black_bowl_1")
    handle0 = handle_pos(ep, "top")
    if b0[1] < clear_y or (b0 - handle0)[1] < pre_clear:
        if grasp_front(ep, "akita_black_bowl_1", se.BOWL_GRASP_OFFSET, record=record):
            se.carry_place(ep, np.array([b0[0], b0[1] + move_y, b0[2]]), record=record, above=0.05)
            b0 = ep.obj_pos("akita_black_bowl_1")
    opened = open_top_to(ep, pull, record=record)
    if opened < pull - 0.03:
        return False
    # 열린 서랍 앞판·손잡이(이제 y≈0.02)에 걸리지 않게: 앞쪽 높은 곳을 거쳐 그릇 위에서 수직으로
    b = ep.obj_pos("akita_black_bowl_1")
    mat = ep.home[1].copy()
    se.servo(ep, np.array([sx.eef_pos(ep.obs)[0], 0.12, 1.19]), mat, -1.0, tol=0.03, record=record)
    se.servo(ep, np.array([b[0], 0.10, 1.19]), mat, -1.0, tol=0.03, record=record)
    close = (b0 - handle0)[1] < 0.135
    if close:
        # 가까운 그릇: 옆 테두리로 집어 앞쪽(+10cm) 빈 자리에 내려놓고, 앞 테두리로 다시 잡는다.
        # 옆으로 쥔 채 서랍에 넣으면 손이 x 로 길어져 서랍 옆벽에 걸렸다
        if grasp_front(ep, "akita_black_bowl_1", se.BOWL_GRASP_OFFSET, record=record, side_first=True):
            b1 = ep.obj_pos("akita_black_bowl_1")
            se.carry_place(ep, np.array([b0[0], b0[1] + move_y, b0[2]]), record=record, above=0.04)
    if not grasp_front(ep, "akita_black_bowl_1", se.BOWL_GRASP_OFFSET, record=record):
        return False
    c = region_pos(ep, "wooden_cabinet_1_top_region")
    P = np.array([c[0], c[1] + y_front, TOP_FLOOR_Z + BOWL_REST_DZ])
    # 높이 들고 넘어간다: 고쳐 잡은 그릇은 기울어 손 아래로 더 처지는데, 낮게 넘어가면
    # 서랍 앞판(윗면 1.12)에 부딪혀 서랍을 도로 닫아 버렸다 (ep13: 서랍 0.4cm)
    se.carry_place(ep, P, record=record, above=carry_above)
    return True


def grasp_front(ep, obj, off, record=None, above=0.08, side_first=False):
    """se.grasp_obj 와 같지만 **앞쪽(+y, 서랍 반대편) 테두리부터** 잡는다.
    열린 위 서랍이 그릇 위로 튀어나와, 뒤쪽 테두리로는 손이 내려가지 못한다 (T3 첫 시도 0/10)."""
    down = ep.home[1].copy()
    se.hold(ep, -1.0, 6, record)
    cands = se.grasp_candidates(ep, obj, off, down)
    if side_first:
        # 그릇이 캐비닛에 가까우면(손잡이보다 13cm 안쪽) 옆쪽 테두리, 그중 서랍에서 먼(−x) 쪽부터.
        # 손가락이 x 로 벌어져 손 폭이 y 로 좁아져서 서랍 앞판에 걸리지 않는다
        o = ep.obj_pos(obj)
        side = [c for c in cands if abs(c[0][0] - o[0]) > abs(c[0][1] - o[1])]
        cands = sorted(side, key=lambda c: c[0][0]) + [c for c in cands if not any(c is s for s in side)]
    else:
        cands = sorted(cands, key=lambda c: -c[0][1])
    for g, gmat in cands[:2]:
        if not se.servo(ep, g + [0, 0, above], gmat, -1.0, tol=0.025, record=record):
            continue
        if not se.servo(ep, g, gmat, -1.0, tol=0.008, vmax=0.3, record=record):
            se.servo(ep, g + [0, 0, above], gmat, -1.0, tol=0.03, max_steps=30, record=record)
            continue
        se.hold(ep, 1.0, 12, record)
        z0 = ep.obj_pos(obj)[2]
        se.servo(ep, g + [0, 0, 0.10], gmat, 1.0, tol=0.02, record=record)
        if ep.obj_pos(obj)[2] > z0 + 0.04:
            return True
    return False


def open_top_to(ep, target, record=None, press=0.006, above=0.07):
    """위 서랍을 target(m)까지만 연다 — 서랍 위치를 보면서 도달하면 바로 누르기를 멈추고 손을 든다.
    그냥 당기면 미끄러져 14~16cm 까지 나가 T3 에서 손이 다시 막혔다 (30회 중 5회)."""
    mat = _aim(ep, ep.home[1].copy())
    h = handle_pos(ep, "top")
    rel = np.array(PRESS["top"])
    se.hold(ep, -1.0, 3, record)
    here = sx.eef_pos(ep.obs)
    pre = h + rel + [0, 0, above]
    mid = (here + pre) / 2
    mid[2] = max(here[2], pre[2]) + 0.02
    se.servo(ep, mid, mat, -1.0, tol=0.04, record=record)
    se.servo(ep, pre, mat, -1.0, tol=0.01, vmax=0.4, record=record)
    se.servo(ep, h + rel, mat, -1.0, tol=0.006, vmax=0.2, max_steps=60, record=record)
    g = h + rel - [0, 0, press]
    se.PRECISE = True                 # 접촉 구간: DART 잡음 없음
    for _ in range(80):
        opened = -drawer_qpos(ep, "top")
        if opened >= target - 0.005:
            break
        dp = g + [0, opened + 0.02, 0] - sx.eef_pos(ep.obs)     # 서랍보다 2cm 앞을 목표로 천천히
        a = np.concatenate([np.clip(dp / sx.POS_SCALE, -0.25, 0.25), [0, 0, 0], [-1.0]]).astype(np.float32)
        if record is not None:
            record(ep.obs, a)
        ep.step(se._exec(a), "E")
    se.PRECISE = False                # 접촉 구간 끝: 다시 잡음 허용
    # 손을 **똑바로 위로** 뺀다 (살짝 뒤로). 대각선으로 빼면 손끝이 막대를 한 번 더 끌어 14~16cm 까지 열렸다
    se.servo(ep, sx.eef_pos(ep.obs) + [0, -0.005, 0.05], mat, -1.0, tol=0.008, vmax=0.3, max_steps=30, record=record)
    return -drawer_qpos(ep, "top")


# ───────────────────────────────────────────────────────────────────────────
# T7 스토브 켜기 (diag_policy_contact.py, T7 ep20)
#   원래 모델: 손잡이 몸체 기준 [-0.004, 0.026, 0.024] 에서 그리퍼를 닫아 손잡이(g5)를 감싸고,
#   손목을 z 축으로 약 +32도 돌린다 → 관절 0.52 rad 에서 켜짐 판정
# ───────────────────────────────────────────────────────────────────────────
STOVE_REL = np.array([-0.004, 0.026, 0.024])


def body_pos(ep, name):
    m, d = ep.inner.sim.model, ep.inner.sim.data
    return np.array(d.body_xpos[m.body_name2id(name)])


def turn_on_stove(ep, record=None, turn_deg=60, above=0.08):
    """손목 40도로는 손잡이가 27~29도(0.47~0.51 rad)만 돌아 켜짐 기준(약 0.5)에 걸렸다 (7/20) → 60도."""
    mat = _aim(ep, ep.home[1].copy())
    k = body_pos(ep, "flat_stove_1_button")
    g = k + STOVE_REL
    se.hold(ep, -1.0, 3, record)
    here = sx.eef_pos(ep.obs)
    pre = g + [0, 0, above]
    mid = (here + pre) / 2
    mid[2] = max(here[2], pre[2]) + 0.02
    se.servo(ep, mid, mat, -1.0, tol=0.04, record=record)
    se.servo(ep, pre, mat, -1.0, tol=0.01, vmax=0.4, record=record)
    se.servo(ep, g, mat, -1.0, tol=0.006, vmax=0.2, max_steps=60, record=record)
    se.hold(ep, 1.0, 6, record)
    m0 = ep.obs["robot_state"]["eef"]["mat"].copy()
    for frac in np.linspace(0.25, 1.0, 4):
        tgt = Rotation.from_euler("z", turn_deg * frac, degrees=True).as_matrix() @ m0
        se.servo(ep, g, tgt, 1.0, tol=0.01, vmax=0.3, max_steps=15, record=record)
    se.hold(ep, -1.0, 5, record)
    se.servo(ep, sx.eef_pos(ep.obs) + [0, 0, 0.07], ep.obs["robot_state"]["eef"]["mat"].copy(), -1.0,
             tol=0.02, max_steps=25, record=record)


def turn_on_stove_piper(ep, record=None, turn_deg=60, above=0.08):
    """PiPER 스토브 켜기. 손잡이가 밑동 옆 23cm 라 손을 똑바로 아래로 하면 관절 여유 0 (손이 5cm 앞에서 멈춤, 0/10).
    밑동 방향과 수직인 축으로 손을 기울인 방향 중 (위 → 쥐는 점) 길의 관절 여유가 가장 큰 것을 고르고,
    그 방향에서 60° 돌리는 길도 이어지는지 확인한다 (역기구학: −15° 여유 10°, 30° 여유 9°)."""
    import piper_sim.ik as pik
    k = body_pos(ep, "flat_stove_1_button")
    g = k + STOVE_REL
    m, d = ep.inner.sim.model, ep.inner.sim.data
    base = np.array(d.body_xpos[m.body_name2id("robot0_base")])
    radial = (g - base) * [1, 1, 0]
    axis = np.cross([0, 0, 1.0], radial / np.linalg.norm(radial))
    M0 = ep.home[1].copy()
    ik = pik.ArmIK(ep)
    best = None
    for tilt in (-15, 15, 30, -30, 0):
        Rt = Rotation.from_rotvec(axis * np.radians(tilt)).as_matrix()
        for M in (Rt @ M0, Rt @ M0 @ _RZ180):
            b = pik.plan_branch(ep, [g + [0, 0, above], g], [M])
            if b is None:
                continue
            q, e = ik.solve(g, M, b[2])
            ok = e < 0.003
            for th in (20, 40, turn_deg):
                q2, e2 = ik.solve(g, Rotation.from_euler("z", th, degrees=True).as_matrix() @ M, q)
                if e2 > 0.003 or np.abs(q2 - q).max() > np.radians(30):
                    ok = False
                    break
                q = q2
            if ok and (best is None or b[0] > best[0]):
                best = (b[0], M, b[2])
    if best is None:
        return turn_on_stove(ep, record, turn_deg, above)
    _, M, q_pre = best
    se.hold(ep, -1.0, 3, record)
    pik.servo_to_q(ep, q_pre, -1.0, record=record, tol=0.01)
    se.servo(ep, g, M, -1.0, tol=0.006, vmax=0.2, max_steps=60, record=record)
    se.hold(ep, 1.0, 6, record)
    for frac in np.linspace(0.25, 1.0, 4):
        tgt = Rotation.from_euler("z", turn_deg * frac, degrees=True).as_matrix() @ M
        se.servo(ep, g, tgt, 1.0, tol=0.01, vmax=0.3, max_steps=15, record=record)
    se.hold(ep, -1.0, 5, record)
    se.servo(ep, sx.eef_pos(ep.obs) + [0, 0, 0.07], ep.obs["robot_state"]["eef"]["mat"].copy(), -1.0,
             tol=0.02, max_steps=25, record=record)


# ───────────────────────────────────────────────────────────────────────────
# T5 접시를 스토브 앞으로 밀기 (diag_policy_contact.py, T5 ep100)
#   원래 모델: 벌린 그리퍼를 접시 안쪽(중심 기준 [0.014, 0.019, 0.015])까지 내리고,
#   살짝 누르며 목표 쪽으로 끈다. 접시가 손을 따라온다 (손보다 약 5cm 뒤처짐)
#   여기에 접시 위치 되먹임을 더한다: 매 스텝 "접시 → 목표" 방향으로 손을 움직인다
# ───────────────────────────────────────────────────────────────────────────
PLATE_REL = np.array([0.014, 0.019, 0.015])


def site_pos(ep, name):
    m, d = ep.inner.sim.model, ep.inner.sim.data
    return np.array(d.site_xpos[m.site_name2id(name)])


def push_plate(ep, record=None, press=0.006, speed=0.8, tol=0.012, max_steps=160, above=0.08, lead=0.07):
    """lead: 손이 접시보다 얼마나 앞서 끌까. 원래 모델은 5~7cm 앞서 안쪽 테두리를 걸어 끈다.
    처음에 3cm 로 했더니 손이 테두리에 걸리지 못하고 혼자 미끄러져 0/20 이었다."""
    mat = ep.home[1].copy()
    goal = site_pos(ep, "main_table_stove_front_region")
    p0 = ep.obj_pos("plate_1")
    g = p0 + PLATE_REL
    se.hold(ep, -1.0, 3, record)
    here = sx.eef_pos(ep.obs)
    pre = g + [0, 0, above]
    mid = (here + pre) / 2
    mid[2] = max(here[2], pre[2]) + 0.02
    se.servo(ep, mid, mat, -1.0, tol=0.04, record=record)
    se.servo(ep, pre, mat, -1.0, tol=0.01, vmax=0.4, record=record)
    se.servo(ep, g - [0, 0, press], mat, -1.0, tol=0.006, vmax=0.25, max_steps=50, record=record)
    se.PRECISE = True                 # 접촉 구간: DART 잡음 없음
    for _ in range(max_steps):
        p = ep.obj_pos("plate_1")
        err = goal[:2] - p[:2]
        if np.linalg.norm(err) < tol:
            break
        hand = sx.eef_pos(ep.obs)
        # 손은 "접시 위치 + 집는 오프셋 + 목표 방향으로 조금 앞" 을 목표로, 높이는 접시 기준으로 눌러 둔다
        dirn = err / (np.linalg.norm(err) + 1e-6)
        # 원래 모델처럼 목표 방향으로 **일정한 속도**로 누르며 끈다 (P 제어만 쓰면 손이 앞선 간격에
        # 도달하는 순간 힘이 0 이 되어 목표 5.7cm 앞에서 멈췄다). 옆으로 벗어나면 바로잡는 항을 더한다
        tgt = p[:2] + PLATE_REL[:2] + dirn * lead
        v = dirn * speed * np.clip(np.linalg.norm(err) / 0.05, 0.4, 1.0)
        v = v + np.clip((tgt - hand[:2]) / sx.POS_SCALE, -0.3, 0.3) * 0.5
        vz = np.clip((p[2] + PLATE_REL[2] - press - hand[2]) / sx.POS_SCALE, -0.4, 0.4) - 0.2
        rerr = Rotation.from_matrix(mat @ ep.obs["robot_state"]["eef"]["mat"].T).as_rotvec()
        a = np.concatenate([np.clip(v, -1, 1), [np.clip(vz, -0.5, 0.5)],
                            np.clip(rerr / sx.ROT_SCALE, -0.3, 0.3), [-1.0]]).astype(np.float32)
        if record is not None:
            record(ep.obs, a)
        ep.step(se._exec(a), "E")
    se.PRECISE = False                # 접촉 구간 끝: 다시 잡음 허용
    se.servo(ep, sx.eef_pos(ep.obs) + [0, 0, 0.07], mat, -1.0, tol=0.02, max_steps=25, record=record)


def push_plate_regrip(ep, record=None, speed=1.0, lead=0.05, press=0.006, tol=0.012, max_steps=300,
                      regrips=2, stall_steps=15, above=0.08):
    """push_plate + **다시 잡기**: 접시가 15스텝 동안 3mm 도 안 움직이면 손을 들어
    남은 방향 쪽 안쪽 테두리(접시 중심에서 그 방향으로 4.5cm)에 다시 걸고 계속 끈다.
    (속도·간격만 조정해서는 26/30 에서 멈췄다 — 접시가 목표 6cm 앞에서 서는 경우)"""
    mat = _aim(ep, ep.home[1].copy())
    goal = site_pos(ep, "main_table_stove_front_region")
    p0 = ep.obj_pos("plate_1")
    g = p0 + PLATE_REL
    se.hold(ep, -1.0, 3, record)
    here = sx.eef_pos(ep.obs)
    pre = g + [0, 0, above]
    mid = (here + pre) / 2
    mid[2] = max(here[2], pre[2]) + 0.02
    se.servo(ep, mid, mat, -1.0, tol=0.04, record=record)
    se.servo(ep, pre, mat, -1.0, tol=0.01, vmax=0.4, record=record)
    se.servo(ep, g - [0, 0, press], mat, -1.0, tol=0.006, vmax=0.25, max_steps=50, record=record)
    hist = []
    left = regrips
    se.PRECISE = True                 # 접촉 구간: DART 잡음 없음
    for _ in range(max_steps):
        p = ep.obj_pos("plate_1")
        err = goal[:2] - p[:2]
        if np.linalg.norm(err) < tol:
            break
        hist.append(p[:2].copy())
        dirn = err / (np.linalg.norm(err) + 1e-6)
        if left > 0 and len(hist) > stall_steps and np.linalg.norm(hist[-1] - hist[-1 - stall_steps]) < 0.003:
            left -= 1
            hist = []
            up = sx.eef_pos(ep.obs) + [0, 0, 0.05]
            se.servo(ep, up, mat, -1.0, tol=0.01, vmax=0.4, max_steps=20, record=record)
            ng = np.concatenate([p[:2] + dirn * 0.045, [p[2] + PLATE_REL[2]]])
            se.servo(ep, ng + [0, 0, 0.05], mat, -1.0, tol=0.01, vmax=0.4, max_steps=30, record=record)
            se.servo(ep, ng - [0, 0, press], mat, -1.0, tol=0.006, vmax=0.25, max_steps=30, record=record)
            continue
        hand = sx.eef_pos(ep.obs)
        tgt = p[:2] + PLATE_REL[:2] + dirn * lead
        v = dirn * speed * np.clip(np.linalg.norm(err) / 0.05, 0.4, 1.0)
        v = v + np.clip((tgt - hand[:2]) / sx.POS_SCALE, -0.3, 0.3) * 0.5
        vz = np.clip((p[2] + PLATE_REL[2] - press - hand[2]) / sx.POS_SCALE, -0.4, 0.4) - 0.2
        rerr = Rotation.from_matrix(mat @ ep.obs["robot_state"]["eef"]["mat"].T).as_rotvec()
        a = np.concatenate([np.clip(v, -1, 1), [np.clip(vz, -0.5, 0.5)],
                            np.clip(rerr / sx.ROT_SCALE, -0.3, 0.3), [-1.0]]).astype(np.float32)
        if record is not None:
            record(ep.obs, a)
        ep.step(se._exec(a), "E")
    se.PRECISE = False                # 접촉 구간 끝: 다시 잡음 허용
    se.servo(ep, sx.eef_pos(ep.obs) + [0, 0, 0.07], mat, -1.0, tol=0.02, max_steps=25, record=record)


# ───────────────────────────────────────────────────────────────────────────
# T6 치즈를 그릇에 (diag_policy_contact.py, T6 ep0)
#   원래 모델: 처음 자세 방향 그대로, 치즈 몸체 기준 [-0.010, -0.003, +0.002] 에서 쥐고(관절 0.018) 들어 올린다
# ───────────────────────────────────────────────────────────────────────────
CHEESE_REL = np.array([-0.010, -0.003, 0.002])


def cheese_carry_to_bowl(ep, record=None, drop_dz=0.06):
    """이미 쥔 치즈를 놓지 않고 그릇 위로 옮겨 떨어뜨린다 (DAgger 넘겨받기용, 10/2)."""
    mat = _aim(ep, ep.home[1].copy())
    se.hold(ep, 1.0, 4, record)
    se.servo(ep, sx.eef_pos(ep.obs) + [0, 0, 0.06], mat, 1.0, tol=0.02, vmax=0.4, max_steps=30, record=record)
    rel = sx.eef_pos(ep.obs) - ep.obj_pos("cream_cheese_1")
    b = ep.obj_pos("akita_black_bowl_1")
    se.servo(ep, b + rel + [0, 0, drop_dz + 0.06], mat, 1.0, tol=0.015, vmax=0.4, record=record)
    se.servo(ep, b + rel + [0, 0, drop_dz], mat, 1.0, tol=0.008, vmax=0.25, record=record)
    se.hold(ep, -1.0, 8, record)
    se.servo(ep, sx.eef_pos(ep.obs) + [0, 0, 0.07], mat, -1.0, tol=0.02, max_steps=25, record=record)
    return True


def cheese_to_bowl(ep, record=None, above=0.10, drop_dz=0.06):
    mat = _aim(ep, ep.home[1].copy())
    c = ep.obj_pos("cream_cheese_1")
    g = c + CHEESE_REL
    se.hold(ep, -1.0, 3, record)
    here = sx.eef_pos(ep.obs)
    pre = g + [0, 0, above]
    mid = (here + pre) / 2
    mid[2] = max(here[2], pre[2]) + 0.02
    se.servo(ep, mid, mat, -1.0, tol=0.04, record=record)
    se.servo(ep, pre, mat, -1.0, tol=0.01, vmax=0.4, record=record)
    se.servo(ep, g, mat, -1.0, tol=0.006, vmax=0.25, max_steps=60, record=record)
    se.hold(ep, 1.0, 10, record)
    z0 = ep.obj_pos("cream_cheese_1")[2]
    se.servo(ep, g + [0, 0, above], mat, 1.0, tol=0.015, vmax=0.4, record=record)
    if ep.obj_pos("cream_cheese_1")[2] < z0 + 0.04:
        return False
    rel = sx.eef_pos(ep.obs) - ep.obj_pos("cream_cheese_1")
    b = ep.obj_pos("akita_black_bowl_1")
    se.servo(ep, b + rel + [0, 0, drop_dz + 0.06], mat, 1.0, tol=0.015, vmax=0.4, record=record)
    se.servo(ep, b + rel + [0, 0, drop_dz], mat, 1.0, tol=0.008, vmax=0.25, record=record)
    se.hold(ep, -1.0, 8, record)
    se.servo(ep, sx.eef_pos(ep.obs) + [0, 0, 0.07], mat, -1.0, tol=0.02, max_steps=25, record=record)
    return True


# ───────────────────────────────────────────────────────────────────────────
# T9 와인병을 선반에 (record_demos 의 T9 ep107·110·114 성공 궤적)
#   쥐기: 손목을 약 60도 기울여(euler xyz [-120, 6, -179]) 병 몸체 기준 [-0.003, -0.010, 0.10] 을 쥔다
#   놓기: 손목을 더 눕혀 [-157, -8, -178], 병이 선반 몸체 기준 [0.068, 0.070, 0.245] 에 오게 (선반 영역 y 폭 ±2.2cm)
#   병이 손에 어떻게 쥐였는지는 들어 올린 직후 **실제로 재서** 놓는 손 위치를 거꾸로 계산한다
# ───────────────────────────────────────────────────────────────────────────
WINE_GRASP_REL = np.array([-0.003, -0.010, 0.10])
WINE_GRASP_EUL = [-120, 6, -179]
WINE_RACK_EUL = [-157, -8, -178]
WINE_RACK_REL = np.array([0.068, 0.070, 0.245])


def _eul(e):
    return Rotation.from_euler("xyz", e, degrees=True).as_matrix()


def grasp_wine(ep, record=None, back=0.08):
    w = ep.obj_pos("wine_bottle_1")
    gm = _aim(ep, _eul(WINE_GRASP_EUL))
    g = w + WINE_GRASP_REL
    approach = gm[:, 2]                         # 손끝이 향하는 방향
    se.hold(ep, -1.0, 3, record)
    if Rotation.from_matrix(ep.obs["robot_state"]["eef"]["mat"].T @ gm).magnitude() > np.radians(90):
        rotate_staged(ep, gm, record, lift=0.04)
    here = sx.eef_pos(ep.obs)
    pre = g - approach * back + [0, 0, 0.03]
    mid = (here + pre) / 2
    mid[2] = max(here[2], pre[2]) + 0.02
    se.servo(ep, mid, gm, -1.0, tol=0.04, record=record)
    se.servo(ep, pre, gm, -1.0, tol=0.01, vmax=0.4, record=record)
    se.servo(ep, g, gm, -1.0, tol=0.006, vmax=0.2, max_steps=60, record=record)
    se.hold(ep, 1.0, 10, record)
    z0 = ep.obj_pos("wine_bottle_1")[2]
    se.servo(ep, g + [0, 0, 0.12], gm, 1.0, tol=0.015, vmax=0.3, record=record)
    return ep.obj_pos("wine_bottle_1")[2] > z0 + 0.05


def place_wine(ep, bottle_target, place_eul, record=None, above=0.08):
    """병이 bottle_target 에 오도록 손을 놓는다. 손−병 관계는 지금 실제로 잰 값(손 좌표계)."""
    Rh = ep.obs["robot_state"]["eef"]["mat"].copy()
    r = Rh.T @ (ep.obj_pos("wine_bottle_1") - sx.eef_pos(ep.obs))
    Rt = _aim(ep, _eul(place_eul))
    hand_t = bottle_target - Rt @ r
    here = sx.eef_pos(ep.obs)
    # 수직으로 먼저 올리지 않고 목적지 쪽으로 비스듬히 올라간다. 와인병 자리는 초기 자세 바로 아래라
    # 수직으로 올리면 손이 초기 자세 7cm 안까지 들어갔다 (전환 뒤 와인병→선반 7cm 규칙에 걸려 1~2/10)
    mid = here + 0.5 * (hand_t - here)
    up = np.array([mid[0], mid[1], max(here[2], hand_t[2] + above)])
    se.servo(ep, up, Rh, 1.0, tol=0.02, vmax=0.4, record=record)
    se.servo(ep, hand_t + [0, 0, above], Rt, 1.0, tol=0.012, vmax=0.35, record=record)
    se.servo(ep, hand_t, Rt, 1.0, tol=0.006, vmax=0.2, max_steps=60, record=record)
    se.hold(ep, -1.0, 10, record)
    se.servo(ep, sx.eef_pos(ep.obs) + [0, 0.05, 0.05], ep.obs["robot_state"]["eef"]["mat"].copy(), -1.0,
             tol=0.02, max_steps=25, record=record)


def wine_to_rack(ep, record=None):
    if not grasp_wine(ep, record):
        return False
    m, d = ep.inner.sim.model, ep.inner.sim.data
    rack = np.array(d.body_xpos[m.body_name2id("wine_rack_1_main")])
    place_wine(ep, rack + WINE_RACK_REL, WINE_RACK_EUL, record)
    return True


# T2 와인병을 캐비닛 위에: 쥔 방향 그대로(병이 선 채) 캐비닛 윗면 가운데로.
#   병 몸체 높이는 탁자 위에서 0.899 (탁자면 0.900) → 윗면(top_side, z 1.127) 기준 −0.001, 1cm 위에서 놓는다
def wine_to_cabinet(ep, record=None, drop=0.01):
    if not grasp_wine(ep, record):
        return False
    top = site_pos(ep, "wooden_cabinet_1_top_side")
    place_wine(ep, top + [0, 0, -0.001 + drop], WINE_GRASP_EUL, record)
    return True


# ───────────────────────────────────────────────────────────────────────────
# PiPER 와인병 (10/3). 병: 높이 15.7cm, 지름 4.3cm, 몸체 원점 = 바닥.
#   집기: 병 너머(+x)에서 로봇 쪽으로 60° 숙인 손으로 바닥 7cm 높이를 쥔다 (손가락은 수평으로 닫힘).
#        병 원점이 로봇 밑동에서 19cm 라 수평으로 쥐는 방향은 모두 손이 닿지 않았다.
#   놓기: 병의 목표 자세(바닥 위치 + 병 축 방향)를 정하고, 쥔 손−병 관계를 그대로 옮겨 손 목표를 계산한다.
#        병 축 둘레로 도는 각은 자유라 30° 간격으로 바꿔 보며 관절 여유가 가장 큰 것을 고른다.
#   선반(T9): Panda 시범이 성공한 최종 자세 그대로 — 바닥이 선반 몸체 기준 (+7.5, +6.2, +24.9)cm,
#        병 축 (0, −0.86, 0.51) (선반 홈을 따라 59° 누움, 목이 −y 위쪽). 측정: scratchpad/panda_t9.py
# ───────────────────────────────────────────────────────────────────────────
WINE_GRASP_DZ = 0.07
# PiPER 배치는 선반을 180° 돌려(yaw 0) 홈이 로봇 쪽으로 올라가게 했다 (병목이 +y). 원래 방향(yaw π)이면
#   병목이 로봇 반대쪽이라 쥔 손의 손목이 로봇 반대편에 있어야 해서 손 목표가 20~40cm 닿지 않았다.
#   Panda 값(선반 몸체 기준, yaw π)을 선반 좌표로 바꿔 yaw 0 에 맞춘 것: 바닥 (−7.5, −6.2, +24.9)cm
#   + 홈을 따라 3cm 더 아래(앞 턱 쪽): LIBERO 판정은 영역 회전을 (회전행렬 @ 차이)로 계산해 yaw 0 에서는
#   Panda 자리 그대로면 기울어진 영역 두께 방향 3.1cm(한계 2.2cm)로 실패했다. 아래로 1.1cm 이상 내려야 통과
WINE_RACK_BOTTOM = np.array([-0.075, -0.062, 0.249]) + 0.03 * np.array([0.0, -0.86, -0.51])
WINE_RACK_AXIS = np.array([0.0, 0.86, 0.51]) / np.linalg.norm([0.0, 0.86, 0.51])


def _side_mats(yaw_deg, pitch_deg):
    zE = np.array([np.cos(np.radians(yaw_deg)) * np.cos(np.radians(pitch_deg)),
                   np.sin(np.radians(yaw_deg)) * np.cos(np.radians(pitch_deg)), -np.sin(np.radians(pitch_deg))])
    xE = np.cross([0, 0, 1.0], zE)
    xE /= np.linalg.norm(xE)
    M = np.column_stack([xE, np.cross(zE, xE), zE])
    return [M, M @ _RZ180]


def _place_margin(ep, M, g, place, above=0.05):
    """병을 손 방향 M, 손 위치 g 로 쥔다고 할 때, 놓을 자세(place = (바닥, 축, 떨어지는 방향))에서
    병 축 둘레 회전(30° 간격) 중 관절 여유가 가장 큰 값. 못 놓으면 None."""
    import piper_sim.ik as pik
    Rb = se.obj_rot(ep, "wine_bottle_1").as_matrix()
    w = ep.obj_pos("wine_bottle_1")
    R_hb, p_hb = Rb.T @ M, Rb.T @ (g - w)
    bottom_t, axis_t, lift_dir = place
    best = None
    for psi in range(0, 360, 30):
        Rbt = _align(axis_t) @ Rotation.from_rotvec([0, 0, np.radians(psi)]).as_matrix()
        Rht = Rbt @ R_hb
        if Rht[:, 2] @ [0, 0, -1.0] < -0.2:
            continue
        pht = bottom_t + Rbt @ p_hb
        b = pik.plan_branch(ep, [pht + lift_dir * above, pht], [Rht], n_seeds=4)
        if b is not None and (best is None or b[0] > best):
            best = b[0]
    return best


def wine_grasp_piper(ep, record=None, back=0.10, lift=0.12, place=None):
    """place 를 주면 쥐는 방향을 고를 때 놓을 자세에서도 손이 닿는지 같이 본다 (T2: 쥐기만 보고 고르면
    캐비닛 위에서 손 회전 관절이 한계 5° 안이라 10장면 중 5번 못 놓음)."""
    import piper_sim.ik as pik
    se.hold(ep, -1.0, 4, record)
    w = ep.obj_pos("wine_bottle_1")
    g = w + [0, 0, WINE_GRASP_DZ]
    best = None
    # 역기구학 지도 (scratchpad/wine_reach.py): 병이 밑동에 가까워 수평으로는 어느 방향도 안 되고,
    # 병 너머(+x)에서 로봇 쪽으로 60° 숙여 내려오는 방향(yaw 180±30)만 관절 여유 16~20°
    for yaw in (180, 150, -150, 120, -120):
        for pitch in (60, 50):
            for M in _side_mats(yaw, pitch):
                ap = M[:, 2]
                b = pik.plan_branch(ep, [g - ap * back, g - ap * back / 2, g, g + [0, 0, lift]], [M])
                if b is None:
                    continue
                score = b[0]
                if place is not None:
                    pm = _place_margin(ep, M, g, place)
                    if pm is None:
                        continue
                    score = min(score, pm)
                if best is None or score > best[0] + np.radians(3):
                    best = (score, M, b[2])
        if best is not None and best[0] > np.radians(15):
            break                                    # 앞 방향(병 +y 쪽)이 충분히 되면 그대로
    if best is None:
        return False
    _, M, q_pre = best
    ap = M[:, 2]
    pik.servo_to_q(ep, q_pre, -1.0, record=record, tol=0.01)
    se.servo(ep, g - ap * back / 2, M, -1.0, tol=0.008, vmax=0.3, record=record)
    se.servo(ep, g, M, -1.0, tol=0.006, vmax=0.2, max_steps=60, record=record)
    se.hold(ep, 1.0, 12, record)
    z0 = ep.obj_pos("wine_bottle_1")[2]
    se.servo(ep, g + [0, 0, lift], M, 1.0, tol=0.015, vmax=0.3, record=record)
    return ep.obj_pos("wine_bottle_1")[2] > z0 + 0.05


def _align(a):
    """e_z 를 a 로 보내는 회전 (최소 회전)."""
    a = a / np.linalg.norm(a)
    v = np.cross([0, 0, 1.0], a)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if a[2] > 0 else Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
    ang = np.arccos(np.clip(a[2], -1, 1))
    return Rotation.from_rotvec(v / np.linalg.norm(v) * ang).as_matrix()


def wine_place_piper(ep, bottom_t, axis_t, lift_dir, record=None, above=0.08, keepout=(), z_safe=None):
    """쥔 병을 바닥 bottom_t, 축 axis_t 자세로 놓는다. lift_dir: 놓는 면에서 떨어지는 방향(위 접근용)."""
    import piper_sim.ik as pik
    Rh = ep.obs["robot_state"]["eef"]["mat"].copy()
    ph = sx.eef_pos(ep.obs)
    Rb = se.obj_rot(ep, "wine_bottle_1").as_matrix()
    pb = ep.obj_pos("wine_bottle_1")
    R_hb = Rb.T @ Rh                                   # 병 좌표계에서 본 손
    p_hb = Rb.T @ (ph - pb)
    best = None
    for psi in range(0, 360, 30):
        Rbt = _align(axis_t) @ Rotation.from_rotvec([0, 0, np.radians(psi)]).as_matrix()
        Rht = Rbt @ R_hb
        if Rht[:, 2] @ [0, 0, -1.0] < -0.2:            # 손끝이 위를 보면(아래에서 받치는 자세) 뺀다
            continue
        pht = bottom_t + Rbt @ p_hb
        b = pik.plan_branch(ep, [pht + lift_dir * above, pht + lift_dir * above / 2, pht], [Rht])
        if b is not None and (best is None or b[0] > best[0]):
            best = (b[0], Rht, b[2], pht)
    if best is None:
        return False
    _, Rht, q_above, pht = best
    if z_safe is not None:
        if not pik.transit(ep, pht + lift_dir * 0.005, Rht, 1.0, z_safe, record=record, down_dir=lift_dir, above=above):
            return False
    else:
        pik.servo_to_q(ep, q_above, 1.0, record=record, tol=0.012, keepout=keepout,
                       lift_z=max(1.25, pht[2] + above))
        se.servo(ep, pht + lift_dir * above / 2, Rht, 1.0, tol=0.01, vmax=0.3, record=record)
        se.servo(ep, pht + lift_dir * 0.005, Rht, 1.0, tol=0.006, vmax=0.2, max_steps=60, record=record)
    se.hold(ep, -1.0, 10, record)
    se.servo(ep, sx.eef_pos(ep.obs) - Rht[:, 2] * 0.06 + lift_dir * 0.03, Rht, -1.0, tol=0.02, max_steps=30,
             record=record)
    return True


def wine_to_rack_piper(ep, record=None):
    m, d = ep.inner.sim.model, ep.inner.sim.data
    rack = np.array(d.body_xpos[m.body_name2id("wine_rack_1_main")])
    normal = np.array([0.0, -0.515, 0.857])           # 선반 홈 면에서 떨어지는 방향 (축에 수직, 위)
    if not wine_grasp_piper(ep, record, place=(rack + WINE_RACK_BOTTOM, WINE_RACK_AXIS, normal)):
        return False
    # 선반 위 모서리(1.25m) + 든 병이 손 아래로 처지는 길이(약 8cm) 위로 지나간다
    return wine_place_piper(ep, rack + WINE_RACK_BOTTOM, WINE_RACK_AXIS, normal, record, z_safe=1.36)


def wine_to_cabinet_piper(ep, record=None):
    m, d = ep.inner.sim.model, ep.inner.sim.data
    sid = m.site_name2id("wooden_cabinet_1_top_side")
    c = np.array(d.site_xpos[sid])
    half = np.abs(np.array(d.site_xmat[sid]).reshape(3, 3)) @ np.array(m.site_size[sid])
    bottom = np.array([c[0] - 0.4 * half[0], c[1], c[2] + 0.004])    # 윗면 가운데에서 로봇 쪽으로 (손이 덜 뻗게)
    up = np.array([0, 0, 1.0])
    if not wine_grasp_piper(ep, record, place=(bottom, up, up)):
        return False
    # 든 병 바닥이 손보다 7.4cm 아래 → 캐비닛 윗면(1.128m) 앞 모서리에 걸리지 않게 손을 1.24m 위로 지나가게
    #   (관절 길로 곧장 가면 병이 캐비닛 앞에 부딪혀 탁자에 떨어졌다, 5장면 중 5번)
    return wine_place_piper(ep, bottom, up, up, record, above=0.05, z_safe=c[2] + WINE_GRASP_DZ + 0.04)


# T1·T4·T8 그릇 옮기기: 집기(grasp_obj) + 놓기(carry_place, 목적지는 원래 모델이 놓던 위치 실측값)
def bowl_task(ep, task, record=None):
    if not grasp_bowl_safe(ep, record=record):
        return False
    se.carry_place(ep, se.bowl_place_target(ep, task), record=record)     # PiPER 캐비닛 위 자리는 se 에서 읽음
    return True


def push_plate_regrip2(ep, record=None, speed=1.0, lead=0.05, press=0.006, tol=0.012, max_steps=320,
                       regrips=5, stall_steps=15, above=0.08, lost=0.03):
    """push_plate_regrip 의 개선판 (2026-10-02).

    실패 원인 (평가 장면 20~49 에서 30번 중 2번, 전환 A4→B5 5번, A8→B5 3번):
      마지막에 옆(−x)으로 고쳐 미는 단계에서, 손을 미는 방향으로 일정 속도로 보내는 힘이 너무 커서
      접시가 따라오지 않아도 손만 달려 나갔다. 손이 접시 중심에서 10cm 밖(가장자리 너머)까지 가서
      옆의 크림치즈 근처에 걸린 채 멈췄고, 접시는 목표에서 4.4cm(영역 ±4cm 밖)에 남았다.
    고친 것:
      ① 손이 있어야 할 자리(접시 + 미는 점)에서 3cm 넘게 벗어나면 접시를 놓친 것으로 보고 바로 다시 잡는다
      ② 다시 잡기 2번 → 5번
      ③ 손이 제자리에서 2cm 넘게 벗어나 있으면 앞으로 보내는 속도를 30%로 줄여 손을 먼저 제자리로"""
    mat = _aim(ep, ep.home[1].copy())
    goal = site_pos(ep, "main_table_stove_front_region")
    p0 = ep.obj_pos("plate_1")
    g = p0 + PLATE_REL
    se.hold(ep, -1.0, 3, record)
    here = sx.eef_pos(ep.obs)
    pre = g + [0, 0, above]
    mid = (here + pre) / 2
    mid[2] = max(here[2], pre[2]) + 0.02
    se.servo(ep, mid, mat, -1.0, tol=0.04, record=record)
    se.servo(ep, pre, mat, -1.0, tol=0.01, vmax=0.4, record=record)
    se.servo(ep, g - [0, 0, press], mat, -1.0, tol=0.006, vmax=0.25, max_steps=50, record=record)
    hist = []
    left = regrips
    se.PRECISE = True                 # 접촉 구간: DART 잡음 없음
    for _ in range(max_steps):
        p = ep.obj_pos("plate_1")
        err = goal[:2] - p[:2]
        if np.linalg.norm(err) < tol:
            break
        hist.append(p[:2].copy())
        dirn = err / (np.linalg.norm(err) + 1e-6)
        hand = sx.eef_pos(ep.obs)
        tgt = p[:2] + PLATE_REL[:2] + dirn * lead
        off = np.linalg.norm(tgt - hand[:2])
        stalled = len(hist) > stall_steps and np.linalg.norm(hist[-1] - hist[-1 - stall_steps]) < 0.003
        if left > 0 and (stalled or off > lost + lead):
            left -= 1
            hist = []
            up = hand + [0, 0, 0.05]
            se.servo(ep, up, mat, -1.0, tol=0.01, vmax=0.4, max_steps=20, record=record)
            ng = np.concatenate([p[:2] + dirn * 0.045, [p[2] + PLATE_REL[2]]])
            se.servo(ep, ng + [0, 0, 0.05], mat, -1.0, tol=0.01, vmax=0.4, max_steps=30, record=record)
            se.servo(ep, ng - [0, 0, press], mat, -1.0, tol=0.006, vmax=0.25, max_steps=30, record=record)
            continue
        ff = speed * np.clip(np.linalg.norm(err) / 0.05, 0.4, 1.0) * (1.0 if off < 0.02 + lead else 0.3)
        v = dirn * ff + np.clip((tgt - hand[:2]) / sx.POS_SCALE, -0.3, 0.3) * 0.5
        vz = np.clip((p[2] + PLATE_REL[2] - press - hand[2]) / sx.POS_SCALE, -0.4, 0.4) - 0.2
        rerr = Rotation.from_matrix(mat @ ep.obs["robot_state"]["eef"]["mat"].T).as_rotvec()
        a = np.concatenate([np.clip(v, -1, 1), [np.clip(vz, -0.5, 0.5)],
                            np.clip(rerr / sx.ROT_SCALE, -0.3, 0.3), [-1.0]]).astype(np.float32)
        if record is not None:
            record(ep.obs, a)
        ep.step(se._exec(a), "E")
    se.PRECISE = False                # 접촉 구간 끝: 다시 잡음 허용
    se.servo(ep, sx.eef_pos(ep.obs) + [0, 0, 0.07], mat, -1.0, tol=0.02, max_steps=25, record=record)


def push_plate_outer(ep, record=None, tol=0.012, tries=2, gap=0.085, low=0.012, speed=0.15, max_steps=80):
    """접시가 목표 영역 밖에 멈췄을 때만 하는 마무리 (2026-10-02).

    안쪽 테두리를 끄는 방식은 옆(−x)으로 고칠 때 손가락이 낮은 테두리(중심에서 4.8cm, 높이 1cm 남짓)
    위로 넘어가 접시가 안 움직였다 (평가 장면 33: 손 6cm 이동, 접시 0cm). 그래서 그리퍼를 닫아 막대처럼
    만들고, 접시 **바깥** 반대편(중심에서 8.5cm)에서 탁자 가까이(접시 기준 +1.2cm) 내려 테두리를 민다.
    한 번 밀 때 방향과 거리를 처음에 정하고, 접시가 그만큼 가면 바로 멈춘다
    (목표를 따라가며 밀었더니 지나쳐 6cm 넘어가고 옆 물체에 올라탔다)."""
    goal = site_pos(ep, "main_table_stove_front_region")
    mat = _aim(ep, ep.home[1].copy())
    for _ in range(tries):
        p0 = ep.obj_pos("plate_1")
        err = goal[:2] - p0[:2]
        dist = np.linalg.norm(err)
        if dist < tol:
            return True
        dirn = err / (dist + 1e-6)
        start = np.r_[p0[:2] - dirn * gap, p0[2] + low]
        se.servo(ep, sx.eef_pos(ep.obs) + [0, 0, 0.06], mat, 1.0, tol=0.02, max_steps=25, record=record)
        se.servo(ep, start + [0, 0, 0.07], mat, 1.0, tol=0.012, vmax=0.4, max_steps=50, record=record)
        se.servo(ep, start, mat, 1.0, tol=0.008, vmax=0.25, max_steps=40, record=record)
        se.PRECISE = True
        for _ in range(max_steps):
            p = ep.obj_pos("plate_1")
            moved = float(np.dot(p[:2] - p0[:2], dirn))
            if moved >= dist - 0.004:                      # 정한 거리만큼 갔으면 멈춘다
                break
            hand = sx.eef_pos(ep.obs)
            side = (hand[:2] - p[:2]) - np.dot(hand[:2] - p[:2], dirn) * dirn   # 미는 선에서 옆으로 벗어난 만큼
            v = dirn * speed - np.clip(side / sx.POS_SCALE, -0.2, 0.2)
            vz = np.clip((p0[2] + low - hand[2]) / sx.POS_SCALE, -0.3, 0.3)
            rerr = Rotation.from_matrix(mat @ ep.obs["robot_state"]["eef"]["mat"].T).as_rotvec()
            a = np.concatenate([v, [vz], np.clip(rerr / sx.ROT_SCALE, -0.3, 0.3), [1.0]]).astype(np.float32)
            if record is not None:
                record(ep.obs, a)
            ep.step(se._exec(a), "E")
        se.PRECISE = False
        se.servo(ep, sx.eef_pos(ep.obs) - np.r_[dirn * 0.02, 0] + [0, 0, 0.07], mat, -1.0, tol=0.02, max_steps=25,
                 record=record)
    p = ep.obj_pos("plate_1")
    return np.linalg.norm(goal[:2] - p[:2]) < tol


def push_plate_v3(ep, record=None):
    """원래 방식(push_plate_regrip) 그대로 하고, 목표 영역 밖에 멈췄을 때만 바깥에서 밀어 마무리."""
    push_plate_regrip(ep, record)
    goal = site_pos(ep, "main_table_stove_front_region")
    if np.linalg.norm(goal[:2] - ep.obj_pos("plate_1")[:2]) > 0.012:
        push_plate_outer(ep, record)


EXPERT = {
    0: lambda ep, rec=None: open_drawer_grasp(ep, "middle", record=rec) if _robot(ep) == "piper" else open_middle_hook(ep, rec),
    1: lambda ep, rec=None: bowl_task(ep, 1, rec),
    2: lambda ep, rec=None: wine_to_cabinet_piper(ep, rec) if _robot(ep) == "piper" else wine_to_cabinet(ep, rec),
    3: lambda ep, rec=None: drawer_bowl_piper(ep, rec) if _robot(ep) == "piper" else drawer_bowl(ep, rec, y_front=0.03, pull=0.10),
    4: lambda ep, rec=None: bowl_task(ep, 4, rec),
    5: lambda ep, rec=None: push_plate_v3(ep, rec),        # 10/2: 바깥에서 밀어 마무리 (30장면 T5 28→30, A4→B5 25→30, A8→B5 27→30)
    6: lambda ep, rec=None: cheese_to_bowl(ep, rec),
    7: lambda ep, rec=None: turn_on_stove_piper(ep, rec) if _robot(ep) == "piper" else turn_on_stove(ep, rec),
    8: lambda ep, rec=None: bowl_task(ep, 8, rec),
    9: lambda ep, rec=None: wine_to_rack_piper(ep, rec) if _robot(ep) == "piper" else wine_to_rack(ep, rec),
}


def _grasp_bowl_piper(ep, obj, cands, record=None, above=0.08, min_margin=np.radians(8), keepout=()):
    """PiPER 그릇 집기: 후보 테두리마다 (위 → 잡는 점 → 들어 올린 점) 길을 역기구학으로 풀어
    관절 한계·손목 특이 자세 여유가 min_margin 이상인 첫 후보(정렬 순서)를 관절 길로 따라간다.
    손끝 변화량만으로 가면 서랍을 연 뒤의 꺾인 자세에서 손목이 특이 자세로 내려가 헛잡았다 (T3 ep2001)."""
    import piper_sim.ik as pik
    plans = []
    m, d = ep.inner.sim.model, ep.inner.sim.data
    base = np.array(d.body_xpos[m.body_name2id("robot0_base")])
    for g, gm in cands:
        # 손을 똑바로 아래로만 두면 로봇 가까운 낮은 곳에서 5번 관절이 0도(특이 자세)가 된다 → 15° 기울인 방향도 후보로
        radial = (g - base) * [1, 1, 0]
        axis = np.cross([0, 0, 1.0], radial / (np.linalg.norm(radial) + 1e-9))
        mats = []
        for tilt in (0, 15, -15):
            Rt = Rotation.from_rotvec(axis * np.radians(tilt)).as_matrix()
            mats += [Rt @ gm, Rt @ gm @ _RZ180]
        b = pik.plan_branch(ep, [g + [0, 0, above], g, g + [0, 0, 0.10]], mats)
        if b is not None:
            plans.append((b[0] < min_margin, b, g))
    plans.sort(key=lambda x: x[0])                 # 여유 충분한 후보 먼저, 그 안에서는 원래 순서
    for _, (mg, gm, q_above), g in plans[:3]:
        pik.servo_to_q(ep, q_above, -1.0, record=record, tol=0.01, keepout=keepout, lift_z=1.20)
        if not se.servo(ep, g, gm, -1.0, tol=0.008, vmax=0.3, record=record):
            se.servo(ep, g + [0, 0, above], gm, -1.0, tol=0.03, max_steps=30, record=record)
            continue
        se.hold(ep, 1.0, 12, record)
        z0 = ep.obj_pos(obj)[2]
        se.servo(ep, g + [0, 0, 0.10], gm, 1.0, tol=0.02, record=record)
        if ep.obj_pos(obj)[2] > z0 + 0.04:
            return True
        se.hold(ep, -1.0, 8, record)
    return False


def grasp_bowl_safe(ep, obj="akita_black_bowl_1", off=None, record=None, above=0.08, prefer=None, keepout=()):
    """se.grasp_obj 와 같지만 손목 방향을 고를 때 7번 관절 한계를 본다 (_aim), 많이 돌아야 하면 나눠 돌린다.
    서랍을 열며 손목을 크게 돌려 둔 뒤 그릇을 다시 집을 때, 원래 방식은 손목이 한계 쪽으로 돌다
    그릇에 못 갔다 (A8→B0 뒤 원래 일로 돌아가기 0/10)."""
    off = se.BOWL_GRASP_OFFSET if off is None else off
    down = ep.home[1].copy()
    se.hold(ep, -1.0, 6, record)
    cands = []
    for g, gm in se.grasp_candidates(ep, obj, off, down):
        gm2 = _aim(ep, gm)
        rel = Rotation.from_matrix(ep.obs["robot_state"]["eef"]["mat"].T @ gm2)
        pred = _j7(ep) + rel.as_rotvec()[2]
        cands.append((abs(pred) > _jlimit(ep), -round(g[2], 3), rel.magnitude(), g, gm2))
    if prefer is not None:                         # 예: PiPER 서랍에 넣을 때 캐비닛에서 먼 쪽(−x) 테두리
        cands.sort(key=lambda c: (c[0], prefer(c[3] - ep.obj_pos(obj)), c[2]))
    else:
        cands.sort(key=lambda c: c[:3])
    if _robot(ep) == "piper":
        return _grasp_bowl_piper(ep, obj, [(c[3], c[4]) for c in cands], record, above, keepout=keepout)
    for _, _, rot, g, gm in cands[:3]:
        if rot > np.radians(90):
            rotate_staged(ep, gm, record, lift=0.04)
        if not se.servo(ep, g + [0, 0, above], gm, -1.0, tol=0.025, record=record):
            continue
        if not se.servo(ep, g, gm, -1.0, tol=0.008, vmax=0.3, record=record):
            se.servo(ep, g + [0, 0, above], gm, -1.0, tol=0.03, max_steps=30, record=record)
            continue
        se.hold(ep, 1.0, 12, record)
        z0 = ep.obj_pos(obj)[2]
        se.servo(ep, g + [0, 0, 0.10], gm, 1.0, tol=0.02, record=record)
        if ep.obj_pos(obj)[2] > z0 + 0.04:
            return True
    return False
