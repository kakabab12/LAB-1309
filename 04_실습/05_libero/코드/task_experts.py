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


def open_middle_hook(ep, record=None, pull=0.15):
    mat = _demo_mat(0, 102, 104)
    h = handle_pos(ep, "middle")
    se.hold(ep, -1.0, 3, record)
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


# T3 "위 서랍을 열고 그릇을 넣어라": 위 서랍 열기 → 그릇 집기 → 서랍 안(앞쪽, 캐비닛 윗판 밖)에 놓기
TOP_FLOOR_Z = 1.064      # 위 서랍 바닥 윗면 (geom g16: 중심 1.060, 두께 반 0.004)
BOWL_REST_DZ = -0.002    # 탁자 위 그릇 몸체 높이 0.898 − 탁자면 0.900


def region_pos(ep, site):
    m, d = ep.inner.sim.model, ep.inner.sim.data
    return np.array(d.site_xpos[m.site_name2id(site)])


def drawer_bowl(ep, record=None, y_front=0.05, pull=0.10):
    """pull: 서랍을 얼마나 열까. 끝까지(16cm) 열면 앞판이 그릇 위로 와서 손이 내려가지 못한다 (0/10)."""
    opened = open_top_to(ep, pull, record=record)
    if opened < pull - 0.03:
        return False
    # 열린 서랍 앞판·손잡이(이제 y≈0.02)에 걸리지 않게: 앞쪽 높은 곳을 거쳐 그릇 위에서 수직으로
    b = ep.obj_pos("akita_black_bowl_1")
    mat = ep.home[1].copy()
    se.servo(ep, np.array([sx.eef_pos(ep.obs)[0], 0.12, 1.19]), mat, -1.0, tol=0.03, record=record)
    se.servo(ep, np.array([b[0], 0.10, 1.19]), mat, -1.0, tol=0.03, record=record)
    if not grasp_front(ep, "akita_black_bowl_1", se.BOWL_GRASP_OFFSET, record=record):
        return False
    c = region_pos(ep, "wooden_cabinet_1_top_region")
    P = np.array([c[0], c[1] + y_front, TOP_FLOOR_Z + BOWL_REST_DZ])
    se.carry_place(ep, P, record=record, above=0.06)
    return True


def grasp_front(ep, obj, off, record=None, above=0.08):
    """se.grasp_obj 와 같지만 **앞쪽(+y, 서랍 반대편) 테두리부터** 잡는다.
    열린 위 서랍이 그릇 위로 튀어나와, 뒤쪽 테두리로는 손이 내려가지 못한다 (T3 첫 시도 0/10)."""
    down = ep.home[1].copy()
    se.hold(ep, -1.0, 6, record)
    for g, gmat in sorted(se.grasp_candidates(ep, obj, off, down), key=lambda c: -c[0][1])[:2]:
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
    mat = ep.home[1].copy()
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
    for _ in range(80):
        opened = -drawer_qpos(ep, "top")
        if opened >= target - 0.005:
            break
        dp = g + [0, opened + 0.02, 0] - sx.eef_pos(ep.obs)     # 서랍보다 2cm 앞을 목표로 천천히
        a = np.concatenate([np.clip(dp / sx.POS_SCALE, -0.25, 0.25), [0, 0, 0], [-1.0]]).astype(np.float32)
        if record is not None:
            record(ep.obs, a)
        ep.step(se._exec(a), "E")
    # 손을 **똑바로 위로** 뺀다 (살짝 뒤로). 대각선으로 빼면 손끝이 막대를 한 번 더 끌어 14~16cm 까지 열렸다
    se.servo(ep, sx.eef_pos(ep.obs) + [0, -0.005, 0.05], mat, -1.0, tol=0.008, vmax=0.3, max_steps=30, record=record)
    return -drawer_qpos(ep, "top")
