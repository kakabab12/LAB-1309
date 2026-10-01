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


def _j7(ep):
    m, d = ep.inner.sim.model, ep.inner.sim.data
    return float(d.qpos[m.jnt_qposadr[m.joint_name2id("robot0_joint7")]])


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
        ok = abs(pred) < J7_LIMIT - J7_MARGIN
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


# T3 "위 서랍을 열고 그릇을 넣어라": 위 서랍 열기 → 그릇 집기 → 서랍 안(앞쪽, 캐비닛 윗판 밖)에 놓기
TOP_FLOOR_Z = 1.064      # 위 서랍 바닥 윗면 (geom g16: 중심 1.060, 두께 반 0.004)
BOWL_REST_DZ = -0.002    # 탁자 위 그릇 몸체 높이 0.898 − 탁자면 0.900


def region_pos(ep, site):
    m, d = ep.inner.sim.model, ep.inner.sim.data
    return np.array(d.site_xpos[m.site_name2id(site)])


def drawer_bowl(ep, record=None, y_front=0.05, pull=0.10, clear_y=-1.0, move_y=0.10, carry_above=0.12):
    """pull: 서랍을 얼마나 열까. 끝까지(16cm) 열면 앞판이 그릇 위로 와서 손이 내려가지 못한다 (0/10)."""
    # 그릇이 캐비닛 쪽에 가까이 있으면(앞쪽 테두리 y < clear_y) 먼저 앞으로 옮겨 둔다.
    # 그대로 두면 열린 서랍 앞판에 손이 걸리거나(5회), 그릇이 서랍 길을 막는다(3회) → 30회 중 22회
    b0 = ep.obj_pos("akita_black_bowl_1")
    handle0 = handle_pos(ep, "top")
    if b0[1] < clear_y:
        if grasp_front(ep, "akita_black_bowl_1", se.BOWL_GRASP_OFFSET, record=record):
            se.carry_place(ep, np.array([b0[0], b0[1] + move_y, b0[2]]), record=record, above=0.05)
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


# T1·T4·T8 그릇 옮기기: 집기(grasp_obj) + 놓기(carry_place, 목적지는 원래 모델이 놓던 위치 실측값)
def bowl_task(ep, task, record=None):
    if not grasp_bowl_safe(ep, record=record):
        return False
    se.carry_place(ep, se.bowl_place_target(ep, task), record=record)
    return True


EXPERT = {
    0: lambda ep, rec=None: open_middle_hook(ep, rec),
    1: lambda ep, rec=None: bowl_task(ep, 1, rec),
    2: lambda ep, rec=None: wine_to_cabinet(ep, rec),
    3: lambda ep, rec=None: drawer_bowl(ep, rec, y_front=0.03, pull=0.10),
    4: lambda ep, rec=None: bowl_task(ep, 4, rec),
    5: lambda ep, rec=None: push_plate_regrip(ep, rec),
    6: lambda ep, rec=None: cheese_to_bowl(ep, rec),
    7: lambda ep, rec=None: turn_on_stove(ep, rec),
    8: lambda ep, rec=None: bowl_task(ep, 8, rec),
    9: lambda ep, rec=None: wine_to_rack(ep, rec),
}


def grasp_bowl_safe(ep, obj="akita_black_bowl_1", off=None, record=None, above=0.08):
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
        cands.append((abs(pred) > J7_LIMIT - J7_MARGIN, -round(g[2], 3), rel.magnitude(), g, gm2))
    cands.sort(key=lambda c: c[:3])
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
