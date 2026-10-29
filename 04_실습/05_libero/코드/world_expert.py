"""
여러 모양·N개 장면의 시범 프로그램 (2026-10-24) — 블록 장면 시범(task_experts.block_task)을 모양·개수에 상관없이

  · 쥐는 방향: 설정의 모양에서 (world.grasp_widths) — 정육면체 90°마다, 막대 짧은 폭, 원기둥 아무 방향(yaw_step 간격),
    각기둥은 면에 수직 (짝수 각형 = 마주 보는 면, 홀수 각형 = 면과 꼭짓점)
  · 피할 물체·지나갈 높이: 이 장면에 있는 물체 전부 (예전에는 블록 4개 이름이 박혀 있었다)
  · 끼어들기 겹치기: A 를 들고 가다 B → A 를 가까운 빈 자리에 내려놓고 B → (B 를 들고 가다 C → …) → 끝나면 쌓아 둔 일로 돌아가
    다시 집어 마친다 (연구 시나리오 research-scenario 의 일반화, 깊이 제한 없음)

  ep = make_episode(sc, ep_idx)          # 장면 sc 로 Episode (switch_experiment.Episode 그대로)
  ok = run_chain(ep, sc, tasks, nested=True, record=rec)
"""
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

import piper_sim.world as W
import scripted_expert as se
import switch_experiment as sx
import task_experts as te

_RZ = lambda deg: Rotation.from_euler("z", deg, degrees=True).as_matrix()  # noqa: E731
STATE = {"objs": ()}


def grasp_z(obj):
    """물체 바닥에서 쥐는 높이 (설정 grasp). z_min 이 있으면 min(max(h/2, z_min), h − top_margin), 없으면 예전 규칙."""
    G = W.CFG["grasp"]
    h = W.height(obj)
    if "z_min" in G:
        return min(max(h / 2, G["z_min"]), h - G.get("top_margin", 0.012))
    return G["z_short"] if te.NOPED and h < G["short_below"] else min(G["z_max"], h / 2)


TOPPLE_DEG = 20.0      # 10/29: 이보다 기운 물체는 넘어진 것으로 보고 toppled_candidates 로 쥔다


def shape_points(obj):
    """그 물체 겉의 꼭짓점·둘레 점 (물체 좌표, 바닥 가운데가 원점 — world._xml 과 같음)."""
    s = W.CFG["shapes"][W.parse(obj)[1]]
    h = W.height(obj)
    if s["type"] == "box":
        a, b, _ = s["size"]
        return np.array([[x, y, z] for x in (-a / 2, a / 2) for y in (-b / 2, b / 2) for z in (0.0, h)])
    if s["type"] == "cylinder":
        r, t = s["radius"], np.linspace(0, 2 * np.pi, 24, endpoint=False)
    else:
        r, t = s["radius"], 2 * np.pi * np.arange(int(s["sides"])) / int(s["sides"])
    ring = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    return np.concatenate([np.c_[ring, np.zeros(len(ring))], np.c_[ring, np.full(len(ring), h)]])


def tilt_deg(ep, obj):
    return float(np.degrees(np.arccos(np.clip(se.obj_rot(ep, obj).apply([0, 0, 1.0])[2], -1, 1))))


def lowest_offset(ep, obj):
    """물체 원점(바닥 가운데) 높이 − 지금 자세에서 가장 낮은 겉 점 높이. 똑바로 서 있으면 0."""
    P = se.obj_rot(ep, obj).apply(shape_points(obj))
    return float(-P[:, 2].min())


def toppled_candidates(ep, obj):
    """넘어진 물체 (10/29): DAgger 6차 이어받기 실패 15장면 중 12장면이 VLA 가 넘어뜨린 목표(기울기 90°)였고, 원래 후보는 물체의
    기운 축을 그대로 따라 손을 옆으로 눕혀 쥐려 해 길을 못 찾았다 (정육면체·막대는 옆으로 누워도 같은 모양인데도).
    지금 자세의 겉 점으로 높이 H 와 수평 방향별 폭을 재, 폭이 그리퍼보다 좁은 방향으로 위에서 내려와 쥔다 (좁은 폭부터).
    쥐는 높이 = 설정 grasp 규칙을 H 에 (min(max(H/2, z_min), H − top_margin))."""
    G = W.CFG["grasp"]
    P = se.obj_rot(ep, obj).apply(shape_points(obj)) + ep.obj_pos(obj)
    zlo, zhi = float(P[:, 2].min()), float(P[:, 2].max())
    H = zhi - zlo
    c = 0.5 * (P.min(axis=0) + P.max(axis=0))
    zg = min(max(H / 2, G.get("z_min", 0.028)), H - G.get("top_margin", 0.012))
    g = np.array([c[0], c[1], zlo + zg])
    ref = float(np.degrees(np.arctan2(te.DOWN_REF[1, 0], te.DOWN_REF[0, 0])))     # 손가락이 벌어지는 축의 처음 방향
    out = []
    for phi in range(0, 180, 15):
        u = np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi)), 0.0])
        w = float((P @ u).max() - (P @ u).min())
        if w < G["gripper_max_width"]:
            out.append((w, g, _RZ(phi - ref) @ te.DOWN_REF))
    out.sort(key=lambda x: x[0])
    return [(gg, M) for _, gg, M in out]


def grasp_candidates(ep, obj):
    if tilt_deg(ep, obj) > TOPPLE_DEG:
        return toppled_candidates(ep, obj)
    o = ep.obj_pos(obj)
    h = W.height(obj)
    R = se.obj_rot(ep, obj)
    g = o + R.apply([0, 0, grasp_z(obj)])
    _, shape, _ = W.parse(obj)
    s = W.CFG["shapes"][shape]
    widths = W.grasp_widths(obj)
    if s["type"] == "cylinder":          # 둘레 어디나 같다 → 후보 수만 줄인다 (설정 yaw_step)
        step = s.get("yaw_step", 45)
        widths = {k: v for k, v in widths.items() if k % step == 0} or widths
    out = []
    for yaw, w in widths.items():
        if w < W.CFG["grasp"]["gripper_max_width"]:
            out.append((g, R.as_matrix() @ _RZ(yaw) @ te.DOWN_REF))
    return out


def keepout(ep, exclude=(), margin=0.04, top=0.03):
    out = []
    for o in STATE["objs"]:
        if o in exclude:
            continue
        c = ep.obj_pos(o)
        a, b, h = W.size_box(o)
        r = float(np.hypot(a, b) / 2) + margin
        out.append((np.array([c[0] - r, c[1] - r, 0.80]), np.array([c[0] + r, c[1] + r, c[2] + h + top])))
    return out


def install():
    """task_experts 의 블록 장면 부분을 이 장면 것으로 바꾼다 (PIPER_BLOCKS 없이)."""
    te._is_blocks = lambda: True
    te.block_size = W.size_box
    te.block_grasp_candidates = grasp_candidates
    te.block_keepout = keepout


class Goal:
    """switch_experiment.GoalChecker 와 같은 꼴 (LIBERO On 술어)."""

    def __init__(self, obj, dst, language=""):
        self.goal = [["on", obj, dst]]
        self.language = language or W.instruction(obj, dst)

    def __call__(self, env):
        inner = env._env.env
        return all(inner._eval_predicate(s) for s in self.goal)


def make_episode(sc, ep_idx, tasks=None, args=None):
    install()
    a = args or sx.fill_defaults(sx.default_args())
    a.task_a, a.task_b, a.strategy = 0, None, "none"
    o, d = (tasks[0][0], tasks[0][1]) if tasks else (sc.objects[0], (sc.pads or sc.objects)[0])
    r = SimpleNamespace(args=a, suite=W.Suite(sc, (o, d), W.instruction(o, d)), chk_a=Goal(o, d), chk_b=None,
                        policy=None, rf=None, noise_shift=None, assist=None)
    ep = sx.Episode(r, ep_idx)
    ep.inner.horizon = 100000
    ep.obs = W.place(ep.env, sc)
    STATE["objs"] = tuple(sc.objects)
    ep.scene = sc
    return ep


def target(ep, dst):
    """dst 위에 놓을 때 옮기는 물체 바닥 자리."""
    return ep.obj_pos(dst) + [0, 0, W.height(dst) + 0.002]


def z_safe(ep, obj):
    tops = [ep.obj_pos(o)[2] + W.height(o) for o in STATE["objs"] if o != obj]
    return max(tops + [W.TABLE_Z]) + min(0.025, W.height(obj) / 2) + 0.03


def why(ep, obj):
    """쥐기 실패 원인을 볼 값: 물체 기울기(도), 밑동에서 거리·각도, 높이."""
    R = se.obj_rot(ep, obj)
    tilt = float(np.degrees(np.arccos(np.clip(R.apply([0, 0, 1.0])[2], -1, 1))))
    p = ep.obj_pos(obj)
    b = np.array(ep.scene.base)
    d = p[:2] - b
    return {"기울기": round(tilt, 1), "거리": round(float(np.linalg.norm(d)), 3),
            "각도": round(float(np.degrees(np.arctan2(d[1], d[0]))), 1), "높이": round(float(p[2] - W.TABLE_Z), 3)}


def grasp(ep, obj, dest=None, record=None):
    """놓을 자리(dest)에서도 닿는 쥐는 방향을 먼저 찾고, 없으면 놓을 자리를 묻지 않고 쥔다 (놓을 때 carry_place 가
    물체 수직축 둘레로 손을 돌려 닿는 자세를 다시 찾는다). 빈 자리에 내려놓은 블록을 다시 쥘 때 dest 조건으로
    후보가 모두 빠져 '다시 못 쥠' 이 났다 (world_chain1 k=2: 10장면 중 3)."""
    if te.grasp_bowl_safe(ep, obj, record=record, dest=dest):
        return True
    return dest is not None and te.grasp_bowl_safe(ep, obj, record=record, dest=None)


def place(ep, obj, P, record=None):
    if tilt_deg(ep, obj) > TOPPLE_DEG:                 # 10/29: 누운 물체는 원점(바닥 가운데)이 옆에 있어 가장 낮은 점이 판에 닿게
        P = np.asarray(P, dtype=float) + [0, 0, lowest_offset(ep, obj)]
    se.carry_place(ep, P, record=record, above=0.06, obj=obj, z_safe=z_safe(ep, obj))


def free_spot(ep, obj, avoid=(), near=None, rng=None, tries=3000):
    """책상에서 다른 물체·판·앞으로 쓸 자리에서 떨어진 빈 자리 (손이 닿는 범위 안). near 가 있으면 가까운 쪽부터."""
    rng = rng or np.random.default_rng(len(ep.acts))
    reach = W.reach_map()
    base = np.array(ep.scene.base)
    tb = W.CFG.get("table", {})
    half = np.array(tb.get("size", [9, 9])) / 2 - tb.get("margin", 0.0)
    fr = W.footprint(obj)
    others = [(ep.obj_pos(n)[:2], W.footprint(n)) for n in ep.scene.names if n != obj] + list(avoid)
    gap = W.CFG.get("putdown_gap", 0.05)
    cands = []
    for _ in range(tries):
        p = W._ws_sample(rng, reach, fr, 0.0) + base
        if np.any(np.abs(p) > half - fr):
            continue
        if all(np.linalg.norm(p - q) > fr + fq + gap for q, fq in others):
            cands.append(p)
        if len(cands) >= 40:
            break
    if not cands:
        return None
    if near is not None:
        cands.sort(key=lambda p: np.linalg.norm(p - near[:2]))
    return np.array([cands[0][0], cands[0][1], W.TABLE_Z + 0.002])


def carry_toward(ep, P, steps=30, record=None):
    """쥔 채 목적지 쪽으로 조금 (끼어들기가 '옮기는 도중'에 오게)."""
    h = sx.eef_pos(ep.obs)
    M = ep.obs["robot_state"]["eef"]["mat"].copy()
    tgt = h + 0.35 * (np.array([P[0], P[1], h[2]]) - h)
    se.servo(ep, tgt, M, 1.0, max_steps=steps, tol=0.02, record=record)


def run_chain(ep, tasks, nested=True, record=None, log=None, say=None):
    """tasks = [(물체, 놓을 곳, 지시문), …]. nested=True 면 i 번째 일을 들고 가는 도중 i+1 번째가 끼어든다 (깊이 = len-1).
    nested=False 면 차례로. say(지시문): 지금 로봇이 따르는 지시가 바뀔 때마다 (학습 데이터를 지시문별 구간으로 나눔).
    끼어든 순간부터 A 를 내려놓는 것까지는 새 지시(B) 구간이고, B 가 끝나 A 를 다시 집는 것부터는 다시 A 구간이다.
    돌려줌: 끝났을 때 모든 일이 되어 있는가, 일별 결과"""
    log = log if log is not None else []
    say = say or (lambda lang: None)

    def do(i):
        o, d, lang = tasks[i]
        say(lang)
        P = target(ep, d)
        if not grasp(ep, o, dest=P, record=record):
            log.append((i, "못 쥠", str(o), why(ep, o)))
            return False
        if nested and i + 1 < len(tasks):
            carry_toward(ep, P, record=record)
            spot = free_spot(ep, o, avoid=[(target(ep, t[1])[:2], W.footprint(t[1])) for t in tasks],
                             near=sx.eef_pos(ep.obs))
            if spot is None:
                log.append((i, "빈 자리 없음"))
                return False
            say(tasks[i + 1][2])                       # 끼어들기: 여기부터 새 지시 (A 내려놓기 포함)
            place(ep, o, spot, record=record)
            log.append((i, "내려놓음", why(ep, o)))
            do(i + 1)                                   # 끼어든 일 (그 안에서 또 끼어들 수 있음)
            say(lang)                                   # 돌아오기
            P = target(ep, d)
            if not grasp(ep, o, dest=P, record=record):
                log.append((i, "다시 못 쥠", str(o), why(ep, o)))
                return False
        place(ep, o, target(ep, d), record=record)
        log.append((i, "놓음"))
        return True

    if nested:
        do(0)
    else:
        for i in range(len(tasks)):
            o, d, lang = tasks[i]
            say(lang)
            if grasp(ep, o, dest=target(ep, d), record=record):
                place(ep, o, target(ep, d), record=record)
                log.append((i, "놓음"))
            else:
                log.append((i, "못 쥠", str(o), why(ep, o)))
    for _ in range(15):
        ep.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32), "E")
    each = [bool(Goal(o, d)(ep.env)) for o, d, _ in tasks]
    return all(each), each


def planner_reach(n_r=13, n_t=25, min_margin_deg=8, above=0.08):
    """팔 닿는 범위를 시범 프로그램이 실제로 쓰는 길 찾기(관절 한계 여유 + 팔-책상·물체 충돌)로 잰다 (10/24).
    역기구학만 본 reach_map 은 밑동 15~19cm 도 닿는다고 했지만, 거기 놓인 블록은 쥐는 길을 못 찾았다 (world_chain2).
    격자마다 정육면체 하나를 놓고 쥐는 후보(모양 대칭 × 기울기 0·±15·±25·±35°)로 위 → 쥐는 점 → 들어 올린 점 길이 있으면 닿음."""
    import json
    import hashlib
    from pathlib import Path
    import piper_sim.piper_robot as pr
    import piper_sim.ik as pik
    reach0 = W.reach_map()
    j1lo, j1hi = reach0["joint1_deg"]
    a0 = W.CFG["workspace"].get("angle_margin", 10)
    thetas = np.linspace(j1lo + a0, j1hi - a0, n_t)
    rs = np.linspace(*W.CFG["workspace"]["r_search"], n_r)
    col = next(iter(W.CFG["object_colors"]))
    obj = W.inst(W.obj_cat(next(iter(W.CFG["shapes"])), col))
    sc = W.Scene(seed=0, objects=[obj], pads=[], pose={obj: (0.3, 0.0, 0.0)}, base=tuple(W.CFG.get("robot_base", [0, 0])))
    ep = make_episode(sc, 0)
    inner = ep.inner
    jnt = inner.objects_dict[obj].joints[0]
    base = np.array(sc.base)
    ok = np.zeros((n_t, n_r), bool)
    for i, th in enumerate(thetas):
        t = np.radians(th)
        for k, r in enumerate(rs):
            qp = inner.sim.data.get_joint_qpos(jnt).copy()
            qp[:3] = [base[0] + r * np.cos(t), base[1] + r * np.sin(t), W.TABLE_Z + 0.001]
            qp[3:] = [np.cos(t / 2), 0, 0, np.sin(t / 2)]
            inner.sim.data.set_joint_qpos(jnt, qp)
            inner.sim.forward()
            hit = False
            for g, gm in grasp_candidates(ep, obj):
                radial = (g[:2] - base) / (np.linalg.norm(g[:2] - base) + 1e-9)
                axis = np.array([-radial[1], radial[0], 0.0])
                for tilt in (0, 15, -15, 25, -25, 35, -35):
                    Rt = Rotation.from_rotvec(axis * np.radians(tilt)).as_matrix()
                    for M in (Rt @ gm, Rt @ gm @ _RZ(180)):
                        b = pik.plan_branch(ep, [g + [0, 0, above], g, g + [0, 0, 0.10]], [M], n_seeds=4)
                        if b is not None and b[0] >= np.radians(min_margin_deg):
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    break
            ok[i, k] = hit
        print(f"  θ {th:6.1f}°: " + "".join("O" if v else "." for v in ok[i]), flush=True)
    r_min, r_max = [], []
    for i in range(n_t):
        ks = np.nonzero(ok[i])[0]
        if len(ks):
            best, cur = (ks[0], ks[0]), [ks[0], ks[0]]
            for kk in ks[1:]:
                cur = [cur[0], kk] if kk == cur[1] + 1 else [kk, kk]
                if cur[1] - cur[0] > best[1] - best[0]:
                    best = tuple(cur)
            r_min.append(float(rs[best[0]])); r_max.append(float(rs[best[1]]))
        else:
            r_min.append(None); r_max.append(None)
    key = W.planner_key()
    d = {"key": key, "method": "planner", "theta": thetas.tolist(), "r_min": r_min, "r_max": r_max,
         "joint1_deg": [j1lo, j1hi], "grid_r": rs.tolist(), "ok": ok.astype(int).tolist()}
    f = W.planner_file(key)
    f.write_text(json.dumps(d))
    ep.env.close()
    print("저장:", f)
    return d
