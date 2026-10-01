"""
6차 전환·재개 시범 — 전부 전문가 방식 (2026-10-01)

한 시범 = 세 구간, 구간마다 그 지시문으로 따로 저장한다
  ① A  : 전문가가 A 를 하다가 물체를 쥔 지 3스텝째에 멈춘다 (평가 때 전환 시점과 같다)
  ② B  : 같은 물체를 다른 곳에 놓는 일이면 쥔 채 옮기고(방향 틀기),
          아니면 쥔 물체를 그 자리에 바로 세워 내려놓고 전문가가 B 를 한다
  ③ A' : B 를 마치면 전문가가 다시 A 를 마무리한다 (원래 일로 돌아가기)

⛔ 처음 자세로 돌아가지 않는다: 모든 이동은 지금 자리에서 다음 목표로 바로 간다.
   전환 뒤 손이 처음 자세 7cm 안으로 들어간 시범은 버린다.
"""
import numpy as np

import collect_expert as ce
import scripted_expert as se
import sim_only as so
import switch_experiment as sx
import task_experts as te
from collect_v6 import ep_range, settle

OBJ = {1: "akita_black_bowl_1", 3: "akita_black_bowl_1", 4: "akita_black_bowl_1", 8: "akita_black_bowl_1",
       2: "wine_bottle_1", 9: "wine_bottle_1", 6: "cream_cheese_1"}
BOWL_DST = {1, 4, 8}
WINE_DST = {2, 9}


class Stop(Exception):
    pass


def run_until_held(ep, task, rec, extra=3, limit=400):
    """전문가 A 를 돌리다가 물체를 쥔 지 extra 스텝째에 멈춘다. 쥐었으면 True."""
    orig = ep.step
    st = {"t": None, "n": 0}

    def step(a, phase):
        orig(a, phase)
        st["n"] += 1
        if st["t"] is None and ep.holding():
            st["t"] = st["n"]
        if (st["t"] is not None and st["n"] >= st["t"] + extra) or st["n"] >= limit:
            raise Stop
    ep.step = step
    try:
        te.EXPERT[task](ep, rec)
    except Stop:
        pass
    finally:
        ep.step = orig
    return st["t"] is not None and ep.holding()


def put_down_any(ep, obj, z_rest, record=None, shift=(0.0, 0.0)):
    """쥔 물체를 내려 탁자에 놓고 손을 든다 (물체를 처음 높이로). shift 만큼 옆으로 옮겨 놓을 수 있다.
    B 가 서랍 일이면 서랍이 나올 자리(캐비닛 앞)를 비켜 앞쪽에 놓는다 — 그대로 두면 열린 서랍이
    그릇 위로 튀어나와, 나중에 원래 일로 돌아갈 때 그릇을 다시 집지 못했다 (A8→B0 재개 0/10)."""
    mat = ep.obs["robot_state"]["eef"]["mat"].copy()
    hand = sx.eef_pos(ep.obs)
    dz = ep.obj_pos(obj)[2] - (z_rest + 0.004)
    if shift[0] or shift[1]:
        se.servo(ep, hand + [shift[0], shift[1], 0.015], mat, 1.0, tol=0.01, vmax=0.3, max_steps=40, record=record)
        hand = sx.eef_pos(ep.obs)
        dz = ep.obj_pos(obj)[2] - (z_rest + 0.004)
    se.servo(ep, hand - [0, 0, dz], mat, 1.0, tol=0.006, vmax=0.25, max_steps=60, record=record)
    se.hold(ep, -1.0, 8, record)
    se.servo(ep, sx.eef_pos(ep.obs) + [0, 0, 0.08], mat, -1.0, tol=0.02, max_steps=25, record=record)


def do_b(ep, at, bt, rec, z_rest):
    obj = OBJ.get(at)
    se.hold(ep, 1.0, 6, rec)          # 쥔 지 3스텝이면 아직 덜 쥐었다 — 마저 쥔다 (사람도 쥐는 동작을 끝낸다)
    if obj == "akita_black_bowl_1" and bt in BOWL_DST:
        ok, _ = se.redirect(ep, bt, record=rec)                    # 쥔 채 새 목적지로
        return
    if obj == "wine_bottle_1" and bt in WINE_DST:
        m, d = ep.inner.sim.model, ep.inner.sim.data
        if bt == 9:
            rack = np.array(d.body_xpos[m.body_name2id("wine_rack_1_main")])
            te.place_wine(ep, rack + te.WINE_RACK_REL, te.WINE_RACK_EUL, rec)
        else:
            top = te.site_pos(ep, "wooden_cabinet_1_top_side")
            te.place_wine(ep, top + [0, 0, 0.009], te.WINE_GRASP_EUL, rec)
        return
    shift = (0.0, 0.08) if bt in (0, 3) and ep.obj_pos(obj)[1] < 0.08 else (0.0, 0.0)
    put_down_any(ep, obj, z_rest, rec, shift)                     # 내려놓고 (서랍 일이면 서랍 앞을 비켜서)
    te.EXPERT[bt](ep, rec)                                        # B 를 한다


def run(a, out, stats):
    pairs = [tuple(map(int, p.split(":"))) for p in a.pairs.split(",")]
    for at, bt in pairs:
        r = so.SimRunner(at, bt)
        chk_a, chk_b = r.chk_a, r.chk_b
        kb = kr = 0
        for i in ep_range(a.episodes):
            if (out / f"SB{at}{bt}_ep{i}.npz").exists() or (out / f"SX{at}{bt}_ep{i}.txt").exists():
                continue                                    # 이미 해 본 장면은 건너뛴다 (세션이 끊겨도 이어서)
            ep = sx.Episode(r, i)
            obj = OBJ[at]
            z_rest = ep.obj_pos(obj)[2]
            ra = ce.Rec()
            if not run_until_held(ep, at, ra):
                ep.env.close()
                continue
            t_sw = len(ep.log["pos"])
            rb = ce.Rec()
            do_b(ep, at, bt, rb, z_rest)
            ok_b = settle(ep, chk_b)
            mh = se.min_home_dist(ep, t_sw)
            stats["tried"] += 1
            if not ok_b or mh < 0.07:
                (out / f"SX{at}{bt}_ep{i}.txt").write_text("fail")   # 실패도 표시해 두어 다시 하지 않는다
                ep.env.close()
                continue
            ra.save(out / f"SA{at}{bt}_ep{i}.npz", chk_a.language, source="expert_switch_a", pair=f"{at}:{bt}")
            rb.save(out / f"SB{at}{bt}_ep{i}.npz", chk_b.language, source="expert_switch_b", pair=f"{at}:{bt}",
                    min_home_cm=round(100 * mh, 1))
            stats["frames"] += len(ra) + len(rb)
            kb += 1
            # ③ 원래 일로 돌아가기 — B 가 A 의 물체를 써 버린 경우(같은 물체를 다른 곳에, 그릇을 서랍에)는 뺀다
            same_obj = OBJ.get(bt) == obj
            if not same_obj and not chk_a(ep.env):
                rr = ce.Rec()
                t_r = len(ep.log["pos"])
                te.EXPERT[at](ep, rr)
                if settle(ep, chk_a) and chk_b(ep.env) and se.min_home_dist(ep, t_r) >= 0.07:
                    rr.save(out / f"SR{at}{bt}_ep{i}.npz", chk_a.language, source="expert_resume", pair=f"{at}:{bt}")
                    stats["frames"] += len(rr)
                    kr += 1
            ep.env.close()
        stats["per"][f"{at}:{bt}"] = {"b": kb, "resume": kr}
        print(f"  == A{at}→B{bt}: 전환 {kb}, 재개 {kr}", flush=True)
