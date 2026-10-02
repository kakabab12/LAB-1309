#!/usr/bin/env python
"""
선생이 실패한 장면을 다시 돌려 왜 실패했는지 본다 (2026-10-02)

  T3: 위 서랍이 얼마나 열렸나, 그릇은 어디 있나 (서랍 안 영역 중심과의 차이), 그릇이 기울었나
  T5: 접시가 목표에서 얼마나 떨어져 멈췄나, 다시 잡기를 몇 번 했나
  공통: GIF 저장 (정면 + 손목)

쓰는 법
  python diag_teacher.py --task 3 --episodes 24 32 36
"""
import argparse

import numpy as np
from PIL import Image

import scripted_expert as se
import sim_only as so
import switch_experiment as sx
import task_experts as te
from collect_v6 import settle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    p.add_argument("--gif", action="store_true")
    p.add_argument("--plate2", action="store_true")
    p.add_argument("--plate3", action="store_true")
    p.add_argument("--piper", action="store_true", help="PiPER 장면 (piper_sim)")
    a = p.parse_args()
    se.DART_SIGMA = 0.0
    if a.piper:
        import piper_sim.piper_robot as pr
        pr.use_piper_in_lerobot()
    if a.plate3:
        te.EXPERT[5] = lambda ep, rec=None: te.push_plate_v3(ep, rec)
    if a.plate2:
        te.EXPERT[5] = lambda ep, rec=None: te.push_plate_regrip2(ep, rec)
    r = so.SimRunner(a.task)
    r.args.video = a.gif
    sx.COLORS.setdefault("E", (150, 80, 200))
    for i in a.episodes:
        ep = sx.Episode(r, i)
        ep.inner.horizon = 4000
        info = {}
        if a.task == 3:
            info["drawer_start_cm"] = round(100 * te.drawer_qpos(ep, "top"), 1)
            info["bowl_start"] = np.round(100 * ep.obj_pos("akita_black_bowl_1"), 1).tolist()
            info["bowl_minus_handle_y_cm"] = round(100 * (ep.obj_pos("akita_black_bowl_1") - te.handle_pos(ep, "top"))[1], 1)
        if a.task in (2, 9):
            info["wine_start"] = np.round(100 * ep.obj_pos("wine_bottle_1"), 1).tolist()
            orig2 = ep.step
            wtrace = []

            def step2(act, phase):
                orig2(act, phase)
                wtrace.append(np.r_[ep.obj_pos("wine_bottle_1"), sx.eef_pos(ep.obs), ep.obs["robot_state"]["gripper"]["qpos"][0]])
            ep.step = step2
        if a.task == 5:
            info["plate_start"] = np.round(100 * ep.obj_pos("plate_1"), 1).tolist()
        if a.task == 5:                          # 접시가 움직인 길을 같이 남긴다
            orig = ep.step
            trace = []

            def step(act, phase):
                orig(act, phase)
                trace.append(np.r_[ep.obj_pos("plate_1")[:2], sx.eef_pos(ep.obs)])
            ep.step = step
        te.EXPERT[a.task](ep, None)
        ok = settle(ep, r.chk_a)
        info["ok"] = bool(ok)
        if a.task == 0:
            info["drawer_mid_cm"] = round(-100 * te.drawer_qpos(ep, "middle"), 1)
            info["hand_end"] = np.round(100 * sx.eef_pos(ep.obs), 1).tolist()
        if a.task == 3:
            c = te.region_pos(ep, "wooden_cabinet_1_top_region")
            b = ep.obj_pos("akita_black_bowl_1")
            info["drawer_end_cm"] = round(100 * te.drawer_qpos(ep, "top"), 1)
            info["bowl_end"] = np.round(100 * b, 1).tolist()
            info["bowl_minus_region_cm"] = np.round(100 * (b - c), 1).tolist()
            info["bowl_tilt_deg"] = round(float(np.degrees(np.arccos(np.clip(
                se.obj_rot(ep, "akita_black_bowl_1").apply([0, 0, 1])[2], -1, 1)))), 1)
        if a.task == 5:
            g = te.site_pos(ep, "main_table_stove_front_region")
            pl = ep.obj_pos("plate_1")
            info["plate_end"] = np.round(100 * pl, 1).tolist()
            info["plate_minus_goal_cm"] = np.round(100 * (pl - g), 1).tolist()
            tr = np.array(trace)
            d = np.linalg.norm(tr[:, :2] - g[:2], axis=1)
            info["closest_cm"] = round(100 * d.min(), 1)
            info["closest_at"] = int(d.argmin())
            info["end_dist_cm"] = round(100 * d[-1], 1)
            info["dist_every50"] = [round(100 * x, 1) for x in d[::50]]
            info["goal"] = np.round(100 * g, 1).tolist()
            info["hand_minus_plate_cm"] = {int(k): np.round(100 * (tr[k, 2:4] - tr[k, :2]), 1).tolist() + [round(100 * float(tr[k, 4]), 1)]
                                           for k in range(150, len(tr), 20)}
            info["plate_xy_cm"] = {int(k): np.round(100 * tr[k, :2], 1).tolist() for k in range(150, len(tr), 20)}
            info["hand_end"] = np.round(100 * sx.eef_pos(ep.obs), 1).tolist()
            info["others"] = {n: np.round(100 * ep.obj_pos(n), 1).tolist() for n in ep.inner.objects_dict
                              if n != "plate_1"}
            m, dd = ep.inner.sim.model, ep.inner.sim.data
            cons = set()
            for c in range(dd.ncon):
                cc = dd.contact[c]
                a1, a2 = m.geom_id2name(cc.geom1) or "", m.geom_id2name(cc.geom2) or ""
                if "plate" in a1 or "plate" in a2:
                    cons.add((a1, a2))
            info["plate_contacts"] = sorted(cons)[:8]
            st = ep.inner.object_sites_dict if hasattr(ep.inner, "object_sites_dict") else {}
            reg = st.get("main_table_stove_front_region")
            if reg is not None and hasattr(reg, "size"):
                info["region_size_cm"] = np.round(100 * np.array(reg.size), 1).tolist()
        if a.task in (2, 9):
            w = ep.obj_pos("wine_bottle_1")
            tgt = te.site_pos(ep, "wooden_cabinet_1_top_side") if a.task == 2 else None
            tilt = float(np.degrees(np.arccos(np.clip(se.obj_rot(ep, "wine_bottle_1").apply([0, 0, 1])[2], -1, 1))))
            wt = np.array(wtrace)
            info["wine_end"] = np.round(100 * w, 1).tolist()
            if tgt is not None:
                info["wine_minus_top_cm"] = np.round(100 * (w - tgt), 1).tolist()
            info["wine_tilt_deg"] = round(tilt, 1)
            info["wine_max_z_cm"] = round(100 * float(wt[:, 2].max()), 1)
            lifted = np.where(wt[:, 2] > wt[0, 2] + 0.03)[0]
            info["lift_step"] = int(lifted[0]) if len(lifted) else None
            rel = np.where(wt[:, 6] > 0.033)[0]          # PiPER 는 다 벌려도 0.035
            info["release_step"] = int(rel[-1]) if len(lifted) and len(rel) else None
            info["wine_z_every40"] = [round(100 * x, 1) for x in wt[::40, 2]]
        info["steps"] = len(ep.log["pos"])
        print(f"T{a.task} ep{i}:", info, flush=True)
        if a.gif and ep.frames:
            fr = [Image.fromarray(np.asarray(f)).resize((512, 256)) for f in ep.frames[::4]]
            tag = "piper_" if a.piper else ""
            fr[0].save(f"outputs/media/teacher_{tag}T{a.task}_ep{i}.gif", save_all=True, append_images=fr[1:], duration=120, loop=0)
        ep.env.close()


if __name__ == "__main__":
    main()
