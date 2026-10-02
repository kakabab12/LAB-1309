#!/usr/bin/env python
"""
선생(스크립트 전문가) 점검 — 학생이 넘을 수 없는 천장 (2026-10-02)

왜
  목표가 항목마다 95% 이상으로 올라갔다. 학생은 선생을 따라 배우므로 선생이 95%를 넘지 못하는 항목은
  학생도 넘기 어렵다. 그래서 **평가와 같은 장면(20~49)** 에서, **노이즈 없이**, 28개 항목을 모두 잰다.
  실패하면 어느 단계에서 왜 실패했는지 남긴다.

  단독:  te.EXPERT[t]
  전환:  A 를 하다 쥔 지 3스텝째 → 내려놓기/방향 틀기 → B (switch_v6.do_b). 처음 자세 7cm 안이면 실패
  재개:  B 를 마친 뒤 te.EXPERT[A]. 끝났을 때 B 도 유지되어야 하고, 처음 자세 7cm 안이면 실패

쓰는 법
  python teacher_audit.py --tasks 0 1 2 --pairs 8:0,8:3 --episodes 20-49 --out outputs/teacher_audit/p1.json
"""
import argparse
import json
import time
from pathlib import Path

import collect_expert as ce
import scripted_expert as se
import sim_only as so
import switch_experiment as sx
import switch_v6 as sv
import task_experts as te
from collect_v6 import settle

RESUME = {(8, 5), (8, 7), (8, 9), (4, 5), (4, 9), (1, 7)}


def ep_list(s):
    out = []
    for part in s.split(","):
        a, b = part.split("-") if "-" in part else (part, part)
        out += list(range(int(a), int(b) + 1))
    return out


def standalone(task, eps, rec):
    r = so.SimRunner(task)
    res = []
    for i in eps:
        ep = sx.Episode(r, i)
        ep.inner.horizon = 4000
        why = ""
        try:
            te.EXPERT[task](ep, None)
            ok = settle(ep, r.chk_a)
            why = "" if ok else "끝났는데 목표 미달"
        except Exception as e:                  # 선생 코드가 예외로 멈춘 것도 실패로 센다
            ok, why = False, f"예외: {type(e).__name__}: {e}"[:120]
        res.append({"episode": i, "ok": bool(ok), "why": why, "steps": len(ep.log["pos"])})
        ep.env.close()
    k = sum(r_["ok"] for r_ in res)
    rec[f"T{task}"] = {"ok": k, "n": len(res), "eps": res}
    print(f"== T{task}: {k}/{len(res)}", flush=True)


def switch(at, bt, eps, rec):
    r = so.SimRunner(at, bt)
    resb, resr = [], []
    for i in eps:
        ep = sx.Episode(r, i)
        ep.inner.horizon = 4000
        obj = sv.OBJ[at]
        z = ep.obj_pos(obj)[2]
        try:
            if not sv.run_until_held(ep, at, ce.Rec()):
                resb.append({"episode": i, "ok": False, "why": "A 에서 물체를 못 쥠"})
                ep.env.close()
                continue
            t_sw = len(ep.log["pos"])
            sv.do_b(ep, at, bt, ce.Rec(), z)
            okb = settle(ep, r.chk_b)
            mh = se.min_home_dist(ep, t_sw)
            why = "" if okb and mh >= 0.07 else ("B 목표 미달" if not okb else f"처음 자세 {100 * mh:.1f}cm")
            resb.append({"episode": i, "ok": bool(okb and mh >= 0.07), "why": why, "min_home_cm": round(100 * mh, 1)})
            if (at, bt) in RESUME and okb and mh >= 0.07 and not r.chk_a(ep.env):
                t_r = len(ep.log["pos"])
                te.EXPERT[at](ep, None)
                oka = settle(ep, r.chk_a)
                bkeep = bool(r.chk_b(ep.env))
                mh2 = se.min_home_dist(ep, t_r)
                ok = oka and bkeep and mh2 >= 0.07
                why = "" if ok else ("A 목표 미달" if not oka else ("B 가 풀림" if not bkeep else f"처음 자세 {100 * mh2:.1f}cm"))
                resr.append({"episode": i, "ok": bool(ok), "why": why, "min_home_cm": round(100 * mh2, 1)})
            elif (at, bt) in RESUME:
                resr.append({"episode": i, "ok": False, "why": "B 실패로 재개 못 함"})
        except Exception as e:
            resb.append({"episode": i, "ok": False, "why": f"예외: {type(e).__name__}: {e}"[:120]})
        ep.env.close()
    kb = sum(x["ok"] for x in resb)
    rec[f"A{at}→B{bt}"] = {"ok": kb, "n": len(resb), "eps": resb}
    line = f"== A{at}→B{bt}: 전환 {kb}/{len(resb)}"
    if (at, bt) in RESUME:
        kr = sum(x["ok"] for x in resr)
        rec[f"A{at}→B{bt}→A{at}"] = {"ok": kr, "n": len(resr), "eps": resr}
        line += f"  재개 {kr}/{len(resr)}"
    print(line, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, nargs="*", default=[])
    p.add_argument("--pairs", default="")
    p.add_argument("--episodes", default="20-49")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    se.DART_SIGMA = 0.0
    eps = ep_list(a.episodes)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = json.loads(out.read_text()) if out.exists() else {}
    t0 = time.time()
    for t in a.tasks:
        if f"T{t}" not in rec:
            standalone(t, eps, rec)
            out.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    for s in [x for x in a.pairs.split(",") if x]:
        at, bt = map(int, s.split(":"))
        if f"A{at}→B{bt}" not in rec:
            switch(at, bt, eps, rec)
            out.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(f"끝 {time.time() - t0:.0f}초", flush=True)


if __name__ == "__main__":
    main()
