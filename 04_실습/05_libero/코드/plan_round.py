#!/usr/bin/env python
"""
다음 라운드 설정을 결과에서 정한다 (2026-10-02)

  - 계산 간격 n_act: run_v6c.sh 의 빠른 비교 결과 (계속 계산 1 vs 10)
  - A2C2 를 쓸지: 같은 항목에서 모델+A2C2 평균이 모델 혼자 이상이면 쓴다
  - A2C2 보정 차원: 그리퍼 보정을 끈(posrot) 시험이 A8→B7 을 2/10 이상 살리고 T8 을 1/10 넘게 깎지 않으면 posrot
  - DAgger 할 항목: 쓰기로 한 설정에서 95% 미만인 항목 (단독 / 전환 B / 재개)

출력: bash 에서 eval 할 수 있는 변수들
  python plan_round.py --base v6clat --a2c2 v6ca > outputs/v6/round2.env
"""
import argparse
import glob
import json
import re
from pathlib import Path

import scoreboard as sb


def rows(prefix):
    return sb.collect([f"outputs/{prefix}_forget"], [f"outputs/{prefix}_switch"])


def rate(dirs, pat, key):
    k = n = 0
    for d in dirs:
        for f in glob.glob(f"{d}/{pat}"):
            for e in json.load(open(f))["episodes"]:
                if key == "b_success" and not e.get("switched"):
                    continue
                k += bool(e.get(key))
                n += 1
    return k, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="v6clat")
    p.add_argument("--a2c2", default="v6ca")
    p.add_argument("--log", default="outputs/v6/run_v6c_post.log")
    p.add_argument("--posrot", default=None, help="그리퍼 보정을 끈 평가 폴더 (기본: <a2c2>_posrot)")
    p.add_argument("--target", type=float, default=0.95)
    a = p.parse_args()
    out = {}
    m = re.findall(r"n_act (\d+) 로 정식 평가", Path(a.log).read_text()) if Path(a.log).exists() else []
    out["NA"] = m[-1] if m else "10"
    rb, ra = rows(a.base), rows(a.a2c2)
    common = [(x, y) for x, y in zip(rb, ra) if x[4] and y[4]]
    mb = sum(x[3] / x[4] for x, _ in common) / max(len(common), 1)
    ma = sum(y[3] / y[4] for _, y in common) / max(len(common), 1)
    out["USE_A2C2"] = "1" if common and ma >= mb else "0"
    out["MEAN_BASE"], out["MEAN_A2C2"] = f"{mb:.3f}", f"{ma:.3f}"
    # 보정 차원
    pr = a.posrot or f"outputs/{a.a2c2}_posrot"
    b7_all = rate([f"outputs/{a.a2c2}_switch"], "A8_B7_*", "b_success")
    b7_pr = rate([pr], "A8_B7_*", "b_success")
    t8_all = rate([f"outputs/{a.a2c2}_forget"], "A8_Bnone_*", "a_success")
    t8_pr = rate([pr], "A8_Bnone_*", "a_success")
    dims = "all"
    if b7_pr[1] and t8_pr[1] and b7_pr[0] - b7_all[0] >= 2 and t8_pr[0] >= t8_all[0] - 1:
        dims = "posrot"
    out["DIMS"] = dims
    # 95% 미만 항목
    best = ra if out["USE_A2C2"] == "1" else rb
    tasks, pairs, resume = [], [], []
    for g, key, _, s, n in best:
        if n == 0 or s / n >= a.target:
            continue
        if g == "단독":
            tasks.append(key[1:])
        elif g == "전환":
            x = key.replace("A", "").replace("B", "").split("→")
            pairs.append(f"{x[0]}:{x[1]}")
        else:
            x = key.replace("A", "").replace("B", "").split("→")
            resume.append(f"{x[0]}:{x[1]}")
    out["TASKS"] = " ".join(tasks)
    out["PAIRS"] = ",".join(pairs)
    out["RESUME"] = ",".join(resume)
    for k, v in out.items():
        print(f'{k}="{v}"')


if __name__ == "__main__":
    main()
