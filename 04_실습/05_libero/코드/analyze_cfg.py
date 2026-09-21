#!/usr/bin/env python
"""
지시문 증폭 (instruction CFG) 평가 — 효과가 있나, 그리고 **제약을 지키나**

무엇을 보나
  ① B 성공률: 증폭이 실제로 새 지시를 따르게 하는가
  ② **부드러움(jerk)**: 증폭은 동작을 키운다 (w=1.5 에서 크기 0.69→0.89).
     성공률이 올라도 동작이 거칠어지면 **"사람처럼 자연스럽게" 제약을 깬다.**
     성공률만 보고 좋다고 하면 안 된다
  ③ 실패 유형: 맴돌기(idling)가 줄었는지, 아니면 다른 식으로 실패하는지

  기준선(w=1)은 다시 돌리지 않는다 — w=1 이 원본과 비트 단위로 같음을
  test_cfg_identity.py 로 확인했으므로 outputs/switch 의 결과가 곧 기준선이다.

읽는 법
  성공 ↑ + jerk 그대로  →  ✅ 쓸 수 있다
  성공 ↑ + jerk ↑↑      →  ⚠️ 제약 위반. 자연스러움 지표를 같이 보고해야 한다
  성공 그대로            →  ❌ CMI 를 키워도 행동이 안 바뀐다 = 가설이 틀렸다
"""
import glob
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

BASE_DIR = "outputs/switch"
CFG_DIR = "outputs/cfg"
PAIRS = [("A8_B7", "8→7"), ("A4_B7", "4→7"), ("A1_B5", "1→5"), ("A2_B5", "2→5")]


def sign_test(w, l):
    k = w + l
    if not k:
        return 1.0
    return min(1.0, 2 * sum(comb(k, i) for i in range(max(w, l), k + 1)) / 2 ** k)


def load(path):
    if not Path(path).exists():
        return None
    return {e["episode"]: e for e in json.load(open(path))["episodes"] if e.get("switched")}


def ws_present(tim):
    out = set()
    for f in glob.glob(f"{CFG_DIR}/A*_{tim}_flush_cfg*.json"):
        s = Path(f).stem.split("_cfg")[1]
        out.add(float(s[:-1] if s.endswith("a") else s))
    return sorted(out)


def main():
    tim = sys.argv[1] if len(sys.argv) > 1 else "grasp3"
    ws = ws_present(tim)
    if not ws:
        print(f"{CFG_DIR} 에 결과가 없습니다 (시점 {tim}).")
        return

    print(f"== B 성공률 — 전환 시점 {tim}\n")
    hdr = f"{'쌍':<8}{'기준(w=1)':>11}" + "".join(f"{'w=' + f'{w:g}':>9}" for w in ws)
    print(hdr)
    tot = {0: [0, 0], **{w: [0, 0] for w in ws}}
    per = {}
    for pk, pl in PAIRS:
        base = load(f"{BASE_DIR}/{pk}_{tim}_flush.json")
        if not base:
            continue
        kb = sum(bool(e.get("b_success")) for e in base.values())
        tot[0][0] += kb; tot[0][1] += len(base)
        line = f"{pl:<8}{100 * kb / len(base):10.0f}%"
        per[pk] = {"base": base}
        for w in ws:
            c = load(f"{CFG_DIR}/{pk}_{tim}_flush_cfg{w:g}.json")
            if c:
                k = sum(bool(e.get("b_success")) for e in c.values())
                tot[w][0] += k; tot[w][1] += len(c)
                line += f"{100 * k / len(c):8.0f}%"
                per[pk][w] = c
            else:
                line += "        -"
        print(line)
    line = f"{'합계':<8}{100 * tot[0][0] / max(tot[0][1], 1):10.0f}%"
    for w in ws:
        line += f"{100 * tot[w][0] / max(tot[w][1], 1):8.0f}%"
    print(line)
    print(f"{'':8}{'(' + str(tot[0][0]) + '/' + str(tot[0][1]) + ')':>11}"
          + "".join(f"{'(' + str(tot[w][0]) + '/' + str(tot[w][1]) + ')':>9}" for w in ws))

    print("\n== 짝 비교 (같은 에피소드끼리)")
    for w in ws:
        win = loss = 0
        for pk in per:
            c = per[pk].get(w)
            if not c:
                continue
            b = per[pk]["base"]
            for ep in set(b) & set(c):
                x, y = bool(c[ep].get("b_success")), bool(b[ep].get("b_success"))
                win += x and not y
                loss += y and not x
        p = sign_test(win, loss)
        mark = "✅" if p < 0.05 and win > loss else ("❌" if p < 0.05 else "–")
        print(f"  w={w:g}: 증폭만 성공 {win}, 기준만 성공 {loss},  p = {p:.3f}  {mark}")

    print("\n== ⚠️ 자연스러움 (jerk 중앙값, 낮을수록 부드럽다)")
    print(f"{'쌍':<8}{'기준':>9}" + "".join(f"{'w=' + f'{w:g}':>9}" for w in ws))
    allj = {0: [], **{w: [] for w in ws}}
    for pk, pl in PAIRS:
        if pk not in per:
            continue
        line = f"{pl:<8}"
        js = [e["jerk"] for e in per[pk]["base"].values() if e.get("jerk") is not None]
        allj[0] += js
        line += f"{np.median(js):9.2f}" if js else "        -"
        for w in ws:
            c = per[pk].get(w)
            js = [e["jerk"] for e in c.values() if e.get("jerk") is not None] if c else []
            allj[w] += js
            line += f"{np.median(js):9.2f}" if js else "        -"
        print(line)
    line = f"{'합계':<8}{np.median(allj[0]):9.2f}"
    for w in ws:
        line += f"{np.median(allj[w]):9.2f}" if allj[w] else "        -"
    print(line)

    b0 = np.median(allj[0])
    print()
    for w in ws:
        if not allj[w]:
            continue
        r = np.median(allj[w]) / b0
        note = ("부드러움 유지" if r < 1.15 else
                "⚠️ 눈에 띄게 거칠어짐" if r < 1.5 else "❌ 크게 거칠어짐 — 제약 위반")
        print(f"  w={w:g}: 기준 대비 jerk {r:.2f}배 — {note}")

    # ---- 맴돌기: 성공률보다 민감한 1차 지표 ----
    print("\n== ⭐ 맴돌기 비율 (B 구간, 2초 동안 이동 범위 3cm 미만)")
    print("   실패의 80% 가 맴돌기다. 성공률은 0/1 이라 둔하지만 이건 연속적으로 재므로")
    print("   **같은 표본에서 더 민감하다.** 성공률이 안 움직여도 여기가 움직이면 방향은 맞다.")
    try:
        import subprocess
        out = subprocess.run([sys.executable, "idling_by_cond.py", tim],
                             capture_output=True, text=True, timeout=300)
        body = out.stdout.split("\n")
        start = next((i for i, l in enumerate(body) if l.startswith("조건")), None)
        if start is not None:
            print()
            for l in body[start:]:
                if l.strip():
                    print("  " + l)
        else:
            print("  (궤적이 없어 계산하지 못했습니다)")
    except Exception as e:
        print(f"  (계산 실패: {type(e).__name__}: {e})")

    print("\n== 실패 유형")
    for w in [0] + ws:
        cnt = {}
        for pk in per:
            d = per[pk]["base"] if w == 0 else per[pk].get(w)
            if not d:
                continue
            for e in d.values():
                if not e.get("b_success"):
                    cnt[e.get("b_failure_type", "?")] = cnt.get(e.get("b_failure_type", "?"), 0) + 1
        lbl = "기준" if w == 0 else f"w={w:g}"
        tt = sum(cnt.values())
        print(f"  {lbl:<7} 실패 {tt}회  " +
              "  ".join(f"{k} {v}" for k, v in sorted(cnt.items(), key=lambda x: -x[1])))

    print("\n== 판정")
    best = max(ws, key=lambda w: tot[w][0] / max(tot[w][1], 1))
    bb = tot[0][0] / max(tot[0][1], 1)
    bw = tot[best][0] / max(tot[best][1], 1)
    if bw > bb + 0.05:
        print(f"가장 좋은 w={best:g}: {100 * bb:.0f}% → {100 * bw:.0f}% ({100 * (bw - bb):+.0f}%p)")
        print("→ 위 짝 비교 p 값과 jerk 비율을 함께 보고 판단할 것. 성공률만으로 결론 내지 말 것")
    else:
        print(f"어떤 w 도 기준({100 * bb:.0f}%)을 의미 있게 넘지 못함 (최고 w={best:g}, {100 * bw:.0f}%)")
        print("→ CMI 를 키워도 행동이 안 바뀐다면, '지시문 채널이 원인' 가설을 다시 봐야 한다")


if __name__ == "__main__":
    main()
