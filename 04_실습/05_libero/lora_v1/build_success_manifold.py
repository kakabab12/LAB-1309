#!/usr/bin/env python
"""
성공 궤적 지도 만들기 — Best-of-N 선택(bon 전략)의 기준.

모든 성공 에피소드의 끝단 위치를 지시문(태스크)별로 모아 둔다.
bon 전략은 후보 동작 묶음 중 "이 태스크를 성공한 궤적들이 지나간 곳"에 가까이 가는 후보를 고른다.

참고: Q-Planning (arXiv 2608.21204) — 정책은 얼리고 후보 N개 중 점수로 고르기.
      원 논문은 1B 파라미터 Q 함수로 점수를 매기지만, 1080 Ti 에서는 불가능해서
      "성공 궤적과의 거리"라는 학습 없는 점수로 바꿨다.

출력: data/success_manifold.npz  (태스크 지시문 → (M, 3) 위치 배열)
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MAX_POINTS_PER_TASK = 6000


def main():
    pts = defaultdict(list)
    # 1) 학습 데이터 (모두 성공 궤적): data/robust, data/v2
    for d in ["data/robust", "data/v2"]:
        for f in Path(d).glob("episodes/*.npz"):
            z = np.load(f, allow_pickle=True)
            pts[str(z["task"])].append(z["eef_pos"].astype(np.float32))
    # 2) 실험 로그의 성공 구간: A 단독 성공(none), B 성공 구간
    for f in Path("outputs/switch").glob("A*_B*_*.json"):
        d = json.load(open(f))
        a = d["args"]
        for e in d["episodes"]:
            tag = f"A{a['task_a']}_B{a['task_b']}_{a['switch_at'].replace(':', '')}_{a['strategy']}_ep{e['episode']}"
            tf = Path("outputs/switch/traj") / f"{tag}.npz"
            if not tf.exists():
                continue
            z = np.load(tf)
            pos, ph = z["pos"], z["phase"]
            if e.get("a_success") and a["strategy"] == "none":
                pts[e["task_a"]].append(pos[ph == "A"].astype(np.float32))
            if e.get("b_success"):
                pts[e["task_b"]].append(pos[ph == "B"].astype(np.float32))

    out, stats = {}, {}
    rng = np.random.default_rng(0)
    for task, arrs in pts.items():
        allp = np.concatenate([x for x in arrs if len(x)], axis=0)
        if len(allp) > MAX_POINTS_PER_TASK:
            allp = allp[rng.choice(len(allp), MAX_POINTS_PER_TASK, replace=False)]
        out[task] = allp
        stats[task] = {"episodes": len(arrs), "points": int(len(allp))}
    Path("data").mkdir(exist_ok=True)
    np.savez("data/success_manifold.npz", tasks=np.array(list(out.keys()), dtype=object),
             **{f"t{i}": v for i, v in enumerate(out.values())})
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
