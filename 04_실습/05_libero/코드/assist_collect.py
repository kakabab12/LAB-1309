#!/usr/bin/env python
"""보조 증류 시범 모으기 (2026-10-22) — VLA + 보조 5가지로 평가와 똑같이 돌리고, 성공한 구간의 (관측, 실제로 보낸 동작)을 시범으로 저장

왜
  VLA + 보조는 28항목 93.8% 인데 VLA 혼자는 25.5% (np1 과 같음). 보조가 하는 일(마지막 1~2cm 맞추기, 막힘 풀기, 손가락 방향,
  빈 자리에 내려놓기, 블록 위로 데려가기)을 VLA 가 배우지 못했다. 보조가 고친 동작을 그대로 정답으로 삼아 VLA 를 더 학습하면
  (성공한 것만 골라 따라 하기, filtered behavior cloning) 보조 없이도 그 일을 하게 될 거라는 가설.

무엇을 저장하나
  평가(switch_experiment)와 같은 흐름 — 지연 9스텝, keep 전환, 쥔 지 3스텝에 새 지시 — 을 학습용 장면(3000번대)에서 돌린다.
  Episode.step 을 감싸 단계(A·B·A2)마다 (그 순간 관측, 보조까지 거친 동작)을 모은다.
  혼자 하는 과제: A 가 3cm 성공이면 A 구간. 일 바꾸기: B 가 성공이면 B 구간 (A 를 쥔 채 시작 — 내려놓기 보조가 빈 자리에 내려놓는
  동작도 들어감), 돌아오기 쌍이면 A2 가 진짜 성공일 때 A2 구간. 전환 전 A 구간은 혼자 하는 과제와 같아 저장하지 않는다.
  처음 자세 7cm 안에 들어간 장면은 버린다.

  python assist_collect.py --data-out data/piper_np_assist1 -- (eval_piper 인자: --policy ... --task-a 8 --task-b 5 ... --start-episode 3300)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import piper_sim.piper_robot as pr  # noqa: E402

pr.use_piper_in_lerobot()
import collect_expert as ce  # noqa: E402
import switch_experiment as sx  # noqa: E402

RESUME = {(8, 5), (8, 7), (8, 9), (4, 5), (4, 9), (1, 7)}
_orig_step = sx.Episode.step


def _step(self, action, phase):
    if not hasattr(self, "_recs"):
        self._recs = {}
        self._near_home = False
    self._recs.setdefault(phase, ce.Rec())(self.obs, action)
    out = _orig_step(self, action, phase)
    if phase != "A" and np.linalg.norm(sx.eef_pos(self.obs) - self.home[0]) < 0.07:
        self._near_home = True
    return out


def main():
    argv = sys.argv[1:]
    i = argv.index("--data-out")
    out = Path(argv[i + 1]) / "episodes"
    argv = argv[:i] + argv[i + 2:]
    if "--" in argv:
        argv.remove("--")
    if "--max-steps" not in argv:
        argv += ["--max-steps", "600"]
    out.mkdir(parents=True, exist_ok=True)
    sx.Episode.step = _step
    last = {}                          # 지금 장면 (지난 장면은 놓아 환경이 쌓이지 않게)
    orig_init = sx.Episode.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        last["ep"] = self
    sx.Episode.__init__ = init
    args = sx.parse_args(argv)
    runner = sx.Runner(args)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ta, tb = args.task_a, args.task_b
    n_saved = 0
    for idx in range(args.start_episode, args.start_episode + args.episodes):
        rec = runner.episode(idx)
        ep = last.pop("ep")
        recs = getattr(ep, "_recs", {})
        save = []
        if tb is None:
            if rec.get("a_success"):
                save.append(("A", runner.chk_a.language))
        elif not getattr(ep, "_near_home", False):
            if rec.get("b_success"):
                save.append(("B", runner.chk_b.language))
            if (ta, tb) in RESUME and rec.get("a_resume_genuine"):
                save.append(("A2", runner.chk_a.language))
        tag = f"T{ta}" if tb is None else f"A{ta}B{tb}"
        for ph, lang in save:
            r = recs.get(ph)
            if r is None or len(r) < 10:
                continue
            # 이름이 D 로 시작하면 train_lora 가 교정 시범(DN/DS/DR)과 같이 --dagger-frac 몫으로 뽑는다 (학생이 실제로 가는 상태라서)
            r.save(out / f"DA{tag}_{ph}_ep{idx}.npz", lang, source="assist", episode=idx, phase=ph,
                   assist_steps=str(rec.get("assist_steps", {})))
            n_saved += 1
        print(f"장면 {idx} {tag}: 저장 {[p for p, _ in save]}", flush=True)
        ep._recs = {}
    print(f"SUMMARY 보조 시범 {tag} 저장 {n_saved}", flush=True)


if __name__ == "__main__":
    main()
