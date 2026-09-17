#!/usr/bin/env python
"""SmolVLA 1회 추론(동작 chunk 50개) 시간 측정 — LAB05 6단계 bench_policy 방식 (synchronize 필수)."""
import os, sys, time, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch
from types import SimpleNamespace
import switch_experiment as sx

label = sys.argv[1] if len(sys.argv) > 1 else "측정"
r = sx.Runner(SimpleNamespace(policy="HuggingFaceVLA/smolvla_libero", task_a=7, strategy="none"))
ep = sx.Episode(r, 0)
for _ in range(5):
    ep.infer_chunk(r.chk_a.language)  # 워밍업
ts = []
for _ in range(30):
    torch.cuda.synchronize(); t = time.perf_counter()
    ep.infer_chunk(r.chk_a.language)
    torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1000)
# 시뮬레이터 한 스텝(렌더링 포함) 시간
act = np.zeros(7); act[6] = -1
ss = []
for _ in range(30):
    t = time.perf_counter(); ep.step(act, "A"); ss.append((time.perf_counter() - t) * 1000)
ts = np.sort(ts)
res = {"label": label, "infer_median_ms": round(float(np.median(ts)), 1), "infer_p90_ms": round(float(ts[int(len(ts) * 0.9)]), 1),
       "infer_steps_at_20Hz": round(float(np.median(ts)) / 50, 2), "sim_step_median_ms": round(float(np.median(ss)), 1),
       "vram_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2), "gpu": torch.cuda.get_device_name(0)}
print(json.dumps(res, ensure_ascii=False))
ep.env.close()
