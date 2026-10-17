"""못 쥔 장면 vs 쥔 장면: 블록 방향(장면 처음)과 손가락 방향의 어긋남 (0~45도, 블록은 90도 대칭)"""
import sys, json, glob, os
from pathlib import Path
sys.path.insert(0, "/home/user/smolVLA")
import numpy as np
from scipy.spatial.transform import Rotation as R
import piper_sim.piper_robot as pr
pr.use_piper_in_lerobot()
import switch_experiment as sx
import piper_sim.blocks as pb
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv
sx.FIXED_STATES_OFF = True
suite = benchmark.get_benchmark_dict()[sx.SUITE]()
def block_yaw(task, ep_idx, obj):
    env = LiberoEnv(task_suite=suite, task_id=task, task_suite_name=sx.SUITE, obs_type="pixels_agent_pos",
                    episode_index=ep_idx, init_states=False)
    env.reset(seed=0 + ep_idx)
    inner = env._env.env
    q = inner.sim.data.body_xquat[inner.obj_body_id[obj]]
    y = R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[2]
    env.close()
    return y
def finger_yaw(rv):
    M = R.from_rotvec(rv).as_matrix()      # 손가락이 벌어지는 방향 = 손 프레임 y (piper: link6 y) — 수평 성분의 각도
    v = M[:, 1]
    return np.arctan2(v[1], v[0])
def mis(a, b):
    d = np.degrees(a - b) % 90
    return min(d, 90 - d)
F, S = [], []
BASE = sys.argv[1] if len(sys.argv) > 1 else "piper_als2apr"
for kind in ["forget", "switch"]:
    for f in sorted(glob.glob(f"outputs/{BASE}_{kind}/*.json")):
        J = json.load(open(f)); ta, tb = J["args"]["task_a"], J["args"]["task_b"]; tag = os.path.basename(f)[:-5]
        for e in J["episodes"]:
            tr = np.load(f"outputs/{BASE}_{kind}/traj/{tag}_ep{e['episode']}.npz", allow_pickle=True)
            ph, t, objs = ("A", ta, e["objects_final"]) if tb is None else ("B", tb, e.get("objects_at_b_end"))
            if objs is None: continue
            obj = pb.TASKS[t][0]; idx = np.nonzero(tr["phase"] == ph)[0]
            if len(idx) == 0: continue
            segs = [s for s in e["hold_segs"] if s[2] == obj and s[0] >= idx[0]]
            if segs:
                i = segs[0][0]; lst = S
            else:
                hand = tr["pos"][idx]; dxy = np.linalg.norm(hand[:, :2] - np.array(objs[obj])[:2], axis=1)
                nr = idx[dxy < 0.01]
                if len(nr) < 20: continue
                i = nr[np.argmin(tr["pos"][nr, 2])]; lst = F
            if len(lst) >= 40: continue
            by = block_yaw(ta, e["episode"], obj)
            lst.append(mis(finger_yaw(tr["eef_rotvec"][i]), by))
F, S = np.array(F), np.array(S)
print("못 쥠: 손가락-블록 면 어긋남", len(F), "중앙 %.0f°, 15° 넘음 %.0f%%" % (np.median(F), 100 * (F > 15).mean()), np.percentile(F, [10, 50, 90]).round(0))
print("쥔 순간", len(S), "중앙 %.0f°, 15° 넘음 %.0f%%" % (np.median(S), 100 * (S > 15).mean()), np.percentile(S, [10, 50, 90]).round(0))
