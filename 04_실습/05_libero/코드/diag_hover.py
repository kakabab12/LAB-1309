"""블록 위에서 못 쥘 때 무엇이 막나 (10/21, 받침대 없음). stuck_contacts 는 손-블록 1cm 안만 찍어,
   np1 + 보조의 굳은 장면(손이 블록에서 2cm 옆·6cm 위, 손 기울기 72° 로 멈춤)을 놓쳤다.
   B·A2 에서 지시문 블록 옆 4cm 안·10cm 위 안이면 25스텝마다: 손 기울기, 접촉, 한계 10° 안의 관절, 보조 동작 z
   python diag_hover.py -- (eval_piper 인자)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.spatial.transform import Rotation as R
import piper_sim.piper_robot as pr
pr.use_piper_in_lerobot()
import switch_experiment as sx
import piper_sim.blocks as pb

TARGET = {}
orig_step = sx.Episode.step
cnt = {}


def step(self, action, phase):
    out = orig_step(self, action, phase)
    t = TARGET.get(phase)
    if t is None:
        return out
    obj = pb.TASKS[t][0]
    hand = sx.eef_pos(self.obs); o = self.obj_pos(obj)
    if np.linalg.norm(hand[:2] - o[:2]) > 0.04 or hand[2] - o[2] > 0.10:
        return out
    cnt[phase] = cnt.get(phase, 0) + 1
    if cnt[phase] % 25 != 1:
        return out
    m, d = self.inner.sim.model, self.inner.sim.data
    pairs = set()
    for i in range(d.ncon):
        c = d.contact[i]
        a, b = m.geom_id2name(c.geom1) or f"#{c.geom1}", m.geom_id2name(c.geom2) or f"#{c.geom2}"
        if "robot0" in a + b or "gripper" in a + b:
            pairs.add(tuple(sorted((a, b))))
    lim = []
    for j in m.joint_names:
        if not j.startswith("robot0_joint"):
            continue
        jid = m.joint_name2id(j); lo, hi = m.jnt_range[jid]
        q = float(d.qpos[m.jnt_qposadr[jid]])
        gap = np.degrees(min(q - lo, hi - q))
        if gap < 10:
            lim.append(f"{j[7:]} {np.degrees(q):.0f}° (한계까지 {gap:.1f}°)")
    q = float(self.obs["robot_state"]["gripper"]["qpos"][0])
    A = getattr(self.r, "assist", None)
    if A is not None:       # 보조가 기억한 블록 위치 (손이 8cm 안이면 갱신 안 됨) — 블록이 밀려 움직였는데 옛 자리로 맞추고 있나
        import grasp_assist as ga
        est = A.pos.get({v: k for k, v in ga.NAMES.items()}[obj])
        if est is not None:
            lim.append(f"보조 추정-실제 {100*np.linalg.norm(est - o[:2]):.1f}cm 추정-손 {100*np.linalg.norm(est - hand[:2]):.1f}cm dmode {A.dmode}")
        import objdet           # 지금 정면 사진으로 새로 잰 위치는 손이 가까워도 맞나 (손에 가려 틀리면 기억을 쓰는 게 맞다)
        f = objdet.detect(np.asarray(self.obs["pixels"]["image"]), np.asarray(self.obs["pixels"]["image2"]))
        col = {v: k for k, v in ga.NAMES.items()}[obj]
        k = objdet.OBJECTS.index(col)
        if f[2 * k] >= 0 and col in A.C:
            u, v = f[2 * k], f[2 * k + 1]
            xy = A.C[col] @ np.array([u, v, 1.0, u * u, u * v, v * v])
            lim.append(f"새로 잰-실제 {100*np.linalg.norm(xy - o[:2]):.1f}cm")
        else:
            lim.append("새로 잰 것 없음(가려짐)")
    print(f"[{phase} {len(self.acts)}] 손-블록 xy {100*np.linalg.norm(hand[:2]-o[:2]):.2f}cm 높이 {100*(hand[2]-o[2]):.1f}cm "
          f"기울기 {sx_tilt(self):.0f}° 손가락 어긋남 {finger_mis(self, obj):.0f}° 그리퍼 {q:.4f} 동작 z {action[2]:+.2f} 그리퍼명령 {action[6]:+.1f} "
          f"관절 {lim or '-'} 접촉 {sorted(pairs)}", flush=True)
    return out


def sx_tilt(ep):
    """손 축(그리퍼 z)이 똑바로 아래에서 몇 도 기울었나 — 평가 기록 eef_rotvec 과 같은 값."""
    m = np.asarray(ep.obs["robot_state"]["eef"]["mat"]).reshape(3, 3)
    return float(np.degrees(np.arccos(np.clip(-m[2, 2], -1, 1))))


def finger_mis(ep, obj):
    """손가락이 벌어지는 방향(손 y)과 블록 면의 어긋남 [도]. 정육면체는 90° 대칭, 노란 막대는 짧은 축(몸 y)과 180° 대칭."""
    m = np.asarray(ep.obs["robot_state"]["eef"]["mat"]).reshape(3, 3)
    f = np.arctan2(m[1, 1], m[0, 1])
    qb = ep.inner.sim.data.body_xquat[ep.inner.obj_body_id[obj]]
    b = R.from_quat([qb[1], qb[2], qb[3], qb[0]]).as_matrix()
    by = np.arctan2(b[1, 1], b[0, 1])
    per = 180 if obj == "wine_bottle_1" else 90
    dd = np.degrees(f - by) % per
    return float(min(dd, per - dd))


sx.Episode.step = step
if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    a = sx.build_parser().parse_args(argv + ["--max-steps", "600"])
    if a.task_b is not None:
        TARGET["B"] = a.task_b; TARGET["A2"] = a.task_a
    sys.argv = [sys.argv[0]] + argv + ["--max-steps", "600"]
    sx.main()
