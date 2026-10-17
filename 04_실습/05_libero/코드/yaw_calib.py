"""손목 카메라 사진 속 블록 기울기(minAreaRect) ↔ 손을 세계 z 축으로 돌린 각도. 부호·크기 맞추기."""
import sys
sys.path.insert(0, "/home/user/smolVLA")
import numpy as np, cv2
from scipy import ndimage
from scipy.spatial.transform import Rotation as R
import piper_sim.piper_robot as pr
pr.use_piper_in_lerobot()
import switch_experiment as sx
import scripted_expert as se
import task_experts as te
import objdet

def blob_angle(img, color):
    m = objdet.masks(img)[color]
    lab, n = ndimage.label(m)
    if n == 0: return None, 0
    sizes = ndimage.sum(m, lab, range(1, n + 1)); k = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(lab == k)
    (cx, cy), (w, h), ang = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
    return ((ang + 45) % 90) - 45, len(xs)

ra = sx.default_args(); ra.task_a, ra.strategy, ra.policy = 0, "none", "outputs/piper_als2_model/merged"
runner = sx.Runner(ra)
for ep_idx, obj, color in ((1000, "blue_block_1", "blue"), (1003, "blue_block_1", "blue"), (1001, "blue_block_1", "blue")):
    ep = sx.Episode(runner, ep_idx)
    by = np.degrees(se.obj_rot(ep, obj).as_euler("xyz")[2])
    g, M = te.block_grasp_candidates(ep, obj)[0]
    print(f"장면 {ep_idx} 블록 yaw {by:.1f}°")
    for dz in (0.06, 0.03):
        for d in (0, 10, 20, -10, -20):
            Mt = R.from_euler("z", d, degrees=True).as_matrix() @ M
            se.servo(ep, g + [0, 0, dz], Mt, -1.0, tol=0.003, max_steps=80)
            cur = ep.obs["robot_state"]["eef"]["mat"]
            act_yaw = np.degrees(R.from_matrix(cur @ M.T).as_rotvec()[2])
            a, n = blob_angle(ep.obs["pixels"]["image2"], color)
            print(f"  높이 {dz*100:.0f}cm 돌림 {d:+d}° (실제 {act_yaw:+.1f}°) → 사진 기울기 {a if a is None else round(a,1)}° 픽셀 {n}")
