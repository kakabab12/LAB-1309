"""넓은 민감도: 여러 순간에서 빨강 블록 (손 기준) x 를 +5cm, 정면 카메라 보임/가림. 20·49번째 동작 목표 x,y 변화 [cm]"""
import sys, io, numpy as np, torch
sys.path.insert(0, "/home/user/smolVLA")
from PIL import Image
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.env_processor import LiberoProcessorStep
import ttrtc
ttrtc.patch()
P = sys.argv[1]
if (__import__("pathlib").Path(P) / "sufstate").exists():
    ttrtc.enable_sufstate()
pol = SmolVLAPolicy.from_pretrained(P); pol.config.device = "cpu"; pol.eval()
pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=P, preprocessor_overrides={"device_processor": {"device": "cpu"}})
env_step = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
f = "data/piper_al_normal_t/episodes/N8_ep3100.npz"
z = np.load(f, allow_pickle=True); rf = np.load(f.replace(".npz", ".relfeat.npy"))
a = z["action"][:, 6]; g = int(np.nonzero((a[1:] > 0) & (a[:-1] <= 0))[0][0]) + 1
dec = lambda b: np.asarray(Image.open(io.BytesIO(bytes(b))).convert("RGB"))
print("쥐는 스텝", g)
for t in (0, 15, g - 40, g - 25):
    for mask in (False, True):
        obs = {"pixels": {"image": dec(z["img"][t])[None], "image2": dec(z["wrist"][t])[None]},
               "robot_state": {"eef": {"pos": z["eef_pos"][t:t+1], "quat": z["eef_quat"][t:t+1]}, "gripper": {"qpos": z["grip"][t:t+1]}}}
        outs = []
        for shift in (0.0, 0.05):
            b = preprocess_observation(obs); b["task"] = [str(z["task"])]
            b = env_step(b)
            feat = rf[t].copy(); feat[0] += shift
            st = b["observation.state"]; b["observation.state"] = torch.cat([st, torch.as_tensor(feat)[None].to(st.dtype)], -1)
            b = pre(b)
            if mask:
                b["observation.images.image_padding_mask"] = torch.tensor([False])
            torch.manual_seed(0)
            with torch.no_grad():
                outs.append(post(pol.predict_action_chunk(b))[0].numpy())
        d = 100 * (outs[1][:, :2] - outs[0][:, :2])
        print(f"t={t:3d} 정면 {'가림' if mask else '보임'}: 20번째 x {d[19,0]:+.2f} y {d[19,1]:+.2f}, 49번째 x {d[48,0]:+.2f} y {d[48,1]:+.2f} cm", flush=True)
