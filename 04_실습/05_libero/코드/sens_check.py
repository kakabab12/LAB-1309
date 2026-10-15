"""민감도 확인 (10/15): 시범 한 장면에서 '손 기준 빨강 블록 x' 만 +2cm 옮겨 넣으면 모델의 손 목표 위치가 따라 움직이나.
   alr1 은 0.1cm 도 안 움직였다 (입력을 무시). 상태를 동작 쪽에도 넣은 모델(sufstate)은 따라가야 한다. CPU 로 돈다.
   CUDA_VISIBLE_DEVICES= python sens_check.py outputs/piper_als1_model/merged"""
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
for dt in (-25, -10):
    t = g + dt
    obs = {"pixels": {"image": dec(z["img"][t])[None], "image2": dec(z["wrist"][t])[None]},
           "robot_state": {"eef": {"pos": z["eef_pos"][t:t+1], "quat": z["eef_quat"][t:t+1]}, "gripper": {"qpos": z["grip"][t:t+1]}}}
    outs = []
    for shift in (0.0, 0.02):
        b = preprocess_observation(obs); b["task"] = [str(z["task"])]
        b = env_step(b)
        feat = rf[t].copy(); feat[0] += shift      # 빨강 블록 x (손 기준) 만 2cm 옮김
        st = b["observation.state"]; b["observation.state"] = torch.cat([st, torch.as_tensor(feat)[None].to(st.dtype)], -1)
        b = pre(b)
        torch.manual_seed(0)
        with torch.no_grad():
            ch = pol.predict_action_chunk(b)
        act = post(ch)[0].numpy()
        outs.append(act)
    d = outs[1][:, :3] - outs[0][:, :3]
    print(f"쥐기 {dt}스텝 전: 손 {z['eef_pos'][t].round(3)}, 목표 위치 변화 (블록 x +2cm 일 때) 앞 1·5·10·20번째 동작 x: "
          f"{[round(100*float(d[i,0]),2) for i in (0,4,9,19)]} cm, y: {[round(100*float(d[i,1]),2) for i in (0,4,9,19)]} cm", flush=True)
