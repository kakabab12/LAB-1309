import sys, io, numpy as np, torch
sys.path.insert(0, "/home/user/smolVLA")
from PIL import Image
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.env_processor import LiberoProcessorStep
import ttrtc
ttrtc.patch(); ttrtc.enable_sufstate()
P = sys.argv[1]
pol = SmolVLAPolicy.from_pretrained(P); pol.config.device = "cpu"; pol.eval()
pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=P, preprocessor_overrides={"device_processor": {"device": "cpu"}})
env_step = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
f = "data/piper_al_normal_t/episodes/N8_ep3100.npz"
z = np.load(f, allow_pickle=True); rf = np.load(f.replace(".npz", ".relfeat.npy"))
dec = lambda b: np.asarray(Image.open(io.BytesIO(bytes(b))).convert("RGB"))
t = 0
obs = {"pixels": {"image": dec(z["img"][t])[None], "image2": dec(z["wrist"][t])[None]},
       "robot_state": {"eef": {"pos": z["eef_pos"][t:t+1], "quat": z["eef_quat"][t:t+1]}, "gripper": {"qpos": z["grip"][t:t+1]}}}
print("relfeat t0", rf[t][:8].round(3))
seen = {}
orig = pol.model.embed_prefix
def spy(images, img_masks, lang_tokens, lang_masks, state=None):
    seen["state"] = state.detach().clone(); seen["masks"] = [m.tolist() for m in img_masks]
    return orig(images, img_masks, lang_tokens, lang_masks, state=state)
pol.model.embed_prefix = spy
for name, shift in (("원래", 0.0), ("빨강 +5cm", 0.05), ("빨강 +15cm", 0.15)):
    b = preprocess_observation(obs); b["task"] = [str(z["task"])]; b = env_step(b)
    feat = rf[t].copy(); feat[0] += shift
    st = b["observation.state"]; b["observation.state"] = torch.cat([st, torch.as_tensor(feat)[None].to(st.dtype)], -1)
    b = pre(b)
    b["observation.images.image_padding_mask"] = torch.tensor([False]); b["observation.images.image2_padding_mask"] = torch.tensor([False])
    torch.manual_seed(0)
    with torch.no_grad():
        out = post(pol.predict_action_chunk(b))[0].numpy()
    print(name, "모델이 받은 상태 8~10칸", seen["state"][0, 8:11].numpy().round(3), "마스크", seen["masks"], "49번째 목표 xy", (100*out[48,:2]).round(2), "cm")
