"""Evaluate trained ACT stage policies in the MuJoCo bin-stacking cell.

Two protocols (unseen seeds, disjoint from the demonstration seeds):
  per-stage : each stage policy starts from a correctly prepared scene (earlier bins at their
              targets +-5 mm), N trials per stage.
  chained   : stage 1 -> 2 -> 3 policies run back-to-back on the same scene, like the real
              MCP-triggered cell (WAIT -> RUNNING -> RETURN_HOME per run), N sequences.
Each run: 270 policy steps at 30 Hz, then a 1 s return to the home pose, then settle.

usage: python eval_act.py --ckpt runs/s1_n100/ckpt_080000 runs/s2_n100/ckpt_080000 runs/s3_n100/ckpt_080000 \
                          --trials 50 --out results/eval/n100
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

import preprocess

from expert import ScriptedExpert
from lerobot.policies.act.modeling_act import ACTPolicy
from stack_env import BIN_H, BIN_NAMES, TARGETS, StackEnv, sample_pick, sample_scene

EVAL_SEED0 = 500000
HORIZON = 270


def load_policy(path: str, device: str, temporal_ensemble: float | None, n_action_steps: int | None = None) -> ACTPolicy:
    policy = ACTPolicy.from_pretrained(path)
    if n_action_steps is not None:
        policy.config.n_action_steps = n_action_steps  # re-plan every k steps instead of executing the whole chunk
    if temporal_ensemble is not None:
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        policy.config.temporal_ensemble_coeff = temporal_ensemble
        policy.config.n_action_steps = 1
        policy.temporal_ensembler = ACTTemporalEnsembler(temporal_ensemble, policy.config.chunk_size)
    meta = Path(path).parent / "train_meta.json"  # camera preprocessing the policy was trained with
    policy.preprocess = json.loads(meta.read_text()).get("preprocess") if meta.exists() else None
    policy.to(device).eval()
    return policy


def to_batch(obs: dict, device: str) -> dict:
    b = {"observation.state": torch.from_numpy(obs["state"]).unsqueeze(0)}
    for c in ("front", "top"):
        b[f"observation.images.{c}"] = torch.from_numpy(obs[c]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return {k: v.to(device) for k, v in b.items()}


class Recorder:
    def __init__(self, env: StackEnv, enabled: bool):
        self.enabled = enabled
        self.frames = []
        if enabled:
            self.big = mujoco.Renderer(env.m, 300, 400)
            self.small = mujoco.Renderer(env.m, 150, 200)
        self.env = env

    def snap(self, label: str = "") -> None:
        if not self.enabled:
            return
        d = self.env.d
        self.big.update_scene(d, "overview")
        a = self.big.render()
        self.small.update_scene(d, "front")
        b = self.small.render()
        self.small.update_scene(d, "top")
        c = self.small.render()
        self.frames.append(Image.fromarray(np.concatenate([a, np.concatenate([b, c], 0)], 1)))

    def save(self, path: Path) -> None:
        if self.enabled and self.frames:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.frames[0].save(path, save_all=True, append_images=self.frames[1:], duration=100, loop=0, optimize=True)
        self.frames = []

    def close(self) -> None:
        if self.enabled:
            self.big.close()
            self.small.close()


def run_stage(env, policy, home, device, rec: Recorder, frame_every: int = 4, obs_delay: int = 0,
              horizon: int = HORIZON) -> dict:
    """obs_delay > 0: the policy sees the camera images and joint state from that many control steps ago
    (sensing/communication latency; robustness test only)."""
    policy.reset()
    obs = env.observe()
    hist = [obs] * (obs_delay + 1)
    infer_ms = []
    bin0 = env.bin_state(env.active_bin).pos.copy()
    grasp_site, opened = None, False
    pre = getattr(policy, "preprocess", None)
    for t in range(horizon):
        x = hist[0] if obs_delay else obs
        if pre:
            x = {**x, **{c: preprocess.apply(pre, x[c]) for c in ("front", "top")}}
        b = to_batch(x, device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            a = policy.select_action(b)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        infer_ms.append(dt)
        act = a.squeeze(0).cpu().numpy()
        env.step(act)
        obs = env.observe()
        if obs_delay:
            hist = hist[1:] + [obs]
        # where does the policy close the gripper, relative to where the bin really was?
        opened = opened or act[5] > 0.2
        if grasp_site is None and opened and act[5] < 0.0:
            grasp_site = env.d.site_xpos[env.site].copy()
        if t % frame_every == 0:
            rec.snap()
    q0 = env.qpos()  # RETURN_HOME, as in the real runner
    for i in range(30):
        env.step(q0 + (home - q0) * (i + 1) / 30)
        if i % frame_every == 0:
            rec.snap()
    env.settle(15)
    return {"infer_ms_max": float(np.max(infer_ms)), "infer_ms_mean": float(np.mean(infer_ms)),
            "bin0": bin0.tolist(), "grasp_site": None if grasp_site is None else grasp_site.tolist()}


def follow_slopes(trials: list[dict]) -> dict:
    """Slope of grasp position vs true bin position (1 = the policy tracks the bin it sees)."""
    pts = [(t["bin0"], t["grasp_site"]) for t in trials if t.get("grasp_site") is not None]
    if len(pts) < 5:
        return {}
    b = np.array([p[0] for p in pts])
    g = np.array([p[1] for p in pts])
    return {"slope_x": float(np.polyfit(b[:, 0], g[:, 0], 1)[0]), "slope_y": float(np.polyfit(b[:, 1], g[:, 1], 1)[0]),
            "n": len(pts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs=3, required=True, help="stage 1, 2, 3 checkpoint dirs")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--start", type=int, default=0, help="first trial index (to split a long evaluation)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--mode", choices=["per_stage", "chained", "both"], default="both")
    ap.add_argument("--temporal-ensemble", type=float, default=None)
    ap.add_argument("--n-action-steps", type=int, default=None, help="execute only the first k actions of each chunk")
    ap.add_argument("--gifs", type=int, default=3, help="record this many trials per protocol as GIF")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = StackEnv(render=True)
    home = ScriptedExpert(env).home_q()
    policies = [load_policy(p, device, args.temporal_ensemble, args.n_action_steps) for p in args.ckpt]
    rec = Recorder(env, enabled=args.gifs > 0)
    results = {"ckpt": args.ckpt, "trials": args.trials, "temporal_ensemble": args.temporal_ensemble,
               "n_action_steps": args.n_action_steps,
               "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}

    if args.mode in ("per_stage", "both"):
        per = {}
        for stage in (1, 2, 3):
            rs = []
            for i in range(args.start, args.start + args.trials):
                rng = np.random.default_rng(EVAL_SEED0 + 1000 * stage + i)
                env.reset(sample_scene(stage, rng), home)
                refs = {n: env.bin_state(n).pos.copy() for n in BIN_NAMES[: stage - 1]}
                rec.enabled = i < args.gifs
                info = run_stage(env, policies[stage - 1], home, device, rec)
                r = env.evaluate_stage(stage, ref_positions=refs)
                r.update(info, trial=i)
                rs.append(r)
                rec.save(out / "gifs" / f"per_stage_s{stage}_t{i:02d}_{'ok' if r['success'] else 'fail'}.gif")
            per[stage] = {"success_rate": float(np.mean([r["success"] for r in rs])),
                          "n": len(rs),
                          "xy_err_mm_mean_success": float(np.mean([r["xy_err_mm"] for r in rs if r["success"]] or [np.nan])),
                          "follow": follow_slopes(rs),
                          "trials": rs}
            print(f"[per-stage] stage {stage}: {per[stage]['success_rate']*100:.1f}% ({sum(r['success'] for r in rs)}/{len(rs)})", flush=True)
        results["per_stage"] = per

    if args.mode in ("chained", "both"):
        seqs = []
        for i in range(args.start, args.start + args.trials):
            rng = np.random.default_rng(EVAL_SEED0 + 90000 + i)
            env.reset(sample_scene(1, rng), home)
            rec.enabled = i < args.gifs
            seq = []
            for stage in (1, 2, 3):
                if stage > 1:
                    xy, yaw = sample_pick(rng)
                    env.spawn_next_bin(stage, xy, yaw)
                refs = {n: env.bin_state(n).pos.copy() for n in BIN_NAMES[: stage - 1]}
                info = run_stage(env, policies[stage - 1], home, device, rec)
                r = env.evaluate_stage(stage, ref_positions=refs)
                r.update(info)
                seq.append(r)
            ok = [s["success"] for s in seq]
            seqs.append({"trial": i, "stages": seq, "cumulative": [all(ok[: k + 1]) for k in range(3)]})
            rec.save(out / "gifs" / f"chained_t{i:02d}_{''.join('o' if o else 'x' for o in ok)}.gif")
        cum = np.array([s["cumulative"] for s in seqs], dtype=float).mean(0)
        results["chained"] = {"cumulative_success": cum.tolist(), "n": len(seqs), "sequences": seqs}
        print(f"[chained] cumulative 1층 {cum[0]*100:.1f}% -> 옆 {cum[1]*100:.1f}% -> 2층 {cum[2]*100:.1f}%", flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    rec.close()
    env.close()


if __name__ == "__main__":
    main()
