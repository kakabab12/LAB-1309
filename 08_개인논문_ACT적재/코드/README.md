# 코드 (시뮬레이션 ACT 적재 실험)

작업 사본은 연구실 PC `~/ACT/act_sim/` 에 있고, 이 폴더는 그 사본입니다. 데이터셋(HDF5)과 체크포인트는 크기 때문에 올리지 않습니다.

## 파일

| 파일 | 하는 일 |
|---|---|
| `stack_env.py` | MuJoCo 셀: SO-101(손목 카메라), 저울, 분류 통 3개, 카메라 `front`/`top`, 성공 판정 |
| `expert.py` | 역기구학 + 최소 저크 궤적으로 통 벽을 집어 옮기는 스크립트 전문가 |
| `gen_data.py` | 단계별 시연 생성 → `data/stage{1,2,3}.hdf5` |
| `train_act.py` | LeRobot 0.3.3 ACT 학습 (단계별 모델) |
| `eval_act.py` | 단계별 / 연속 평가, 결과 JSON·GIF |
| `batch_expert.py` | 전문가 성능 확인 |
| `make_gif.py` | 전문가 연속 3단계 GIF |

## 설치 (Ubuntu 22.04, Python 3.10, GTX 1080 Ti 기준)

```bash
uv venv --system-site-packages --python /usr/bin/python3.10 .venv   # 시스템 torch 2.7.1+cu126 재사용
uv pip install --python .venv/bin/python mujoco==3.3.5 h5py
uv pip install --python .venv/bin/python lerobot==0.3.3 --no-deps
uv pip install --python .venv/bin/python "draccus==0.10.0" einops "huggingface-hub[hf-transfer]>=0.34.2,<1.0" \
    safetensors "datasets==3.6.0" "diffusers>=0.27.2,<0.36" "imageio[ffmpeg]<3" "gymnasium<1.0" \
    "torchcodec<0.6" opencv-python-headless pyserial "av>=14.2" pynput termcolor deepdiff jsonlines
git clone --depth 1 --filter=blob:none --sparse https://github.com/TheRobotStudio/SO-ARM100.git so_arm100
git -C so_arm100 sparse-checkout set Simulation/SO101
```

## 실행

```bash
.venv/bin/python batch_expert.py 20                                   # 전문가 확인
.venv/bin/python gen_data.py --stage 1 --episodes 200 --out data      # 단계 2, 3 도 같이
.venv/bin/python train_act.py --stage 1 --episodes 100 --steps 30000 --out runs/s1_n100
.venv/bin/python eval_act.py --ckpt runs/s1_n100/ckpt_030000 runs/s2_n100/ckpt_030000 runs/s3_n100/ckpt_030000 \
    --trials 50 --out results/eval/n100
```
