"""
정렬 보조 (2026-10-15) — VLA 가 블록을 쥐러 내려갈 때 앞뒤좌우만 블록 가운데로 맞춰 준다

왜
  al1 실패의 절반이 '쥐었다 빠짐'이고, 그중 대부분은 쥠이 1스텝만 잡혔다 — 손가락이 블록 모서리를 스치며 오므려
  블록이 튕겨 나갔다. 영상을 보면 학생은 손목 화면에서 블록을 가운데로 못 맞춘 채 내려간다 (10/14).
  시범은 블록 가운데에서 ±0.2mm 로 쥔다. 학생이 0.45초 늦게 보고 1~2cm 어긋나는 것을 카메라 계산으로 메운다.

무엇을 쓰나 (실물에도 있는 것만)
  정면 카메라 사진 → 색 검출(objdet) → 2차 변환식(front_calib.json, 평가 장면에서 평균 0.09cm·최대 0.24cm) → 블록의 책상 위 x, y
  로봇 손 위치 (로봇 상태). 손이 블록 가까이 오면 정면 카메라에서 블록이 손에 가릴 수 있어, 손이 8cm 넘게 떨어져 있을 때 잰 값을 기억해 쓴다.
  어느 블록인지는 지시문의 색 낱말 ("put the red block ...") 로 정한다.

무엇을 하나 (그리퍼가 열려 있고, 손이 그 블록에서 옆으로 4cm·위로 10cm 안일 때만)
  ① 앞뒤좌우 = 블록 가운데로 (위치 명령의 x, y 만 바꿈. 내려가기·손목·그리퍼는 VLA 그대로)
  ② 0.6cm 넘게 어긋나 있으면 쥐는 높이 3cm 아래에서는 조금씩 들어 올린다 (윗면에 걸린 손가락을 빼고 옆으로 맞추게)
  ③ 0.8cm 넘게 어긋나 있으면 오므리기를 미룬다 (맞춰지면 VLA 명령대로)

놓기 보조 (10/16, place=True, 평가 --place-assist): 지시문의 블록을 쥐고 있을 때만
  ④ 놓을 곳(판 가운데, 쌓기면 아래 블록 가운데)에서 옆으로 5cm 안이면 앞뒤좌우를 그 가운데로
  ⑤ 놓을 곳에서 3cm 넘게 떨어져 있으면 손을 벌리지 않는다 (al1 실패의 20% 가 판 밖에 놓기 — 대부분 판까지 가기 전에 놓음)
  일 바꾸기 때 쥐고 있던 다른 블록(새 지시의 대상이 아님)을 아무 데나 내려놓는 것은 막지 않는다.
  어느 블록을 쥐었나: 오므려 무언가를 쥔 순간(손가락 0.005~0.028) 손에서 4cm 안의 가장 가까운 블록. 판은 장면 처음에 잰 값.

쥐기 마무리 (10/16, complete=True, 평가 --grasp-complete):
  ⑥ 손이 열린 채 지시문의 블록 위 1cm 안·높이 10cm 안에서 20스텝 동안 1cm 도 안 내려가면 (맴돌기, 9/18 policy idling)
     그 자리에서 똑바로 내려가 쥐고 5cm 들어 올린 뒤 VLA 에 돌려준다.
     일 바꾸기 뒤 B 블록 바로 위에서 300스텝 넘게 맴돌다 시간이 다 된 장면이 있었다 (al1 + 놓기 보조, A8→B5 장면 1003).

내려가기 재촉 (10/17, descend=True, 평가 --descend-assist):
  ⑦ 손이 열린 채 지시문의 블록 위 0.8cm 안으로 맞춰졌는데 20스텝 동안 1cm 도 안 내려가면, 쥐는 높이까지 내려가게 한다
     (그동안 오므리지 않음). 바닥에서 10스텝을 기다려도 VLA 가 오므리지 않으면 오므린다. 들어 올리기·옮기기는 VLA.
     als2 실패 나누기: '블록 위에서 못 쥠' 은 손이 블록 중심 4~7cm 위에서 멈춘다 (⑥ 쥐기 마무리는 1cm 위 조건이라 거의 안 켜졌고,
     켜졌을 때는 들어 올린 뒤 VLA 가 옮기기를 시작하지 못했다).
다가가기 (10/17, approach=True, 평가 --approach-assist):
  ⑧ 빈손인데 ⓐ 지시문의 놓을 곳 6cm 안에 있거나 ⓑ 60스텝(3초) 동안 지시문의 블록에 1cm 도 다가가지 않으면 (그 블록에서 6cm 넘게),
     높이 13cm 로 올라 그 블록 위로 가서 블록 위 8cm 에서 VLA 에 돌려준다. 돌아오기 실패의 대부분이 원래 블록을 집지 않고
     빈손으로 놓을 판 위를 서성였다 (als1 + 보조, als2: 마지막 10초에 손이 70~110cm 움직임).
손목 맞추기 (10/17, yaw=True, 평가 --yaw-assist):
  ⑨ 블록 위 2.5cm 안·높이 1~9cm 에서 손목 카메라 사진의 블록 덩어리 기울기(minAreaRect)를 재, 손가락이 블록 면과
     나란해지도록 손을 세계 z 축으로 돌린다. 6° 넘게 어긋나 있으면 블록 윗면 근처(쥐는 높이 4.5cm 위)보다 내려가지 않고
     오므리지 않는다. 4° 안으로 맞으면 놓아준다. 정육면체는 90° 대칭, 노란 막대는 긴 축이 사진 세로(손가락이 사진 가로로
     벌어짐)가 되게. 시뮬에서 맞춤: 손을 +δ 돌리면 사진 기울기가 −δ (1:1) → 사진에서 m 이면 손을 +m 돌린다.
     als2 + 보조 다 켬에서 '블록 위에서 못 쥠' (B 53, 돌아오기 42) 은 손가락-블록 면 어긋남 중앙 18° (쥔 장면 4°) —
     벌린 손가락이 블록 모서리에 얹혀 내려가지 못했다.
쥐기 확정 (10/17, close=True, 평가 --close-assist, ⑦ 내려가기 재촉과 같이 켬):
  ⑩ 쥐는 높이(0.8cm 안)에 맞춰 내려왔으면 VLA 가 손을 올리지 못하게 잡고, 3스텝 뒤 12스텝 동안 오므린다. 그 뒤는 VLA.
     (평가 기록의 물체 높이는 블록 바닥 기준이다.) als2 + 보조 다 켬의 '블록 위에서 못 쥠' 장면을 다시 돌려 보니 손은 쥐는 높이까지
     내려오고 아무것에도 닿지 않았는데, 바닥에서 VLA 가 그리퍼를 벌린 채 위로 +1.0 을 명령해 2.7 ↔ 3.9cm(바닥 위)를 오르내렸다.
막힘 풀기 (10/21, unjam=True, 평가 --unjam-assist): 받침대 없는 장면
  ⑪ 빈손으로 블록 옆 4cm·위 10cm 안에서 1초(20스텝) 동안 손이 3mm 도 안 움직이고, 손목 관절(4·5·6번) 하나가 한계 3° 안이면,
     손을 쥐는 높이 7cm 위까지 올리면서 처음 자세의 손 방향으로 되돌린 뒤(최대 60스텝), 정면 카메라로 블록 위치를 새로 재고 VLA 에 돌려준다.
     np1 + 보조 '블록 위에서 못 쥠' 8장면을 다시 돌려 보니 모두 같은 순서였다: 손가락이 블록 윗면·모서리나 책상에 걸림 →
     내려가기 재촉이 계속 누름 → 손이 70° 넘게 기울며 4·5·6번 관절이 한계에 박힘 → 끝까지 못 움직임. 블록이 손가락에 밀려
     움직였는데 기억한 위치(손이 8cm 안이면 갱신 안 함)는 옛 자리라, 보조가 블록 옆 2cm 를 누르던 장면도 있다 (A8→B0 장면 1007).
  ⑨ 손목 맞추기는 6번 관절 한계를 본다: 돌릴 방향이 관절을 한계 10° 안으로 넣으면 대칭인 반대 방향으로 돌린다
     (정육면체 90°, 막대 180°). 6번 관절 +δ ≈ 손가락 방향 세계 z −δ·cos(기울기) (받침대 없는 처음 자세에서 +11.5° → −8.6°).
내려놓기 (10/22, putdown=True, 평가 --putdown-assist): 받침대 없는 장면
  ⑫ 지시문의 블록이 아닌 블록을 쥐고 있으면 (일 바꾸기 직후), 빈 자리 — 다른 물체에서 9cm, 지시문의 블록·놓을 곳에서 12cm,
     로봇 밑동에서 18~40cm, 되도록 그 블록이 처음 놓여 있던 자리 — 로 10cm 높이에서 옮겨 내려놓고 6cm 든 뒤 VLA 에 돌려준다.
     np2 + 막힘 풀기 평가에서 일 바꾸기 140장면 중 59장면은 VLA 가 A 를 B 의 놓을 곳(6cm 안)에 내려놓았다. 그때 B 성공 32/59
     (다른 곳이면 78/81), 돌아오기 17/47 (다른 곳이면 67/73) — A 가 가장 가까운 물체에서 9cm 넘게 떨어져 있으면 돌아오기 52/54.
     시범은 A 를 쥔 자리에 그대로 내려놓는다 (switch_v6.put_down_any).
쥔 블록 알아보기 (10/22, fresh_held=True, 평가 --fresh-held):
  ⑬ 무엇을 쥐었나를 쥐는 순간 '지금 사진'으로 잰 위치로 정하고 (블록이 아직 책상 위라 정확), 쥐고 있는 블록은 기억을 고치지 않는다.
     np2 + 보조 5가지의 돌아오기 '판 밖에 놓음' (원래 블록이 목표에서 30~55cm) 을 다시 돌려 보니 (A1→B7 장면 1001), 원래 블록(빨강)을
     제대로 쥐었는데 보조의 기억 속 노란 막대 자리(옮기는 동안 공중의 막대를 책상 위로 잘못 읽은 '유령' 자리)가 손에 더 가까워
     노랑을 쥐었다고 보고, 내려놓기(⑫)가 빨강을 빈 자리로 가져가 버렸다. 내려놓기는 지시문의 블록이 손에서 6cm 넘게 떨어져
     책상에 보일 때만 켠다 (안 보이면 손에 있을 수 있다).
  ⑫' 옮기는 도중 새 지시 (10/22, putdown_near=True, 평가 --putdown-near): 사용자 "어느 정도 수행했다가 다른 곳에 놨다가 다른 태스크하고
     다시 하는 걸로". 처음 자리로 되돌려 놓지 않고, 지금 손에서 가장 가까운 빈 자리에 내려놓는다 (처음 자리도 후보 중 하나).
엉뚱한 블록 막기 (10/22, wrong_guard=True, 평가 --wrong-guard):
  ⑭ 돌아오기에서 B 로 쌓아 둔 블록(파랑 위 초록)을 VLA 가 집어 가, 내려놓기(⑫)가 그것을 빈 자리에 놓아 B 를 망쳤다
     (np2 + 보조 5가지: A4→B9·A8→B9 에서 B 를 끝까지 지킨 장면 40 중 1). 두 가지로 막는다.
     ① 빈손으로 지시문의 블록이 아닌 블록 바로 위(2.5cm 안)에서 오므리려 하고 지시문의 블록은 4cm 넘게 떨어져 보이면, 오므리지 않고 조금 든다
     ② 그래도 같은 지시 아래에서 엉뚱한 블록을 쥐었으면, 빈 자리가 아니라 쥐었던 그 자리·그 높이에 도로 놓는다 (쌓여 있던 블록은 다시 쌓임).
        지시가 바뀌어 쥐고 있던 블록(일 바꾸기)만 빈 자리로.
     ①② 로도 쌓은 B 가 그대로 무너져 다시 돌려 보니 (A4→B9 장면 1000, diag_stack.py): 초록을 파랑 위에 올려 닿는 순간 평가가
     B 성공으로 보고 돌아오기(A2)로 넘어갔는데 손은 아직 초록을 쥐고 있었다 → 보조가 '지시문(빨강)이 아닌 블록' 으로 보고 빈 자리로
     옮겨 가 쌓은 것이 무너졌다. '앞 지시의 놓을 곳 위면 그 자리에서 놓기' 는 옮기는 도중(쥔 뒤 1.5초) 이미 목적지 근처인 장면(A4: 빨강
     → 가까운 회색 판)에서 A 를 미리 놓아 버려 뺐다. 대신 평가가 B 성공 뒤 손을 벌릴 때까지 기다린다 (switch_experiment --release-wait,
     한국어 실행기·실물은 원래 손을 벌려야 끝으로 본다).
판 지키기 (10/22 밤, pad_safe=True, 평가 --pad-safe): 새 기준(옮기는 도중 바꿈) 돌아오기 실패를 같은 장면으로 다시 돌려 보니 (diag_pad.py,
  A4→B9 장면 1000) 빨강을 판에서 9.5cm 떨어진 빈 자리에 내려놨다가 다시 쥐자마자, 놓기 보조(④)가 아직 책상 높이인 손을 판 쪽으로
  끌어 빨강이 책상 위를 미끄러지며 판을 6cm 밀었다 (판에 닿은 것: 빨강). 블록은 판의 옛 자리에 놓였고 판은 밀려 있어 실패.
  ⑮ 지시문의 블록을 쥐었는데 놓을 곳에서 3cm 넘게 떨어져 있고 쥐는 높이 3cm 위보다 낮으면, 옆으로 끌지 않고 먼저 똑바로 든다
  ⑮' 내려놓기(⑫) 빈 자리는 그 블록의 원래 목표(앞 지시의 놓을 곳)에서 13cm 넘게 (없으면 예전 조건). ⑮ 없이 이것만: putdown_far=True, --putdown-far
     (10/23: 고친 보조 28항목 A8→B0 에서 ⑮ 먼저 들기가 45스텝 넘게 켜진 장면은 B 성공 2/7, 짧으면 29/31 → 따로 잴 수 있게 나눔)
판 따라가기 (pad_track=True, 평가 --pad-track):
  ⑯ 판도 밀리면 기억을 고친다: 정면 카메라로 잰 판이 기억과 1.5cm 넘게 5스텝 연속 다르고, 그 판 8cm 안에 블록이 없고(가림), 손이 10cm 넘게 떨어져 있을 때
기억 고치기 (mem_refresh=True, 평가 --mem-refresh):
  ⑰ 손이 8cm 안이어도, 블록이 손 바로 밑(3cm 안)이 아니고 사진 위치가 기억과 2.5cm 넘게 10스텝 연속 다르면 사진 위치로 고친다.
     A4→B9 장면 1004: 헛쥐기로 빨강이 6cm 밀렸는데 기억(손 8cm 안이면 안 고침)은 옛 자리라 정렬 보조가 손을 빈 자리에 400스텝 붙잡았다.
  넓은 변환식: 블록을 새 빈 자리에 내려놓는 새 기준에서는 블록별 변환식(처음 자리 근처로만 맞춤)이 2~3cm 틀려(diag_pad: 2.2·3.2cm)
  정렬 보조가 블록 옆을 짚었다 → PIPER_FRONT_CALIB=piper_sim/front_calib_wide.json 과 같이 쓴다.
쥐기 확정 기다리기 (10/23 새벽, close_wait=N, 평가 --close-wait N):
  ⑩' 쥐는 높이에서 3스텝 만에 대신 오므리지 않고 N 스텝 기다린다. 그동안 VLA 가 스스로 오므리면 VLA 에 맡긴다 (예전 ⑩ 은 그 3스텝 동안
     VLA 의 오므리기를 막았다). 고친 보조로 A8→B0 장면 1010 을 다시 돌려 보니(diag_held.py) ⑩ 이 VLA 보다 먼저 파랑을 쥐어 주자,
     VLA 는 쥔 직후부터 계속 벌리려 했고 ⑤ 가 500스텝 막는 동안 파랑을 든 채 헤맸다 (고치기 전에는 VLA 가 스스로 쥐어 성공한 장면).
다시 쥐기 허용 (10/23 새벽, regrasp_ok=True, 평가 --regrasp-ok):
  ⑱ ⑤(놓을 곳에서 멀면 못 벌림)와 ⑮(먼저 들기)는 쥔 자리 3cm 안·쥔 높이 5cm 위까지에서 VLA 가 벌리려 하면 막지 않는다 (다시 쥐기).
     ⑩' 로도 A8→B0 1010 의 VLA 는 쥐어 준 뒤 계속 벌리려 했다 → 쥔 자리에서는 놓게 두고 VLA 가 다시 쥐게 한다. 판에서 먼 공중 놓기는 그대로 막는다.
  (⑩'·⑱ 은 몇 장면으로 차이가 안 보였다. A8→B0·A8→B7 40장면 짝 비교에서 ⑩·⑮ 를 빼고 ⑮'·⑯·⑰·넓은 변환식만 둔 것이 가장 좋았다)
막힌 쥐기 마무리 (10/23 03시, stuck_grasp=N, 평가 --stuck-grasp N):
  ⑲ 빈손으로 지시문의 블록 바로 위(1cm 안)·쥐는 높이 1.5cm 안에서 벌린 채 머문 스텝이 지시마다 누적 N 이 되면, ⑥ 처럼 똑바로 내려가
     오므리고(10스텝) 5cm 든 뒤 VLA 에 돌려준다. 손 안 잡는 고침의 노란 막대 B 실패 5개 중 4개가 막대를 한 번도 못 쥐고 쥐는 높이에서
     벌린 채 67~231스텝 머물렀다 (A8→B7). 바로 쥐어 주는 ⑩ 과 달리 VLA 가 오래(N=40, 2초) 못 쥘 때만.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

import objdet

HERE = Path(__file__).resolve().parent
NAMES = {"red": "akita_black_bowl_1", "green": "cream_cheese_1", "blue": "blue_block_1", "yellow": "wine_bottle_1"}
TABLE_Z = 0.90          # 책상 높이 (실물: 로봇 밑동 기준으로 한 번 잼)
GRASP_H = {"red": 0.0225, "green": 0.0225, "blue": 0.0225, "yellow": 0.020}   # 쥐는 높이 = 블록 높이 절반 (시범과 같음)
if __import__("os").environ.get("PIPER_NOPED") == "1":      # 10/20 받침대 없음: 막대는 2.8cm 에서 쥔다 (task_experts)
    GRASP_H["yellow"] = 0.028
LIMIT_DEG = {3: 100.0, 4: 70.0, 5: 120.0}       # PiPER 손목 관절 4·5·6번 한계 [도] (build_piper_xml URDF_RANGE_DEG, 대칭)


def _feats(u, v):
    return np.array([u, v, 1.0, u * u, u * v, v * v])


class GraspAssist:
    def __init__(self, calib=None, radius=0.04, height=0.10,
                 stop_err=0.006, close_err=0.008, vmax=0.4, place=False, place_radius=0.05, release_err=0.03,
                 complete=False, idle_steps=20, descend=False, approach=False, yaw=False, close=False, unjam=False,
                 putdown=False, fresh_held=False, putdown_near=False, wrong_guard=False, pad_safe=False, pad_track=False,
                 mem_refresh=False, close_wait=3, regrasp_ok=False, putdown_far=False, stuck_grasp=0):
        # 10/22: PIPER_FRONT_CALIB=piper_sim/front_calib_wide.json 이면 책상 전체로 맞춘 변환식 (블록 공통, calib_front_wide.py).
        #   기본(front_calib.json)은 블록마다 처음 배치로만 맞춰, 블록이 제자리를 떠나면 평균 1.8cm·최대 13.6cm 틀렸다
        calib = calib or os.environ.get("PIPER_FRONT_CALIB", str(HERE / "piper_sim" / "front_calib.json"))
        self.C = {k: np.array(v) for k, v in json.loads(Path(calib).read_text()).items()}
        self.radius, self.height, self.stop_err, self.close_err, self.vmax = radius, height, stop_err, close_err, vmax
        self.place, self.place_radius, self.release_err = place, place_radius, release_err
        self.n_place = 0
        self.n_hold_release = 0
        self.complete, self.idle_steps = complete, idle_steps
        self.n_complete = 0
        self.ep = None
        self.n_used = 0
        self.descend, self.approach, self.yaw, self.close = descend, approach, yaw, close
        self.unjam, self.putdown, self.fresh_held, self.putdown_near = unjam, putdown, fresh_held, putdown_near
        self.wrong_guard = wrong_guard
        self.pad_safe, self.pad_track, self.mem_refresh = pad_safe, pad_track, mem_refresh
        self.close_wait, self.regrasp_ok, self.putdown_far = close_wait, regrasp_ok, putdown_far
        self.stuck_grasp = stuck_grasp
        self.stat = {}

    def stats(self):
        """이번 장면에서 보조가 손댄 스텝 수 (평가 기록용)."""
        return dict(self.stat)

    def _target(self, instruction):
        m = re.search(r"put the (red|green|blue|yellow) block", instruction.lower())
        return m.group(1) if m else None

    def _dest(self, instruction):
        """놓을 곳 색 (판이면 보라·회색·주황·흰색, 쌓기면 아래 블록 색)."""
        m = re.search(r" on the (red|green|blue|yellow|purple|gray|orange|white) (square|block)", instruction.lower())
        return m.group(1) if m else None

    def _update(self, ep):
        """손이 8cm 넘게 떨어진 블록만 위치를 새로 잰다 (가까우면 손에 가려 중심이 틀어질 수 있다)."""
        f = objdet.detect(np.asarray(ep.obs["pixels"]["image"]), np.asarray(ep.obs["pixels"]["image2"]))
        hand = np.asarray(ep.obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        self.fresh = {}
        for k, c in enumerate(objdet.OBJECTS):
            if f[2 * k] < 0 or c not in self.C:
                continue
            xy = self.C[c] @ _feats(f[2 * k], f[2 * k + 1])
            self.fresh[c] = xy
            if c in objdet.PADS:                # 판은 움직이지 않는다 — 처음 값 (블록을 올리면 가려져 틀어진다)
                self.pos.setdefault(c, xy)
                if self.pad_track:              # ⑯ 밀린 판은 고친다 (블록이 가까우면 가려지니 안 고침)
                    self._track_pad(c, xy, hand)
            elif self.mem_refresh and c in self.pos and np.linalg.norm(xy - self.pos[c]) > 0.025 \
                    and np.linalg.norm(xy - hand[:2]) > 0.03 and not (self.fresh_held and c == self.held):
                self.mr_n[c] = self.mr_n.get(c, 0) + 1          # ⑰ 손 8cm 안이어도 계속 다르게 보이면 고침
                if self.mr_n[c] >= 10 or np.linalg.norm(xy - hand[:2]) > 0.08:
                    self.pos[c], self.mr_n[c] = xy, 0
            elif (np.linalg.norm(xy - hand[:2]) > 0.08 or c not in self.pos) and not (self.fresh_held and c == self.held):
                self.pos[c] = xy
            if c in objdet.BLOCKS:
                self.pos0.setdefault(c, xy)     # 장면 처음 자리 (⑫ 내려놓을 자리 1순위)
            if self.mem_refresh and c in objdet.BLOCKS and (c not in self.pos or np.linalg.norm(xy - self.pos[c]) <= 0.025):
                self.mr_n[c] = 0

    def _track_pad(self, c, xy, hand):
        """⑯ 판 기억 고치기: 1.5cm 넘게 5스텝 연속 다르고, 판 8cm 안에 블록 없고, 손 10cm 밖."""
        if np.linalg.norm(xy - self.pos[c]) < 0.015 or np.linalg.norm(xy - hand[:2]) < 0.10 \
                or any(b in self.fresh and np.linalg.norm(self.fresh[b] - xy) < 0.08 for b in objdet.BLOCKS):
            self.pt_n[c] = 0
            return
        self.pt_n[c] = self.pt_n.get(c, 0) + 1
        if self.pt_n[c] >= 5:
            self.pos[c], self.pt_n[c] = xy, 0
            self.stat["pad_track"] = self.stat.get("pad_track", 0) + 1

    def __call__(self, ep, act, instruction):
        if ep is not self.ep:              # 새 장면
            self.ep, self.pos, self.held = ep, {}, None
            self.prim, self.zhist = None, []
            self.dmode, self.dbottom, self.amode, self.dhist, self.dhist_a = False, 0, False, [], []
            self.stat = {"align": 0, "place": 0, "descend": 0, "close": 0, "approach": 0, "yaw": 0}
            if self.unjam:
                self.stat.update(unjam=0, unjam_n=0)
            if self.putdown:
                self.stat.update(putdown=0, putdown_n=0)
            self.pos0, self.pd = {}, None
            self.yaw_bad = False
            self.nclose = 0
            self.ujmode, self.ujhist = 0, []
            self.mr_n, self.pt_n, self.sg_n = {}, {}, {}
            self.home_mat = np.asarray(ep.obs["robot_state"]["eef"]["mat"], dtype=np.float64).reshape(3, 3)   # 장면 처음 = 처음 자세
            self.home_xy = np.asarray(ep.obs["robot_state"]["eef"]["pos"], dtype=np.float64)[:2].copy()
        self._update(ep)
        q = float(ep.obs["robot_state"]["gripper"]["qpos"][0])
        hand = np.asarray(ep.obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        if q > 0.028 or q < 0.005:         # 벌렸거나 빈손으로 끝까지 닫힘 → 쥔 것 없음
            self.held = None
        elif self.held is None:            # 방금 무언가를 쥠 → 손에 가장 가까운 블록
            cand = {b: self.pos[b] for b in objdet.BLOCKS if b in self.pos}
            if self.fresh_held:            # ⑬ 지금 사진으로 잰 위치 우선 (기억은 공중의 블록을 잘못 읽은 '유령' 자리일 수 있다)
                cand.update({b: v for b, v in self.fresh.items() if b in objdet.BLOCKS})
            near = [(np.linalg.norm(xy - hand[:2]), b) for b, xy in cand.items()]
            if near and min(near)[0] < 0.04:
                self.held = min(near)[1]
                self.grasp_at = (hand.copy(), instruction)        # ⑭ 어디서·어떤 지시 아래 쥐었나
        c = self._target(instruction)
        if instruction != getattr(self, "cur_instr", None):          # 바로 앞 지시 (⑭③)
            self.prev_instr, self.cur_instr = getattr(self, "cur_instr", None), instruction
        if self.prim is not None:          # ⑥ 쥐기 마무리 중
            return self._primitive(act, hand, q)
        if self.pd is not None:            # ⑫ 내려놓는 중
            return self._putdown(act, hand)
        if c is None or c not in self.pos:
            return act
        if self.putdown and q <= 0.028 and self.held is not None and self.held != c and \
                (not self.fresh_held or (c in self.fresh and np.linalg.norm(self.fresh[c] - hand[:2]) > 0.06)):
            ga = getattr(self, "grasp_at", None)
            if self.wrong_guard and ga is not None and ga[1] == instruction:     # ⑭② 같은 지시 아래 엉뚱한 블록 → 쥐었던 자리·높이에 도로
                spot, z_back = ga[0][:2].copy(), float(ga[0][2])
            else:
                spot, z_back = self._free_spot(instruction, hand, self.held), None
            if spot is not None:
                self.pd = {"spot": spot, "held": self.held, "phase": "carry", "n": 0, "z_back": z_back}
                self.stat["putdown_n"] += 1
                return self._putdown(act, hand)
        if q <= 0.028:
            return self._place(act, instruction, c, hand) if self.place else act
        e = self.pos[c] - hand[:2]
        d = float(np.linalg.norm(e))
        dz = hand[2] - (TABLE_Z + GRASP_H[c])           # 쥐는 높이 위로 얼마나
        if self.wrong_guard and self.held is None and act[6] > 0 and hand[2] < TABLE_Z + 0.12:
            # ⑭① 지시문의 블록이 아닌 블록 바로 위에서 오므리려 함 (지시문의 블록은 4cm 넘게 떨어져 보임) → 오므리지 않고 조금 든다
            others = [self.fresh.get(b, self.pos.get(b)) for b in objdet.BLOCKS if b != c]
            near_other = min((np.linalg.norm(v - hand[:2]) for v in others if v is not None), default=1.0)
            fc = self.fresh.get(c, self.pos.get(c))
            if near_other < 0.025 and fc is not None and np.linalg.norm(fc - hand[:2]) > 0.04:
                act = np.array(act, dtype=np.float32, copy=True)
                act[6], act[2] = -1.0, max(float(act[2]), 0.3)
                self.stat["guard"] = self.stat.get("guard", 0) + 1
                return act
        if self.unjam and self.held is None:
            a2 = self._unjam(ep, act, c, hand, d, dz)
            if a2 is not None:
                return a2
        if self.approach and self.held is None:
            a2 = self._approach(act, instruction, c, hand, d)
            if a2 is not None:
                return a2
        if d > self.radius or dz > self.height or dz < -0.02:
            self.dmode, self.dhist = False, []
            return act
        act = np.array(act, dtype=np.float32, copy=True)
        act[:2] = np.clip(e / 0.05, -self.vmax, self.vmax)          # ① 앞뒤좌우 = 블록 가운데
        if d > self.stop_err and dz < 0.03:
            act[2] = max(act[2], 0.15)                              # ② 많이 어긋나면 조금 들어 올린다 (손가락이 윗면에 걸려 옆으로 못 움직였다)
        if d > self.close_err and act[6] > 0:
            act[6] = -1.0                                           # ③ 오므리기를 미룬다
        self.n_used += 1
        self.stat["align"] = self.stat.get("align", 0) + 1
        if self.stuck_grasp and self.held is None and q > 0.03 and d < 0.01 and dz < 0.015:
            k = (instruction, c)                                    # ⑲ 블록 바로 위 쥐는 높이에서 벌린 채 머문 스텝 (지시마다 누적)
            self.sg_n[k] = self.sg_n.get(k, 0) + 1
            if self.sg_n[k] >= self.stuck_grasp:
                self.sg_n[k] = 0
                self.prim = {"c": c, "phase": "down", "n": 0, "xy": self.pos[c].copy(), "z": TABLE_Z + GRASP_H[c]}
                self.stat["stuck_grasp"] = self.stat.get("stuck_grasp", 0) + 1
                return self._primitive(act, hand, q)
        if self.yaw:
            act = self._yaw(ep, act, c, d, dz)
        if self.descend and not getattr(self, "yaw_bad", False):
            act = self._descend(act, hand, d, dz)
        if self.complete:
            self.zhist.append(hand[2])
            if d < 0.01 and 0.005 < dz and len(self.zhist) >= self.idle_steps \
                    and self.zhist[-self.idle_steps] - hand[2] < 0.01:     # ⑥ 블록 위에서 내려가지 않고 맴돎
                self.prim = {"c": c, "phase": "down", "n": 0, "xy": self.pos[c].copy(),
                             "z": TABLE_Z + GRASP_H[c]}
                self.n_complete += 1
                self.zhist = []
                return self._primitive(act, hand, q)
        return act

    def _wrist_angle(self, img, c):
        """손목 사진에서 색 c 덩어리의 어긋남 [도]. 손을 이만큼(세계 z, +) 돌리면 맞는다. 덩어리가 작으면 None."""
        import cv2
        m = objdet.masks(np.asarray(img))[c]
        lab, n = ndimage.label(m)
        if n == 0:
            return None
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        if sizes[k - 1] < 1500:            # 일부만 보이면 기울기가 틀린다 (시뮬 맞춤에서 1,454픽셀일 때 43°)
            return None
        ys, xs = np.nonzero(lab == k)
        (_, _), (w, h), ang = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        if c == "yellow":                  # 막대: 긴 축이 사진 세로(90°)가 되게
            longang = ang if w >= h else ang + 90
            return ((longang - 90 + 90) % 180) - 90
        return ((ang + 45) % 90) - 45

    def _yaw(self, ep, act, c, d, dz):
        """⑨ 손가락을 블록 면과 나란히."""
        if d > 0.025 or not (0.01 < dz < 0.09):
            self.yaw_bad = False
            return act
        m = self._wrist_angle(ep.obs["pixels"]["image2"], c)
        if m is None:
            return act
        if abs(m) > 6:
            self.yaw_bad = True
        elif abs(m) < 4:
            self.yaw_bad = False
        if not self.yaw_bad:
            return act
        if self.unjam:                     # 막힘 풀기를 켜면 6번 관절 한계도 본다
            m = self._yaw_turn(ep, m, c)
        act[5] = float(np.clip(np.radians(m) / 0.5, -0.4, 0.4))     # 세계 z 축 회전 (회전 1.0 = 0.5rad)
        if dz < 0.045:
            act[2] = max(act[2], 0.0)
        act[6] = -1.0
        self.stat["yaw"] += 1
        return act

    def _yaw_turn(self, ep, m, c):
        """돌릴 각도 m[도, 세계 z] 이 6번 관절을 한계 10° 안으로 넣으면 대칭인 반대 방향 (정육면체 m∓90, 막대 m∓180).
        6번 관절 +δ ≈ 손가락 방향 세계 z −δ·cos(기울기) → 돌린 뒤 6번 관절 ≈ 지금 − m / cos(기울기)."""
        j6 = float(np.degrees(ep.obs["robot_state"]["joints"]["pos"][5]))
        k = max(abs(float(np.asarray(ep.obs["robot_state"]["eef"]["mat"]).reshape(3, 3)[2, 2])), 0.3)
        per = 180.0 if c == "yellow" else 90.0
        for mm in (m, m - np.sign(m) * per):
            if abs(j6 - mm / k) < LIMIT_DEG[5] - 10:
                return mm
        return m

    def _descend(self, act, hand, d, dz):
        """⑦ 맞춰졌는데 내려가지 않으면 쥐는 높이까지 내려가게, 바닥에서 오래 안 쥐면 오므림."""
        self.dhist.append(hand[2])
        if not self.dmode and d < self.close_err and dz > 0.015 and len(self.dhist) >= self.idle_steps \
                and self.dhist[-self.idle_steps] - hand[2] < 0.01:
            self.dmode, self.dbottom = True, 0
        if not self.dmode:
            return act
        if d > 0.015:                      # 많이 어긋나면 멈추고 VLA·정렬에 맡김
            self.dmode = False
            return act
        if dz > 0.008:
            act[2] = float(np.clip(-dz / 0.05, -0.3, -0.05))
            act[6] = -1.0
            self.stat["descend"] += 1
            if self.close:
                self.dbottom, self.nclose = 0, 0
            return act
        if self.close:                     # ⑩ 바닥: 올리지 못하게, 3스텝 뒤 12스텝 오므림, 그 뒤 VLA
            self.dbottom += 1
            act[2] = min(act[2], 0.0)
            if self.close_wait > 3 and self.dbottom <= self.close_wait and act[6] > 0:
                self.dmode, self.dhist, self.dbottom, self.nclose = False, [], 0, 0     # ⑩' 기다리는 동안 VLA 가 스스로 오므리면 맡김
                return act
            if self.dbottom > self.close_wait:
                act[6] = 1.0
                self.nclose += 1
                self.stat["close"] += 1
                if self.nclose >= 12:
                    self.dmode, self.dhist, self.dbottom, self.nclose = False, [], 0, 0
            else:
                act[6] = -1.0
            return act
        self.dbottom += 1
        if act[6] <= 0 and self.dbottom > 10:
            act[6] = 1.0
            self.stat["close"] += 1
        if act[6] > 0:
            self.dmode, self.dhist = False, []
        return act

    def _unjam(self, ep, act, c, hand, d, dz):
        """⑪ 블록 위에서 손목 관절이 한계에 박혀 멈췄으면 위로 빼며 처음 자세의 손 방향으로 되돌린다. 안 하면 None."""
        jp = np.degrees(np.asarray(ep.obs["robot_state"]["joints"]["pos"], dtype=np.float64))
        margin = min(lim - abs(jp[k]) for k, lim in LIMIT_DEG.items())
        self.ujhist = (self.ujhist + [hand.copy()])[-20:]
        if not self.ujmode:
            if d > self.radius or dz > self.height or len(self.ujhist) < 20 or margin > 3:
                return None
            h = np.array(self.ujhist[-20:])
            if np.linalg.norm(h.max(0) - h.min(0)) > 0.003:
                return None
            self.ujmode = 1
            self.stat["unjam_n"] += 1
        self.ujmode += 1
        m = np.asarray(ep.obs["robot_state"]["eef"]["mat"], dtype=np.float64).reshape(3, 3)
        rv = Rotation.from_matrix(self.home_mat @ m.T).as_rotvec()       # 세계 기준, 지금 → 처음 자세 방향
        a = np.array(act, dtype=np.float32, copy=True)
        a[:2] = 0.0
        a[2] = 0.4 if dz < 0.07 else 0.0
        a[3:6] = np.clip(rv / 0.5, -0.3, 0.3)
        a[6] = -1.0
        self.stat["unjam"] += 1
        if (dz >= 0.07 and np.degrees(np.linalg.norm(rv)) < 10) or self.ujmode > 60:
            self.ujmode, self.ujhist = 0, []
            self.dmode, self.dhist, self.yaw_bad = False, [], False
            if c in getattr(self, "fresh", {}):    # 블록 7cm 위에서 새로 잰 위치 (기억은 손가락에 밀린 블록을 모른다)
                self.pos[c] = self.fresh[c]
        return a

    def _free_spot(self, instruction, hand, held):
        """⑫ 쥔 블록을 내려놓을 빈 자리 (책상 위 x, y). 처음 자리 → 손에서 가까운 격자 순. 없으면 None."""
        c, dst = self._target(instruction), self._dest(instruction)
        others = {k: v for k, v in self.pos.items() if k != held}
        base = np.array([-0.33, 0.0])       # 로봇 밑동 (layout_noped)

        own = None                          # ⑮' 쥔 블록의 원래 목표 (앞 지시의 놓을 곳)
        pi = getattr(self, "prev_instr", None)
        if (self.pad_safe or self.putdown_far) and pi and self._target(pi) == held and self._dest(pi) in others:
            own = others[self._dest(pi)]

        def ok(p, strict=False):
            if strict and own is not None and np.linalg.norm(p - own) < 0.13:
                return False
            if min((np.linalg.norm(p - v) for v in others.values()), default=1.0) < 0.09:
                return False
            if min((np.linalg.norm(p - others[k]) for k in (c, dst) if k in others), default=1.0) < 0.12:
                return False
            r = np.linalg.norm(p - base)
            return 0.18 < r < 0.40          # 처음 자세(책상 위 18cm)와는 옮기는 높이 10cm 에서도 8cm 넘게 떨어진다
        cands = [self.pos0[held]] if held in self.pos0 and not self.putdown_near else []
        xs, ys = np.meshgrid(np.arange(-0.16, 0.07, 0.02), np.arange(-0.26, 0.27, 0.02))
        grid = np.stack([xs.ravel(), ys.ravel()], 1)
        if self.putdown_near and held in self.pos0:
            grid = np.vstack([grid, self.pos0[held][None]])
        cands += list(grid[np.argsort(np.linalg.norm(grid - hand[:2], axis=1))])
        for strict in ([True, False] if own is not None else [False]):
            for p in cands:
                if ok(np.asarray(p), strict):
                    return np.array(p, dtype=np.float64)
        return None

    def _putdown(self, act, hand):
        """⑫ 빈 자리 위로 (10cm 높이) → 내림 → 벌림 8스텝 → 6cm 듦 → VLA 에 돌려줌. 손목 방향은 그대로 둔다."""
        pd = self.pd
        pd["n"] += 1
        self.stat["putdown"] += 1
        a = np.zeros(7, dtype=np.float32)
        z_travel, z_rest = TABLE_Z + 0.10, TABLE_Z + GRASP_H[pd["held"]] + 0.004
        if pd.get("z_back") is not None:                     # ⑭② 쥐었던 높이 (다른 블록 위였으면 그 위) + 0.4cm
            z_rest = pd["z_back"] + 0.004
            z_travel = max(z_travel, z_rest + 0.06)
        e = pd["spot"] - hand[:2]
        d = float(np.linalg.norm(e))
        if pd["phase"] == "carry":
            tgt = np.array([pd["spot"][0], pd["spot"][1], z_travel])
            if hand[2] < z_travel - 0.02 and d > 0.03:      # 낮으면 먼저 올라감 (다른 블록에 걸리지 않게)
                tgt[:2] = hand[:2]
            a[:3] = np.clip((tgt - hand) / 0.05, -self.vmax, self.vmax)
            a[6] = 1.0
            if d < 0.01 or pd["n"] > 120:
                pd["phase"], pd["n"] = "down", 0
        elif pd["phase"] == "down":
            a[:3] = np.clip((np.array([pd["spot"][0], pd["spot"][1], z_rest]) - hand) / 0.05, -0.3, 0.3)
            a[6] = 1.0
            if abs(hand[2] - z_rest) < 0.004 or pd["n"] > 60:
                pd["phase"], pd["n"] = "open", 0
        elif pd["phase"] == "open":
            a[6] = -1.0
            if pd["n"] >= 8:
                pd["phase"], pd["n"], pd["z_up"] = "up", 0, hand[2] + 0.06
        else:
            a[:3] = np.clip((np.array([hand[0], hand[1], pd["z_up"]]) - hand) / 0.05, -0.3, 0.3)
            a[6] = -1.0
            if hand[2] > pd["z_up"] - 0.005 or pd["n"] > 20:
                self.pd = None
                self.pos[pd["held"]] = pd["spot"].copy()     # 손에 가려 못 잰 동안은 놓은 자리로 기억
        return a

    def _approach(self, act, instruction, c, hand, d):
        """⑧ 빈손으로 놓을 곳 근처에 있거나 블록에 다가가지 않으면 블록 위로 데려간다. 안 하면 None."""
        self.dhist_a.append(d)
        dst = self._dest(instruction)
        at_dst = dst is not None and dst in self.pos and np.linalg.norm(self.pos[dst] - hand[:2]) < 0.06
        stuck = len(self.dhist_a) > 60 and self.dhist_a[-61] - d < 0.01
        if not self.amode and d > 0.06 and (at_dst or stuck):
            self.amode = True
        if not self.amode:
            return None
        if d < 0.02:                       # 블록 위에 왔다 → VLA·정렬 보조에 돌려줌
            self.amode, self.dhist_a = False, []
            return None
        z = TABLE_Z + (0.13 if d > 0.05 else GRASP_H[c] + 0.08)
        tgt = np.array([self.pos[c][0], self.pos[c][1], z])
        if hand[2] < TABLE_Z + 0.11 and d > 0.05:      # 낮으면 먼저 올라감 (쌓은 블록 9cm 위로)
            tgt[:2] = hand[:2]
        a = np.array(act, dtype=np.float32, copy=True)
        a[:3] = np.clip((tgt - hand) / 0.05, -self.vmax, self.vmax)
        a[6] = -1.0
        self.stat["approach"] += 1
        return a

    def _primitive(self, act, hand, q):
        """⑥ 쥐기 마무리: 똑바로 내려가기 → 오므리기 10스텝 → 5cm 들어 올리기 → VLA 로 돌려줌."""
        pr = self.prim
        pr["n"] += 1
        a = np.zeros(7, dtype=np.float32)
        a[3:6] = act[3:6]                   # 손목 방향은 VLA 그대로
        tgt = np.array([pr["xy"][0], pr["xy"][1], pr["z"]])
        if pr["phase"] == "down":
            a[:3] = np.clip((tgt - hand) / 0.05, -0.3, 0.3)
            a[6] = -1.0
            if abs(hand[2] - pr["z"]) < 0.004 or pr["n"] > 40:
                pr["phase"], pr["n"] = "close", 0
        elif pr["phase"] == "close":
            a[6] = 1.0
            if pr["n"] >= 10:
                pr["phase"], pr["n"], pr["z_up"] = "lift", 0, hand[2] + 0.05
        else:
            a[:3] = np.clip((np.array([hand[0], hand[1], pr["z_up"]]) - hand) / 0.05, -0.3, 0.3)
            a[6] = 1.0
            if hand[2] > pr["z_up"] - 0.005 or pr["n"] > 20:
                self.prim = None
        return a

    def _place(self, act, instruction, c, hand):
        dst = self._dest(instruction)
        if self.held != c or dst is None or dst not in self.pos:
            return act                      # 지시문의 블록을 쥔 게 아니면 (일 바꾸기 때 내려놓기 등) 손대지 않는다
        e = self.pos[dst] - hand[:2]
        d = float(np.linalg.norm(e))
        act = np.array(act, dtype=np.float32, copy=True)
        ga = getattr(self, "grasp_at", None)
        if self.regrasp_ok and act[6] < 0 and ga is not None and np.linalg.norm(hand[:2] - ga[0][:2]) < 0.03 \
                and hand[2] < ga[0][2] + 0.05:
            self.stat["regrasp"] = self.stat.get("regrasp", 0) + 1     # ⑱ 쥔 자리에서 다시 쥐려고 벌리는 것은 둔다
            return act
        if self.pad_safe and d > self.release_err and hand[2] < TABLE_Z + GRASP_H[c] + 0.03:
            act[:2], act[2], act[6] = 0.0, max(float(act[2]), 0.3), 1.0     # ⑮ 책상 높이면 옆으로 끌지 않고 먼저 든다
            self.stat["lift"] = self.stat.get("lift", 0) + 1
            return act
        if d < self.place_radius:
            act[:2] = np.clip(e / 0.05, -self.vmax, self.vmax)        # ④ 놓을 곳 가운데로
            self.n_place += 1
            self.stat["place"] = self.stat.get("place", 0) + 1
        if d > self.release_err and act[6] < 0:
            act[6] = 1.0                                              # ⑤ 놓을 곳에서 멀면 벌리지 않는다
            self.n_hold_release += 1
        return act
