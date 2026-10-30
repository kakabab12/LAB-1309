# 학회 포스터 (2026 대한전자공학회 추계학술대회, 포스터 세션)

| 파일 | 용도 |
|---|---|
| [포스터_ACT적재_90x120cm.pptx](포스터_ACT적재_90x120cm.pptx) | 고칠 때 (PowerPoint, 슬라이드 1장 = 가로 90 cm × 세로 120 cm, 글꼴 맑은 고딕) |
| [포스터_ACT적재_90x120cm.pdf](포스터_ACT적재_90x120cm.pdf) | 인쇄소에 보낼 파일 (90 × 120 cm, 글꼴 포함) |
| [포스터_미리보기.png](포스터_미리보기.png) | 한눈에 보기 |

## 학회 규정 (학회 홈페이지 "논문발표 안내", 2026-10-31 확인)

학회가 주는 포스터 **파일 양식은 없음** (2023~2026 학술대회 페이지, 학회 자료실·논문양식 게시판 모두 확인). 정해진 것은 아래 규정과 패널 견본 그림뿐.

| 항목 | 규정 | 이 포스터 |
|---|---|---|
| 패널 | 가로 100 cm × 세로 180 cm | 90 × 120 cm 한 장 (전지 1장 방식) |
| 제목·저자·소속 칸 | 가로 90 cm × 세로 20 cm | 맨 위 90 × 20 cm 남색 띠 |
| 제목·저자 글자 | 25~30 mm | 제목 84 pt (29.6 mm), 저자 72 pt (25.4 mm) |
| 각 장 제목 | 15~20 mm | 50 pt (17.6 mm) |
| 발표 내용 | 7~10 mm | 본문 22~24 pt (7.8~8.5 mm), 주석·참고문헌 20 pt (7.1 mm) |
| 부착 | 발표 5분 전 직접 부착, 끝나고 5분 안에 제거, 저자 1명은 포스터 앞에서 답변 | |
| 논문번호 | 학회가 패널 위에 붙임 | |

지원 프로그램(RISE) 규정에 따라 포스터 오른쪽 아래에 사사 문구를 넣음.

## 다시 만들기 (연구실 PC)

```bash
cd ~/ACT/act_sim
systemd-run --user --unit=act-poster-render -p MemoryMax=4G paper/poster/run_renders.sh   # 시뮬레이션 장면 1600×1200 렌더
.venv/bin/python paper/poster/fig_poster.py      # 손목 각도 그림, 규칙 개념도 (인쇄 크기)
python3 paper/poster/prep_images.py               # 배경 정리·자르기
PYTHONPATH=<python-pptx> python3 paper/poster/make_poster.py ~/ACT/포스터/포스터_ACT적재_90x120cm.pptx
```
