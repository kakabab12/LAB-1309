#!/usr/bin/env python
"""03_일지/*.md 를 읽어 README.md 의 목록 표를 다시 만든다.

각 일지에서 읽는 것
  - 제목        : 첫 번째 `# ` 줄
  - 한 줄 요약  : `> 한 줄 요약: ...` 줄 (예전 형식 `> **한 줄 요약**:` 도 읽는다)
  - 핵심 수치   : `## 측정한 수치` 표의 앞 세 줄 (9월 ~ 10/1 일지에만 있다)

연구 일지는 달마다 표 하나로 나눠 싣는다.

사용법:  python 03_일지/make_index.py
"""

import re
from itertools import groupby
from pathlib import Path

HERE = Path(__file__).parent
HEAD = """# 일지

두 가지로 나눠 씁니다.

- 연구 일지 (`03_일지/YYYY-MM-DD.md`): 전환 실험, 학습, 분석
- 공부 일지 (`03_일지/공부/YYYY-MM-DD.md`): 코랩 실습, 공부 순서 진도

[TEMPLATE.md](TEMPLATE.md)를 복사해 날짜 이름으로 저장하고, 다 쓰면 `python 03_일지/make_index.py` 로 아래 목록을 갱신합니다.

일지를 쓰는 이유는 세 가지입니다. 막혔던 것과 해결 방법이 남아 있으면 같은 문제로 두 번 고생하지 않고,
측정한 수치가 쌓이면 그대로 논문 재료가 되고, 교수님이나 선배에게 진행 상황을 설명할 때 바로 쓸 수 있습니다.

## 시기별로 보기

| 시기 | 실험 무대 | 무엇을 했나 |
|---|---|---|
| 9/14 ~ 10/3 | LIBERO 시뮬레이터, Franka Panda 팔 | 일을 바꾸면 왜 실패하는지 찾기, 시범 프로그램, 실제 로봇과 같은 지연 넣기 |
| 10/4 ~ 10/24 | PiPER 디지털 트윈, 색 블록 4개 · 색 판 4개 | 28개 항목 (혼자 10, 일 바꾸기 12, 돌아오기 6), VLA + 보조, 옮기는 도중 끼어들기 |
| 10/24 ~ | PiPER 디지털 트윈, 모양 5가지 · 물체 최대 14개 · 로봇 둘레 | 설정 파일로 만드는 장면, 물체 찾기, DAgger, 지시문을 읽지 않는 다듬기 |
"""


def parse(path):
    t = path.read_text()
    title = next((l[2:].strip() for l in t.splitlines() if l.startswith("# ")), path.stem)
    title = re.sub(r"^\d{4}-\d{2}-\d{2}\s*(?:[—-]\s*)?", "", title)
    # 요약이 여러 줄에 걸쳐 있으면(인용문이 이어지면) 이어 붙인다
    m = re.search(r">\s*(?:\*\*)?한 줄 요약(?:\*\*)?\s*:\s*(.+(?:\n>.*)*)", t)
    summary = re.sub(r"\s*\n>\s*", " ", m.group(1)).strip() if m else ""
    nums, last_label = [], ""
    sec = re.search(r"## 측정한 수치\n(.*?)(?=\n## |\Z)", t, re.S)
    if sec:
        for row in sec.group(1).splitlines():
            cells = [c.strip() for c in row.strip().strip("|").split("|")] if row.startswith("|") else []
            if len(cells) < 2 or "---" in row or cells[0] in ("항목",):
                continue
            label = last_label if cells[0] in ("〃", "''", "") else cells[0]
            last_label = label
            cond = f" ({cells[2]})" if len(cells) > 2 and cells[2] else ""
            nums.append(f"{label}: {cells[1]}{cond}".replace("**", ""))
    return {"date": path.stem, "file": path.name, "title": title, "summary": summary, "nums": nums[:3]}


def table(rows, base=""):
    out = ["| 날짜 | 제목 | 한 줄 요약 |", "|---|---|---|"]
    for r in rows:
        title = r["title"].replace("[공부] ", "")
        out.append(f"| [{r['date'][5:].replace('-', '/')}]({base}{r['file']}) | {title} | {r['summary']} |")
    return out


def collect(folder):
    logs = sorted((p for p in folder.glob("*.md") if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.stem)), reverse=True)
    return [parse(p) for p in logs]


def main():
    research = collect(HERE)
    study = collect(HERE / "공부")
    out = [HEAD, f"\n## 연구 일지 (최신순, {len(research)}개)"]
    for month, rows in groupby(research, key=lambda r: r["date"][:7]):
        rows = list(rows)
        y, m = month.split("-")
        out.append(f"\n### {y}년 {int(m)}월 ({len(rows)}개)\n")
        out += table(rows)
    out.append(f"\n## 공부 일지 (코랩 실습, 최신순, {len(study)}개)\n")
    out += table(study, base="공부/")
    out.append("\n## 핵심 수치 모음\n")
    out.append("9월 ~ 10/1 일지의 `## 측정한 수치` 표에서 앞 세 줄씩 모았습니다.\n")
    for name, rows in (("연구", research), ("공부", study)):
        rows = [r for r in rows if r["nums"]]
        if not rows:
            continue
        out.append(f"<details>\n<summary>{name} 일지 ({len(rows)}개)</summary>\n")
        for r in rows:
            out.append(f"- {r['date']}")
            out += [f"  - {n}" for n in r["nums"]]
        out.append("\n</details>\n")
    (HERE / "README.md").write_text("\n".join(out) + "\n")
    print(f"연구 일지 {len(research)}개, 공부 일지 {len(study)}개로 README.md 갱신")


if __name__ == "__main__":
    main()
