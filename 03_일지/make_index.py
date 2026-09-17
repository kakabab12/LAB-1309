#!/usr/bin/env python
"""03_일지/*.md 를 읽어 README.md 의 목록 표를 다시 만든다.

각 일지에서 읽는 것
  - 제목        : 첫 번째 `# ` 줄
  - 한 줄 요약  : `> **한 줄 요약**: ...` 줄
  - 핵심 수치   : `## 측정한 수치` 표에서 **굵게** 표시한 항목
  - 남은 할 일  : `## 다음에 할 것` 의 체크 안 된 항목 수

사용법:  python 03_일지/make_index.py
"""

import re
from pathlib import Path

HERE = Path(__file__).parent
HEAD = """# 일지

두 종류로 나눠 씁니다.

- **연구 일지** (`03_일지/YYYY-MM-DD.md`) — 전환 실험, 학습, 분석
- **공부 일지** (`03_일지/공부/YYYY-MM-DD.md`) — 코랩 실습, 공부 순서 진도

[TEMPLATE.md](TEMPLATE.md)를 복사해서 날짜 이름으로 저장하세요.
쓰고 나면 `python 03_일지/make_index.py` 로 아래 목록을 갱신합니다.

**왜 쓰나**
- 막혔던 것과 해결 방법이 쌓이면 **같은 문제로 두 번 고생하지 않습니다**
- 측정한 수치가 쌓이면 **나중에 그대로 논문 재료**가 됩니다
- 교수님이나 선배에게 진행 상황을 설명할 때 바로 씁니다
"""


def parse(path):
    t = path.read_text()
    title = next((l[2:].strip() for l in t.splitlines() if l.startswith("# ")), path.stem)
    title = re.sub(r"^\d{4}-\d{2}-\d{2}\s*[—-]\s*", "", title)
    m = re.search(r">\s*\*\*한 줄 요약\*\*:\s*(.+)", t)
    summary = m.group(1).strip() if m else ""
    nums, last_label = [], ""
    sec = re.search(r"## 측정한 수치\n(.*?)(?=\n## |\Z)", t, re.S)
    if sec:
        for row in sec.group(1).splitlines():
            cells = [c.strip() for c in row.strip().strip("|").split("|")] if row.startswith("|") else []
            if len(cells) < 2 or "---" in row or cells[0] in ("항목",):
                continue
            label = last_label if cells[0] in ("〃", "''", "") else cells[0]
            last_label = label
            if "**" in cells[1]:
                cond = f" ({cells[2]})" if len(cells) > 2 and cells[2] else ""
                nums.append(f"{label}: {cells[1]}{cond}")
    todo = len(re.findall(r"- \[ \]", re.search(r"## 다음에 할 것\n(.*?)(?=\n## |\Z)", t, re.S).group(1))) \
        if "## 다음에 할 것" in t else 0
    return {"date": path.stem, "file": path.name, "title": title, "summary": summary,
            "nums": nums[:3], "todo": todo}


def table(rows, base=""):
    out = []
    if not rows:
        return ["_아직 없습니다._"]
    span = f"{rows[-1]['date']} ~ {rows[0]['date']}"
    out.append(f"{len(rows)}개, {span}\n")
    out.append("| 날짜 | 제목 | 한 줄 요약 | 남은 할 일 |")
    out.append("|---|---|---|---|")
    for r in rows:
        title = r["title"].replace("[공부] ", "")
        out.append(f"| [{r['date']}]({base}{r['file']}) | {title} | {r['summary']} | {r['todo']}개 |")
    return out


def collect(folder):
    logs = sorted((p for p in folder.glob("*.md") if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.stem)), reverse=True)
    return [parse(p) for p in logs]


def main():
    research = collect(HERE)
    study = collect(HERE / "공부")
    out = [HEAD, "\n## 🔬 연구 일지 (최신순)\n"]
    out += table(research)
    out.append("\n## 📘 공부 일지 — 코랩 실습 (최신순)\n")
    out += table(study, base="공부/")
    out.append("\n## 핵심 수치 모음 (연구)\n")
    for r in research:
        if r["nums"]:
            out.append(f"- **{r['date']}** — " + " · ".join(r["nums"]))
    out.append("\n## 핵심 수치 모음 (공부)\n")
    for r in study:
        if r["nums"]:
            out.append(f"- **{r['date']}** — " + " · ".join(r["nums"]))
    (HERE / "README.md").write_text("\n".join(out) + "\n")
    print(f"연구 일지 {len(research)}개, 공부 일지 {len(study)}개로 README.md 갱신")


if __name__ == "__main__":
    main()
