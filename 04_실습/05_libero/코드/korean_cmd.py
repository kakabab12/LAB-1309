"""한국어 명령 해석기 (2026-10-22) — 사람이 친 한국어 문장을 VLA 가 아는 영어 지시문과 '끼어들기·돌아오기' 순서로 바꾼다

왜
  VLA 는 "put the red block on the purple square" 같은 영어 지시문 한 가지 꼴만 배웠다. 사람은 한국어로,
  "잠깐, 초록 블록 먼저 흰 판에 놓고 하던 거 이어서 해줘" 처럼 끼어들기와 돌아오기를 한 문장에 섞어 말한다.
  사용자 (10/22): "90% 넘으면 해석기 시작하자" → VLA + 보조 28항목 93.8% 에서 시작.

어떻게 (규칙, 인터넷·LLM 없이 실물 PC 에서 돈다)
  ① 문장을 조각으로 나눈다: "~하고 / ~한 다음 / ~하고 나서 / 그리고 / , "
  ② 조각마다 색 낱말을 찾는다. 뒤에 '판·사각형·네모·자리·패드' 가 오면 놓을 판, '블록(막대) 위' 면 쌓을 블록,
     나머지 첫 블록 색이 옮길 블록. "노란 막대"·"막대" 는 노랑.
  ③ 끼어들기 낱말(잠깐·먼저·그 전에·우선) 이 있으면 지금 하던 일을 쌓아 두고 새 일을 하고, 돌아오기 낱말(이어서·하던 거·
     원래·다시·계속) 이 있으면 끝난 뒤 쌓아 둔 일로 돌아간다. 끼어들기 낱말만 있어도 돌아온다 ("먼저" = 그다음 하던 일).
  ④ 놓을 곳이 없으면 되묻는다. 배운 10가지 지시문이 아니면 알려 주고 그래도 한다 (정확도는 낮을 수 있음).
  ⑤ 가리키기 (10/22): "저거·이거·그거" 는 화면에서 마지막으로 누른 블록, "저기·여기·거기" 는 그다음 누른 판(또는 블록 위)으로
     바꿔 읽는다 (korean_live --show 창에서 정면 카메라 화면을 누름 → 그 자리의 색 덩어리). "그거 말고" 는 가리키기가 아니다.

  python korean_cmd.py            # 예시 문장 시험
  python korean_cmd.py "빨간 블록을 보라 판에 올려줘"
"""
import re
import sys

BLOCK_WORDS = {
    "red": ["빨간", "빨강", "붉은", "레드"],
    "green": ["초록", "녹색", "연두", "그린"],
    "blue": ["파란", "파랑", "푸른", "블루"],
    "yellow": ["노란", "노랑", "옐로", "막대"],
}
PAD_WORDS = {
    "purple": ["보라", "퍼플"],
    "gray": ["회색", "그레이", "잿빛"],
    "orange": ["주황", "오렌지"],
    "white": ["흰", "하얀", "흰색", "화이트", "하양"],
}
PAD_NOUN = r"(?:색)?\s*(?:판|사각형|네모|자리|패드|칸|매트)"
STACK_NOUN = r"(?:색)?\s*(?:블록|블럭|막대|상자|거|것)?\s*(?:위|위에|위로|꼭대기)"
INTERRUPT = ["잠깐", "먼저", "그 전에", "그전에", "우선", "일단", "그거 말고 잠깐"]
RESUME = ["이어서", "하던 거", "하던거", "하던 일", "원래", "계속", "다시 하던", "마저"]
CANCEL = ["그거 말고", "그만하고", "대신", "취소하고", "바꿔서"]
STOP = ["멈춰", "멈춰줘", "그만", "정지", "스톱"]
RESUME_ONLY = ["이어서 해", "하던 거 해", "하던거 해", "계속 해", "계속해", "다시 해", "마저 해"]

# VLA 가 배운 10가지 (piper_sim.blocks.TASKS)
KNOWN = {
    "put the blue block on the white square", "put the red block on the orange square",
    "put the yellow block on the purple square", "put the red block on the blue block",
    "put the red block on the gray square", "put the green block on the white square",
    "put the green block on the red block", "put the yellow block on the white square",
    "put the red block on the purple square", "put the green block on the blue block",
}
POINT_OBJ = ["저거", "이거", "그거", "저것", "이것", "그것", "저 블록", "이 블록", "그 블록", "저 막대", "이 막대"]
POINT_DST = ["저기", "여기", "거기", "저 판", "이 판", "그 판", "저쪽", "이쪽", "저 위", "그 위"]
SAY_COLOR = {"red": "빨간", "green": "초록", "blue": "파란", "yellow": "노란",
             "purple": "보라", "gray": "회색", "orange": "주황", "white": "흰"}
KO = {"red": "빨강", "green": "초록", "blue": "파랑", "yellow": "노란 막대",
      "purple": "보라 판", "gray": "회색 판", "orange": "주황 판", "white": "흰 판"}


def josa(word, a, b):
    """받침 있으면 a (을·이·은), 없으면 b (를·가·는)."""
    ch = word.strip()[-1]
    return a if "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 else b


def _has(s, words):
    return any(w in s for w in words)


def _mentions(s):
    """(위치, 종류, 색) 목록. 종류: pad / stack / block."""
    out = []
    for col, ws in PAD_WORDS.items():
        for w in ws:
            for m in re.finditer(re.escape(w) + PAD_NOUN, s):
                out.append((m.start(), "pad", col, m.end()))
    for col, ws in BLOCK_WORDS.items():
        for w in ws:
            for m in re.finditer(re.escape(w), s):
                tail = s[m.end():m.end() + 12]
                if re.match(r"\s*" + STACK_NOUN, tail) and w != "막대":
                    out.append((m.start(), "stack", col, m.end()))
                else:
                    out.append((m.start(), "block", col, m.end()))
    out.sort()
    # 같은 자리에서 겹친 것(판 낱말 '흰' 과 블록 낱말) 은 판·쌓기를 우선
    keep, last_end = [], -1
    for m in out:
        if m[0] < last_end:
            continue
        keep.append(m)
        last_end = m[3]
    # 판 낱말은 블록 낱말과 겹치지 않으니 따로 보지만, '흰 판' 의 '흰' 이 블록 색이 아니므로 그대로
    return keep


def parse_clause(s):
    """조각 하나 → ('do', 지시문) / ('ask', 물음) / None."""
    ms = _mentions(s)
    blocks = [m for m in ms if m[1] == "block"]
    dests = [m for m in ms if m[1] in ("pad", "stack")]
    if not blocks and not dests:
        return None
    if not blocks:
        return ("ask", "어떤 블록을 옮길까요? (빨강·초록·파랑·노란 막대)")
    obj = blocks[0][2]
    if not dests:
        # "빨간 블록을 파란 블록에" 처럼 '위' 없이 블록 둘이면 두 번째를 쌓을 곳으로
        if len(blocks) >= 2 and blocks[1][2] != obj:
            dests = [(blocks[1][0], "stack", blocks[1][2], blocks[1][3])]
        else:
            return ("ask", f"{KO[obj]}{josa(KO[obj], '을', '를')} 어디에 놓을까요? (보라·회색·주황·흰 판, 또는 다른 블록 위)")
    kind, col = dests[0][1], dests[0][2]
    if kind == "stack" and col == obj:
        return ("ask", "같은 블록 위에는 놓을 수 없어요. 어디에 놓을까요?")
    if kind == "stack" and col == "yellow":
        return ("ask", "노란 막대 위에는 쌓을 수 없어요. 어디에 놓을까요?")
    noun = "square" if kind == "pad" else "block"
    return ("do", f"put the {obj} block on the {col} {noun}")


def split_clauses(s):
    parts = re.split(r"\s*(?:,|\.|그리고|하고 나서|하고나서|한 다음에?|한 뒤에?|한 후에?|놓고|올리고|올려놓고|쌓고|두고|하고)\s*", s)
    return [p for p in parts if p and p.strip()]


def resolve_pointing(text, point):
    """'저거'·'저기' 를 화면에서 누른 블록·판의 색 낱말로 바꾼다. point = {"obj": 블록 색, "dst": 판 또는 블록 색} (없으면 None).
    가리킨 게 없는데 '저거' 를 쓰면 None (되물음)."""
    s = text
    guard = s.replace("그거 말고", "\0")          # '그거 말고' 는 가리키기가 아니다
    if _has(guard, POINT_OBJ):
        obj = (point or {}).get("obj")
        if obj is None:
            return None, "어느 블록인지 화면에서 눌러 주세요."
        noun = "막대" if obj == "yellow" else "블록"
        for w in sorted(POINT_OBJ, key=len, reverse=True):
            guard = guard.replace(w, f"{SAY_COLOR[obj]} {noun}")
    if _has(guard, POINT_DST):
        dst = (point or {}).get("dst")
        if dst is None:
            return None, "어디에 놓을지 화면에서 판(또는 블록)을 눌러 주세요."
        rep = f"{SAY_COLOR[dst]} 판" if dst in PAD_WORDS else f"{SAY_COLOR[dst]} 블록 위"
        for w in sorted(POINT_DST, key=len, reverse=True):
            guard = guard.replace(w, rep)
    return guard.replace("\0", "그거 말고"), None


def interpret(text, point=None):
    """한국어 한 문장 → 명령 목록. 각 명령은 dict:
       {"type": "do", "instr": 영어}            지금 일을 버리고 이것을 한다
       {"type": "interrupt", "instr": 영어}     지금 일을 쌓아 두고 이것을 한 뒤 돌아온다
       {"type": "resume"}                       쌓아 둔 일로 돌아간다
       {"type": "stop"} / {"type": "ask", "msg": 물음}
       각 명령에 "known" (배운 10가지인가) 와 "say" (한국어로 확인할 말) 를 붙인다."""
    s = text.strip()
    if not s:
        return []
    s, ask = resolve_pointing(s, point)
    if ask:
        return [{"type": "ask", "msg": ask, "say": ask}]
    if _has(s, STOP) and not _mentions(s) and not _has(s, RESUME):
        return [{"type": "stop", "say": "멈춥니다."}]
    if _has(s, RESUME_ONLY) and not _mentions(s):
        return [{"type": "resume", "say": "하던 일로 돌아갑니다."}]
    interrupt = _has(s, INTERRUPT)
    resume = _has(s, RESUME)
    cancel = _has(s, CANCEL) and not interrupt
    tasks = []
    for c in split_clauses(s):
        r = parse_clause(c)
        if r is None:
            continue
        if r[0] == "ask":
            return [{"type": "ask", "msg": r[1], "say": r[1]}]
        if r[1] not in tasks:
            tasks.append(r[1])
    if not tasks:
        return [{"type": "ask", "msg": "무엇을 어디에 놓을까요? 예: 빨간 블록을 보라 판에 올려줘", "say": "무엇을 어디에 놓을까요?"}]
    out = []
    for i, t in enumerate(tasks):
        typ = "interrupt" if (interrupt or resume) and not cancel else "do"
        if i > 0:
            typ = "then"                       # 앞 일 다음에 이어서 할 일
        out.append({"type": typ, "instr": t, "known": t in KNOWN, "say": _say(t)})
    if resume and not interrupt and not cancel and len(out) == 1:
        out[0]["type"] = "interrupt"           # "초록 블록 흰 판에 놓고 하던 거 이어서" = 끼어들기
    return out


def _say(instr):
    m = re.match(r"put the (\w+) block on the (\w+) (square|block)", instr)
    obj, col, kind = m.groups()
    dst = KO[col] if kind == "square" else f"{KO[col]} 블록 위"
    return f"{KO[obj]}{josa(KO[obj], '을', '를')} {dst}에 놓습니다."


class Controller:
    """명령 목록을 받아 지금 VLA 에 줄 지시문을 정한다. 실물·시뮬 같은 쓰임.
       interrupt: 지금 일을 쌓아 두고 새 일 → 끝나면 쌓아 둔 일로 (연구 시나리오: A 쥔 직후 B → 다시 A)."""

    def __init__(self):
        self.current, self.queue, self.stack, self.log = None, [], [], []
        self.point = {"obj": None, "dst": None}

    def click(self, color):
        """화면에서 누른 것: 블록이면 '저거', 판이면 '저기'. 블록을 이미 골랐으면 두 번째 블록은 '저기'(그 위)."""
        if color in PAD_WORDS or (self.point["obj"] is not None and color != self.point["obj"]):
            self.point["dst"] = color
        else:
            self.point = {"obj": color, "dst": None}
        return dict(self.point)

    def command(self, text, t=0):
        cmds = interpret(text, self.point)
        self.log.append((t, text, cmds))
        if not any(c["type"] == "ask" for c in cmds):
            self.point = {"obj": None, "dst": None}        # 누른 것은 다음 명령 하나에만 쓴다
        for c in cmds:
            if c["type"] == "stop":
                self.current, self.queue = None, []
            elif c["type"] == "resume":
                if self.stack:
                    self.current = self.stack.pop()
            elif c["type"] == "interrupt":
                if self.current is not None:
                    self.stack.append(self.current)
                self.current, self.queue = c["instr"], []
            elif c["type"] == "do":
                self.current, self.queue, self.stack = c["instr"], [], []
            elif c["type"] == "then":
                self.queue.append(c["instr"])
        return cmds

    def done(self):
        """지금 일이 끝났을 때 부른다 → 다음 일 (이어서 할 일 → 쌓아 둔 일 → 없음)."""
        if self.queue:
            self.current = self.queue.pop(0)
        elif self.stack:
            self.current = self.stack.pop()
        else:
            self.current = None
        return self.current


EXAMPLES = [
    "빨간 블록을 보라 판에 올려줘",
    "파란 거 흰 판에 놔",
    "초록 블록을 파란 블록 위에 쌓아줘",
    "노란 막대를 흰색 판으로 옮겨줘",
    "잠깐, 초록 블록 먼저 흰 판에 놓고 하던 거 이어서 해줘",
    "그 전에 노란 막대를 흰 판에 놓아줘",
    "초록 블록 흰 판에 놓고 이어서 해",
    "그거 말고 빨간 블록을 회색 판에 놔",
    "빨간 블록 집어줘",
    "이어서 해줘",
    "멈춰",
    "파란 블록을 보라 판에 올려줘",
    "빨간 블록을 주황 판에 놓고 초록 블록을 빨간 블록 위에 쌓아줘",
    "빨간색 블록 보라색 판 위에 올려놔",
    "빨간 블록을 파란 블록에 올려줘",
    "노란 막대 집어서 보라 판에",
    "흰 판에 초록 블록 놔줘",
]

POINT_EXAMPLES = [  # (누른 것, 문장)
    ({"obj": "green", "dst": "white"}, "저거 저기에 놔줘"),
    ({"obj": "yellow", "dst": None}, "이거 보라 판에 올려"),
    ({"obj": "red", "dst": "blue"}, "잠깐, 그거 먼저 저 위에 쌓고 이어서 해"),
    ({"obj": None, "dst": None}, "저거 흰 판에 놔"),
    ({"obj": "blue", "dst": None}, "그거 말고 빨간 블록을 회색 판에 놔"),
]

if __name__ == "__main__":
    for point, text in ([(None, t) for t in sys.argv[1:]] or [(None, t) for t in EXAMPLES] + POINT_EXAMPLES):
        cmds = interpret(text, point)
        if point:
            print(f"(누름: {point})", end=" ")
        print(f"「{text}」")
        for c in cmds:
            extra = "" if c.get("known", True) else "  (배운 적 없는 조합 — 정확도 낮을 수 있음)"
            print(f"   {c['type']:9s} {c.get('instr', c.get('msg', ''))}{extra}")
