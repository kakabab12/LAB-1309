"""여러 모양·N개 장면용 한국어 명령 해석기 (2026-10-24) — 색·모양·판 낱말을 설정 파일(piper_sim/worlds/*.json)에서 읽는다

  korean_cmd.py (블록 4개·판 4개 낱말이 박힌 것) 와 같은 규칙(끼어들기·돌아오기·되묻기·멈춤·가리키기)을 쓰되,
  물체 = 색 낱말(+ 모양 낱말), 판 = 판 색 낱말 + 판 낱말, 쌓기 = 물체 + '위'. 장면(scene)이 있으면 그 장면의 실제 물체로 바꾸고,
  모양만 말해도("원기둥") 그 모양이 장면에 하나면 알아듣는다. 둘 이상이면 되묻는다.
  지시문은 world.instruction (설정의 영어 틀) — VLA 가 배운 꼴 그대로.

  python world_cmd.py "잠깐, 하늘색 원기둥 먼저 남색 판에 놓고 하던 거 이어서 해줘"
"""
import re
import sys

import korean_cmd as kc
import piper_sim.world as W

STACK_TAIL = r"\s*(?:색)?\s*(?:블록|블럭|것|거)?\s*(?:위|위에|위로|꼭대기)"


def vocab():
    C = W.CFG
    obj_col = {c: v["ko"] for c, v in C["object_colors"].items()}
    pad_col = {c: v["ko"] for c, v in C["pad_colors"].items()}
    shapes = {s: v["ko"] for s, v in C["shapes"].items()}
    pad_noun = r"(?:색)?\s*(?:" + "|".join(sorted(map(re.escape, C["pad"]["ko"]), key=len, reverse=True)) + r")"
    return obj_col, pad_col, shapes, pad_noun


def _mentions(s):
    """[(시작, 종류(pad|stack|obj), 색 또는 None, 모양 또는 None, 끝)]."""
    obj_col, pad_col, shapes, pad_noun = vocab()
    out = []
    for col, ws in pad_col.items():
        for w in ws:
            for m in re.finditer(re.escape(w) + pad_noun, s):
                out.append((m.start(), "pad", col, None, m.end()))
    shape_re = [(sh, w) for sh, ws in shapes.items() for w in sorted(ws, key=len, reverse=True)]
    for col, ws in obj_col.items():
        for w in ws:
            for m in re.finditer(re.escape(w), s):
                end, shape = m.end(), None
                tail = s[end:end + 10]
                for sh, sw in shape_re:                      # 색 바로 뒤 모양 낱말 ("빨간 원기둥")
                    mm = re.match(r"\s*(?:색\s*)?" + re.escape(sw), tail)
                    if mm:
                        shape, end = sh, end + mm.end()
                        break
                kind = "stack" if re.match(STACK_TAIL, s[end:end + 12]) else "obj"
                out.append((m.start(), kind, col, shape, end))
    for sh, sw in shape_re:                                  # 색 없이 모양만 ("원기둥을")
        for m in re.finditer(re.escape(sw), s):
            kind = "stack" if re.match(STACK_TAIL, s[m.end():m.end() + 12]) else "obj"
            out.append((m.start(), kind, None, sh, m.end()))
    out.sort(key=lambda x: (x[0], -(x[4] - x[0])))
    keep, last_end = [], -1
    for m in out:                                            # 겹치면 먼저·길게 맞은 것
        if m[0] < last_end:
            continue
        keep.append(m)
        last_end = m[4]
    return keep


def _resolve(col, shape, scene, kind):
    """(색, 모양) → 장면의 인스턴스 이름. 장면이 없으면 설정에서 만든 이름 (모양 모르면 None)."""
    if kind == "pad":
        name = W.inst(W.pad_cat(col))
        return (name, None) if scene is None or name in scene.pads else (None, f"장면에 {_ko_pad(col)}이 없어요.")
    if scene is None:
        if col and shape:
            return W.inst(W.obj_cat(shape, col)), None
        return None, "어떤 물체인지 색과 모양으로 말해 주세요."
    cands = [o for o in scene.objects if (col is None or W.parse(o)[2] == col) and (shape is None or W.parse(o)[1] == shape)]
    if len(cands) == 1:
        return cands[0], None
    if not cands:
        return None, "그런 물체가 장면에 없어요."
    return None, "어느 것인지 색까지 말해 주세요: " + ", ".join(ko(o) for o in cands)


def ko(name):
    kind, shape, col = W.parse(name)
    if kind == "pad":
        return _ko_pad(col)
    return f"{W.CFG['object_colors'][col]['ko'][0]} {W.CFG['shapes'][shape]['ko'][0]}"


def _ko_pad(col):
    return f"{W.CFG['pad_colors'][col]['ko'][0]} {W.CFG['pad']['ko'][0]}"


def parse_clause(s, scene=None):
    """조각 → ('do', (물체, 놓을 곳, 지시문)) / ('ask', 물음) / None."""
    ms = _mentions(s)
    objs = [m for m in ms if m[1] == "obj"]
    dests = [m for m in ms if m[1] in ("pad", "stack")]
    if not objs and not dests:
        return None
    if not objs:
        return ("ask", "무엇을 옮길까요?")
    o, err = _resolve(objs[0][2], objs[0][3], scene, "obj")
    if err:
        return ("ask", err)
    if not dests and len(objs) >= 2:
        dests = [(objs[1][0], "stack", objs[1][2], objs[1][3], objs[1][4])]
    if not dests:
        return ("ask", f"{ko(o)}{kc.josa(ko(o), '을', '를')} 어디에 놓을까요?")
    d = dests[0]
    t, err = _resolve(d[2], d[3], scene, "pad" if d[1] == "pad" else "obj")
    if err:
        return ("ask", err)
    if t == o:
        return ("ask", "같은 물체 위에는 놓을 수 없어요.")
    if d[1] == "stack" and not W.CFG["shapes"][W.parse(t)[1]].get("stack_on"):
        return ("ask", f"{ko(t)} 위에는 쌓을 수 없어요. 어디에 놓을까요?")
    return ("do", (o, t, W.instruction(o, t)))


def interpret(text, scene=None):
    """korean_cmd.interpret 와 같은 꼴의 명령 목록. "instr" 는 영어 지시문, "task" 는 (물체, 놓을 곳)."""
    s = text.strip()
    if not s:
        return []
    if kc._has(s, kc.STOP) and not _mentions(s) and not kc._has(s, kc.RESUME):
        return [{"type": "stop", "say": "멈춥니다."}]
    if kc._has(s, kc.RESUME_ONLY) and not _mentions(s):
        return [{"type": "resume", "say": "하던 일로 돌아갑니다."}]
    interrupt, resume = kc._has(s, kc.INTERRUPT), kc._has(s, kc.RESUME)
    cancel = kc._has(s, kc.CANCEL) and not interrupt
    tasks = []
    for c in kc.split_clauses(s):
        r = parse_clause(c, scene)
        if r is None:
            continue
        if r[0] == "ask":
            return [{"type": "ask", "msg": r[1], "say": r[1]}]
        if r[1] not in tasks:
            tasks.append(r[1])
    if not tasks:
        return [{"type": "ask", "msg": "무엇을 어디에 놓을까요?", "say": "무엇을 어디에 놓을까요?"}]
    out = []
    for i, (o, t, instr) in enumerate(tasks):
        typ = "interrupt" if (interrupt or resume) and not cancel else "do"
        if i > 0:
            typ = "then"
        out.append({"type": typ, "instr": instr, "task": (o, t), "say": f"{ko(o)}{kc.josa(ko(o), '을', '를')} {ko(t)}{'' if W.parse(t)[0] == 'pad' else ' 위'}에 놓습니다."})
    if resume and not interrupt and not cancel and len(out) == 1:
        out[0]["type"] = "interrupt"
    return out


class Controller(kc.Controller):
    """korean_cmd.Controller 와 같은 쌓기(끼어들기)·줄(차례로) 규칙. 장면을 알고 있다."""

    def __init__(self, scene=None):
        super().__init__()
        self.scene = scene

    def command(self, text, t=0):
        cmds = interpret(text, self.scene)
        self.log.append((t, text, cmds))
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


if __name__ == "__main__":
    sc = W.sample_scene(int(sys.argv[2]) if len(sys.argv) > 2 else 5, n_objects=8, n_pads=4)
    print("장면:", [ko(o) for o in sc.objects], "/", [ko(p) for p in sc.pads])
    texts = [sys.argv[1]] if len(sys.argv) > 1 and sys.argv[1] else [
        f"{ko(sc.objects[0])}{kc.josa(ko(sc.objects[0]), '을', '를')} {ko(sc.pads[0])}에 올려줘",
        f"잠깐, {ko(sc.objects[1])} 먼저 {ko(sc.pads[1])}에 놓고 하던 거 이어서 해줘",
        f"{W.CFG['shapes'][W.parse(sc.objects[2])[1]]['ko'][0]}{kc.josa(W.CFG['shapes'][W.parse(sc.objects[2])[1]]['ko'][0], '을', '를')} {ko(sc.pads[0])}에 놔",
        f"{ko(sc.objects[3])}{kc.josa(ko(sc.objects[3]), '을', '를')} {ko(sc.objects[4])} 위에 쌓아줘",
        "이어서 해줘", "멈춰"]
    for t in texts:
        print(f"{t}\n   → {interpret(t, sc)}")
