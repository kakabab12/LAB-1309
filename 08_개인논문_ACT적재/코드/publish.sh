#!/usr/bin/env bash
# After each evaluation: analysis -> figures -> paper (docx/pdf) -> LAB-1309 sync -> commit/push.
#   ./publish.sh "commit message"   (rollout figure uses the best finished model set, GPU, ~3 min)
set -u
cd "$(dirname "$0")"
MSG=${1:-"08_개인논문_ACT적재: 결과 갱신"}
PY=.venv/bin/python

$PY analyze.py > results/analysis.md 2>&1 || true
$PY paper/fig_scaling.py > /dev/null 2>&1 || true

# rollout figure from the finished model set with the highest chained success (checkpoints from its results.json)
BEST=$($PY - <<'EOF2'
import json, pathlib
best = None
for p in pathlib.Path("results/eval").glob("*/results.json"):
    if "_ckpt" in p.parent.name or p.parent.name.endswith(("_p0", "_p1", "_p2", "_p3", "_retry")):
        continue
    r = json.loads(p.read_text())
    if "chained" not in r or len(r.get("ckpt", [])) != 3:
        continue
    c = r["chained"]["cumulative_success"][2]
    if best is None or c > best[0]:
        best = (c, " ".join(r["ckpt"]))
print(best[1] if best else "")
EOF2
)
if [ -n "$BEST" ] && { [ ! -f paper/fig4_rollout.json ] || ! grep -q "$(echo $BEST | cut -d' ' -f3)" paper/fig4_rollout.json; }; then
  systemd-run --user --wait --collect --unit=act-rollout -p MemoryMax=2500M -p Nice=10 \
    -p WorkingDirectory=$PWD --setenv=MUJOCO_GL=egl --setenv=MALLOC_MMAP_THRESHOLD_=1048576 \
    $PWD/.venv/bin/python paper/fig_rollout.py --ckpt $BEST > /dev/null 2>&1 || true
fi

$PY paper/build_paper.py --out paper/out/paper.docx > /dev/null 2>&1
(cd paper/out && timeout 180 soffice --headless --convert-to pdf paper.docx > /dev/null 2>&1)
$PY paper/build_paper.py --content content_v2 --out paper/out/paper_v2.docx > /dev/null 2>&1
(cd paper/out && timeout 180 soffice --headless --convert-to pdf paper_v2.docx > /dev/null 2>&1)
./sync_repo.sh > /dev/null 2>&1
cp results/analysis.md "/home/user/ACT/LAB-1309/08_개인논문_ACT적재/결과/수치/분석요약.md" 2>/dev/null || true

cd /home/user/ACT/LAB-1309
git add 08_개인논문_ACT적재
if ! git diff --cached --quiet; then
  git commit -q -m "$MSG

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  git pull --rebase -q && git push -q origin main
fi
git log --oneline -1
