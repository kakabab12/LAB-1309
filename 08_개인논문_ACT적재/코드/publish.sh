#!/usr/bin/env bash
# After each evaluation: analysis -> figures -> paper (docx/pdf) -> LAB-1309 sync -> commit/push.
#   ./publish.sh "commit message"   (rollout figure uses the best finished model set, GPU, ~3 min)
set -u
cd "$(dirname "$0")"
MSG=${1:-"08_개인논문_ACT적재: 결과 갱신"}
PY=.venv/bin/python

$PY analyze.py > results/analysis.md 2>&1 || true
$PY paper/fig_scaling.py > /dev/null 2>&1 || true

# rollout figure from the best available full model set (prefer larger data)
for name in dart200 n1000 n500 n200 n100; do
  c=runs/s1_${name}/ckpt_030000
  if [ -f runs/s3_${name}/ckpt_030000/model.safetensors ] && [ -f results/eval/${name}/results.json ]; then
    if [ ! -f paper/fig4_rollout.json ] || ! grep -q "s1_${name}/" paper/fig4_rollout.json; then
      systemd-run --user --wait --collect --unit=act-rollout -p MemoryMax=2500M -p Nice=10 \
        -p WorkingDirectory=$PWD --setenv=MUJOCO_GL=egl --setenv=MALLOC_MMAP_THRESHOLD_=1048576 \
        $PWD/.venv/bin/python paper/fig_rollout.py --ckpt $c runs/s2_${name}/ckpt_030000 runs/s3_${name}/ckpt_030000 \
        > /dev/null 2>&1 || true
    fi
    break
  fi
done

$PY paper/build_paper.py --out paper/out/paper.docx > /dev/null 2>&1
(cd paper/out && timeout 180 soffice --headless --convert-to pdf paper.docx > /dev/null 2>&1)
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
