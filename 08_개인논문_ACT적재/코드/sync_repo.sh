#!/usr/bin/env bash
# Copy results (numbers, figures, small GIFs, code) into the LAB-1309 paper folder.
# Commit/push is done separately after review.
set -eu
cd "$(dirname "$0")"
R=/home/user/ACT/LAB-1309/08_개인논문_ACT적재
mkdir -p "$R/결과/수치" "$R/결과/그림" "$R/결과/gif" "$R/코드" "$R/논문"

# numbers: data generation summaries, training logs/meta, evaluation results
for f in data/*_gen.json; do
  [ -f "$f" ] && python3 -c "import json,sys; d=json.load(open('$f')); json.dump(d['summary'], open('$R/결과/수치/$(basename $f)','w'), indent=1, ensure_ascii=False)"
done
for d in runs/*/; do
  n=$(basename "$d")
  [ -f "$d/train_meta.json" ] && cp "$d/train_meta.json" "$R/결과/수치/train_${n}_meta.json"
  [ -f "$d/train_log.json" ] && cp "$d/train_log.json" "$R/결과/수치/train_${n}_log.json"
done
for d in results/eval/*/; do
  n=$(basename "$d")
  [ -f "$d/results.json" ] && cp "$d/results.json" "$R/결과/수치/eval_${n}.json"
  # every evaluation GIF, shrunk for the repo (originals stay on the lab PC)
  for g in $(ls "$d"/gifs/*.gif 2>/dev/null); do
    out="$R/결과/gif/${n}_$(basename "$g")"
    [ -f "$out" ] || .venv/bin/python - "$g" "$out" <<'EOF'
import sys
from PIL import Image, ImageSequence
src, dst = sys.argv[1], sys.argv[2]
im = Image.open(src)
fr = [f.convert("RGB").resize((300, 150)).quantize(colors=96) for i, f in enumerate(ImageSequence.Iterator(im)) if i % 2 == 0]
fr[0].save(dst, save_all=True, append_images=fr[1:], duration=200, loop=0, optimize=True)
EOF
  done
done

# dataset previews, analysis summary, logs
mkdir -p "$R/결과/시연데이터_미리보기" "$R/결과/로그"
cp results/preview/* "$R/결과/시연데이터_미리보기/" 2>/dev/null || true
[ -f results/summary.json ] && cp results/summary.json "$R/결과/수치/summary.json"
.venv/bin/python analyze.py > "$R/결과/수치/분석요약.md" 2>/dev/null || true
cp logs/queue.log "$R/결과/로그/" 2>/dev/null || true
for f in logs/train_*.log logs/eval_*.log logs/gen_*.log; do [ -f "$f" ] && tail -n 40 "$f" > "$R/결과/로그/$(basename "$f")"; done

# figures and paper
cp paper/fig*.png "$R/결과/그림/" 2>/dev/null || true
cp results/stack_*.png results/pallet_expert_final.png "$R/결과/그림/" 2>/dev/null || true
[ -f paper/out/paper.docx ] && cp paper/out/paper.docx "$R/논문/"
[ -f paper/out/paper.pdf ] && cp paper/out/paper.pdf "$R/논문/"

# code
cp stack_env.py expert.py gen_data.py train_act.py eval_act.py make_gif.py batch_expert.py run_queue3.sh sync_repo.sh analyze.py dataset_preview.py convert_jpeg.py run_queue4.sh publish.sh stack_photos.py "$R/코드/"
mkdir -p "$R/코드/paper" && cp paper/*.py "$R/코드/paper/"
echo "synced to $R"
