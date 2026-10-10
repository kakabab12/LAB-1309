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

# robustness results (per model tag) and their GIFs; retry evaluations
for d in results/robust/*/; do
  [ -d "$d" ] || continue
  t=$(basename "$d"); mkdir -p "$R/결과/수치/robust/$t"
  cp "$d"*.json "$R/결과/수치/robust/$t/" 2>/dev/null || true
  for g in $(ls "$d"gifs/*.gif 2>/dev/null); do
    out="$R/결과/gif/robust_${t}_$(basename "$g")"
    [ -f "$out" ] || .venv/bin/python -c "
import sys
from PIL import Image, ImageSequence
im = Image.open(sys.argv[1])
fr = [f.convert('RGB').resize((300, 150)).quantize(colors=96) for i, f in enumerate(ImageSequence.Iterator(im)) if i % 2 == 0]
fr[0].save(sys.argv[2], save_all=True, append_images=fr[1:], duration=200, loop=0, optimize=True)" "$g" "$out"
  done
done
[ -f results/robust_summary.json ] && cp results/robust_summary.json "$R/결과/수치/"
[ -f results/verify_il.json ] && cp results/verify_il.json "$R/결과/수치/"
[ -f results/demo_audit.json ] && cp results/demo_audit.json results/demo_audit.png "$R/결과/수치/" && cp results/demo_audit.png "$R/결과/그림/"

# large showcase GIFs (make_showcase.py), copied as they are
mkdir -p "$R/결과/gif/showcase"
cp results/showcase/*.gif "$R/결과/gif/showcase/" 2>/dev/null || true
cp results/pallet_*.gif "$R/결과/gif/" 2>/dev/null || true

# dataset previews, analysis summary, logs
mkdir -p "$R/결과/시연데이터_미리보기" "$R/결과/로그"
cp results/preview/* "$R/결과/시연데이터_미리보기/" 2>/dev/null || true
[ -f results/summary.json ] && cp results/summary.json "$R/결과/수치/summary.json"
for f in results/grasp_tolerance*.json results/grasp_offsets.json results/wall_rule_*.json results/wallswitch_*.json results/grip_check_*.json results/expert_eval_*.json results/stage3_inspect.json results/wide_expert_*.json results/pallet/*.json; do [ -f "$f" ] && cp "$f" "$R/결과/수치/"; done
.venv/bin/python analyze.py > "$R/결과/수치/분석요약.md" 2>/dev/null || true
cp logs/queue.log "$R/결과/로그/" 2>/dev/null || true
for f in logs/train_*.log logs/eval_*.log logs/gen_*.log; do [ -f "$f" ] && tail -n 40 "$f" > "$R/결과/로그/$(basename "$f")"; done

# figures and paper
cp paper/fig*.png "$R/결과/그림/" 2>/dev/null || true
cp results/stack_*.png results/pallet_*.png results/design_versions.png results/grip_check_*.png results/stage3_inspect.png results/robust_views.png results/clahe_views.png results/robust_views_beyond.png results/v7_views.png results/side_offset_v5_v8.png "$R/결과/그림/" 2>/dev/null || true
# clutter test (table with other objects): keep-out map, condition views, per-condition results and GIFs
cp results/clutter_keepout*.png results/clutter_chained.png results/clutter_views.png results/object_views.png results/cup_jam_debug.png results/factory_views.png results/pallet_slot5_offsets.png results/pallet_state_noise.png results/pallet_slot5_fix.png results/paper_v2_pages.png results/demo_audit_general.png "$R/결과/그림/" 2>/dev/null || true
for d in results/clutter/*/; do
  [ -d "$d" ] || continue
  t=$(basename "$d"); mkdir -p "$R/결과/수치/clutter/$t" "$R/결과/gif/clutter/$t"
  cp "$d"*.json "$R/결과/수치/clutter/$t/" 2>/dev/null || true
  for g in "$d"gifs/*_chained_t0[01]_*.gif; do
    [ -f "$g" ] || continue
    out="$R/결과/gif/clutter/$t/$(basename "$g")"
    [ -f "$out" ] || .venv/bin/python shrink_gif.py "$g" "$out" 480
  done
done
# scripted-expert GIFs of the generalisation / factory scenes
mkdir -p "$R/결과/gif/expert"
for g in results/gifs_expert/*.gif; do
  [ -f "$g" ] || continue
  out="$R/결과/gif/expert/$(basename "$g")"
  [ "$out" -nt "$g" ] || .venv/bin/python shrink_gif.py "$g" "$out" 480
done
[ -f paper/out/paper.docx ] && cp paper/out/paper.docx "$R/논문/"
[ -f paper/out/paper.pdf ] && cp paper/out/paper.pdf "$R/논문/"
[ -f paper/out/paper_v2.docx ] && cp paper/out/paper_v2.docx "$R/논문/"
[ -f paper/out/paper_v2.pdf ] && cp paper/out/paper_v2.pdf "$R/논문/"

# code
cp stack_env.py expert.py gen_data.py train_act.py eval_act.py make_gif.py batch_expert.py sync_repo.sh analyze.py dataset_preview.py \
   convert_jpeg.py publish.sh stack_photos.py grasp_tolerance.py grasp_offsets.py wall_rule_test.py design_photos.py grip_check.py inspect_stage3.py expert_eval.py robust_eval.py preprocess.py wide_expert_test.py eval_retry.py robust_summary.py merge_eval.py make_showcase.py fig_side_offset.py fig_pallet_slot5_fix.py verify_il.py make_object_stl.py fig_pallet_layout.py demo_audit.py pallet_slot_check.py run_pallet_v2.sh run_pallet_k25.sh run_pallet_fix.sh run_pallet_fix2.sh gen_pallet_more.sh pallet_grid.py gen_pallet.py pallet_eval.py \
   clutter_map.py clutter_eval.py clutter_views.py run_clutter.sh run_general.sh run_integrated.sh run_factory_eval.sh run_clutter_eval2.sh run_noise_check.sh run_pallet_2x2.sh run_pallet_v3.sh orchestrate.py expert_gif.py shrink_gif.py run_pallet_noise.sh factory.py factory_views.py object_views.py demo_audit_general.py health_check.py clutter_chart.py run_regif.sh run_pallet_s5fix.sh pallet_slot5_diag.py run_int80.sh \
   run_queue3.sh run_queue4.sh run_queue6.sh run_queue7.sh run_queue8.sh gen_all_v2.sh gen_stage_v2.sh gen_more_v2.sh gen_v4.sh gen_v5.sh gen_v6.sh run_queue9.sh run_queue10.sh run_post.sh run_post2.sh gen_pallet_all.sh run_queue_pallet.sh gen_v7.sh run_queue11.sh run_queue12.sh "$R/코드/" 2>/dev/null || true
mkdir -p "$R/코드/paper/templates" && cp paper/*.py "$R/코드/paper/" && cp paper/templates/*.docx "$R/코드/paper/templates/" 2>/dev/null || true
echo "synced to $R"
