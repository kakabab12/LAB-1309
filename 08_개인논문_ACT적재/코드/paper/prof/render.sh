#!/usr/bin/env bash
# Page-count preview of a .docx as Word would show it. HY신명조 is not installed on this PC; the user's Word showed
# 6 pages for the 10-11 v2 file, which is what the plain Noto Serif CJK substitution gives (the 1.2 em proxy gave 5),
# so the Noto Serif rendering ("word") is the estimate to trust; "tight" is the optimistic one.
# usage: paper/prof/render.sh file.docx  -> <dir>/_render/<name>_word.pdf, <name>_tight.pdf
set -eu
f=$(readlink -f "$1"); d=$(dirname "$f")/_render; n=$(basename "$f" .docx); mkdir -p "$d"
here=$(dirname "$(readlink -f "$0")")
for v in word tight; do
  cp "$f" "$d/${n}_$v.docx"
  conf=$here/fonts_word.conf; [ $v = tight ] && conf=$here/../fonts_proxy/fonts.conf
  (cd "$d" && FONTCONFIG_FILE=$conf timeout 300 soffice --headless --convert-to pdf "${n}_$v.docx" > /dev/null 2>&1; rm -f "${n}_$v.docx")
  echo "$v: $(pdfinfo "$d/${n}_$v.pdf" | grep Pages | tr -s ' ')"
done
