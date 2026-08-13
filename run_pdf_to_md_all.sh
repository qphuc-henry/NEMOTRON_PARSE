#!/usr/bin/env bash
set -u

PDF=${1:?Usage: $0 FILE.pdf [OUTPUT.md]}
OUT=${2:-"${PDF%.pdf}.md"}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PAGES=$(pdfinfo "$PDF" | awk '/^Pages:/ {print $2}')
WORK=$(mktemp -d /tmp/nemotron-pages.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

START=$(date +%s.%N)
for ((page=0; page<PAGES; page++)); do
  t0=$(date +%s.%N)
  echo "[page $((page+1))/$PAGES]"
  python "$SCRIPT_DIR/pdf_to_md.py" --pdf "$PDF" --page "$page" \
    --output "$WORK/page_$(printf '%03d' "$page").md" || exit $?
  t1=$(date +%s.%N)
  awk -v a="$t0" -v b="$t1" 'BEGIN {printf "elapsed=%.2fs\n", b-a}'
done

{
  printf '# %s\n\n' "$(basename "$PDF")"
  for ((page=0; page<PAGES; page++)); do
    [ "$page" -gt 0 ] && printf '\n\n---\n\n'
    cat "$WORK/page_$(printf '%03d' "$page").md"
  done
} > "$OUT"

END=$(date +%s.%N)
awk -v a="$START" -v b="$END" -v out="$OUT" 'BEGIN {printf "output=%s\ntotal_elapsed=%.2fs\n", out, b-a}'
