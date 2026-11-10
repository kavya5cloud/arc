#!/usr/bin/env bash
# Regenerate every raw result and every reported number from scratch, then verify.
# Tested with Python 3.13.18 and numpy 2.5.3 on 4 CPUs (about 20 minutes).
set -euo pipefail
cd "$(dirname "$0")"
OUT=fresh
rm -rf "$OUT" && mkdir -p "$OUT/res2" "$OUT/res3" "$OUT/res4" "$OUT/res5"

for s in 1 2 3; do
  python3 pilot.py $s 300 > "$OUT/res_$s.json" &
  python3 pilot2.py $s 300 2 3 random,single,curiosity,ours-full,ours-s8,ours-s16,ours-s32 > "$OUT/res2/h288_s$s.json" &
  python3 pilot2.py $s 300 3 4 random,single,curiosity,ours-full,ours-s8,ours-s16,ours-s32 > "$OUT/res2/h768_s$s.json" &
  python3 pilot2.py $s 300 4 5 random,single,curiosity,ours-s8,ours-s16,ours-s32 > "$OUT/res2/h1920_s$s.json" &
  wait
  python3 pilot2.py $s 300 2 3 worldcoder,empa,rate-s16 > "$OUT/res3/h288_s$s.json" &
  python3 pilot2.py $s 300 3 4 worldcoder,empa,rate-s16 > "$OUT/res3/h768_s$s.json" &
  python3 pilot2.py $s 300 4 5 worldcoder,empa,rate-s16 > "$OUT/res3/h1920_s$s.json" &
  python3 pilot2.py $s 300 2 3 single,curiosity,empa,worldcoder,rate-s16 hazard > "$OUT/res3/z648_s$s.json" &
  python3 pilot2.py $s 300 3 4 single,curiosity,empa,worldcoder,rate-s16 hazard > "$OUT/res3/z2592_s$s.json" &
  wait
  for c in 0 5 10 20 40; do
    HAZARD_COST=$c python3 pilot2.py $s 300 2 3 worldcoder,rate-s16,ours-s16 hazard > "$OUT/res4/c${c}_s$s.json" &
  done
  wait
  python3 pilot2.py $s 300 2 3 rate-full,rate-s8,rate-s32 hazard > "$OUT/res5/sub_z648_s$s.json" &
  python3 pilot2.py $s 300 3 4 rate-full,rate-s8,rate-s32 hazard > "$OUT/res5/sub_z2592_s$s.json" &
  python3 pilot2.py $s 300 2 3 tau50-s16,tau80-s16 hazard > "$OUT/res5/tau_z648_s$s.json" &
  python3 pilot2.py $s 300 3 4 tau50-s16,tau80-s16 hazard > "$OUT/res5/tau_z2592_s$s.json" &
  wait
done

# Raw outputs are deterministic: they must be byte-identical to the committed ones.
for f in $(cd "$OUT" && find . -name '*.json' | sort); do
  cmp -s "$OUT/$f" "$f" || { echo "DIFFERS: $f"; exit 1; }
done
echo "OK: all raw results byte-identical"
python3 analyze_all.py --check
