#!/bin/bash
# Run ATPlace WL for all 10 cases, save to timestamped folder under atplace/
# Usage: bash run_wl_batch.sh [tag]
#   tag: optional label, e.g. "milp150s" (default: empty)
export GRB_LICENSE_FILE=/data/xinli/gurobi.lic
export LC_ALL=C

SCRIPT="$(dirname "$0")/reproduce.py"
CASE_BASE="$(dirname "$0")/cases"
ATPLACE_DIR=/data/xinli/ThermOpt/atplace
TMPDIR="$(dirname "$0")/.param_tmp"
mkdir -p "$TMPDIR"

TAG="${1:-}"
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${STAMP}${TAG:+_$TAG}"
OUT_BASE="$ATPLACE_DIR/$RUN_NAME"
mkdir -p "$OUT_BASE"

declare -A PAPER_WL
PAPER_WL[Case1]=12.01; PAPER_WL[Case2]=15.35; PAPER_WL[Case3]=34.09; PAPER_WL[Case4]=54.47
PAPER_WL[Case5]=47.89; PAPER_WL[Case6]=29.27; PAPER_WL[Case7]=12.29
PAPER_WL[Case8]=8.41;  PAPER_WL[Case9]=43.21; PAPER_WL[Case10]=25.74

echo "=== ATPlace WL batch: $RUN_NAME ===" | tee "$OUT_BASE/run.log"
echo "Started: $(date)" | tee -a "$OUT_BASE/run.log"
echo "gurobi.env: $(cat "$(dirname "$0")/gurobi.env" | tr '\n' ' ')" | tee -a "$OUT_BASE/run.log"

for case in Case1 Case2 Case3 Case4 Case5 Case6 Case7 Case8 Case9 Case10; do
    param_tmp="$TMPDIR/${case}_wl.json"
    python3 -c "
import json
d = json.load(open('$CASE_BASE/$case/reproduce.json'))
wl = d.get('wl', {})
wl['ILPsolver'] = 'grb'
wl['thermal_solver'] = 'hotspot'
wl['temp_aware_opt'] = False
print(json.dumps(wl, indent=2))
" > "$param_tmp"

    out_dir="$OUT_BASE/$case"
    echo "" | tee -a "$OUT_BASE/run.log"
    echo "--- $case ($(date +%H:%M:%S)) ---" | tee -a "$OUT_BASE/run.log"

    (cd "$(dirname "$0")" && conda run -n atplace_py39 python "$SCRIPT" \
        --case "$case" --mode wl \
        --case-dir "$CASE_BASE/$case" \
        --param-file "$param_tmp" \
        --out-dir "$out_dir") 2>&1 | \
        grep -E '"twl_m"|"runtime_s"|"layout_chiplets"|Error No final' | \
        tee -a "$OUT_BASE/run.log"

    if [ -f "$out_dir/summary.json" ]; then
        twl=$(python3 -c "import json; d=json.load(open('$out_dir/summary.json')); print(f'{d[\"twl_m\"]:.3f}')")
        rt=$(python3 -c  "import json; d=json.load(open('$out_dir/summary.json')); print(f'{d[\"runtime_s\"]:.0f}')")
        paper=${PAPER_WL[$case]}
        ratio=$(python3 -c "print(f'{$twl/$paper:.3f}')")
        echo "  => WL=${twl}m  paper=${paper}m  ratio=${ratio}  time=${rt}s" | tee -a "$OUT_BASE/run.log"
    else
        echo "  => FAILED" | tee -a "$OUT_BASE/run.log"
    fi
done

echo "" | tee -a "$OUT_BASE/run.log"
echo "=== Done: $(date) ===" | tee -a "$OUT_BASE/run.log"
echo "Results in: $OUT_BASE" | tee -a "$OUT_BASE/run.log"

# Summary table
echo ""
python3 << PYEOF
import json, os
paper = {"Case1":12.01,"Case2":15.35,"Case3":34.09,"Case4":54.47,
         "Case5":47.89,"Case6":29.27,"Case7":12.29,
         "Case8":8.41,"Case9":43.21,"Case10":25.74}
base = "$OUT_BASE"
print(f"{'Case':6} | {'WL (m)':>8} | {'Paper':>8} | {'Ratio':>6} | {'Time':>7}")
print("-"*48)
for c in [f"Case{i}" for i in range(1,11)]:
    p = f"{base}/{c}/summary.json"
    if os.path.exists(p):
        d = json.load(open(p))
        print(f"{c:6} | {d['twl_m']:>8.3f} | {paper[c]:>8.2f} | {d['twl_m']/paper[c]:>6.3f} | {d['runtime_s']:>6.0f}s")
    else:
        print(f"{c:6} | {'FAIL':>8} | {paper[c]:>8.2f} |    --- |")
PYEOF
