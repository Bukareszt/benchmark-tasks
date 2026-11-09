#!/usr/bin/env bash
# usage: run_bg.sh LABEL MODE STEPS [VAR=val ...] : starts run_demo.sh detached; result in ~/robocurve/eval3/LABEL.log
L=$1; M=$2; S=$3; shift 3
cd ~/robocurve/eval3 && nohup setsid env "$@" bash ~/robocurve/eval3/run_demo.sh "$L" "$M" "$S" > ~/robocurve/eval3/$L.log 2>&1 < /dev/null &
echo started $L
