#!/usr/bin/env bash
# usage: run_batch.sh "LABEL MODE STEPS [VAR=val ...]" ...   runs each with run_demo.sh in turn, in the background caller
for spec in "$@"; do
  set -- $spec; L=$1; M=$2; S=$3; shift 3
  env "$@" bash ~/robocurve/eval2/run_demo.sh "$L" "$M" "$S" > ~/robocurve/eval2/$L.log 2>&1
  echo "$L $(grep -o 'task_score: [0-9.]*' ~/robocurve/eval2/$L.log) $(grep '^exit' ~/robocurve/eval2/$L.log)"
done
