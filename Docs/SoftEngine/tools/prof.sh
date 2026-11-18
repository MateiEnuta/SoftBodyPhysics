#!/bin/bash
# prof.sh "scenario steps" ... [-- key=value...]
S=/Users/enutamatei/GitHub/SoftBodyPhysics/Docs/SoftEngine/tools
extra=""; list=()
for a in "$@"; do if [[ "$a" == *=* ]]; then extra="$extra $a"; else list+=("$a"); fi; done
for sc in "${list[@]}"; do set -- $sc
  out=$(python3 $S/sb.py -t 600 run $1 $2 check=0 tag=prof safety=0 profile=1 maxSeconds=120 $extra)
  echo "$1: $(echo "$out" | grep -o 'ms avg=[0-9.]* max=[0-9.]*') | $(echo "$out" | grep stageMs | sed 's/stageMs //') | $(echo "$out" | grep -o 'pairsMax=[0-9]* pieceContacts=[0-9]* activeContacts=[0-9]*')"
done
