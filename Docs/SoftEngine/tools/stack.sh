#!/bin/bash
# stack.sh <scenario> <steps> key=value... -> one line: settings, then per-body maxSpeed at a few steps
S=/Users/enutamatei/GitHub/SoftBodyPhysics/Docs/SoftEngine/tools
D=/Users/enutamatei/GitHub/SoftBodyPhysics/Temp/SoftBridge
sc=$1; steps=$2; shift; shift
out=$(python3 $S/sb.py -t 300 run $sc $steps check=10 tag=st safety=0 maxSeconds=60 "$@")
echo "== $sc $* :: $(echo "$out" | grep -o 'avg=[0-9.]*') $(echo "$out" | grep -o 'minDistance=[0-9.]*') $(echo "$out" | grep -o 'total cuts=[0-9]*') $(echo "$out" | grep -o 'activeContacts=[0-9]*') $(echo "$out" | grep -o 'crossings=[0-9]*' | head -1)"
python3 - "$steps" <<'PY'
import csv, sys
D='/Users/enutamatei/GitHub/SoftBodyPhysics/Temp/SoftBridge/'
steps=int(sys.argv[1])
rows=list(csv.DictReader(open(D+'st_bodies.csv')))
bodies=[]
for r in rows:
    if r['body'] not in bodies: bodies.append(r['body'])
# average of maxSpeed over the last third, per body, and final position
for b in bodies:
    rs=[r for r in rows if r['body']==b]
    last=[float(r['maxSpeed']) for r in rs if int(r['step'])>steps*2//3]
    mid=[float(r['maxSpeed']) for r in rs if steps//3<int(r['step'])<=steps*2//3]
    e=rs[-1]
    print('   %-9s nodeSpeed mid=%.2f last=%.2f  end pos=(%s,%s,%s)' % (b, sum(mid)/max(len(mid),1), sum(last)/max(len(last),1), e['x'], e['y'], e['z']))
PY
