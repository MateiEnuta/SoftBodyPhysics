#!/usr/bin/env python3
"""sleepy.py "scenario steps" key=value... -> how sleeping develops, and who is still awake at the end"""
import sys, csv, re
sys.path.insert(0, '/Users/enutamatei/GitHub/SoftBodyPhysics/Docs/SoftEngine/tools')
import sb
D = '/Users/enutamatei/GitHub/SoftBodyPhysics/Temp/SoftBridge/'
out = sb.call('run %s check=%s tag=sy safety=0 maxSeconds=200 %s' % (sys.argv[1], '25', ' '.join(sys.argv[2:])), 900)
print(' '.join(re.findall(r'ms avg=\S+ max=\S+|minDistance=\S+|crossings=\d+|total cuts=\d+|^error.*', out)))
rows = list(csv.DictReader(open(D + 'sy_steps.csv')))
n = len(rows); k = n // 6
print(' | '.join('%d-%d: %.2f ms, %s asleep, %d changes' % (a + 1, a + k, sum(float(x['ms']) for x in rows[a:a + k]) / k, rows[a + k - 1]['asleep'], sum(int(x['sleepChanges']) for x in rows[a:a + k])) for a in range(0, n - k + 1, k)))
b = list(csv.DictReader(open(D + 'sy_bodies.csv')))
names = []
for r in b:
    if r['body'] not in names: names.append(r['body'])
awake = []
for nm in names:
    rs = [r for r in b if r['body'] == nm and int(r['step']) > n - 100]
    if float(rs[-1]['maxSpeed']) == 0: continue
    d = sum((float(rs[-1][c]) - float(rs[0][c])) ** 2 for c in 'xyz') ** 0.5
    sp = sorted(float(r['maxSpeed']) for r in rs)
    awake.append('%s (centre %.2f m, node speed %.2f)' % (nm.replace('new_', ''), d, sp[len(sp) // 2]))
print('awake at end, last 100 steps:', '; '.join(awake) if awake else 'nobody')
