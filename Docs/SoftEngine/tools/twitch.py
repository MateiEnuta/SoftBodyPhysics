#!/usr/bin/env python3
"""twitch.py "scenario steps" body,body key=value... -> median / max node speed of the bodies over the last 100 steps"""
import sys, csv, re
sys.path.insert(0, '/Users/enutamatei/GitHub/SoftBodyPhysics/Docs/SoftEngine/tools')
import sb
D = '/Users/enutamatei/GitHub/SoftBodyPhysics/Temp/SoftBridge/'
sc, names, extra = sys.argv[1], sys.argv[2].split(','), ' '.join(sys.argv[3:])
out = sb.call('run %s check=0 tag=tw safety=0 maxSeconds=120 %s' % (sc, extra), 600)
rows = list(csv.DictReader(open(D + 'tw_bodies.csv')))
last = max(int(r['step']) for r in rows)
line = '%-22s %-28s ms %s cuts %s |' % (sc, extra, re.search(r'ms avg=(\S+)', out).group(1), re.search(r'total cuts=(\d+)', out).group(1))
for n in names:
    sp = sorted(float(r['maxSpeed']) for r in rows if r['body'] == n and int(r['step']) > last - 100)
    y = [r['y'] for r in rows if r['body'] == n][-1]
    line += ' %s: med %.2f max %.2f y=%s |' % (n[-12:], sp[len(sp) // 2], sp[-1], y)
print(line)
