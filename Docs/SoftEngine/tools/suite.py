#!/usr/bin/env python3
"""suite.py [-s name,name] [key=value ...]  -> one line per scenario for the given settings"""
import sys, re, csv
sys.path.insert(0, '/Users/enutamatei/GitHub/SoftBodyPhysics/Docs/SoftEngine/tools')
import sb
D = '/Users/enutamatei/GitHub/SoftBodyPhysics/Temp/SoftBridge/'
RUNS = [('drop', 150, 1), ('rest', 150, 1), ('slide', 150, 1), ('midair', 40, 1), ('headon', 100, 1), ('tower', 175, 5),
        ('smash', 100, 2), ('fast', 100, 2), ('squeeze', 300, 5), ('pile', 300, 10), ('coils', 300, 10)]
args = sys.argv[1:]
only = None
if args and args[0] == '-s': only = args[1].split(','); args = args[2:]

def body_at(tag, body, step):
    for x in csv.DictReader(open(D + tag + '_bodies.csv')):
        if x['body'] == body and int(x['step']) == step: return x
print('%-8s %6s %6s %7s %5s %9s %6s %6s %5s %6s %7s %7s  %s' % ('scene', 'msAvg', 'msMax', 'minDist', 'cross', 'inv max/end', 'edge', 'cuts', 'frz', 'hullPts', 'pieceC', 'activeC', 'result'))
for name, steps, check in RUNS:
    if only and name not in only: continue
    out = sb.call('run %s %d check=%d tag=suite_%s maxSeconds=45 %s' % (name, steps, check, name, ' '.join(args)), 900)
    if not out.startswith('ok'): print(name, out[:300]); continue
    m = re.search(r'STOPPED after step (\d+)', out)
    if m: steps = int(m.group(1)); print('   (stopped at step %d)' % steps)
    g = lambda k: re.search(r'\b' + k + r'=(\S+)', out).group(1)
    t = 'suite_' + name
    if name == 'drop': b = body_at(t, 'ball', steps); res = 'drift xz=%s,%s y=%s' % (b['x'], b['z'], b['y'])
    elif name == 'rest': b = body_at(t, 'cube', steps); res = 'y=%s maxSpeed=%s' % (b['y'], b['maxSpeed'])
    elif name == 'slide': b = body_at(t, 'cube', steps); res = 'x=%s (0.917) vx=%s' % (b['x'], b['vx'])
    elif name == 'midair': b = body_at(t, 'cube', min(40, steps)); res = 'cube vy=%s (-17.85) y=%s (11.06)' % (b['vy'], b['y'])
    elif name == 'headon': b = body_at(t, 'left', steps); c = body_at(t, 'right', steps); res = 'vx=%s,%s p=%s' % (b['vx'], c['vx'], g('momentum'))
    elif name == 'tower': b = body_at(t, 'cylinder', steps); res = 'cyl y=%s x=%s' % (b['y'], b['x'])
    elif name in ('smash', 'squeeze'): b = body_at(t, 'cube', steps); res = 'cube y=%s vy=%s' % (b['y'], b['vy'])
    elif name == 'fast': b = body_at(t, 'ball', steps); res = 'ball y=%s vy=%s' % (b['y'], b['vy'])
    else: res = 'kinetic=' + re.search(r'kinetic=([^ ]+)', out).group(1)
    print('%-8s %6s %6s %7s %5s %9s %6s %6s %5s %6s %7s %7s  %s' % (name, g('avg'), g('max'), g('minDistance'), g('crossings'),
          g('invertedMax') + '/' + g('invertedEnd'), g('maxEdge'), g('cuts'), g('freezes'), g('hullPoints'), g('pieceContacts'), g('activeContacts'), res + (' NAN!' if g('nanNodes') != '0' else '')))
    sys.stdout.flush()
