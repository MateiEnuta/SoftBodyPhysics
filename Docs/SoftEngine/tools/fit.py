import csv, re, sys
sys.path.insert(0, '.')
import sb
D = '/Users/enutamatei/GitHub/SoftBodyPhysics/Temp/SoftBridge/'
names = ['drop', 'midair', 'headon', 'tower', 'smash', 'fast', 'squeeze']
cells = {}
for n in names:
    out = sb.call('run %s 2 check=0 tag=x' % n, 120)
    cells[n] = int(re.search(r'cells=(\d+)', out).group(1))
print(cells)
F = ['cells', 'pairs', 'pieceContacts', 'activeContacts', 'cuts', 'extraPasses']
X, Y, tag = [], [], []
for n in names:
    for r in list(csv.DictReader(open(D + 'suite_%s_steps.csv' % n)))[1:]:
        if int(r['asleep']) > 0: continue
        X.append([cells[n] * 4, int(r['pairs']), int(r['pieceContacts']), int(r['activeContacts']), int(r['cuts']), int(r['passes']) - 4])
        Y.append(float(r['ms'])); tag.append(n)
def solve(X, Y, use):
    n = len(use)
    A = [[sum(x[use[i]] * x[use[j]] for x in X) for j in range(n)] for i in range(n)]
    b = [sum(x[use[i]] * y for x, y in zip(X, Y)) for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda k: abs(A[k][i])); A[i], A[p] = A[p], A[i]; b[i], b[p] = b[p], b[i]
        for k in range(i + 1, n):
            f = A[k][i] / A[i][i]
            for j in range(i, n): A[k][j] -= f * A[i][j]
            b[k] -= f * b[i]
    w = [0] * n
    for i in reversed(range(n)): w[i] = (b[i] - sum(A[i][j] * w[j] for j in range(i + 1, n))) / A[i][i]
    return w
for use in ([0, 1, 2, 3, 4, 5], [0, 1, 2, 4, 5], [0, 1, 2, 4]):
    w = solve(X, Y, use)
    print('fit', {F[u]: '%.3g' % (c * 1000) for u, c in zip(use, w)}, '(microseconds per unit)')
    pred = [sum(c * x[u] for u, c in zip(use, w)) for x in X]
    for n in names:
        idx = [i for i, t in enumerate(tag) if t == n]
        worst = max(idx, key=lambda i: Y[i])
        over = sorted(idx, key=lambda i: -Y[i])[:5]
        print('   %-8s avg real %.2f est %.2f | worst real %.1f est %.1f | 5 worst real %s est %s' % (n, sum(Y[i] for i in idx) / len(idx), sum(pred[i] for i in idx) / len(idx), Y[worst], pred[worst], ' '.join('%.0f' % Y[i] for i in over), ' '.join('%.0f' % pred[i] for i in over)))
