using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace SoftEngine
{
    /*
    Contact solve: a few sweeps over the contacts, all bodies at the same time.
    Its job is to leave nothing for the limit step to do.

    One sweep is one run of this job, one body per work item. A body only ever moves
    its own nodes, so all bodies run in parallel. What it needs to know about the
    other side of its contacts it reads from `from`, where every node was when the
    sweep began, and it writes to `to`. The next sweep swaps the two.

    Every contact wants its two spots to end at least `target` apart along its plane:

        Target(gap) = no closer than they started, and never further than skin,
                      plus a part of the way back out to skin while they are inside it

    so pieces further apart than skin may close in down to skin, and pieces already
    closer are held where they are and moved back out, quickly when they are deep in
    and slower and slower as they get to skin (a fixed step every substep keeps
    kicking a body that just lies there).

    Both bodies of a contact see the same thing at the start of a sweep, how much
    the spots are short of the target, and each takes its part of it: shareA for
    side A and the rest for side B, the lighter side moves more. Together that is
    exactly what is missing. A body goes through its contacts one after the other,
    and each one first looks how far its spot has already been moved in this sweep
    by the ones before it, so several contacts that want the same thing do not add
    up.

    A spot is a fixed place in its cell and moves with the cell's 8 nodes. A push
    moves every node by its share of the spot, light or heavy:

        node[k] += share[k] * push / (sum of share^2)

    so the cell keeps its shape and the elastic solve has nothing to undo in the
    next substep. (Splitting by inverse mass is what a real point force would do,
    but the nodes on the outside of a body are lighter than the ones inside, they
    move further, the cell is bent a little by every push and a stiff body that only
    lies there keeps buzzing.)

    Firm contact (see PlaneJob): the same idea with the real hull points instead of
    one spot. What is missing is measured between the lowest point of A and the
    highest of B, each side takes its part, and moves its whole cell by it (the
    spot of a firm contact is the middle of the cell, where all 8 nodes have the
    same share), until low - high >= target, which is what the limit step checks.

    A push is never taken back. (Giving back what turned out to be too much works
    when one loop goes through all contacts, as it used to. With two bodies doing
    it at the same time, each from its own side, the result depended on whether the
    number of sweeps was odd or even, and a stack of cubes kept buzzing.)

    Floor contacts work per hull point, against the floor that does not move. Every
    point below its target asks its nodes for the lift that would bring it back (the
    same lift for each of them, for the same reason), and every node then takes the
    largest lift asked of it. That satisfies all floor contacts at once and does not
    depend on the order they are in, so a body that lands straight does not get
    pushed to one side.

    Friction runs in the last sweep, when it is known how hard every contact pushed
    in total: the sideways slide of the substep is taken back, at most friction *
    push (Coulomb). Between pieces that is the slide of the two spots against each
    other, and each side again takes its part (see Rub). On the floor it is done
    per node, every node loses slide in proportion to how much the floor lifted it.

    The sweeps alternate direction, a fixed direction makes the contacts that come
    first carry more than the last ones. In the last sweep the firm contacts come
    after all others: they are the ones closest to the limit, so nothing may move
    their points after they had their last say.

    The last sweep also tells which bodies it pushed further than expected
    (pushedFar), those get a second look before the limit step (LatePieceJob).
    */
    [BurstCompile]
    public struct ContactSweepJob : IJobParallelFor
    {
        // A contact inside skin moves back out by this share of the missing distance per substep (at most pushOut)
        private const float PushOutShare = 0.2f;

        public int sweep;
        public bool last;
        public bool forward;        // the order this sweep goes through a body's contacts in
        public float skin, pushOut, regionMargin;
        [ReadOnly] public NativeArray<Contact> contacts;
        [ReadOnly] public NativeArray<int> halves, bodyStart;
        [ReadOnly] public NativeArray<BodyRange> bodies;
        [ReadOnly] public NativeArray<Piece> pieces;
        [ReadOnly] public NativeArray<EmbeddedVertex> hullVerts;
        [ReadOnly] public NativeArray<int> cellNodes;
        [ReadOnly] public NativeArray<float3> prev, expected, hullPos;
        [ReadOnly] public NativeArray<float3> from;
        [NativeDisableParallelForRestriction] public NativeArray<float3> to;
        [NativeDisableParallelForRestriction] public NativeArray<float> pushed;         // per half, total push so far
        [NativeDisableParallelForRestriction] public NativeArray<float> lift, liftGrip; // per node, largest lift asked in this sweep and the friction of who asked
        [NativeDisableParallelForRestriction] public NativeArray<float> floorGrip;      // per node, friction * total lift
        [NativeDisableParallelForRestriction] public NativeArray<bool> pushedFar;       // per body, set by the last sweep: a node ended further than regionMargin from where it was heading

        // How far apart two sides that start `gap` apart have to end
        public static float Target(float gap, float skin, float pushOut)
        {
            return math.min(gap, skin) + math.clamp(PushOutShare * (skin - gap), 0f, pushOut);
        }

        public void Execute(int body)
        {
            int start = bodyStart[body], end = bodyStart[body + 1];
            if (last) pushedFar[body] = false;
            if (start == end) return;

            int firstNode = bodies[body].nodeStart, nodeEnd = firstNode + bodies[body].nodeCount;
            for (int i = firstNode; i < nodeEnd; i++) to[i] = from[i];
            if (sweep == 0)
                for (int i = firstNode; i < nodeEnd; i++) floorGrip[i] = 0f;

            bool floor = false;
            for (int round = 0; round < (last ? 2 : 1); round++)
            for (int step = start; step < end; step++)
            {
                int half = halves[forward ? step : start + end - 1 - step];
                Contact c = contacts[half >> 1];
                if (c.kind == ContactKind.Floor) floor = true;
                else if (last && c.firm != (round == 1)) continue;
                else if (c.firm) pushed[half] += PushHull(c, (half & 1) == 0);
                else pushed[half] += PushSpot(c, (half & 1) == 0);
            }

            if (floor)
            {
                for (int step = start; step < end; step++)
                    if (contacts[halves[step] >> 1].kind == ContactKind.Floor) AskFloorLift(contacts[halves[step] >> 1]);
                for (int i = firstNode; i < nodeEnd; i++)
                {
                    if (lift[i] == 0f) continue;
                    to[i] += new float3(0f, lift[i], 0f);
                    floorGrip[i] += liftGrip[i] * lift[i];
                    lift[i] = 0f;
                }
            }
            if (!last) return;

            for (int step = start; step < end; step++)
            {
                int half = halves[step];
                if (contacts[half >> 1].kind == ContactKind.Pieces && pushed[half] > 0f) Rub(contacts[half >> 1], (half & 1) == 0, pushed[half]);
            }
            for (int i = firstNode; floor && i < nodeEnd; i++)
            {
                if (floorGrip[i] == 0f) continue;
                float3 slide = to[i] - prev[i];
                slide.y = 0f;
                float length = math.length(slide);
                if (length > 1e-9f) to[i] -= slide * (math.min(length, floorGrip[i]) / length);
            }

            float furthest = 0f;
            for (int i = firstNode; i < nodeEnd; i++) furthest = math.max(furthest, math.distancesq(to[i], expected[i]));
            pushedFar[body] = furthest > regionMargin * regionMargin;
        }

        // One side's part of one push. Returns the push
        private float PushSpot(in Contact c, bool sideA)
        {
            float3 n = (float3)c.n;
            int cellA = pieces[c.a].cell * 8, cellB = pieces[c.b].cell * 8;

            // What was missing when the sweep began
            float3 apart = 0f;      // spot of A minus spot of B
            for (int k = 0; k < 4; k++)
            {
                apart += c.a0[k] * from[cellNodes[cellA + k]] + c.a1[k] * from[cellNodes[cellA + k + 4]];
                apart -= c.b0[k] * from[cellNodes[cellB + k]] + c.b1[k] * from[cellNodes[cellB + k + 4]];
            }
            float missing = c.target - math.dot(n, apart);

            // This side's part of it, less what its spot has been moved since then
            int cell = sideA ? cellA : cellB;
            float3 dir = sideA ? n : -n;
            float4 low = sideA ? c.a0 : c.b0, high = sideA ? c.a1 : c.b1;
            float moved = 0f;
            for (int k = 0; k < 4; k++)
            {
                int n0 = cellNodes[cell + k], n1 = cellNodes[cell + k + 4];
                moved += low[k] * math.dot(dir, to[n0] - from[n0]) + high[k] * math.dot(dir, to[n1] - from[n1]);
            }
            float push = (sideA ? c.shareA : 1f - c.shareA) * missing - moved;
            if (push <= 1e-6f) return 0f;

            MoveSpot(cell, low, high, dir * (push / (sideA ? c.spreadA : c.spreadB)));
            return push;
        }

        // One side's part of keeping the hull points of the two pieces apart: its whole cell moves. Returns the push
        private float PushHull(in Contact c, bool sideA)
        {
            Piece own = pieces[sideA ? c.a : c.b], other = pieces[sideA ? c.b : c.a];
            float3 dir = sideA ? (float3)c.n : -(float3)c.n;

            // What was missing when the sweep began
            float low = Lowest(own, dir, from);
            float missing = c.target - low - Lowest(other, -dir, from);
            if (missing <= 0f) return 0f;

            // This side's part of it, less what its lowest point has been moved since then
            float push = low + (sideA ? c.shareA : 1f - c.shareA) * missing - Lowest(own, dir, to);
            if (push <= 1e-6f) return 0f;

            MoveSpot(own.cell * 8, sideA ? c.a0 : c.b0, sideA ? c.a1 : c.b1, dir * (push / (sideA ? c.spreadA : c.spreadB)));
            return push;
        }

        // The lowest hull point of a piece along dir
        private float Lowest(in Piece piece, float3 dir, NativeArray<float3> x)
        {
            float lowest = float.MaxValue;
            for (int v = piece.hullStart; v < piece.hullStart + piece.hullCount; v++)
                lowest = math.min(lowest, math.dot(dir, Geometry.Point(hullVerts[v], x)));
            return lowest;
        }

        /*
        Friction of one side of a Pieces contact. Both sides see the same slide, the one
        of the two spots against each other when this sweep began, and each wants its
        own spot to have taken back its part of it by the end of the sweep. What the
        spot has been moved sideways since then (by the pushes and by the friction of
        the contacts before this one) counts, so contacts that share nodes do not
        each take the slide back again.
        */
        private void Rub(in Contact c, bool sideA, float push)
        {
            float3 n = (float3)c.n;
            int own = pieces[sideA ? c.a : c.b].cell * 8, other = pieces[sideA ? c.b : c.a].cell * 8;
            float4 low = sideA ? c.a0 : c.b0, high = sideA ? c.a1 : c.b1;
            float4 otherLow = sideA ? c.b0 : c.a0, otherHigh = sideA ? c.b1 : c.a1;

            float3 slide = 0f, moved = 0f;
            for (int k = 0; k < 4; k++)
            {
                int n0 = cellNodes[own + k], n1 = cellNodes[own + k + 4];
                slide += low[k] * (from[n0] - prev[n0]) + high[k] * (from[n1] - prev[n1]);
                moved += low[k] * (to[n0] - from[n0]) + high[k] * (to[n1] - from[n1]);
                n0 = cellNodes[other + k]; n1 = cellNodes[other + k + 4];
                slide -= otherLow[k] * (from[n0] - prev[n0]) + otherHigh[k] * (from[n1] - prev[n1]);
            }
            float3 back = -(sideA ? c.shareA : 1f - c.shareA) * slide - moved;
            back -= n * math.dot(n, back);
            float length = math.length(back);
            if (length < 1e-9f) return;

            // Coulomb: at most friction * push
            MoveSpot(own, low, high, back * (math.min(length, c.friction * push) / length / (sideA ? c.spreadA : c.spreadB)));
        }

        // Every node of the cell moves by its share of the spot. move is how far the spot goes, over the sum of the squared shares
        private void MoveSpot(int cell, float4 low, float4 high, float3 move)
        {
            for (int k = 0; k < 4; k++)
            {
                to[cellNodes[cell + k]] += move * low[k];
                to[cellNodes[cell + k + 4]] += move * high[k];
            }
        }

        private void AskFloorLift(in Contact c)
        {
            Piece piece = pieces[c.a];
            for (int v = piece.hullStart; v < piece.hullStart + piece.hullCount; v++)
            {
                EmbeddedVertex e = hullVerts[v];
                float push = Target(hullPos[v].y - c.planeD, skin, pushOut) - (Geometry.Point(e, to).y - c.planeD);
                if (push <= 1e-6f) continue;

                for (int k = 0; k < 4; k++)
                {
                    if (e.w[k] <= 0f || push <= lift[e.nodes[k]]) continue;
                    lift[e.nodes[k]] = push;
                    liftGrip[e.nodes[k]] = c.friction;
                }
            }
        }
    }

    /*
    Limit step: the guarantee. Last thing that moves nodes in a substep.

    Every contact is checked with the real hull points, where the nodes are now:

        low - high >= limit,    limit = min(slab at the start, hardSkin)

    If that holds, the two sides are apart along n at the start and at the end of
    straight paths, so they were apart all the way and end at least limit apart.
    Pieces are convex and the plane is the best one there is at the start of every
    substep, so the distance between two pieces can never shrink below hardSkin,
    and whatever is inside the hulls, the real surface, cannot touch.

    A contact that fails is cut: the room (start slab - limit) is split between the
    sides by how much each came closer, and every hull point past its side's share
    is brought back to it. How, depends on how often the list has been gone through:

        slide       the point's nodes lose (part of) what they moved towards the
                    other side, and nothing else. They stay free to move along the
                    plane and away from it, so nothing sticks to what it touches.
                    (A node never leaves the box around its own path, the boxes of
                    the broad phase count on that.)
        stop        the nodes go back along their own path of this substep.
        freeze      the nodes of both cells go back to where they started.

    SoftSolver says how many passes slide and stop (fewer when a step runs out of
    time: sliding is the gentle way and the slow one).

    Taking away motion towards one plane can break a contact that was fine, so the
    whole list is checked again until one full pass finds nothing. Sliding nearly
    always settles in a pass or two, but nothing says it has to end. Stopping and
    freezing only ever shorten the paths, a frozen contact holds whatever else
    happens, so this ends. What is left has passed the same check that defines
    "safe", with the positions that are stored.

    Nothing here can add energy: a node only ever moves less than it wanted to.
    */
    [BurstCompile]
    public struct LimitJob : IJob
    {
        private const int MaxPasses = 40;
        private const double Cushion = 1e-5;     // cuts aim this far past the limit, so rounding cannot leave them short

        public float hardSkin;
        public int slidePasses, stopPasses;     // how many passes slide, how many after that stop, the rest freeze
        public int bodyCount;
        [ReadOnly] public NativeList<Contact> contacts;
        [ReadOnly] public NativeArray<Piece> pieces;
        [ReadOnly] public NativeArray<EmbeddedVertex> hullVerts;
        [ReadOnly] public NativeArray<int> cellNodes;
        [ReadOnly] public NativeArray<float3> prev;
        public NativeArray<float3> pos;
        public NativeArray<float3> goal;        // where the contact solve left the nodes
        public NativeArray<int> stats;

        public void Execute()
        {
            goal.CopyFrom(pos);

            // After the first pass only the contacts of bodies that had a cut in the pass before can have changed
            NativeArray<bool> cut = new NativeArray<bool>(bodyCount, Allocator.Temp), cutBefore = new NativeArray<bool>(bodyCount, Allocator.Temp);
            for (int pass = 0; pass < MaxPasses; pass++)
            {
                bool clean = true;
                for (int i = 0; i < contacts.Length; i++)
                {
                    Contact c = contacts[i];
                    int bodyA = pieces[c.a].body, bodyB = c.kind == ContactKind.Pieces ? pieces[c.b].body : bodyA;
                    if (pass > 0 && !cutBefore[bodyA] && !cutBefore[bodyB]) continue;
                    if (Check(c, pass)) continue;

                    clean = false;
                    cut[bodyA] = true;
                    cut[bodyB] = true;
                }

                stats[SoftStats.LimitPasses]++;
                if (clean) return;
                cutBefore.CopyFrom(cut);
                for (int b = 0; b < bodyCount; b++) cut[b] = false;
            }

            // Never seen, but then everything in a contact stays where it started
            for (int i = 0; i < contacts.Length; i++)
            {
                Freeze(pieces[contacts[i].a].cell);
                if (contacts[i].kind == ContactKind.Pieces) Freeze(pieces[contacts[i].b].cell);
            }
        }

        // True when the contact holds as it is
        private bool Check(in Contact c, int pass)
        {
            bool floor = c.kind == ContactKind.Floor;
            Piece A = pieces[c.a], B = floor ? default : pieces[c.b];

            // Cheap test first: a point moves along n at least as much as the slowest node of its cell.
            // slab is the float copy of the start slab, so leave a little room
            float3 n = (float3)c.n;
            float least = Geometry.LeastMotion(A.cell, n, cellNodes, pos, prev), most = floor ? 0f : -Geometry.LeastMotion(B.cell, -n, cellNodes, pos, prev);
            if (c.slab + least - most >= hardSkin + 1e-4f) return true;

            double low = Lowest(A, c.n, pos), high = floor ? c.planeD : -Lowest(B, -c.n, pos);
            if (low - high >= hardSkin) return true;

            double lowStart = Lowest(A, c.n, prev), highStart = floor ? c.planeD : -Lowest(B, -c.n, prev);
            double limit = math.min(lowStart - highStart, hardSkin);
            if (low - high >= limit) return true;

            stats[SoftStats.Cuts]++;
            if (pass >= slidePasses + stopPasses)
            {
                stats[SoftStats.Freezes]++;
                Freeze(A.cell);
                if (!floor) Freeze(B.cell);
                return false;
            }

            double room = lowStart - highStart - limit;
            double closerA = math.max(lowStart - low, 0.0), closerB = math.max(high - highStart, 0.0);
            double levelA = lowStart - room * closerA / (closerA + closerB);
            Cut(A, c.n, levelA + Cushion, pass < slidePasses);
            if (!floor) Cut(B, -c.n, -(levelA - limit) + Cushion, pass < slidePasses);
            return false;
        }

        // Every hull point of the piece that ends below level along dir is brought back up to it
        private void Cut(in Piece piece, double3 dir, double level, bool slide)
        {
            float3 along = (float3)dir;
            for (int v = piece.hullStart; v < piece.hullStart + piece.hullCount; v++)
            {
                EmbeddedVertex e = hullVerts[v];
                double end = math.dot(dir, Geometry.PointD(e, pos));
                if (end >= level) continue;

                // Slide: how far the point came down because of the nodes that moved down, and the part of that to take back
                float4 down = 0f;
                for (int k = 0; k < 4; k++) down[k] = math.max(-math.dot(along, pos[e.nodes[k]] - prev[e.nodes[k]]), 0f);
                float cameDown = math.dot(e.w, down);
                if (slide && cameDown > 1e-9f)
                {
                    float back = math.min((float)(level - end) / cameDown, 1f);
                    for (int k = 0; k < 4; k++)
                    {
                        if (e.w[k] <= 0f || down[k] <= 0f) continue;
                        int node = e.nodes[k];
                        pos[node] = math.clamp(pos[node] + along * (back * down[k]), math.min(prev[node], goal[node]), math.max(prev[node], goal[node]));
                    }
                    continue;
                }

                // Stop: back along the path, by the part of the way the point was not allowed to go
                double start = math.dot(dir, Geometry.PointD(e, prev));
                float keep = start > level ? (float)((start - level) / (start - end)) : 0f;
                for (int k = 0; k < 4; k++)
                    if (e.w[k] > 0f) pos[e.nodes[k]] = keep <= 0f ? prev[e.nodes[k]] : prev[e.nodes[k]] + keep * (pos[e.nodes[k]] - prev[e.nodes[k]]);
            }
        }

        private void Freeze(int cell)
        {
            for (int k = 0; k < 8; k++) pos[cellNodes[cell * 8 + k]] = prev[cellNodes[cell * 8 + k]];
        }

        /*
        The lowest hull point of a piece along dir, for the given node positions. Not
        inlined on purpose: start and end are measured by the very same code, so a
        contact whose nodes are all back at the start measures exactly its start slab.
        */
        [MethodImpl(MethodImplOptions.NoInlining)]
        private double Lowest(in Piece piece, double3 dir, NativeArray<float3> x)
        {
            double lowest = double.MaxValue;
            for (int v = piece.hullStart; v < piece.hullStart + piece.hullCount; v++)
                lowest = math.min(lowest, math.dot(dir, Geometry.PointD(hullVerts[v], x)));
            return lowest;
        }
    }
}
