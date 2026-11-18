using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace SoftEngine
{
    /*
    Detection 1/4: per piece, everything the next jobs need.

        hullPos           its hull points at the start of the substep
        startLo / Hi      box of those points
        centre            their average
        reach             the most any of its points is expected to move this substep
        boxLo / Hi        box the piece is expected to stay in this substep
        escaped           that box sticks out of the one the pair list was made for (see PairsJob)
        travel            the most any of its points has moved in all substeps before this one
        need              travel + reach (see PlaneJob)

    "Expected" means: every node ends within regionMargin of the straight path from
    where it started to where the elastic solve left it. Hull points are averages of
    their cell's nodes, so the piece then stays inside the box of its 8 nodes grown
    by regionMargin. Two pieces whose boxes, grown by another skin / 2, do not
    overlap end more than skin apart. Pieces that do not behave as expected are
    found after the contact solve (LatePieceJob).
    */
    [BurstCompile]
    public struct PieceJob : IJobParallelFor
    {
        public float skin, regionMargin;
        [ReadOnly] public NativeArray<Piece> pieces;
        [ReadOnly] public NativeArray<EmbeddedVertex> hullVerts;
        [ReadOnly] public NativeArray<int> cellNodes;
        [ReadOnly] public NativeArray<float3> pos, prev;
        [ReadOnly] public NativeArray<float> lastMove;      // per node, how far it moved in the substep before this one
        [ReadOnly] public NativeArray<float3> frameLo, frameHi;

        [NativeDisableParallelForRestriction] [WriteOnly] public NativeArray<float3> hullPos;
        [WriteOnly] public NativeArray<float3> startLo, startHi, centre, boxLo, boxHi;
        [WriteOnly] public NativeArray<float> reach, need;
        [WriteOnly] public NativeArray<bool> escaped, late;
        public NativeArray<float> travel;

        public void Execute(int p)
        {
            Piece piece = pieces[p];

            float3 lo = float.MaxValue, hi = float.MinValue, sum = 0f;
            for (int v = piece.hullStart; v < piece.hullStart + piece.hullCount; v++)
            {
                float3 x = Geometry.Point(hullVerts[v], prev);
                hullPos[v] = x;
                lo = math.min(lo, x);
                hi = math.max(hi, x);
                sum += x;
            }
            startLo[p] = lo;
            startHi[p] = hi;
            centre[p] = sum / piece.hullCount;

            float furthest = 0f, before = 0f;
            lo = float.MaxValue; hi = float.MinValue;
            for (int k = 0; k < 8; k++)
            {
                int node = cellNodes[piece.cell * 8 + k];
                lo = math.min(lo, math.min(prev[node], pos[node]));
                hi = math.max(hi, math.max(prev[node], pos[node]));
                furthest = math.max(furthest, math.distance(prev[node], pos[node]));
                before = math.max(before, lastMove[node]);
            }
            lo -= regionMargin + 0.5f * skin;
            hi += regionMargin + 0.5f * skin;
            boxLo[p] = lo;
            boxHi[p] = hi;
            escaped[p] = math.any(lo < frameLo[p]) || math.any(hi > frameHi[p]);
            late[p] = false;

            float soFar = travel[p] + before;
            travel[p] = soFar;
            reach[p] = furthest + regionMargin;
            need[p] = soFar + furthest + regionMargin;
        }
    }

    /*
    After the contact solve: which pieces did not stay where they were expected?

    A contact can push a node much further than regionMargin (something fast and heavy
    hits something light). Such a piece may now reach pieces it was never paired with,
    and pairs that were judged too far apart may not be any more. A piece is late when
    one of its nodes left its region. For those, box and reach are redone from where
    the nodes really are, and the detection jobs run a second time for every pair with
    a late piece in it. The limit step then sees every pair that matters, without the
    boxes of calm pieces having to be large "just in case".

    pos is final here as far as boxes go: the limit step only takes motion away from
    a node, and what is left stays inside the new box.

    One body per work item, and only the bodies the contact solve says it pushed
    far (ContactSweepJob.pushedFar). Nearly always that is none, and none of this runs.
    */
    [BurstCompile]
    public struct LatePieceJob : IJobParallelFor
    {
        public float skin, regionMargin;
        [ReadOnly] public NativeArray<bool> pushedFar;
        [ReadOnly] public NativeArray<BodyRange> bodies;
        [ReadOnly] public NativeArray<Piece> pieces;
        [ReadOnly] public NativeArray<int> cellNodes;
        [ReadOnly] public NativeArray<float3> pos, prev, expected;
        [ReadOnly] public NativeArray<float3> frameLo, frameHi;
        [ReadOnly] public NativeArray<float> travel;
        [NativeDisableParallelForRestriction] public NativeArray<float3> boxLo, boxHi;
        [NativeDisableParallelForRestriction] public NativeArray<float> reach, need;
        [NativeDisableParallelForRestriction] public NativeArray<bool> late, escaped;

        public void Execute(int body)
        {
            if (!pushedFar[body]) return;

            for (int p = bodies[body].pieceStart; p < bodies[body].pieceStart + bodies[body].pieceCount; p++)
            {
                float3 lo = float.MaxValue, hi = float.MinValue;
                float furthest = 0f;
                bool left = false;
                for (int k = 0; k < 8; k++)
                {
                    int node = cellNodes[pieces[p].cell * 8 + k];
                    left |= Geometry.OffPath(pos[node], prev[node], expected[node]) > regionMargin;
                    lo = math.min(lo, math.min(prev[node], pos[node]));
                    hi = math.max(hi, math.max(prev[node], pos[node]));
                    furthest = math.max(furthest, math.distance(prev[node], pos[node]));
                }
                if (!left) continue;

                late[p] = true;
                boxLo[p] = math.min(boxLo[p], lo - 0.5f * skin);
                boxHi[p] = math.max(boxHi[p], hi + 0.5f * skin);
                reach[p] = furthest;
                need[p] = travel[p] + furthest;
                escaped[p] = math.any(boxLo[p] < frameLo[p]) || math.any(boxHi[p] > frameHi[p]);
            }
        }
    }

    /*
    Detection 2/4, broad phase: every pair of pieces whose boxes overlap.

    The list is not made every substep. Every piece gets a frame box: the box it
    stays in this substep, stretched along where its nodes are heading for the next
    `horizon` seconds, plus some slack so that a body that only trembles does not
    count as having left it. The pairs of overlapping frame boxes stay good for as
    long as every piece stays inside its frame box, and PieceJob checks that at the
    start of every substep (escaped). A pile that lies still keeps its list for
    seconds, a falling body gets a new one about once per frame.

    How: boxes inside boxes. A body's pieces come in chunks (the pieces of a block
    of 2 x 2 x 2 cells, see LatticeBuilder), so there are three levels, each with
    the box around everything below it:

        bodies      two bodies whose boxes do not overlap are done
        chunks      of the other body, only those that reach into this body's box,
                    and of those the ones that overlap this chunk
        pieces      of two overlapping chunks

    Most of the surface of most bodies is nowhere near another body, and is dropped
    a whole chunk or a whole body at a time.

    Three jobs:

        PairsJob        decides whether a new list is needed, and makes the boxes for it
        PairSearchJob   one body per work item: its pairs with the bodies after it
        PairListJob     puts those behind each other, body after body

    The list comes out sorted by (first piece, second piece), the same on every
    run. That order is also how a pair finds what was known about it in the list
    before (PairPlane), which is what lets PlaneJob skip most pairs.

    Second run of a substep (onlyLate), after the contact solve: only pairs with a
    late piece in them, see LatePieceJob. While the late pieces are still inside
    their frame boxes those are simply the pairs of the kept list that have a late
    piece. If one got out, the boxes are gone through again, with the boxes the
    late pieces really need.
    */
    [BurstCompile]
    public struct PairsJob : IJob
    {
        public float horizon, grow, slack;      // grow: regionMargin + skin / 2
        public bool onlyLate;
        [ReadOnly] public NativeArray<bool> escaped, late;
        [ReadOnly] public NativeList<int2> keptPairs;       // onlyLate: the list of the first run
        [ReadOnly] public NativeArray<Piece> pieces;
        [ReadOnly] public NativeArray<BodyRange> bodies;
        [ReadOnly] public NativeArray<int2> chunks;         // first piece, number of pieces
        [ReadOnly] public NativeArray<int> cellNodes;
        [ReadOnly] public NativeArray<float3> pos, vel, boxLo, boxHi;

        public NativeArray<float3> frameLo, frameHi;
        [WriteOnly] public NativeArray<float3> chunkLo, chunkHi, bodyLo, bodyHi;
        public NativeArray<int> builtFor;       // [0]: how many pieces there were when the list was made
        [WriteOnly] public NativeArray<bool> search;        // [0]: the boxes have to be gone through, PairSearchJob
        public NativeList<int2> pairs;
        public NativeList<PairPlane> planes;
        public NativeList<bool> touching;
        public NativeList<Contact> found;
        public NativeArray<int> stats;

        public void Execute()
        {
            int count = pieces.Length;
            search[0] = false;

            if (onlyLate)
            {
                bool lateEscaped = false;
                for (int p = 0; p < count; p++) lateEscaped |= late[p] && escaped[p];
                if (!lateEscaped)
                {
                    pairs.Clear();
                    for (int i = 0; i < keptPairs.Length; i++)
                        if (late[keptPairs[i].x] || late[keptPairs[i].y]) pairs.Add(keptPairs[i]);
                    Size(pairs, planes, touching, found);
                    return;
                }
            }
            else
            {
                bool anyEscaped = builtFor[0] != count;
                for (int p = 0; p < count; p++) anyEscaped |= escaped[p];
                if (!anyEscaped)
                {
                    stats[SoftStats.Pairs] += pairs.Length;
                    return;
                }

                builtFor[0] = count;
                for (int p = 0; p < count; p++)
                {
                    float3 lo = float.MaxValue, hi = float.MinValue;
                    for (int k = 0; k < 8; k++)
                    {
                        int node = cellNodes[pieces[p].cell * 8 + k];
                        float3 ahead = pos[node] + vel[node] * horizon;
                        lo = math.min(lo, ahead);
                        hi = math.max(hi, ahead);
                    }
                    frameLo[p] = math.min(boxLo[p], lo - grow) - slack;
                    frameHi[p] = math.max(boxHi[p], hi + grow) + slack;
                }
            }
            search[0] = true;

            // The boxes around the boxes to pair up (frame boxes, and in the second run the real box of every late piece)
            for (int b = 0; b < bodies.Length; b++)
            {
                float3 lowest = float.MaxValue, highest = float.MinValue;
                for (int c = bodies[b].chunkStart; c < bodies[b].chunkStart + bodies[b].chunkCount; c++)
                {
                    float3 lo = float.MaxValue, hi = float.MinValue;
                    for (int p = chunks[c].x; p < chunks[c].x + chunks[c].y; p++)
                    {
                        bool real = onlyLate && late[p];
                        lo = math.min(lo, real ? boxLo[p] : frameLo[p]);
                        hi = math.max(hi, real ? boxHi[p] : frameHi[p]);
                    }
                    chunkLo[c] = lo; chunkHi[c] = hi;
                    lowest = math.min(lowest, lo);
                    highest = math.max(highest, hi);
                }
                bodyLo[b] = lowest; bodyHi[b] = highest;
            }
        }

        // What PlaneJob fills in for every pair, with nothing known yet
        public static void Size(NativeList<int2> pairs, NativeList<PairPlane> planes, NativeList<bool> touching, NativeList<Contact> found)
        {
            planes.Clear();
            planes.Resize(pairs.Length, NativeArrayOptions.ClearMemory);
            touching.Clear();
            touching.Resize(pairs.Length, NativeArrayOptions.ClearMemory);
            found.ResizeUninitialized(pairs.Length);
        }
    }

    // The pairs of one body's pieces with the pieces of the bodies after it (and with its own, if selfCollision is on)
    [BurstCompile]
    public struct PairSearchJob : IJobParallelFor
    {
        public bool selfCollision, onlyLate;
        [ReadOnly] public NativeArray<bool> late;
        [ReadOnly] public NativeArray<Piece> pieces;
        [ReadOnly] public NativeArray<BodyRange> bodies;
        [ReadOnly] public NativeArray<int2> chunks;
        [ReadOnly] public NativeArray<int3> cellCoords;
        [ReadOnly] public NativeArray<bool> bodyStatic;
        [ReadOnly] public NativeArray<float3> boxLo, boxHi, frameLo, frameHi, chunkLo, chunkHi, bodyLo, bodyHi;
        public NativeStream.Writer bodyPairs;

        public void Execute(int a)
        {
            bodyPairs.BeginForEachIndex(a);

            // The chunks that reach into this body's box. Only of the bodies after it, so two bodies meet once
            NativeList<int> region = new NativeList<int>(64, Allocator.Temp), near = new NativeList<int>(16, Allocator.Temp);
            for (int b = selfCollision ? a : a + 1; b < bodies.Length; b++)
            {
                if (bodyStatic[a] && bodyStatic[b]) continue;
                if (!Overlap(bodyLo[a], bodyHi[a], bodyLo[b], bodyHi[b])) continue;
                for (int c = bodies[b].chunkStart; c < bodies[b].chunkStart + bodies[b].chunkCount; c++)
                    if (Overlap(bodyLo[a], bodyHi[a], chunkLo[c], chunkHi[c])) region.Add(c);
            }

            for (int ca = bodies[a].chunkStart; region.Length > 0 && ca < bodies[a].chunkStart + bodies[a].chunkCount; ca++)
            {
                // Of those, the ones that overlap this chunk (of the body's own chunks only itself and later ones)
                near.Clear();
                for (int i = 0; i < region.Length; i++)
                    if (region[i] >= ca && Overlap(chunkLo[ca], chunkHi[ca], chunkLo[region[i]], chunkHi[region[i]])) near.Add(region[i]);

                for (int p = chunks[ca].x; p < chunks[ca].x + chunks[ca].y; p++)
                {
                    bool realP = onlyLate && late[p];
                    float3 lo = realP ? boxLo[p] : frameLo[p], hi = realP ? boxHi[p] : frameHi[p];
                    for (int i = 0; i < near.Length; i++)
                    {
                        int cb = near[i];
                        if (!Overlap(lo, hi, chunkLo[cb], chunkHi[cb])) continue;

                        for (int q = math.max(chunks[cb].x, p + 1); q < chunks[cb].x + chunks[cb].y; q++)
                        {
                            bool realQ = onlyLate && late[q];
                            if (!Overlap(lo, hi, realQ ? boxLo[q] : frameLo[q], realQ ? boxHi[q] : frameHi[q])) continue;
                            if (onlyLate && !realP && !realQ) continue;
                            if (pieces[p].body == pieces[q].body && math.cmax(math.abs(cellCoords[pieces[p].cell] - cellCoords[pieces[q].cell])) < 2) continue;

                            bodyPairs.Write(new int2(p, q));
                        }
                    }
                }
            }
            bodyPairs.EndForEachIndex();
        }

        private static bool Overlap(float3 loA, float3 hiA, float3 loB, float3 hiB) => math.all(loA <= hiB) && math.all(loB <= hiA);
    }

    // The new pair list, from what PairSearchJob found for every body
    [BurstCompile]
    public struct PairListJob : IJob
    {
        public bool onlyLate;
        public NativeStream.Reader bodyPairs;
        public NativeArray<float> travel, need;
        public NativeList<int2> pairs;
        public NativeList<PairPlane> planes;
        public NativeList<bool> touching;
        public NativeList<Contact> found;
        public NativeArray<int> stats;

        public void Execute()
        {
            // The list before this one, to take over what was known about its pairs
            NativeArray<int2> oldPairs = pairs.ToArray(Allocator.Temp);
            NativeArray<PairPlane> oldPlanes = planes.ToArray(Allocator.Temp);

            pairs.Clear();
            for (int body = 0; body < bodyPairs.ForEachCount; body++)
            {
                int count = bodyPairs.BeginForEachIndex(body);
                for (int i = 0; i < count; i++) pairs.Add(bodyPairs.Read<int2>());
                bodyPairs.EndForEachIndex();
            }
            PairsJob.Size(pairs, planes, touching, found);
            if (onlyLate) return;

            // Both lists are sorted the same way, so one walk through them finds every pair that was there before.
            // A new pair starts with nothing known. travel starts again from zero here (it would lose its small
            // digits if it grew forever), so what the old pairs remember of it is taken off their budget
            for (int i = 0, old = 0; i < pairs.Length; i++)
            {
                while (old < oldPairs.Length && Before(oldPairs[old], pairs[i])) old++;
                if (old == oldPairs.Length || math.any(oldPairs[old] != pairs[i])) continue;

                PairPlane plane = oldPlanes[old];
                plane.budget -= travel[pairs[i].x] + travel[pairs[i].y];
                planes[i] = plane;
            }
            for (int p = 0; p < travel.Length; p++)
            {
                need[p] -= travel[p];
                travel[p] = 0f;
            }
            stats[SoftStats.Pairs] += pairs.Length;
        }

        private static bool Before(int2 a, int2 b) => a.x < b.x || (a.x == b.x && a.y < b.y);
    }

    /*
    Detection 3/4, narrow phase: the plane between two pieces, and the contact if they can touch.

    Most pairs of the broad phase are nowhere near touching, and they are not even
    looked at. When a pair was last found too far apart, its budget is the slab that
    was found plus how far the two pieces had travelled until then. No point of a
    piece moves further than the piece's travel says, so as long as

        budget - (travel + reach of A) - (travel + reach of B) > skin

    the two are still more than skin apart at the end of this substep, whatever
    they did in between. That is one subtraction per pair.

    For the rest, the job looks for the unit direction n that keeps all of A's hull
    points furthest above all of B's:

        slab(n) = min over A of n.x  -  max over B of n.x

    slab is a lower bound on the distance between the two hulls for any n, and the
    best n is the direction between their closest points. Pieces are convex, so the
    distance between the hulls is the distance between the pieces.

    The closest points are found with GJK. v is a point of "A minus B" (all
    differences a - b), and the closest such point to the origin is exactly
    onA - onB. Every step asks both pieces for their worst point along v (that is
    also what slab needs), adds the difference to a small simplex of at most 4 such
    points, and moves v to the point of the simplex closest to the origin. This ends
    after a few steps with the exact answer, also when two faces are parallel. The
    search starts from the n of the substep before, which is nearly right.

    When the hulls touch or overlap there is no such n, the best one seen is kept.

    Not every pair that gets this far needs a contact. The nodes of both cells are
    heading somewhere, and none of them is expected to end further than
    regionMargin from its path. If that still leaves the pieces further apart than
    the contact solve and the limit step care about, there is nothing to do for
    this substep.

    Second run of a substep (see PairsJob): the nodes are where the contact solve
    left them, so margin is 0, and the contacts are only there for the limit step.
    */
    [BurstCompile]
    public struct PlaneJob : IJobParallelFor
    {
        private const int MaxSteps = 24;
        private const float Tolerance = 0.005f;     // done when slab is within this share of the distance
        private const float Band = 0.1f;            // share of skin, see FacingSpot
        private const float IdleFrom = 2.1f;        // times skin: a pair that stays this far apart gets no spot (the idle test of MakeContact, with a little room)

        // A contact is firm when the pieces are heading closer than hardSkin + this share of the way up to skin
        private const float FirmFrom = 0.5f;

        // ... unless one of its cells is crumpled: a corner of it is left with less than this share of its volume
        private const float CrumpledVolume = 0.1f;

        public float skin, hardSkin, pushOut, cellSize;
        public float margin;        // how far a node can still end from where it is heading: regionMargin, or 0 in the second run
        [ReadOnly] public NativeArray<int2> pairs;
        [ReadOnly] public NativeArray<Piece> pieces;
        [ReadOnly] public NativeArray<float3> hullPos, startLo, startHi, centre;
        [ReadOnly] public NativeArray<float> reach, travel, need;
        [ReadOnly] public NativeArray<int> cellNodes;
        [ReadOnly] public NativeArray<float> invMass, bodyFriction;
        [ReadOnly] public NativeArray<float3> pos, prev;
        public NativeArray<PairPlane> planes;
        public NativeArray<bool> touching;
        [WriteOnly] public NativeArray<Contact> found;

        public void Execute(int i)
        {
            int a = pairs[i].x, b = pairs[i].y;
            PairPlane plane = planes[i];
            if (plane.budget - need[a] - need[b] > skin) return;

            Piece A = pieces[a], B = pieces[b];
            float limit = reach[a] + reach[b] + skin;
            float travelled = travel[a] + travel[b];
            touching[i] = false;

            float boxGap = math.cmax(math.max(startLo[a] - startHi[b], startLo[b] - startHi[a]));
            if (boxGap > limit) { plane.budget = boxGap + travelled; planes[i] = plane; return; }

            double3x4 simplex = default;        // points of A minus B
            int count = 0;

            double3 v = math.lengthsq(plane.n) > 0.5f ? (double3)plane.n : (double3)(centre[a] - centre[b]);
            float3 bestN = 0f, bestLowA = 0f, bestHighB = 0f;
            float bestSlab = float.MinValue;

            for (int step = 0; step < MaxSteps; step++)
            {
                float length = (float)math.length(v);
                if (length < 1e-9f) break;                      // the hulls touch or overlap
                float3 n = (float3)v / length;

                float slab = Slab(A, B, n, out float3 lowA, out float3 highB);
                if (slab > bestSlab) { bestSlab = slab; bestN = n; bestLowA = lowA; bestHighB = highB; }
                if (bestSlab > limit) { planes[i] = new PairPlane { n = bestN, budget = bestSlab + travelled }; return; }
                if (length - slab < Tolerance * length) break;  // v is (nearly) the closest there is

                double3 worst = (double3)lowA - (double3)highB;
                bool known = false;
                for (int k = 0; k < count; k++) known |= math.all(simplex[k] == worst);
                if (known) break;                               // nothing new to find

                simplex[count] = worst;
                v = Geometry.ClosestOnSimplex(simplex, count + 1, out double4 share);

                // Only the points that carry v stay
                int kept = 0;
                for (int k = 0; k <= count; k++)
                    if (share[k] > 0.0) simplex[kept++] = simplex[k];
                count = kept;
            }

            if (bestSlab == float.MinValue)     // the two centres are the same point, any direction is as good
            {
                bestN = new float3(0f, 1f, 0f);
                bestSlab = Slab(A, B, bestN, out bestLowA, out bestHighB);
            }

            // How much closer the two can get along n: no point moves less than the slowest node of its cell
            float closer = Geometry.LeastMotion(A.cell, bestN, cellNodes, pos, prev) + Geometry.LeastMotion(B.cell, -bestN, cellNodes, pos, prev);
            float atLeast = bestSlab + math.min(closer, 0f);
            if (atLeast > 2f * skin && atLeast - 2f * margin > hardSkin)
            {
                planes[i] = new PairPlane { n = bestN, budget = bestSlab + travelled };
                return;
            }

            planes[i] = new PairPlane { n = bestN, budget = float.NegativeInfinity };
            touching[i] = true;

            // Most contacts are heading well clear of skin. They have nothing to push and need no spot, they are only there for the limit step to check
            if (bestSlab + closer > IdleFrom * skin)
                found[i] = new Contact { kind = ContactKind.Pieces, a = a, b = b, n = math.normalize((double3)bestN), slab = bestSlab, idle = true };
            else
                found[i] = MakeContact(a, b, bestN, bestSlab, closer, FacingSpot(A, B, bestN, bestLowA, bestHighB), bestLowA, bestHighB);
        }

        // slab along n, and the two points that set it: A's lowest and B's highest
        private float Slab(in Piece A, in Piece B, float3 n, out float3 lowA, out float3 highB)
        {
            lowA = 0f; highB = 0f;
            float minA = float.MaxValue, maxB = float.MinValue;
            for (int k = A.hullStart; k < A.hullStart + A.hullCount; k++)
            {
                float d = math.dot(n, hullPos[k]);
                if (d < minA) { minA = d; lowA = hullPos[k]; }
            }
            for (int k = B.hullStart; k < B.hullStart + B.hullCount; k++)
            {
                float d = math.dot(n, hullPos[k]);
                if (d > maxB) { maxB = d; highB = hullPos[k]; }
            }
            return minA - maxB;
        }

        /*
        Where the contact pushes: the middle of the part where the two pieces face each other.

        The two closest points would be the obvious choice, but between two flat faces
        lying on each other there is no single closest point, and the one that is found
        jumps between corners from one substep to the next.

        Every point within Band of its piece's extreme along n counts as facing the
        other piece. The box of those of A is overlapped with the box of those of B
        and the middle of the overlap is the spot (if they do not overlap on an axis,
        the middle of the space between them). A point right at the extreme counts
        fully and one at the edge of the band not at all, so nothing jumps when a
        point comes into the band. For a corner against a face that is the corner, for
        two faces the middle of what they share.
        */
        private float3 FacingSpot(in Piece A, in Piece B, float3 n, float3 lowA, float3 highB)
        {
            float band = Band * skin;
            float minA = math.dot(n, lowA), maxB = math.dot(n, highB);

            float3 loA = float.MaxValue, hiA = float.MinValue, loB = float.MaxValue, hiB = float.MinValue;
            for (int k = A.hullStart; k < A.hullStart + A.hullCount; k++)
            {
                float3 p = lowA + math.saturate(1f - (math.dot(n, hullPos[k]) - minA) / band) * (hullPos[k] - lowA);
                loA = math.min(loA, p);
                hiA = math.max(hiA, p);
            }
            for (int k = B.hullStart; k < B.hullStart + B.hullCount; k++)
            {
                float3 p = highB + math.saturate(1f - (maxB - math.dot(n, hullPos[k])) / band) * (hullPos[k] - highB);
                loB = math.min(loB, p);
                hiB = math.max(hiB, p);
            }
            return 0.5f * (math.max(loA, loB) + math.min(hiA, hiB));
        }

        /*
        The contact of a pair that can touch. The spot is moved onto each side of the
        plane and turned into a spot of each cell (see Geometry.CellCoords). The gap is
        measured between those two cell spots, not between the points they came from:
        in a deformed cell they are not exactly the same place, and the difference
        would be pushed every substep.
        */
        private Contact MakeContact(int a, int b, float3 n, float slab, float closer, float3 spot, float3 lowA, float3 highB)
        {
            int cellA = pieces[a].cell, cellB = pieces[b].cell;
            float3 ua = Geometry.CellCoords(spot + n * math.dot(n, lowA - spot), cellA, cellNodes, prev);
            float3 ub = Geometry.CellCoords(spot + n * math.dot(n, highB - spot), cellB, cellNodes, prev);
            float gap = math.dot(n, Geometry.CellPoint(ua, cellA, cellNodes, prev) - Geometry.CellPoint(ub, cellB, cellNodes, prev));

            Contact c = new Contact
            {
                kind = ContactKind.Pieces, a = a, b = b, n = math.normalize((double3)n), slab = slab,
                friction = math.sqrt(bodyFriction[pieces[a].body] * bodyFriction[pieces[b].body]),
                target = ContactSweepJob.Target(gap, skin, pushOut)
            };

            // Where the elastic solve left the nodes tells which contacts have something to do:
            // a pair that will stay a skin further apart than it has to is left alone by the contact solve
            c.idle = gap + closer > c.target + skin;

            /*
            One spot cannot keep every corner of two pieces apart when they come in fast
            or turn against each other, and what the spot misses is left to the limit
            step, which can only hold back, and holds back the heavy side as much as the
            light one. So a pair that is heading for the limit is firm: it is measured
            at the hull points, and each side answers by moving its whole cell (the spot
            is the middle of the cell).

            Not while one of its cells is crumpled (squashed to a tenth of its volume or
            turned inside out). The surface in such a cell is folded up and its hull is
            much fatter than what is really there.
            */
            if (slab + closer < hardSkin + FirmFrom * (skin - hardSkin) && !Crumpled(cellA) && !Crumpled(cellB))
            {
                c.firm = true;
                c.idle = false;
                c.target = ContactSweepJob.Target(slab, skin, pushOut);
                ua = 0.5f;
                ub = 0.5f;
            }

            float perImpulseA = Spread(cellA, ua, ref c.a0, ref c.a1, ref c.spreadA);
            float perImpulseB = Spread(cellB, ub, ref c.b0, ref c.b1, ref c.spreadB);
            c.shareA = perImpulseA / math.max(perImpulseA + perImpulseB, 1e-30f);
            return c;
        }

        // Measured at two opposite corners of the cell, where it was at the start of the substep
        private bool Crumpled(int cell)
        {
            float3 a = prev[cellNodes[cell * 8]], b = prev[cellNodes[cell * 8 + 7]];
            float atA = math.dot(prev[cellNodes[cell * 8 + 1]] - a, math.cross(prev[cellNodes[cell * 8 + 2]] - a, prev[cellNodes[cell * 8 + 4]] - a));
            float atB = -math.dot(prev[cellNodes[cell * 8 + 6]] - b, math.cross(prev[cellNodes[cell * 8 + 5]] - b, prev[cellNodes[cell * 8 + 3]] - b));
            return math.min(atA, atB) < CrumpledVolume * cellSize * cellSize * cellSize;
        }

        /*
        The share of each of the cell's 8 nodes in the spot u, and the sum of their
        squares. Returns how far a unit impulse at the spot moves it when every node
        moves by its share (see ContactSweepJob): sum of share^2 / sum of share * mass.
        0 for a cell that cannot move.
        */
        private float Spread(int cell, float3 u, ref float4 low, ref float4 high, ref float spread)
        {
            float mass = 0f;
            bool moves = true;
            for (int k = 0; k < 8; k++)
            {
                float share = Geometry.CornerWeight(u, k), w = invMass[cellNodes[cell * 8 + k]];
                if (k < 4) low[k] = share; else high[k - 4] = share;
                spread += share * share;
                if (w <= 0f) moves = false; else mass += share / w;
            }
            return moves ? spread / mass : 0f;
        }
    }

    /*
    Detection 4/4: the contact list of this substep.

        Pieces      one per pair that can get within skin
        Floor       every piece that can reach the floor

    The order is fixed (pairs, then floor).

    The contact solve works one body at a time, so every moving body also gets the
    list of its own sides of the contacts that have something to push: halves, as
    contact * 2 for side A and contact * 2 + 1 for side B, bodyStart[body] says
    where a body's part of the list starts.

    Second run of a substep (onlyLate): adds the contacts of the late pieces to the
    list, for the limit step. Some are in the list already, checking a plane twice
    does no harm.
    */
    [BurstCompile]
    public struct BuildContactsJob : IJob
    {
        public float skin;
        public bool hasFloor;
        public float floorY, floorFriction;
        public bool onlyLate;
        [ReadOnly] public NativeArray<bool> late;
        [ReadOnly] public NativeList<bool> touching;
        [ReadOnly] public NativeList<Contact> found;
        [ReadOnly] public NativeArray<Piece> pieces;
        [ReadOnly] public NativeArray<int> cellNodes;
        [ReadOnly] public NativeArray<float> bodyFriction;
        [ReadOnly] public NativeArray<bool> bodyStatic;
        [ReadOnly] public NativeArray<float3> startLo, pos, prev;
        [ReadOnly] public NativeArray<float> reach;

        public NativeList<Contact> contacts;
        public NativeList<int> halves;
        public NativeList<float> pushed;        // per half, sized here for ContactSweepJob
        public NativeArray<int> bodyStart;
        public NativeArray<int> stats;

        public void Execute()
        {
            if (!onlyLate) contacts.Clear();
            for (int i = 0; i < touching.Length; i++)
            {
                if (!touching[i]) continue;
                contacts.Add(found[i]);
                stats[SoftStats.PieceContacts]++;
            }

            for (int p = 0; hasFloor && p < pieces.Length; p++)
            {
                if (bodyStatic[pieces[p].body] || (onlyLate && !late[p])) continue;
                float gap = startLo[p].y - floorY;
                if (gap - skin > reach[p]) continue;

                contacts.Add(new Contact
                {
                    kind = ContactKind.Floor, a = p, b = -1,
                    n = new double3(0.0, 1.0, 0.0), slab = gap, planeD = floorY,
                    friction = math.sqrt(bodyFriction[pieces[p].body] * floorFriction),
                    idle = gap + Geometry.LeastMotion(pieces[p].cell, new float3(0f, 1f, 0f), cellNodes, pos, prev) > 2f * skin      // a target is never above skin
                });
                stats[SoftStats.FloorContacts]++;
            }
            if (onlyLate) return;

            // Halves per body: count them, turn the counts into where each body starts, fill
            for (int b = 0; b < bodyStart.Length; b++) bodyStart[b] = 0;
            for (int i = 0; i < contacts.Length; i++)
            {
                Contact c = contacts[i];
                if (c.idle) continue;
                stats[SoftStats.ActiveContacts]++;
                if (!bodyStatic[pieces[c.a].body]) bodyStart[pieces[c.a].body + 1]++;
                if (c.kind == ContactKind.Pieces && !bodyStatic[pieces[c.b].body]) bodyStart[pieces[c.b].body + 1]++;
            }
            for (int b = 1; b < bodyStart.Length; b++) bodyStart[b] += bodyStart[b - 1];

            halves.ResizeUninitialized(bodyStart[bodyStart.Length - 1]);
            NativeArray<int> next = new NativeArray<int>(bodyStart.Length, Allocator.Temp);
            for (int i = 0; i < contacts.Length; i++)
            {
                Contact c = contacts[i];
                if (c.idle) continue;
                int bodyA = pieces[c.a].body;
                if (!bodyStatic[bodyA]) halves[bodyStart[bodyA] + next[bodyA]++] = i * 2;
                if (c.kind != ContactKind.Pieces) continue;
                int bodyB = pieces[c.b].body;
                if (!bodyStatic[bodyB]) halves[bodyStart[bodyB] + next[bodyB]++] = i * 2 + 1;
            }

            pushed.Clear();
            pushed.Resize(contacts.Length * 2, NativeArrayOptions.ClearMemory);
        }
    }
}
