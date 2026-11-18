using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace SoftEngine.Testing
{
    /*
    Measures a SoftSolver from the outside, for tests only. It shares nothing with the
    engine's own contact list: every pair of pieces is looked at, so a pair the engine
    missed shows up here.
    */
    [BurstCompile]
    public struct CheckJob : IJob
    {
        public const int MinDistance = 0, Crossings = 1, Inverted = 2, MinJ = 3, MaxEdge = 4, Kinetic = 5, Potential = 6, Elastic = 7,
            MomentumX = 8, MomentumY = 9, MomentumZ = 10, SelfMinDistance = 11, SelfCrossings = 12, LowestVertex = 13, BadNodes = 14, Count = 15;

        public float nearDistance;      // pieces whose boxes are further apart are not measured
        public float cellSize;
        public float3 gravity;
        [ReadOnly] public NativeArray<Piece> pieces;
        [ReadOnly] public NativeArray<EmbeddedVertex> colVerts;
        [ReadOnly] public NativeArray<int3> colTris;
        [ReadOnly] public NativeArray<int> cellNodes, cellBody;
        [ReadOnly] public NativeArray<int3> cellCoords;
        [ReadOnly] public NativeArray<float> cellMu, cellLambda, invMass;
        [ReadOnly] public NativeArray<bool> bodyStatic;
        [ReadOnly] public NativeArray<float3> pos, vel;
        public NativeArray<double> result;

        public void Execute()
        {
            for (int i = 0; i < Count; i++) result[i] = 0.0;
            result[MinDistance] = result[SelfMinDistance] = result[MinJ] = result[LowestVertex] = double.MaxValue;

            // Collision vertices and piece boxes, as they are now
            NativeArray<double3> verts = new NativeArray<double3>(colVerts.Length, Allocator.Temp);
            NativeArray<double3> lo = new NativeArray<double3>(pieces.Length, Allocator.Temp);
            NativeArray<double3> hi = new NativeArray<double3>(pieces.Length, Allocator.Temp);
            for (int p = 0; p < pieces.Length; p++)
            {
                lo[p] = double.MaxValue; hi[p] = double.MinValue;
                for (int v = pieces[p].vertStart; v < pieces[p].vertStart + pieces[p].vertCount; v++)
                {
                    verts[v] = Geometry.PointD(colVerts[v], pos);
                    lo[p] = math.min(lo[p], verts[v]);
                    hi[p] = math.max(hi[p], verts[v]);
                }
                if (!bodyStatic[pieces[p].body]) result[LowestVertex] = math.min(result[LowestVertex], lo[p].y);
            }

            // Distance and crossings between pieces
            for (int p = 0; p < pieces.Length; p++)
            for (int q = p + 1; q < pieces.Length; q++)
            {
                if (math.cmax(math.max(lo[p] - hi[q], lo[q] - hi[p])) > nearDistance) continue;

                bool self = pieces[p].body == pieces[q].body;
                if (self && math.cmax(math.abs(cellCoords[pieces[p].cell] - cellCoords[pieces[q].cell])) < 2) continue;
                if (!self && bodyStatic[pieces[p].body] && bodyStatic[pieces[q].body]) continue;

                for (int ta = pieces[p].triStart; ta < pieces[p].triStart + pieces[p].triCount; ta++)
                for (int tb = pieces[q].triStart; tb < pieces[q].triStart + pieces[q].triCount; tb++)
                {
                    double3 a0 = verts[colTris[ta].x], a1 = verts[colTris[ta].y], a2 = verts[colTris[ta].z];
                    double3 b0 = verts[colTris[tb].x], b1 = verts[colTris[tb].y], b2 = verts[colTris[tb].z];

                    bool crossing = Hits(a0, a1, b0, b1, b2) || Hits(a1, a2, b0, b1, b2) || Hits(a2, a0, b0, b1, b2)
                                 || Hits(b0, b1, a0, a1, a2) || Hits(b1, b2, a0, a1, a2) || Hits(b2, b0, a0, a1, a2);
                    double distance = crossing ? 0.0 : ClosestOnTriangles(a0, a1, a2, b0, b1, b2, out _, out _);

                    if (crossing) result[self ? SelfCrossings : Crossings] += 1.0;
                    int slot = self ? SelfMinDistance : MinDistance;
                    result[slot] = math.min(result[slot], distance);
                }
            }

            // Cells: inverted tets, stretch, elastic energy
            double tetVolume = (double)cellSize * cellSize * cellSize / 6.0;
            for (int cell = 0; cell < cellCoords.Length; cell++)
            {
                if (bodyStatic[cellBody[cell]]) continue;

                for (int k = 0; k < 8; k++)
                for (int axis = 0; axis < 3; axis++)
                {
                    if ((k >> axis & 1) != 0) continue;
                    double length = math.distance(pos[cellNodes[cell * 8 + k]], pos[cellNodes[cell * 8 + (k | 1 << axis)]]);
                    result[MaxEdge] = math.max(result[MaxEdge], length);
                }

                int variant = Lattice.Variant(cellCoords[cell]);
                for (int t = 0; t < 5; t++)
                {
                    int4 tet = Lattice.Tet(variant, t);
                    double3 x0 = pos[cellNodes[cell * 8 + tet.x]], q0 = Lattice.Corner(tet.x) * cellSize;
                    double3x3 now = new double3x3(pos[cellNodes[cell * 8 + tet.y]] - x0, pos[cellNodes[cell * 8 + tet.z]] - x0, pos[cellNodes[cell * 8 + tet.w]] - x0);
                    double3x3 rest = new double3x3(Lattice.Corner(tet.y) * cellSize - q0, Lattice.Corner(tet.z) * cellSize - q0, Lattice.Corner(tet.w) * cellSize - q0);
                    double3x3 F = math.mul(now, math.inverse(rest));

                    double J = math.determinant(F);
                    result[MinJ] = math.min(result[MinJ], J);
                    if (J <= 0.0) result[Inverted] += 1.0;

                    double volume = t == 0 ? 2.0 * tetVolume : tetVolume;
                    double shape = math.max(math.lengthsq(F.c0) + math.lengthsq(F.c1) + math.lengthsq(F.c2) + J * J - 4.0 * J, 0.0);
                    result[Elastic] += 0.5 * volume * (cellMu[cell] * shape + cellLambda[cell] * (J - 1.0) * (J - 1.0));
                }
            }

            // Nodes: energy and momentum
            for (int i = 0; i < pos.Length; i++)
            {
                if (!math.all(math.isfinite(pos[i]))) result[BadNodes] += 1.0;
                if (invMass[i] <= 0f) continue;
                double mass = 1.0 / invMass[i];
                double3 v = vel[i];
                result[Kinetic] += 0.5 * mass * math.lengthsq(v);
                result[Potential] -= mass * math.dot((double3)gravity, (double3)pos[i]);
                result[MomentumX] += mass * v.x;
                result[MomentumY] += mass * v.y;
                result[MomentumZ] += mass * v.z;
            }
        }

        // Closest points of segments (p1, q1) and (p2, q2), Ericson 5.1.9
        private static void ClosestOnSegments(double3 p1, double3 q1, double3 p2, double3 q2, out double3 c1, out double3 c2)
        {
            double3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
            double a = math.dot(d1, d1), e = math.dot(d2, d2), f = math.dot(d2, r);
            double s, t;

            if (a <= 1e-300 && e <= 1e-300) { c1 = p1; c2 = p2; return; }
            if (a <= 1e-300)
            {
                s = 0.0;
                t = math.saturate(f / e);
            }
            else
            {
                double c = math.dot(d1, r);
                if (e <= 1e-300)
                {
                    t = 0.0;
                    s = math.saturate(-c / a);
                }
                else
                {
                    double b = math.dot(d1, d2);
                    double denom = a * e - b * b;
                    s = denom > 1e-12 * a * e ? math.saturate((b * f - c * e) / denom) : 0.0;
                    t = (b * s + f) / e;
                    if (t < 0.0) { t = 0.0; s = math.saturate(-c / a); }
                    else if (t > 1.0) { t = 1.0; s = math.saturate((b - c) / a); }
                }
            }
            c1 = p1 + d1 * s;
            c2 = p2 + d2 * t;
        }

        /*
        Closest points of two triangles that do not intersect: always a vertex of one
        against the other, or an edge against an edge. Returns the distance.
        For intersecting triangles the result is not a separating direction, the caller checks.
        */
        private static double ClosestOnTriangles(
            double3 a0, double3 a1, double3 a2, double3 b0, double3 b1, double3 b2, out double3 onA, out double3 onB)
        {
            double best = double.MaxValue;
            onA = a0; onB = b0;

            for (int i = 0; i < 3; i++)
            {
                double3 p = Pick(i, a0, a1, a2);
                double3 q = Geometry.ClosestOnTriangle(p, b0, b1, b2);
                double d = math.lengthsq(p - q);
                if (d < best) { best = d; onA = p; onB = q; }

                p = Pick(i, b0, b1, b2);
                q = Geometry.ClosestOnTriangle(p, a0, a1, a2);
                d = math.lengthsq(p - q);
                if (d < best) { best = d; onA = q; onB = p; }
            }

            for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                ClosestOnSegments(Pick(i, a0, a1, a2), Pick((i + 1) % 3, a0, a1, a2),
                                  Pick(j, b0, b1, b2), Pick((j + 1) % 3, b0, b1, b2), out double3 p, out double3 q);
                double d = math.lengthsq(p - q);
                if (d < best) { best = d; onA = p; onB = q; }
            }
            return math.sqrt(best);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double3 Pick(int i, double3 a, double3 b, double3 c) => i == 0 ? a : i == 1 ? b : c;

        // True when segment (p, q) goes through triangle (a, b, c)
        private static bool Hits(double3 p, double3 q, double3 a, double3 b, double3 c)
        {
            double3 n = math.cross(b - a, c - a);
            double dp = math.dot(n, p - a), dq = math.dot(n, q - a);
            if (dp * dq >= 0.0) return false;

            double3 x = p + (q - p) * (dp / (dp - dq));
            return math.dot(n, math.cross(b - a, x - a)) >= 0.0 && math.dot(n, math.cross(c - b, x - b)) >= 0.0
                && math.dot(n, math.cross(a - c, x - c)) >= 0.0;
        }
    }
}
