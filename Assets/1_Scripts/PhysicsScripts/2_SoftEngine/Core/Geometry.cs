using System.Runtime.CompilerServices;
using Unity.Collections;
using Unity.Mathematics;

namespace SoftEngine
{
    /*
    Geometry shared by the jobs.

    The limit step works in double. Node positions stay float, but a plane that is
    checked every substep with float rounding loses a little distance every time two
    pieces are pressed together, and over minutes that adds up.
    */
    public static class Geometry
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float3 Point(in EmbeddedVertex v, NativeArray<float3> x)
        {
            return v.w.x * x[v.nodes.x] + v.w.y * x[v.nodes.y] + v.w.z * x[v.nodes.z] + v.w.w * x[v.nodes.w];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double3 PointD(in EmbeddedVertex v, NativeArray<float3> x)
        {
            return (double)v.w.x * (double3)x[v.nodes.x] + (double)v.w.y * (double3)x[v.nodes.y]
                 + (double)v.w.z * (double3)x[v.nodes.z] + (double)v.w.w * (double3)x[v.nodes.w];
        }

        /*
        Where p sits in a cell: 3 numbers in [0, 1], measured along the cell's three edges
        from corner 0 (exact while the cell is a sheared box, close enough otherwise).
        CellPoint turns them back into a position for any node positions, so a contact
        can follow "the same spot of the cell" and push it through all 8 nodes.
        */
        public static float3 CellCoords(float3 p, int cell, NativeArray<int> cellNodes, NativeArray<float3> x)
        {
            float3 corner = x[cellNodes[cell * 8]];
            float3x3 edges = new float3x3(x[cellNodes[cell * 8 + 1]] - corner, x[cellNodes[cell * 8 + 2]] - corner, x[cellNodes[cell * 8 + 4]] - corner);
            if (math.abs(math.determinant(edges)) < 1e-12f) return 0.5f;
            return math.saturate(math.mul(math.inverse(edges), p - corner));
        }

        // Share of corner k in a point with cell coordinates u
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float CornerWeight(float3 u, int k)
        {
            return ((k & 1) != 0 ? u.x : 1f - u.x) * ((k & 2) != 0 ? u.y : 1f - u.y) * ((k & 4) != 0 ? u.z : 1f - u.z);
        }

        public static float3 CellPoint(float3 u, int cell, NativeArray<int> cellNodes, NativeArray<float3> x)
        {
            float3 p = 0f;
            for (int k = 0; k < 8; k++) p += CornerWeight(u, k) * x[cellNodes[cell * 8 + k]];
            return p;
        }

        // The least any node of the cell moved along dir this substep. No point of the cell moved less
        public static float LeastMotion(int cell, float3 dir, NativeArray<int> cellNodes, NativeArray<float3> pos, NativeArray<float3> prev)
        {
            float least = float.MaxValue;
            for (int k = 0; k < 8; k++)
            {
                int node = cellNodes[cell * 8 + k];
                least = math.min(least, math.dot(dir, pos[node] - prev[node]));
            }
            return least;
        }

        // How far x is from the straight path between start and end
        public static float OffPath(float3 x, float3 start, float3 end)
        {
            float3 path = end - start;
            return math.distance(x, start + path * math.saturate(math.dot(x - start, path) / math.max(math.lengthsq(path), 1e-20f)));
        }

        // Closest point to p on triangle (a, b, c). A flat triangle returns a point on one of its edges
        public static double3 ClosestOnTriangle(double3 p, double3 a, double3 b, double3 c)
        {
            double3 w = TriangleWeights(p, a, b, c);
            return a * w.x + b * w.y + c * w.z;
        }

        // The shares of a, b and c in the point of triangle (a, b, c) closest to p, Ericson 5.1.5
        public static double3 TriangleWeights(double3 p, double3 a, double3 b, double3 c)
        {
            double3 ab = b - a, ac = c - a, ap = p - a;
            double d1 = math.dot(ab, ap), d2 = math.dot(ac, ap);
            if (d1 <= 0.0 && d2 <= 0.0) return new double3(1.0, 0.0, 0.0);

            double3 bp = p - b;
            double d3 = math.dot(ab, bp), d4 = math.dot(ac, bp);
            if (d3 >= 0.0 && d4 <= d3) return new double3(0.0, 1.0, 0.0);

            double vc = d1 * d4 - d3 * d2;
            if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return new double3(1.0 - d1 / (d1 - d3), d1 / (d1 - d3), 0.0);

            double3 cp = p - c;
            double d5 = math.dot(ab, cp), d6 = math.dot(ac, cp);
            if (d6 >= 0.0 && d5 <= d6) return new double3(0.0, 0.0, 1.0);

            double vb = d5 * d2 - d1 * d6;
            if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return new double3(1.0 - d2 / (d2 - d6), 0.0, d2 / (d2 - d6));

            double va = d3 * d6 - d5 * d4;
            if (va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0)
            {
                double t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                return new double3(0.0, 1.0 - t, t);
            }

            double sum = va + vb + vc;
            if (sum <= 1e-300) return new double3(1.0, 0.0, 0.0);
            return new double3(va, vb, vc) / sum;
        }

        /*
        Closest point to the origin of a simplex: the first `count` columns of p, so a
        point, a segment, a triangle or a tetrahedron. w gets the share of every column
        in that point (0 for the ones it does not need). When the origin is inside a
        tetrahedron the result is zero and all shares are zero.
        */
        public static double3 ClosestOnSimplex(double3x4 p, int count, out double4 w)
        {
            w = new double4(1.0, 0.0, 0.0, 0.0);
            if (count == 1) return p.c0;

            if (count == 2)
            {
                double3 d = p.c1 - p.c0;
                double t = math.saturate(-math.dot(p.c0, d) / math.max(math.lengthsq(d), 1e-300));
                w = new double4(1.0 - t, t, 0.0, 0.0);
                return p.c0 + d * t;
            }

            if (count == 3)
            {
                double3 t = TriangleWeights(0.0, p.c0, p.c1, p.c2);
                w = new double4(t, 0.0);
                return p.c0 * t.x + p.c1 * t.y + p.c2 * t.z;
            }

            // Tetrahedron: the closest of its faces that have the origin on their outer side
            w = 0.0;
            double3 closest = 0.0;
            double best = double.MaxValue;
            for (int skip = 0; skip < 4; skip++)
            {
                int i = skip == 0 ? 1 : 0, j = skip <= 1 ? 2 : 1, k = skip == 3 ? 2 : 3;       // the face without point `skip`
                double3 normal = math.cross(p[j] - p[i], p[k] - p[i]);
                if (math.dot(normal, -p[i]) * math.dot(normal, p[skip] - p[i]) > 0.0) continue;  // origin on the inner side

                double3 t = TriangleWeights(0.0, p[i], p[j], p[k]);
                double3 q = p[i] * t.x + p[j] * t.y + p[k] * t.z;
                if (math.lengthsq(q) >= best) continue;

                best = math.lengthsq(q);
                closest = q;
                w = 0.0;
                w[i] = t.x; w[j] = t.y; w[k] = t.z;
            }
            return closest;
        }
    }
}
