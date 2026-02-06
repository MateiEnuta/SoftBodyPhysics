using Unity.Mathematics;

public static class CCDMath
{
#region Triangle-Point check
    static bool PointInTriangleBarycentric(double3 p, double3 a, double3 b, double3 c, double epsBary)
    {
        // Barycentric in 3D using dot products (works as long as triangle isn't degenerate)
        double3 v0 = b - a;
        double3 v1 = c - a;
        double3 v2 = p - a;

        double d00 = math.dot(v0, v0);
        double d01 = math.dot(v0, v1);
        double d11 = math.dot(v1, v1);
        double d20 = math.dot(v2, v0);
        double d21 = math.dot(v2, v1);

        double denom = d00 * d11 - d01 * d01;
        if (math.abs(denom) < 1e-30) return false; // degenerate

        double v = (d11 * d20 - d01 * d21) / denom;
        double w = (d00 * d21 - d01 * d20) / denom;
        double u = 1.0 - v - w;

        return (u >= -epsBary) && (v >= -epsBary) && (w >= -epsBary);
    }

    /// <summary>
    /// Boolean vertex–triangle CCD for linear motion over t in [0,1].
    /// Detects if moving point p(t) intersects moving triangle (a(t),b(t),c(t)).
    /// Skips the coplanar-for-entire-step case (cubic identically ~0).
    /// </summary>
    public static bool PointTriangleCCD_Boolean(
        float3 p0, float3 a0, float3 b0, float3 c0,
        float3 p1, float3 a1, float3 b1, float3 c1,
        float epsBary = 1e-6f,
        float epsParallel = 1e-14f,
        float epsPlane = 1e-6f)
    {
        double3 P0 = D(p0), A0 = D(a0), B0 = D(b0), C0 = D(c0);
        double3 P1 = D(p1), A1 = D(a1), B1 = D(b1), C1 = D(c1);

        // Build cubic for:
        // s(t) = (p(t)-a(t)) · ( (b(t)-a(t)) × (c(t)-a(t)) ) = 0
        // With linear vertex motion, s(t) is cubic. [web:63]
        double3 r0 = P0 - A0;
        double3 r1 = (P1 - P0) - (A1 - A0);

        double3 e10 = B0 - A0;                           // b-a at t0
        double3 e11 = (B1 - B0) - (A1 - A0);             // delta(b-a) over step
        double3 e20 = C0 - A0;                           // c-a at t0
        double3 e21 = (C1 - C0) - (A1 - A0);             // delta(c-a) over step

        // n(t) = e1(t) x e2(t) = n0 + n1 t + n2 t^2
        double3 n0 = math.cross(e10, e20);
        double3 n1 = math.cross(e10, e21) + math.cross(e11, e20);
        double3 n2 = math.cross(e11, e21);

        double c0p = math.dot(r0, n0);
        double c1p = math.dot(r0, n1) + math.dot(r1, n0);
        double c2p = math.dot(r0, n2) + math.dot(r1, n1);
        double c3p = math.dot(r1, n2);

        // Entire-step coplanar/degenerate -> skip (per your requirements)
        if (math.abs(c0p) + math.abs(c1p) + math.abs(c2p) + math.abs(c3p) < 1e-18)
            return false;

        double3 roots;
        int3 valid;
        int nRoots = SolveCubic01(c0p, c1p, c2p, c3p, out roots, out valid);
        if (nRoots == 0) return false;

        double epsB = (double)epsBary;
        double epsPar = (double)epsParallel;
        double epsPl = (double)epsPlane;

        if (valid.x != 0 && HitAtT(roots.x)) return true;
        if (valid.y != 0 && HitAtT(roots.y)) return true;
        if (valid.z != 0 && HitAtT(roots.z)) return true;

        return false;

        bool HitAtT(double t)
        {
            // Interpolate configuration at t
            double3 p = math.lerp(P0, P1, t);
            double3 a = math.lerp(A0, A1, t);
            double3 b = math.lerp(B0, B1, t);
            double3 c = math.lerp(C0, C1, t);

            // Reject degenerate triangle
            double3 nrm = math.cross(b - a, c - a);
            double n2len = math.dot(nrm, nrm);
            if (n2len < epsPar) return false;

            // Numeric safety: p should be on the plane at a root, but allow a tolerance
            double planeMeasure = math.abs(math.dot(p - a, nrm));
            if (planeMeasure > epsPl * math.sqrt(n2len)) return false;

            // Inside-triangle test (including edges)
            return PointInTriangleBarycentric(p, a, b, c, epsB);
        }
    }
#endregion

#region Edge-Edge check
    /// <summary>
    /// Boolean edge-edge CCD for linear motion over t in [0,1].
    /// Skips the coplanar-for-entire-step case (cubic identically ~0).
    /// </summary>
    public static bool EdgeEdgeCCD_Boolean(
        float3 a0_0, float3 a1_0, float3 b0_0, float3 b1_0,
        float3 a0_1, float3 a1_1, float3 b0_1, float3 b1_1,
        float epsPos = 1e-5f,
        float epsParallel = 1e-12f)
    {
        // Convert once
        double3 A0_0 = D(a0_0), A1_0 = D(a1_0), B0_0 = D(b0_0), B1_0 = D(b1_0);
        double3 A0_1 = D(a0_1), A1_1 = D(a1_1), B0_1 = D(b0_1), B1_1 = D(b1_1);

        // Build cubic coefficients for s(t) = r(t) · (da(t) × db(t)) = 0
        double3 r0 = B0_0 - A0_0;
        double3 r1 = (B0_1 - B0_0) - (A0_1 - A0_0);

        double3 da0 = A1_0 - A0_0;
        double3 da1 = (A1_1 - A1_0) - (A0_1 - A0_0);

        double3 db0 = B1_0 - B0_0;
        double3 db1 = (B1_1 - B1_0) - (B0_1 - B0_0);

        double3 n0 = math.cross(da0, db0);
        double3 n1 = math.cross(da0, db1) + math.cross(da1, db0);
        double3 n2 = math.cross(da1, db1);

        double c0 = math.dot(r0, n0);
        double c1 = math.dot(r0, n1) + math.dot(r1, n0);
        double c2 = math.dot(r0, n2) + math.dot(r1, n1);
        double c3 = math.dot(r1, n2);

        // If it's identically ~0, that's coplanar/degenerate case, can be ignored
        if (math.abs(c0) + math.abs(c1) + math.abs(c2) + math.abs(c3) < 1e-18)
            return false;

        // Solve cubic in [0,1] with fixed-size outputs
        double3 roots;
        int3 valid;
        int n = SolveCubic01(c0, c1, c2, c3, out roots, out valid);

        if (n == 0) return false;

        double epsPos2 = (double)epsPos * (double)epsPos;
        double epsPar = (double)epsParallel;

        // Fixed-size loop (unrolled via valid mask)
        // root 0
        if (valid.x != 0 && IntersectAtT(roots.x)) return true;
        if (valid.y != 0 && IntersectAtT(roots.y)) return true;
        if (valid.z != 0 && IntersectAtT(roots.z)) return true;

        return false;

        bool IntersectAtT(double t)
        {
            // Lerp endpoints at time t
            double3 a0 = math.lerp(A0_0, A0_1, t);
            double3 a1 = math.lerp(A1_0, A1_1, t);
            double3 b0 = math.lerp(B0_0, B0_1, t);
            double3 b1 = math.lerp(B1_0, B1_1, t);

            double3 da = a1 - a0;
            double3 db = b1 - b0;
            double3 r  = b0 - a0;

            double3 nrm = math.cross(da, db);
            double denom = math.dot(nrm, nrm);

            // Skip near-parallel/coplanar-degenerate validation
            if (denom < epsPar) return false;

            // Line-line intersection parameters in coplanar case
            double u = math.dot(math.cross(r, db), nrm) / denom;
            double v = math.dot(math.cross(r, da), nrm) / denom;

            if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) return false;

            double3 p = a0 + u * da;
            double3 q = b0 + v * db;

            double3 d = p - q;
            return math.dot(d, d) <= epsPos2;
        }
    }
#endregion

#region Cubic root solve
    // --- small helpers ---
    static double3 D(float3 v) => new double3(v.x, v.y, v.z);
    static double Cbrt(double x)
    {
        // Burst-friendly cube root
        if (x == 0.0) return 0.0;
        double ax = math.abs(x);
        double y = math.pow(ax, 1.0 / 3.0);
        return (x < 0.0) ? -y : y;
    }

    static bool In01(double t, double eps) => (t >= -eps) && (t <= 1.0 + eps);

    static double Clamp01(double t) => math.min(1.0, math.max(0.0, t));

    static void AddRootSorted01(double t, double eps, ref double3 roots, ref int count)
    {
        if (!In01(t, eps)) return;
        t = Clamp01(t);

        // Insert-sort into roots[0..count)
        if (count == 0)
        {
            roots.x = t; count = 1; return;
        }
        if (count == 1)
        {
            if (t < roots.x) { roots.y = roots.x; roots.x = t; }
            else roots.y = t;
            count = 2; return;
        }

        // count == 2
        double a = roots.x, b = roots.y;
        if (t < a) { roots.z = b; roots.y = a; roots.x = t; }
        else if (t < b) { roots.z = b; roots.y = t; }
        else roots.z = t;
        count = 3;
    }

    static int SolveLinear01(double c0, double c1, double epsT, out double3 roots)
    {
        roots = 0;
        if (math.abs(c1) < 1e-18) return 0; // constant
        double t = -c0 / c1;
        int count = 0;
        AddRootSorted01(t, epsT, ref roots, ref count);
        return count;
    }

    static int SolveQuadratic01(double c0, double c1, double c2, double epsT, out double3 roots)
    {
        roots = 0;
        if (math.abs(c2) < 1e-18) return SolveLinear01(c0, c1, epsT, out roots);

        // c2 t^2 + c1 t + c0 = 0
        double disc = c1 * c1 - 4.0 * c2 * c0;
        if (disc < 0.0) return 0;

        double s = math.sqrt(disc);

        // Use a more stable quadratic form
        double q = -0.5 * (c1 + math.sign(c1) * s);
        double t0 = q / c2;
        double t1 = (math.abs(q) < 1e-18) ? (-c1 / (2.0 * c2)) : (c0 / q);

        int count = 0;
        AddRootSorted01(t0, epsT, ref roots, ref count);
        AddRootSorted01(t1, epsT, ref roots, ref count);
        return count;
    }

    /// <summary>
    /// Solves c0 + c1 t + c2 t^2 + c3 t^3 = 0, returns roots in [0,1] only.
    /// Output: roots (double3), valid mask (int3), and count.
    /// </summary>
    public static int SolveCubic01(double c0, double c1, double c2, double c3,
                                   out double3 roots, out int3 valid,
                                   double epsT = 1e-12)
    {
        roots = 0;
        valid = 0;

        // Degenerate to quadratic/linear
        if (math.abs(c3) < 1e-18)
        {
            int nQ = SolveQuadratic01(c0, c1, c2, epsT, out roots);
            valid = new int3(nQ > 0 ? 1 : 0, nQ > 1 ? 1 : 0, nQ > 2 ? 1 : 0);
            return nQ;
        }

        // Normalize to monic: t^3 + a t^2 + b t + c = 0
        double inv = 1.0 / c3;
        double a = c2 * inv;
        double b = c1 * inv;
        double c = c0 * inv;

        // Depress: t = y - a/3  => y^3 + p y + q = 0
        double a3 = a / 3.0;
        double p = b - a * a3;                        // b - a^2/3
        double q = (2.0 * a * a * a) / 27.0 - (a * b) / 3.0 + c;

        double halfQ = 0.5 * q;
        double thirdP = p / 3.0;
        double disc = halfQ * halfQ + thirdP * thirdP * thirdP;

        int count = 0;
        if (disc > 0.0)
        {
            // One real root
            double s = math.sqrt(disc);
            double u = Cbrt(-halfQ + s);
            double v = Cbrt(-halfQ - s);
            double y = u + v;
            double t = y - a3;
            AddRootSorted01(t, epsT, ref roots, ref count);
        }
        else
        {
            // Three real roots (or multiple roots when disc == 0)
            // If p is ~0, y^3 + q = 0
            if (math.abs(p) < 1e-18)
            {
                double y = Cbrt(-q);
                double t = y - a3;
                AddRootSorted01(t, epsT, ref roots, ref count);
            }
            else
            {
                double r = 2.0 * math.sqrt(-thirdP);   // 2*sqrt(-p/3)
                double cosArg = (-halfQ) / math.sqrt(-(thirdP * thirdP * thirdP));
                cosArg = math.clamp(cosArg, -1.0, 1.0);
                double phi = math.acos(cosArg);

                const double TWO_PI = 6.283185307179586476925286766559;
                double y0 = r * math.cos(phi / 3.0);
                double y1 = r * math.cos((phi + TWO_PI) / 3.0);
                double y2 = r * math.cos((phi + 2.0 * TWO_PI) / 3.0);

                AddRootSorted01(y0 - a3, epsT, ref roots, ref count);
                AddRootSorted01(y1 - a3, epsT, ref roots, ref count);
                AddRootSorted01(y2 - a3, epsT, ref roots, ref count);
            }
        }

        valid = new int3(count > 0 ? 1 : 0, count > 1 ? 1 : 0, count > 2 ? 1 : 0);
        return count;
    }
#endregion
}
