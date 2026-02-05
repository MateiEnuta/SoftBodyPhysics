using Unity.Mathematics;

public static class CCDMath
{
    // --- small helpers ---
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
}
