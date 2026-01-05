using System;
using System.Collections.Generic;
using UnityEngine;

public class PrimitiveCollisionTest : MonoBehaviour
{
    public Color TriangleColor = Color.red;
    public Color PointColor = Color.green;
    public Color velocityColor = Color.blue;
    public float dt = 0.02f, pointRadius = 0.05f;

    public bool triCase = true;

    public Transform A, B, C;
    public Transform Avel, Bvel, Cvel;
    public Transform P, Pvel;

    private Vector3 triA, triB, triC;    // triangle at t=0
    private Vector3 velA, velB, velC;   // triangle velocities

    private Vector3 pointPos, pointVel;       // point

    public Transform E1, E2, E3, E4;
    public Transform E1vel, E2vel, E3vel, E4vel;

    void OnDrawGizmos()
    {
        if (triCase)
        {
            triA = A.position;
            triB = B.position;
            triC = C.position;
            pointPos = P.position;

            velA = Avel.position;
            velB = Bvel.position;
            velC = Cvel.position;
            pointVel = Pvel.position;

            Gizmos.color = TriangleColor;

            Gizmos.DrawSphere(triA, pointRadius);
            Gizmos.DrawSphere(triB, pointRadius);
            Gizmos.DrawSphere(triC, pointRadius);

            Gizmos.DrawLine(triA, triB);
            Gizmos.DrawLine(triB, triC);
            Gizmos.DrawLine(triC, triA);

            Gizmos.color = PointColor;
            Gizmos.DrawSphere(pointPos, pointRadius);

            Gizmos.color = velocityColor;
            Gizmos.DrawLine(triA, triA + velA * dt);
            Gizmos.DrawLine(triB, triB + velB * dt);
            Gizmos.DrawLine(triC, triC + velC * dt);
            Gizmos.DrawLine(pointPos, pointPos + pointVel * dt);

            Gizmos.DrawLine(triA + velA * dt, triB + velB * dt);
            Gizmos.DrawLine(triB + velB * dt, triC + velC * dt);
            Gizmos.DrawLine(triC + velC * dt, triA + velA * dt);

            bool willCollide = PointTriangleCCD.PointHitsMovingTriangle(
                pointPos, pointVel,
                triA, triB, triC,
                velA, velB, velC,
                dt);

            if (willCollide) Debug.Log("Will collide");
            else Debug.Log("Wouldn't collide");
        }
        else
        {
            Gizmos.color = PointColor;
            Gizmos.DrawLine(E1.position, E2.position);
            Gizmos.DrawLine(E3.position, E4.position);

            Gizmos.color = velocityColor;
            Gizmos.DrawLine(E1.position, E1.position + E1vel.position * dt);
            Gizmos.DrawLine(E2.position, E2.position + E2vel.position * dt);
            Gizmos.DrawLine(E3.position, E3.position + E3vel.position * dt);
            Gizmos.DrawLine(E4.position, E4.position + E4vel.position * dt);

            Gizmos.DrawLine(E1.position + E1vel.position * dt, E2.position + E2vel.position * dt);
            Gizmos.DrawLine(E3.position + E3vel.position * dt, E4.position + E4vel.position * dt);

            bool willCollide = PointTriangleCCD.EdgeHitsMovingEdge(
                E1.position, E2.position, E1vel.position, E2vel.position,
                E3.position, E4.position, E3vel.position, E4vel.position,
                dt
            );

            if (willCollide) Debug.Log("Will collide");
            else Debug.Log("Wouldn't collide");
        }
    }
}

public static class PointTriangleCCD
{
#region triangle
    /// Returns true if moving point P intersects moving triangle ABC at any time t in [0, dt].
    /// Assumes linear vertex motion (constant velocities during the step).
    public static bool PointHitsMovingTriangle(
        Vector3 P0, Vector3 Vp,
        Vector3 A0, Vector3 B0, Vector3 C0,
        Vector3 Va, Vector3 Vb, Vector3 Vc,
        float dt,
        float eps = 1e-6f)
    {
        if (dt <= 0f) return false;

        // Put into relative motion so the point is stationary at the origin.
        Vector3 a0 = A0 - P0;
        Vector3 b0 = B0 - P0;
        Vector3 c0 = C0 - P0;

        Vector3 va = Va - Vp;
        Vector3 vb = Vb - Vp;
        Vector3 vc = Vc - Vp;

        // Quick check at endpoints (handles "already intersecting" and numeric corner cases).
        if (OriginInTriangle(a0, b0, c0, eps)) return true;
        Vector3 aDt = a0 + va * dt;
        Vector3 bDt = b0 + vb * dt;
        Vector3 cDt = c0 + vc * dt;
        if (OriginInTriangle(aDt, bDt, cDt, eps)) return true;

        // Build cubic f(t) = dot(N(t), A(t)) where N(t) = (B-A)x(C-A).
        // Coplanarity with origin occurs when f(t) == 0.
        Vector3 d0 = b0 - a0;
        Vector3 e0 = c0 - a0;
        Vector3 d1 = (vb - va);
        Vector3 e1 = (vc - va);

        Vector3 n0 = Vector3.Cross(d0, e0);
        Vector3 n1 = Vector3.Cross(d0, e1) + Vector3.Cross(d1, e0);
        Vector3 n2 = Vector3.Cross(d1, e1);

        // A(t) = a0 + va*t
        // f(t) = dot(n0 + n1 t + n2 t^2, a0 + va t) = c0 + c1 t + c2 t^2 + c3 t^3
        double c0p = Vector3.Dot(n0, a0);
        double c1p = Vector3.Dot(n0, va) + Vector3.Dot(n1, a0);
        double c2p = Vector3.Dot(n1, va) + Vector3.Dot(n2, a0);
        double c3p = Vector3.Dot(n2, va);

        // Degenerate: if the cubic is ~0, they are (numerically) coplanar for the whole step.
        // The simple "solve cubic then test" approach breaks down in constantly-coplanar cases.
        if (Math.Abs(c0p) < 1e-12 && Math.Abs(c1p) < 1e-12 && Math.Abs(c2p) < 1e-12 && Math.Abs(c3p) < 1e-12)
        {
            // Always coplanar → reduce to 2D point–triangle CCD
            return PointTriangleCCD2D(
                a0, b0, c0,
                va, vb, vc,
                dt, eps);
        }

        // Find real roots in [0, dt].
        var roots = SolveCubicAllReal(c3p, c2p, c1p, c0p);
        foreach (double tD in roots)
        {
            if (tD < -eps || tD > (double)dt + eps) continue;
            float t = Mathf.Clamp((float)tD, 0f, dt);

            Vector3 A = a0 + va * t;
            Vector3 B = b0 + vb * t;
            Vector3 C = c0 + vc * t;

            // Degenerate triangle: area ~ 0, treat as "no hit" unless the origin lies on one of the edges.
            Vector3 N = Vector3.Cross(B - A, C - A);
            if (N.sqrMagnitude <= eps * eps)
            {
                if (OriginOnSegment(A, B, eps) || OriginOnSegment(B, C, eps) || OriginOnSegment(C, A, eps))
                    return true;
                continue;
            }

            if (OriginInTriangle(A, B, C, eps))
            {
                Gizmos.color = Color.green;
                Gizmos.DrawLine(A0 + Va * t, B0 + Vb * t);
                Gizmos.DrawLine(B0 + Vb * t, C0 + Vc * t);
                Gizmos.DrawLine(C0 + Vc * t, A0 + Va * t);
                Gizmos.DrawSphere(P0 + Vp * t, 0.05f);
                return true;
            }
        }

        return false;
    }

    // Static test: origin inside triangle (including boundary), with tolerance.
    private static bool OriginInTriangle(Vector3 A, Vector3 B, Vector3 C, float eps)
    {
        Vector3 N = Vector3.Cross(B - A, C - A);
        float n2 = N.sqrMagnitude;
        if (n2 <= eps * eps) return false;

        // Ensure origin is near the plane.
        float plane = Vector3.Dot(N, -A);
        if (Mathf.Abs(plane) > eps * Mathf.Sqrt(n2)) return false;

        // Barycentric for point P=0: compute using oriented areas.
        // u = (P x C)·N / |N|^2  with A at origin in barycentric derivation, so shift by A:
        // Solve 0 = A + u(B-A) + v(C-A) => -A = u(B-A) + v(C-A)
        Vector3 p = -A;
        Vector3 uVec = B - A;
        Vector3 vVec = C - A;

        // Using the same stable form as area ratios:
        float u = Vector3.Dot(Vector3.Cross(p, vVec), N) / n2;
        float v = Vector3.Dot(Vector3.Cross(uVec, p), N) / n2;
        float w = 1f - u - v;

        return u >= -eps && v >= -eps && w >= -eps;
    }

    private static bool OriginOnSegment(Vector3 A, Vector3 B, float eps)
    {
        Vector3 AB = B - A;
        float ab2 = AB.sqrMagnitude;
        if (ab2 <= eps * eps) return (A.sqrMagnitude <= eps * eps);

        // Collinearity via cross product magnitude.
        float cross2 = Vector3.Cross(AB, -A).sqrMagnitude;
        if (cross2 > (eps * eps) * ab2) return false;

        // Between endpoints via dot.
        float t = Vector3.Dot(-A, AB) / ab2;
        return t >= -eps && t <= 1f + eps;
    }
#endregion

#region Edge
    public static bool EdgeHitsMovingEdge(
        Vector3 A0, Vector3 B0, Vector3 Va, Vector3 Vb,
        Vector3 C0, Vector3 D0, Vector3 Vc, Vector3 Vd,
        float dt,
        float eps = 1e-6f)
    {
        if (dt <= 0f) return false;

        // Relative frame: make A stationary at origin.
        Vector3 a0 = Vector3.zero;
        Vector3 b0 = B0 - A0;
        Vector3 c0 = C0 - A0;
        Vector3 d0 = D0 - A0;

        Vector3 va = Vector3.zero;
        Vector3 vb = Vb - Va;
        Vector3 vc = Vc - Va;
        Vector3 vd = Vd - Va;

        // Endpoint static checks (often catches hits and reduces numeric issues).
        if (SegmentsIntersect3D(a0, b0, c0, d0, eps)) return true;
        Vector3 bDt = b0 + vb * dt;
        Vector3 cDt = c0 + vc * dt;
        Vector3 dDt = d0 + vd * dt;
        if (SegmentsIntersect3D(a0, bDt, cDt, dDt, eps)) return true;

        // Coplanarity function:
        // f(t) = (C(t)-A(t)) · ( (B(t)-A(t)) x (D(t)-C(t)) ) = 0
        // With A(t)=0 in this frame:
        // f(t) = C(t) · ( B(t) x (D(t)-C(t)) )
        //
        // B(t) = b0 + vb t
        // C(t) = c0 + vc t
        // E(t) = (D-C)(t) = (d0-c0) + (vd-vc) t
        Vector3 e0 = d0 - c0;
        Vector3 e1 = (vd - vc);

        // Cross(t) = B(t) x E(t) = n0 + n1 t + n2 t^2
        Vector3 n0 = Vector3.Cross(b0, e0);
        Vector3 n1 = Vector3.Cross(b0, e1) + Vector3.Cross(vb, e0);
        Vector3 n2 = Vector3.Cross(vb, e1);

        // f(t) = C(t) · Cross(t) = (c0+vc t)·(n0+n1 t+n2 t^2)
        //      = k0 + k1 t + k2 t^2 + k3 t^3
        double k0 = Vector3.Dot(c0, n0);
        double k1 = Vector3.Dot(c0, n1) + Vector3.Dot(vc, n0);
        double k2 = Vector3.Dot(c0, n2) + Vector3.Dot(vc, n1);
        double k3 = Vector3.Dot(vc, n2);

        // Degenerate: if the cubic is ~0, they are (numerically) coplanar for the whole step.
        // The simple "solve cubic then test" approach breaks down in constantly-coplanar cases.
        if (Math.Abs(k0) < 1e-12 && Math.Abs(k1) < 1e-12 && Math.Abs(k2) < 1e-12 && Math.Abs(k3) < 1e-12)
        {
            // Always coplanar → reduce to 2D edge–edge CCD
            return EdgeEdgeCCD2D(
                A0, B0, Va, Vb,
                C0, D0, Vc, Vd,
                dt, eps);
        }

        // Solve cubic for candidate coplanarity times
        // Reuse your previous solver (double precision):
        List<double> roots = SolveCubicAllReal(k3, k2, k1, k0);

        foreach (double tD in roots)
        {
            if (tD < -eps || tD > (double)dt + eps) continue;
            float t = Mathf.Clamp((float)tD, 0f, dt);

            Vector3 B = b0 + vb * t;
            Vector3 C = c0 + vc * t;
            Vector3 D = d0 + vd * t;

            if (SegmentsIntersect3D(a0, B, C, D, eps))
            {
                Gizmos.color = Color.green;
                Gizmos.DrawLine(A0 + Va * t, B0 + Vb * t);
                Gizmos.DrawLine(C0 + Vc * t, D0 + Vd * t);
                return true;
            }
        }

        return false;
    }

    // Static 3D segment intersection:
    // Works by projecting coplanar segments to 2D using the dominant normal axis,
    // and also handles colinear overlap with a tolerance.
    private static bool SegmentsIntersect3D(Vector3 A, Vector3 B, Vector3 C, Vector3 D, float eps)
    {
        Vector3 AB = B - A;
        Vector3 CD = D - C;

        Vector3 N = Vector3.Cross(AB, CD);
        float n2 = N.sqrMagnitude;

        // Parallel (or near-parallel)
        if (n2 <= eps * eps)
        {
            // Check colinearity: (C-A) x AB ~ 0
            if (Vector3.Cross(C - A, AB).sqrMagnitude > (eps * eps) * AB.sqrMagnitude)
                return false;

            // Colinear: reduce to 1D overlap by projecting onto the best axis.
            int axis = DominantAxis(AB);
            float a0 = A[axis], a1 = B[axis];
            float c0 = C[axis], c1 = D[axis];

            float amin = Mathf.Min(a0, a1), amax = Mathf.Max(a0, a1);
            float cmin = Mathf.Min(c0, c1), cmax = Mathf.Max(c0, c1);

            return (amax >= cmin - eps) && (cmax >= amin - eps);
        }

        // Check coplanarity: C and D should lie in plane of AB through A
        float distC = Mathf.Abs(Vector3.Dot(N, C - A));
        float distD = Mathf.Abs(Vector3.Dot(N, D - A));
        if (distC > eps * Mathf.Sqrt(n2) || distD > eps * Mathf.Sqrt(n2)) return false;

        // Project to 2D by dropping the dominant axis of N
        int drop = DominantAxis(N);
        Vector2 a2 = DropAxis(A, drop);
        Vector2 b2 = DropAxis(B, drop);
        Vector2 c2 = DropAxis(C, drop);
        Vector2 d2 = DropAxis(D, drop);

        return SegmentsIntersect2D(a2, b2, c2, d2, eps);
    }

    private static int DominantAxis(Vector3 v)
    {
        float ax = Mathf.Abs(v.x), ay = Mathf.Abs(v.y), az = Mathf.Abs(v.z);
        if (ax > ay && ax > az) return 0;
        if (ay > az) return 1;
        return 2;
    }

    private static Vector2 DropAxis(Vector3 p, int drop)
    {
        return drop switch
        {
            0 => new Vector2(p.y, p.z),
            1 => new Vector2(p.x, p.z),
            _ => new Vector2(p.x, p.y),
        };
    }

    // 2D inclusive segment intersection with epsilon
    private static bool SegmentsIntersect2D(Vector2 A, Vector2 B, Vector2 C, Vector2 D, float eps)
    {
        float o1 = Orient(A, B, C);
        float o2 = Orient(A, B, D);
        float o3 = Orient(C, D, A);
        float o4 = Orient(C, D, B);

        // General case
        if (SignDiffOrZero(o1, o2, eps) && SignDiffOrZero(o3, o4, eps))
            return true;

        // Colinear / touching cases
        if (Mathf.Abs(o1) <= eps && OnSegment(A, B, C, eps)) return true;
        if (Mathf.Abs(o2) <= eps && OnSegment(A, B, D, eps)) return true;
        if (Mathf.Abs(o3) <= eps && OnSegment(C, D, A, eps)) return true;
        if (Mathf.Abs(o4) <= eps && OnSegment(C, D, B, eps)) return true;

        return false;
    }

    private static float Orient(Vector2 a, Vector2 b, Vector2 c)
        => (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);

    private static bool SignDiffOrZero(float a, float b, float eps)
        => (a <= eps && b >= -eps) || (a >= -eps && b <= eps);

    private static bool OnSegment(Vector2 a, Vector2 b, Vector2 p, float eps)
    {
        return p.x >= Mathf.Min(a.x, b.x) - eps && p.x <= Mathf.Max(a.x, b.x) + eps &&
               p.y >= Mathf.Min(a.y, b.y) - eps && p.y <= Mathf.Max(a.y, b.y) + eps;
    }
#endregion

#region Helper functions
    /// Solves a*t^3 + b*t^2 + c*t + d = 0, returns all real roots (as doubles).
    private static List<double> SolveCubicAllReal(double a, double b, double c, double d)
    {
        const double EPS = 1e-12;
        var roots = new List<double>();

        if (Math.Abs(a) < EPS)
            return SolveQuadraticAllReal(b, c, d);

        // Depressed cubic: t = x - b/(3a) => x^3 + px + q = 0
        double invA = 1.0 / a;
        double bb = b * invA;
        double cc = c * invA;
        double dd = d * invA;

        double shift = bb / 3.0;
        double p = cc - (bb * bb) / 3.0;
        double q = (2.0 * bb * bb * bb) / 27.0 - (bb * cc) / 3.0 + dd;

        double disc = (q * q) / 4.0 + (p * p * p) / 27.0;

        if (disc > EPS)
        {
            // One real root.
            double sqrtDisc = Math.Sqrt(disc);
            double u = Cbrt(-q / 2.0 + sqrtDisc);
            double v = Cbrt(-q / 2.0 - sqrtDisc);
            roots.Add((u + v) - shift);
            return roots;
        }

        if (Math.Abs(disc) <= EPS)
        {
            // Multiple real roots (at least two equal).
            double u = Cbrt(-q / 2.0);
            roots.Add(2.0 * u - shift);
            roots.Add(-u - shift);
            return roots;
        }

        // Three distinct real roots.
        double r = Math.Sqrt(-(p * p * p) / 27.0);
        double phi = Math.Acos(-q / (2.0 * r));
        double m = 2.0 * Math.Sqrt(-p / 3.0);

        roots.Add(m * Math.Cos(phi / 3.0) - shift);
        roots.Add(m * Math.Cos((phi + 2.0 * Math.PI) / 3.0) - shift);
        roots.Add(m * Math.Cos((phi + 4.0 * Math.PI) / 3.0) - shift);
        return roots;
    }

    private static List<double> SolveQuadraticAllReal(double a, double b, double c)
    {
        const double EPS = 1e-12;
        var roots = new List<double>();

        if (Math.Abs(a) < EPS)
        {
            if (Math.Abs(b) < EPS) return roots;
            roots.Add(-c / b);
            return roots;
        }

        double disc = b * b - 4.0 * a * c;
        if (disc < -EPS) return roots;

        if (Math.Abs(disc) <= EPS)
        {
            roots.Add(-b / (2.0 * a));
            return roots;
        }

        double s = Math.Sqrt(disc);
        roots.Add((-b - s) / (2.0 * a));
        roots.Add((-b + s) / (2.0 * a));
        return roots;
    }

    private static double Cbrt(double x)
    {
        if (x >= 0.0) return Math.Pow(x, 1.0 / 3.0);
        return -Math.Pow(-x, 1.0 / 3.0);
    }
#endregion

#region 2D Coplanar Fallbacks

    private static bool PointTriangleCCD2D(
        Vector3 a0, Vector3 b0, Vector3 c0,
        Vector3 va, Vector3 vb, Vector3 vc,
        float dt, float eps)
    {
        Vector3 n = Vector3.Cross(b0 - a0, c0 - a0);
        if (n.sqrMagnitude <= eps * eps) return false;

        BuildPlaneBasis(n, out Vector3 u, out Vector3 v);

        Vector2 A0 = Project2D(a0, u, v);
        Vector2 B0 = Project2D(b0, u, v);
        Vector2 C0 = Project2D(c0, u, v);

        Vector2 VA = Project2D(va, u, v);
        Vector2 VB = Project2D(vb, u, v);
        Vector2 VC = Project2D(vc, u, v);

        // Edge CCD: point vs each moving edge
        if (PointSegmentCCD2D(A0, B0, VA, VB, dt, eps)) return true;
        if (PointSegmentCCD2D(B0, C0, VB, VC, dt, eps)) return true;
        if (PointSegmentCCD2D(C0, A0, VC, VA, dt, eps)) return true;

        return false;
    }

    private static bool EdgeEdgeCCD2D(
        Vector3 a0, Vector3 b0, Vector3 va, Vector3 vb,
        Vector3 c0, Vector3 d0, Vector3 vc, Vector3 vd,
        float dt, float eps)
    {
        Vector3 n = Vector3.Cross(b0 - a0, d0 - c0);
        if (n.sqrMagnitude <= eps * eps)
            n = Vector3.Cross(b0 - a0, c0 - a0);

        if (n.sqrMagnitude <= eps * eps) return false;

        BuildPlaneBasis(n, out Vector3 u, out Vector3 v);

        Vector2 A0 = Project2D(a0, u, v);
        Vector2 B0 = Project2D(b0, u, v);
        Vector2 C0 = Project2D(c0, u, v);
        Vector2 D0 = Project2D(d0, u, v);

        Vector2 VA = Project2D(va, u, v);
        Vector2 VB = Project2D(vb, u, v);
        Vector2 VC = Project2D(vc, u, v);
        Vector2 VD = Project2D(vd, u, v);

        // Symmetric endpoint–segment CCD
        if (PointSegmentCCD2D(C0 - A0, D0 - A0, VC - VA, VD - VA, dt, eps)) return true;
        if (PointSegmentCCD2D(C0 - B0, D0 - B0, VC - VB, VD - VB, dt, eps)) return true;

        if (PointSegmentCCD2D(A0 - C0, B0 - C0, VA - VC, VB - VC, dt, eps)) return true;
        if (PointSegmentCCD2D(A0 - D0, B0 - D0, VA - VD, VB - VD, dt, eps)) return true;

        return false;
    }

    private static bool PointSegmentCCD2D(
        Vector2 A0, Vector2 B0,
        Vector2 VA, Vector2 VB,
        float dt, float eps)
    {
        Vector2 D0 = B0 - A0;
        Vector2 VD = VB - VA;

        float c0 = Cross2D(A0, D0);
        float c1 = Cross2D(A0, VD) + Cross2D(VA, D0);
        float c2 = Cross2D(VA, VD);

        int roots = SolveQuadratic(c2, c1, c0, out float t0, out float t1);

        for (int i = 0; i < roots; i++)
        {
            float t = (i == 0) ? t0 : t1;
            if (t < -eps || t > dt + eps) continue;

            Vector2 A = A0 + VA * t;
            Vector2 B = B0 + VB * t;
            Vector2 AB = B - A;

            float ab2 = AB.sqrMagnitude;
            if (ab2 <= eps * eps) continue;

            float s = Vector2.Dot(-A, AB) / ab2;
            if (s >= -eps && s <= 1f + eps)
                return true;
        }

        return false;
    }

    private static bool PointInTriangle2D(Vector2 p, Vector2 a, Vector2 b, Vector2 c, float eps)
    {
        float o1 = Orient(a, b, p);
        float o2 = Orient(b, c, p);
        float o3 = Orient(c, a, p);

        bool hasNeg = (o1 < -eps) || (o2 < -eps) || (o3 < -eps);
        bool hasPos = (o1 > eps) || (o2 > eps) || (o3 > eps);

        return !(hasNeg && hasPos);
    }

    private static void BuildPlaneBasis(Vector3 n, out Vector3 u, out Vector3 v)
    {
        n = n.normalized;
        
        // Pick the cardinal axis LEAST parallel to n
        float absX = Mathf.Abs(n.x);
        float absY = Mathf.Abs(n.y);
        float absZ = Mathf.Abs(n.z);
        
        Vector3 a;
        if (absX < absY && absX < absZ)
            a = Vector3.right;      // n.x is smallest
        else if (absY < absZ)
            a = Vector3.up;         // n.y is smallest
        else
            a = Vector3.forward;    // n.z is smallest
        
        u = Vector3.Cross(n, a).normalized;
        v = Vector3.Cross(n, u).normalized;  // Normalize v too for robustness
    }


    private static Vector2 Project2D(Vector3 p, Vector3 u, Vector3 v)
    {
        return new Vector2(Vector3.Dot(p, u), Vector3.Dot(p, v));
    }

    private static float Cross2D(Vector2 a, Vector2 b)
    {
        return a.x * b.y - a.y * b.x;
    }

    private static int SolveQuadratic(float a, float b, float c, out float r0, out float r1)
    {
        r0 = r1 = 0f;
        const float EPS = 1e-6f;

        if (Mathf.Abs(a) < EPS)
        {
            if (Mathf.Abs(b) < EPS) return 0;
            r0 = -c / b;
            return 1;
        }

        float disc = b * b - 4f * a * c;
        if (disc < 0f) return 0;

        float s = Mathf.Sqrt(Mathf.Max(0f, disc));
        r0 = (-b - s) / (2f * a);
        r1 = (-b + s) / (2f * a);
        return (Mathf.Abs(s) < EPS) ? 1 : 2;
    }

#endregion
}
