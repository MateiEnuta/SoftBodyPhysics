using Unity.Mathematics;
using UnityEngine;

// Runs in Edit mode too (so you can step without pressing Play).
[ExecuteAlways]
public class PbdPointTriangleGizmoStepper : MonoBehaviour
{
    public enum CollisionMode
    {
        PointTriangle,
        EdgeEdge
    }

    [Header("Collision Mode")]
    public CollisionMode collisionMode = CollisionMode.PointTriangle;

    [Header("Scene Handles (optional)")]
    public Transform pointT;
    public Transform aT;
    public Transform bT;
    public Transform cT;
    public Transform dT;

    [Header("Simulation State")]
    public Vector3 pointPos = new Vector3(0, 1, 0);
    public Vector3 aPos     = new Vector3(-0.5f, 0, -0.5f);
    public Vector3 bPos     = new Vector3( 0.5f, 0, -0.5f);
    public Vector3 cPos     = new Vector3( 0.0f, 0,  0.5f);
    public Vector3 dPos = new Vector3(1.0f, 0, 0.5f);

    public Vector3 pointVel = new Vector3(0, -1, 0);
    public Vector3 aVel, bVel, cVel, dVel;

    [Header("Mass (inverse mass: 0 = static)")]
    public float pointInvMass = 1.0f;
    public float aInvMass = 1.0f;
    public float bInvMass = 1.0f;
    public float cInvMass = 1.0f;
    public float dInvMass = 1.0f;

    [Header("Collision / Solver")]
    public float dt = 1.0f / 60.0f;
    public float skin = 0.001f;
    public int solverIterations = 1;
    public float simulatedCompliance = 0.01f;

    [Header("Controls")]
    public bool autoStepInPlayMode = true;

    // Debug visualization
    private Vector3 lastClosest;
    private Vector3 lastBary;
    private Vector3 lastN;
    private float lastC;
    private Vector3 initialPointVel;
    private Vector3 initialaPos;
    private Vector3 initialbPos;
    private Vector3 initialcPos;

    void OnEnable()
    {
        initialPointVel = pointVel;
        initialaPos = aPos;
        initialbPos = bPos;
        initialcPos = cPos;
        SyncFromTransforms();
    }

    void Update()
    {
        if (Application.isPlaying && autoStepInPlayMode)
            StepOnce();
    }

    [ContextMenu("Step Once")]
    public void StepOnce()
    {
        // 1) Pull from transforms if assigned (so you can move handles in Scene view).
        SyncFromTransforms();

        // 2) Predict (semi-implicit-ish: just integrate positions from velocities for debugging).
        Vector3 p = pointPos + pointVel * dt;
        Vector3 A = aPos + aVel * dt;
        Vector3 B = bPos + bVel * dt;
        Vector3 C = cPos + cVel * dt;
        Vector3 D = dPos + dVel * dt;

        if (collisionMode == CollisionMode.PointTriangle)
        {
            SolveAttachment(ref A, initialaPos, simulatedCompliance, Time.deltaTime);
            SolveAttachment(ref B, initialbPos, simulatedCompliance, Time.deltaTime);
            SolveAttachment(ref C, initialcPos, simulatedCompliance, Time.deltaTime);
        }
        // else
        // {
        //     SolveAttachment(ref A, initialaPos, simulatedCompliance, Time.deltaTime);
        //     SolveAttachment(ref B, initialbPos, simulatedCompliance, Time.deltaTime);
        // }

        // 3) Solve collision by projection (can run multiple iterations).
        for (int it = 0; it < Mathf.Max(1, solverIterations); it++)
        {
            if (collisionMode == CollisionMode.PointTriangle)
            {
                SolvePointTriangle(ref p, ref A, ref B, ref C);
            }
            else // EdgeEdge
            {
                SolveEdgeEdge(ref A, ref B, ref C, ref D);
            }
        }

        // 4) Commit positions and update velocities (simple finite difference).
        pointVel = (p - pointPos) / dt;
        aVel = (A - aPos) / dt;
        bVel = (B - bPos) / dt;
        cVel = (C - cPos) / dt;
        dVel = (D - dPos) / dt;

        pointPos = p; aPos = A; bPos = B; cPos = C; dPos = D;

        // 5) Push back into transforms.
        SyncToTransforms();
    }

    // For each vertex A, B, C:
    void SolveAttachment(ref Vector3 currentPos, Vector3 targetPos, float compliance, float dt)
    {
        Vector3 diff = currentPos - targetPos;
        float dist = diff.magnitude;
        if (dist < 1e-9f) return;

        // Constraint: C = dist - 0
        // Gradient: n = diff / dist
        // Update: - C / (w + alpha/dt^2)
        
        Vector3 n = diff / dist;
        float alphaTilde = compliance / (dt * dt);
        
        float denom = alphaTilde;
        float dl = -dist / denom; // "lambda" change
        
        currentPos += dl * n;
    }

    [ContextMenu("Reset Velocities")]
    public void ResetVelocities()
    {
        pointVel = new Vector3(0, -1, 0);
        aVel = bVel = cVel = Vector3.zero;
    }

    [ContextMenu("Reset Positions")]
    public void ResetPositions()
    {
        pointPos = new Vector3(0, 1, 0);
        aPos = new Vector3(-0.5f, 0, -0.5f);
        bPos = new Vector3( 0.5f, 0, -0.5f);
        cPos = new Vector3( 0.0f, 0,  0.5f);
        SyncToTransforms();
    }

    void SolvePointTriangle(ref Vector3 p, ref Vector3 A, ref Vector3 B, ref Vector3 C)
    {
        // Closest point on triangle + barycentric coords (clamped)
        Vector3 qc = ClosestPointOnTriangleBarycentric(p, A, B, C, out Vector3 bary);

        // Contact direction
        Vector3 n;
        float Con;

        // Plane normal push (one-sided as requested: no flipping).
        Vector3 triN = Vector3.Cross(B - A, C - A);
        float triNLen = triN.magnitude;
        if (triNLen < 1e-8f) return;
        triN /= triNLen;

        n = triN;
        Con = Vector3.Dot(p - A, n) - skin;  // your original plane form

        // Cache debug
        lastClosest = qc;
        lastBary = bary;
        lastN = n;
        lastC = Con;

        if (Con >= 0f) return; // not penetrating

        // PBD-style single constraint projection:
        // Gradients: point +n, triangle verts -b_i*n
        Vector3 pGrad = n;
        Vector3 AGrad = -bary.x * n;
        Vector3 BGrad = -bary.y * n;
        Vector3 CGrad = -bary.z * n;

        // Denominator = sum(w_i * |grad_i|^2)
        float denom = 0f;
        denom += pointInvMass;
        denom += aInvMass * (bary.x * bary.x);
        denom += bInvMass * (bary.y * bary.y);
        denom += cInvMass * (bary.z * bary.z);
        if (denom < 1e-12f) return;

        float dl = -Con / denom;

        p += pointInvMass * dl * pGrad;
        A += aInvMass * dl * AGrad;
        B += bInvMass * dl * BGrad;
        C += cInvMass * dl * CGrad;
    }

    void SolveEdgeEdge(ref Vector3 A, ref Vector3 B, ref Vector3 C, ref Vector3 D)
    {
        float t1Clamp, t2Clamp;
        Vector3 sepNow = LineLineSeparationUnity(A, B, C, D, out t1Clamp, out t2Clamp);
        float distNow = sepNow.magnitude;
        if (distNow < 1e-8f)
            return;

        Vector3 n = sepNow / distNow;

        // Signed constraint now
        float Cnow = distNow - skin;

        // Interpolated velocities at the current closest points
        Vector3 vA = aVel;
        Vector3 vB = bVel;
        Vector3 vC = cVel;
        Vector3 vD = dVel;

        Vector3 P1 = vA * (1f - t1Clamp) + vB * t1Clamp;
        Vector3 P2 = vC * (1f - t2Clamp) + vD * t2Clamp;

        Vector3 relVel = P2 - P1;
        float relAlongN = Vector3.Dot(relVel, n);

        // Predict distance and constraint at end of step
        float distPred = distNow + relAlongN * dt;
        float Cpred = distPred + skin;
        
        float totalDistance = Cnow - relAlongN * dt;
        if (totalDistance > 0.01f) return;    // re-check; only solve if positive


        // Use your function (convert to float3 if using Unity.Mathematics, or adapt types)
        Vector3 sep = LineLineSeparationUnity(A, B, C, D, out t1Clamp, out t2Clamp);

        float Con = sep.magnitude;
        if (Con < 1e-8f) return;

        n = sep / Con;

        // Contact points on each segment (using clamped t’s)
        P1 = A + t1Clamp * (B - A);
        P2 = C + t2Clamp * (D - C);

        // We want a minimum separation = skin
        float Cdist = Con + skin;
        if (Cdist <= 0f) return; // already separated by >= skin; only push if overlapping

        // Gradients (distance between two points on two segments)
        // p1 = A + t1*(B-A), p2 = C + t2*(D-C), C = |p2 - p1|
        // ∂C/∂A = -(1 - t1)*n, ∂C/∂B = -t1*n, ∂C/∂C = (1 - t2)*n, ∂C/∂D = t2*n
        Vector3 AGrad = -(1f - t1Clamp) * n;
        Vector3 BGrad = -t1Clamp * n;
        Vector3 CGrad = (1f - t2Clamp) * n;
        Vector3 DGrad = t2Clamp * n;

        float denom = 0f;
        denom += aInvMass * (1f - t1Clamp) * (1f - t1Clamp);
        denom += bInvMass * (t1Clamp * t1Clamp);
        denom += cInvMass * (1f - t2Clamp) * (1f - t2Clamp);
        denom += dInvMass * (t2Clamp * t2Clamp);
        if (denom < 1e-12f) return;

        float dl = -Cdist / denom;

        A += aInvMass * dl * AGrad;
        B += bInvMass * dl * BGrad;
        C += cInvMass * dl * CGrad;
        D += dInvMass * dl * DGrad;

        // Debug (reuse your fields)
        lastClosest = (P1 + P2) * 0.5f;
        lastN = n;
        lastC = Cdist;
        lastBary = new Vector3(t1Clamp, t2Clamp, 0f); // store t's for inspection
    }


    // Ericson-style closest point with barycentric output (clamped to triangle).
    static Vector3 ClosestPointOnTriangleBarycentric(Vector3 p, Vector3 a, Vector3 b, Vector3 c, out Vector3 bary)
    {
        Vector3 ab = b - a;
        Vector3 ac = c - a;
        Vector3 ap = p - a;

        float d1 = Vector3.Dot(ab, ap);
        float d2 = Vector3.Dot(ac, ap);

        // Vertex region A
        if (d1 <= 0f && d2 <= 0f)
        {
            bary = new Vector3(1f, 0f, 0f);
            return a;
        }

        Vector3 bp = p - b;
        float d3 = Vector3.Dot(ab, bp);
        float d4 = Vector3.Dot(ac, bp);

        // Vertex region B
        if (d3 >= 0f && d4 <= d3)
        {
            bary = new Vector3(0f, 1f, 0f);
            return b;
        }

        // Edge region AB
        float vc = d1 * d4 - d3 * d2;
        if (vc <= 0f && d1 >= 0f && d3 <= 0f)
        {
            float v = d1 / (d1 - d3);
            bary = new Vector3(1f - v, v, 0f);
            return a + v * ab;
        }

        Vector3 cp = p - c;
        float d5 = Vector3.Dot(ab, cp);
        float d6 = Vector3.Dot(ac, cp);

        // Vertex region C
        if (d6 >= 0f && d5 <= d6)
        {
            bary = new Vector3(0f, 0f, 1f);
            return c;
        }

        // Edge region AC
        float vb = d5 * d2 - d1 * d6;
        if (vb <= 0f && d2 >= 0f && d6 <= 0f)
        {
            float w = d2 / (d2 - d6);
            bary = new Vector3(1f - w, 0f, w);
            return a + w * ac;
        }

        // Edge region BC
        float va = d3 * d6 - d5 * d4;
        if (va <= 0f && (d4 - d3) >= 0f && (d5 - d6) >= 0f)
        {
            Vector3 bc = c - b;
            float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            bary = new Vector3(0f, 1f - w, w);
            return b + w * bc;
        }

        // Face region
        float inv = 1f / (va + vb + vc);
        float v2 = vb * inv;
        float w2 = vc * inv;
        float u2 = 1f - v2 - w2;
        bary = new Vector3(u2, v2, w2);
        return a + ab * v2 + ac * w2;
    }

    // Wrapper around your float3 version
    static Vector3 LineLineSeparationUnity(
        Vector3 p1, Vector3 p2, Vector3 p3, Vector3 p4,
        out float t1Clamp, out float t2Clamp)
    {
        float3 f1 = p1;
        float3 f2 = p2;
        float3 f3 = p3;
        float3 f4 = p4;

        float3 sep = LineLineSeparation(f1, f2, f3, f4, out t1Clamp, out t2Clamp);
        return new Vector3(sep.x, sep.y, sep.z);
    }

    public static float3 LineLineSeparation(
        float3 p1, float3 p2,
        float3 p3, float3 p4,
        out float t1Clamp, out float t2Clamp)
    {
        float3 u = p2 - p1;
        float3 v = p4 - p3;
        float3 w0 = p1 - p3;

        float a = math.dot(u, u);
        float b = math.dot(u, v);
        float c = math.dot(v, v);
        float d = math.dot(u, w0);
        float e = math.dot(v, w0);

        float denom = a * c - b * b;

        float s, t;

        // Unclamped parameters for infinite lines
        if (math.abs(denom) < 1e-6f)
        {
            // Nearly parallel: pick one parameter, slide along v
            s = 0f;
            t = (c > 0f) ? (e / c) : 0f;
        }
        else
        {
            s = (b * e - c * d) / denom;
            t = (a * e - b * d) / denom;
        }

        // Clamp to [0,1] to treat them as segments/rays
        t1Clamp = math.clamp(s, 0f, 1f);
        t2Clamp = math.clamp(t, 0f, 1f);

        float3 pOn1 = p1 + s * u;
        float3 pOn2 = p3 + t * v;

        // Separation vector: apply +0.5*sep to line 1, -0.5*sep to line 2, etc.
        float3 separation = pOn2 - pOn1;

        return separation;
    }

    void OnDrawGizmos()
    {
        // Ensure we have something to draw even in edit mode.
        if (!Application.isPlaying)
            SyncFromTransforms();

        if (collisionMode == CollisionMode.PointTriangle)
        {
            // Triangle
            Gizmos.color = Color.white;
            Gizmos.DrawLine(aPos, bPos);
            Gizmos.DrawLine(bPos, cPos);
            Gizmos.DrawLine(cPos, aPos);

            // Point
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(pointPos, 0.03f);
        }
        else
        {
            // Edge AB
            Gizmos.color = Color.white;
            Gizmos.DrawLine(aPos, bPos);

            // Edge CD
            Gizmos.color = Color.gray;
            Gizmos.DrawLine(cPos, dPos);
        }

        // Closest / normal
        Gizmos.color = Color.cyan;
        Gizmos.DrawWireSphere(lastClosest, 0.02f);

        Gizmos.color = (lastC > 0f) ? Color.red : Color.green;
        Gizmos.DrawLine(lastClosest, lastClosest + lastN * 0.25f);

        // Vertices
        Gizmos.color = new Color(1, 0, 0, 0.6f); Gizmos.DrawWireSphere(aPos, 0.02f);
        Gizmos.color = new Color(0, 1, 0, 0.6f); Gizmos.DrawWireSphere(bPos, 0.02f);
        Gizmos.color = new Color(0, 0, 1, 0.6f); Gizmos.DrawWireSphere(cPos, 0.02f);
        Gizmos.color = new Color(1, 1, 0, 0.6f); Gizmos.DrawWireSphere(dPos, 0.02f);
    }

    void SyncFromTransforms()
    {
        if (pointT) pointPos = pointT.position;
        if (aT) aPos = aT.position;
        if (bT) bPos = bT.position;
        if (cT) cPos = cT.position;
    }

    void SyncToTransforms()
    {
        if (pointT) pointT.position = pointPos;
        if (aT) aT.position = aPos;
        if (bT) bT.position = bPos;
        if (cT) cT.position = cPos;
    }
}
