using UnityEngine;

// Runs in Edit mode too (so you can step without pressing Play).
[ExecuteAlways]
public class PbdPointTriangleGizmoStepper : MonoBehaviour
{
    [Header("Scene Handles (optional)")]
    public Transform pointT;
    public Transform aT;
    public Transform bT;
    public Transform cT;

    [Header("Simulation State")]
    public Vector3 pointPos = new Vector3(0, 1, 0);
    public Vector3 aPos     = new Vector3(-0.5f, 0, -0.5f);
    public Vector3 bPos     = new Vector3( 0.5f, 0, -0.5f);
    public Vector3 cPos     = new Vector3( 0.0f, 0,  0.5f);

    public Vector3 pointVel = new Vector3(0, -1, 0);
    public Vector3 aVel, bVel, cVel;

    [Header("Mass (inverse mass: 0 = static)")]
    public float pointInvMass = 1.0f;
    public float aInvMass = 1.0f;
    public float bInvMass = 1.0f;
    public float cInvMass = 1.0f;

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

        SolveAttachment(ref A, initialaPos, simulatedCompliance, Time.deltaTime);
        SolveAttachment(ref B, initialbPos, simulatedCompliance, Time.deltaTime);
        SolveAttachment(ref C, initialcPos, simulatedCompliance, Time.deltaTime);

        // 3) Solve collision by projection (can run multiple iterations).
        for (int it = 0; it < Mathf.Max(1, solverIterations); it++)
        {
            SolvePointTriangle(ref p, ref A, ref B, ref C);
        }

        // 4) Commit positions and update velocities (simple finite difference).
        pointVel = (p - pointPos) / dt;
        aVel = (A - aPos) / dt;
        bVel = (B - bPos) / dt;
        cVel = (C - cPos) / dt;

        pointPos = p; aPos = A; bPos = B; cPos = C;

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

    void OnDrawGizmos()
    {
        // Ensure we have something to draw even in edit mode.
        if (!Application.isPlaying)
            SyncFromTransforms();

        // Triangle
        Gizmos.color = Color.white;
        Gizmos.DrawLine(aPos, bPos);
        Gizmos.DrawLine(bPos, cPos);
        Gizmos.DrawLine(cPos, aPos);

        // Point
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(pointPos, 0.03f);

        // Closest point and normal debug
        Gizmos.color = Color.cyan;
        Gizmos.DrawWireSphere(lastClosest, 0.02f);

        Gizmos.color = (lastC < 0f) ? Color.red : Color.green;
        Gizmos.DrawLine(lastClosest, lastClosest + lastN * 0.25f);

        // Optional: visualize barycentric weights by tinting vertices
        Gizmos.color = new Color(1, 0, 0, 0.6f); Gizmos.DrawWireSphere(aPos, 0.02f);
        Gizmos.color = new Color(0, 1, 0, 0.6f); Gizmos.DrawWireSphere(bPos, 0.02f);
        Gizmos.color = new Color(0, 0, 1, 0.6f); Gizmos.DrawWireSphere(cPos, 0.02f);
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
