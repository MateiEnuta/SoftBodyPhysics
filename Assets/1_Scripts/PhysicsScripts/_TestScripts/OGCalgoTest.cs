using System.Collections.Generic;
using System.Linq;
using UnityEngine;

/// Minimal demo: spheres + a floor at y = 0, penetration-free by construction.
///
/// Per substep:
///   1. Velocity solve  - speculative, PAIR-RELATIVE contacts. A pair may close its whole gap this substep
///                        (never more). Excess approach speed is removed with an equal & opposite impulse
///                        (inverse-mass weighted), so momentum is exchanged instead of deleted.
///   2. Integrate       - u = h * v  (the displacement every body wants to make this substep)
///   3. Displacement bound - for every pair, the relative path must stay outside the spheres (+ a hard skin).
///                        Violations shrink the pair's RELATIVE displacement, split by inverse mass
///                        (the pair's centre-of-mass motion is preserved).
///   4. Fallback        - if the iterations run out, scale ALL displacements by the worst pair's safe fraction.
///   5. v = u / h       - velocity always comes from the motion that actually happened (no velocity wind-up).
///
/// Setup: put this on an empty GameObject, press Play, enable Gizmos in the Game view (or look at the Scene view).
/// A fast sphere is fired at the pile after `shootDelay` seconds. Right-click the component header for
/// "Reset Scene" / "Shoot Fast Sphere".
public class PenetrationFreeSpheres : MonoBehaviour
{
    [Header("Solver")]
    public int substeps = 8;
    public int velocityIterations = 10;
    public int positionIterations = 8;
    public float skin = 0.01f;                 // gap the velocity solver tries to keep between surfaces
    public Vector3 gravity = new Vector3(0f, -9.81f, 0f);

    [Header("Scene")]
    public int sphereCount = 36;
    public float bulletSpeed = 150f;           // m/s -> ~0.4 m per substep, i.e. more than its own radius
    public float shootDelay = 4f;

    [Header("Debug (read only)")]
    public int fallbackCount;                  // how often the global fallback had to kick in
    public float minGapEver = 67;  // smallest surface gap ever seen; stays > 0 if the guarantee holds

    class Body
    {
        public Vector3 p, v, u;   // position, velocity, displacement planned for this substep
        public float r, w;        // radius, inverse mass
    }

    readonly List<Body> bodies = new List<Body>();
    private Transform[] visualSpheres;
    float[] lambda = new float[0];   // accumulated contact impulses (pairs, then floor), reset every substep
    float timer;
    bool fired;

    float HardSkin => skin * 0.5f;   // the position stage never lets surfaces get closer than this

    // ------------------------------------------------------------------ setup

    void Start() { ResetScene(); }

    [ContextMenu("Reset Scene")]
    void ResetScene()
    {
        bodies.Clear();
        timer = 0f; fired = false; fallbackCount = 0; minGapEver = float.MaxValue;
        for (int i = 0; i < sphereCount; i++)
        {
            float r = Random.Range(0.3f, 0.6f);
            var p = new Vector3((i % 3) * 1.6f + Random.Range(-0.1f, 0.1f),
                                1f + (i / 9) * 1.6f,
                                ((i / 3) % 3) * 1.6f + Random.Range(-0.1f, 0.1f));
            Add(p, Vector3.zero, r);
        }
    }

    [ContextMenu("Shoot Fast Sphere")]
    void Shoot() { Add(new Vector3(-15f, 0.9f, 0), Vector3.right * bulletSpeed, 0.4f); }

    void Add(Vector3 p, Vector3 v, float r)
    {
        bodies.Add(new Body { p = p, v = v, r = r, w = 1f / (r * r * r) });   // mass ~ r^3
    }

    // ------------------------------------------------------------------ simulation

    void FixedUpdate()
    {
        if ((timer += Time.fixedDeltaTime) > shootDelay) { Shoot(); timer = 0; }

        float h = Time.fixedDeltaTime / substeps;
        for (int s = 0; s < substeps; s++) Substep(h);

        if (visualSpheres == null) InitializeVisuals();
        else if (bodies.Count > visualSpheres.Length)
        {
            for (int i = 0; i < visualSpheres.Length; i++) Destroy(visualSpheres[i].gameObject);
            InitializeVisuals();
        }
        for (int i = 0; i < bodies.Count; i++)
        {
            visualSpheres[i].position = bodies[i].p;
        }
    }

    void Substep(float h)
    {
        int n = bodies.Count;
        if (lambda.Length != n * n + n) lambda = new float[n * n + n];
        System.Array.Clear(lambda, 0, lambda.Length);

        foreach (var b in bodies) b.v += b.p.normalized * -10f * h;

        // 1) Velocity solve (projected Gauss-Seidel). Positions p are the start-of-substep positions throughout.
        for (int it = 0; it < velocityIterations; it++)
        {
            for (int i = 0; i < n; i++)
            {
                Body a = bodies[i];
                SolveContact(a, null, Vector3.up, a.p.y - a.r, h, ref lambda[n * n + i]);   // floor

                for (int j = i + 1; j < n; j++)                                              // n^2 "broadphase"
                {
                    Body b = bodies[j];
                    Vector3 d = a.p - b.p;
                    float dist = d.magnitude;
                    SolveContact(a, b, d / dist, dist - a.r - b.r, h, ref lambda[i * n + j]);
                }
            }
        }

        // 2) Integrate + 3/4) bound the displacements
        foreach (var b in bodies) b.u = b.v * h;
        LimitDisplacements();

        // (XPBD internal constraints - joints, cloth, ... - would be solved between steps 2 and 3.)

        // 5) Apply motion, velocity comes from what actually happened
        foreach (var b in bodies) { b.p += b.u; b.v = b.u / h; }

        minGapEver = Mathf.Min(minGapEver, MinGap());
    }

    /// One speculative contact between a and b (b == null -> static floor).
    /// n points from b to a. Rule: relative approach speed <= (gap - skin) / h, i.e. the pair may close
    /// its WHOLE gap this substep but not more. The excess is removed by an impulse on BOTH bodies.
    void SolveContact(Body a, Body b, Vector3 n, float gap, float h, ref float lam)
    {
        float wa = a.w, wb = b != null ? b.w : 0f;
        float vn = Vector3.Dot(b != null ? a.v - b.v : a.v, n);       // < 0 means approaching
        float c = vn + Mathf.Max(gap - skin, 0f) / h;                 // c >= 0 means satisfied
        float dl = -c / (wa + wb);
        float newLam = Mathf.Max(lam + dl, 0f);                       // contacts can only push, never pull
        dl = newLam - lam; lam = newLam;
        a.v += wa * dl * n;
        if (b != null) b.v -= wb * dl * n;
    }

    /// Pair-relative displacement bound. Assumes linear paths during the substep.
    void LimitDisplacements()
    {
        int n = bodies.Count;

        // Run a few iterations just to iron out the deflections. 
        // NO global fallback. NO freezing. Just sliding.
        for (int it = 0; it < positionIterations; it++)
        {
            bool violated = false;
            
            for (int i = 0; i < n; i++)
            {
                Body a = bodies[i];
                
                // Floor sliding
                float drop = Mathf.Max(a.p.y - a.r - HardSkin, 0f);
                if (a.u.y < -drop) { a.u.y = -drop; violated = true; }

                for (int j = i + 1; j < n; j++)
                {
                    Body b = bodies[j];
                    
                    // When do they hit?
                    float s = SafeFraction(a.p - b.p, a.u - b.u, a.r + b.r + HardSkin);
                    if (s >= 1f) continue;
                    
                    violated = true; // We found a crash, meaning we might need another iteration

                    // 1. Safe movement up to impact
                    Vector3 safeU_A = a.u * s;
                    Vector3 safeU_B = b.u * s;
                    
                    // 2. Remaining movement
                    Vector3 remA = a.u * (1f - s);
                    Vector3 remB = b.u * (1f - s);
                    
                    // 3. Normal points from B to A
                    Vector3 impactPosA = a.p + safeU_A;
                    Vector3 impactPosB = b.p + safeU_B;
                    Vector3 normal = (impactPosA - impactPosB).normalized;
                    
                    // 4. Slide Particle A (If it pushes AGAINST the normal, it's penetrating B)
                    float pushA = Vector3.Dot(remA, normal);
                    if (pushA < 0f) 
                    {
                        remA -= pushA * normal; 
                    }

                    // 5. Slide Particle B (If it pushes ALONG the normal, it's penetrating A)
                    float pushB = Vector3.Dot(remB, normal);
                    if (pushB > 0f) 
                    {
                        remB -= pushB * normal; 
                    }
                    
                    // 6. Apply to BOTH
                    a.u = safeU_A + remA;
                    b.u = safeU_B + remB;
                }
            }
            
            // If we made it through all particles and nobody crashed, we can exit early!
            if (!violated) return; 
        }

        for (int i = 0; i < n; i++)
        {
            Body a = bodies[i];
            
            // 1. Find the absolute smallest safe sphere around this particle
            float maxAllowedMove = float.MaxValue;
            
            // Floor gap (w = 0, so particle A gets 100% of the gap)
            float floorGap = (a.p.y - a.r);
            maxAllowedMove = Mathf.Min(maxAllowedMove, Mathf.Max(floorGap, 0f));

            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;
                
                Body b = bodies[j];
                
                // Measure distance using CURRENT positions, not predicted
                float dist = Vector3.Distance(a.p, b.p);
                float gap = dist - a.r - b.r;
                
                // Clamp negative gaps to 0 (if they are already touching, safe sphere is 0)
                float safeGap = Mathf.Max(gap, 0f);
                
                float wSum = a.w + b.w;
                if (wSum > 0f)
                {
                    // OGC Rule: Share the gap based on inverse mass
                    float allowedShare = safeGap * (a.w / wSum);
                    
                    if (allowedShare < maxAllowedMove)
                    {
                        maxAllowedMove = allowedShare;
                    }
                }
            }
            
            // 2. Limit the displacement by modifying ONLY its magnitude
            float u_mag = a.u.magnitude;
            if (u_mag > maxAllowedMove)
            {
                // Preserve the original direction, but cap the speed
                a.u = (a.u / u_mag) * maxAllowedMove;
            }
        }
    }


    /// Largest t in [0,1] such that |d0 + t*delta| stays >= R (d0 = centre offset, delta = relative displacement).
    /// Sliding past each other is allowed; only the part of the motion that would enter the sphere is cut.
    static float SafeFraction(Vector3 d0, Vector3 delta, float R)
    {
        float a = delta.sqrMagnitude;
        float b = Vector3.Dot(d0, delta);
        float c = d0.sqrMagnitude - R * R;
        if (c < 0f) return b >= 0f ? 1f : 0f;     // already inside the hard skin: only allow separating motion
        if (a < 1e-12f || b >= 0f) return 1f;     // not approaching
        float disc = b * b - a * c;
        if (disc < 0f) return 1f;                 // path passes by
        float t = (-b - Mathf.Sqrt(disc)) / a;    // first time the centres are exactly R apart
        return t >= 1f ? 1f : Mathf.Max(t, 0f);
    }

    float MinGap()
    {
        float m = float.MaxValue;
        for (int i = 0; i < bodies.Count; i++)
        {
            Body a = bodies[i];
            m = Mathf.Min(m, a.p.y - a.r);
            for (int j = i + 1; j < bodies.Count; j++)
                m = Mathf.Min(m, Vector3.Distance(a.p, bodies[j].p) - a.r - bodies[j].r);
        }
        return m;
    }

    // ------------------------------------------------------------------ drawing

    private void InitializeVisuals()
    {
        // 1. Find the URP Lit Shader
        Shader urpLitShader = Shader.Find("Universal Render Pipeline/Lit");
        if (urpLitShader == null)
        {
            Debug.LogError("URP Lit shader not found! Make sure URP is installed and active.");
            return;
        }

        // 2. Create the Floor
        GameObject floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
        floor.transform.position = new Vector3(0f, -0.05f, 0f);
        floor.transform.localScale = new Vector3(40f, 0.1f, 40f);
        
        Material floorMat = new Material(urpLitShader);
        floorMat.SetColor("_BaseColor", new Color(0.25f, 0.25f, 0.25f));
        floor.GetComponent<Renderer>().material = floorMat;
        
        // Remove floor collider so it doesn't interfere with your custom physics
        Destroy(floor.GetComponent<Collider>());

        // 3. Create the Spheres
        visualSpheres = new Transform[bodies.Count];
        
        for (int i = 0; i < bodies.Count; i++)
        {
            GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            
            // CRITICAL: Unity scale is diameter, so we multiply radius by 2
            sphere.transform.localScale = Vector3.one * (bodies[i].r * 2f);
            
            // Remove the Unity collider to keep it purely visual
            Destroy(sphere.GetComponent<Collider>());

            // Apply your mass-based coloring
            Color bodyColor = Color.HSVToRGB(Mathf.Repeat(bodies[i].r * 3f, 1f), 0.6f, 1f);
            
            Material mat = new Material(urpLitShader);
            mat.SetColor("_BaseColor", bodyColor);
            
            // URP uses _Smoothness instead of _Glossiness
            mat.SetFloat("_Smoothness", 0.5f); 
            
            sphere.GetComponent<Renderer>().material = mat;
            
            visualSpheres[i] = sphere.transform;
        }
    }
}