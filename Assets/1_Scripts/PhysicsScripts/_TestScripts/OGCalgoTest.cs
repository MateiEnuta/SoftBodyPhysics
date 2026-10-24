using System.Collections.Generic;
using UnityEngine;

/// Minimal demo: spheres + a floor at y = 0, penetration-free by construction.
///
/// Per substep:
///   0. Broadphase      - spatial hash grid -> short list of candidate pairs (replaces the n^2 loops).
///   1. Velocity solve  - speculative, PAIR-RELATIVE contacts. A pair may close its whole gap this substep
///                        (never more). Excess approach speed is removed with an equal & opposite impulse
///                        (inverse-mass weighted), so momentum is exchanged instead of deleted.
///                        The pair list is CLOSED afterwards: bodies that received speed during the solve get
///                        their own pairs added and the solve continues (see closureRounds).
///   2. Integrate       - u = h * v  (the displacement every body wants to make this substep)
///   3. Displacement bound - for every pair, the relative path must stay outside the spheres (+ a hard skin).
///                        Violations are slid along the contact normal. The pair list for this stage is rebuilt
///                        from the ACTUAL displacements, so the guarantee does not depend on step 1's list.
///   4. Fallback        - if the iterations run out, cap every body's displacement to its mass-weighted gap share.
///   5. v = u / h       - velocity always comes from the motion that actually happened (no velocity wind-up).
///
/// Setup: put this on an empty GameObject and press Play (URP or Built-in, it picks a shader automatically).
/// A fast sphere is fired at the pile every `shootDelay` seconds. Right-click the component header for
/// "Reset Scene" / "Shoot Fast Sphere".
public class PenetrationFreeSpheres : MonoBehaviour
{
    [Header("Solver")]
    public int substeps = 8;
    public int velocityIterations = 10;
    public int positionIterations = 8;
    public float skin = 0.01f;                 // gap the velocity solver tries to keep between surfaces
    public int closureRounds = 8;              // max extra pair-list rounds in the velocity solve (0 = old behaviour)
    public int closureIterations = 5;          // PGS sweeps per extra round
    public Vector3 gravity = new Vector3(0f, -9.81f, 0f);

    [Header("Scene")]
    public int sphereCount = 36;
    public float bulletSpeed = 150f;           // m/s -> ~0.4 m per substep, i.e. more than its own radius
    public float shootDelay = 4f;

    [Header("Debug (read only)")]
    public int fallbackCount;                  // how often the fallback cap had to kick in
    public float minGapEver = float.MaxValue;  // smallest surface gap ever seen; stays > 0 if the guarantee holds
    public int bodyCount;
    public int candidatePairs;                 // pairs the velocity solver worked on last substep (n^2/2 before)
    public int closureUsed;                    // extra rounds the last substep needed (0 = nothing new started moving)

    class Body
    {
        public Vector3 p, v, u;   // position, velocity, displacement planned for this substep
        public float r, w;        // radius, inverse mass
    }

    struct Pair
    {
        public int a, b;          // body indices, a < b
        public Vector3 n;         // contact normal, from b to a (start-of-substep positions)
        public float gap;         // surface gap at the start of the substep
        public float lambda;      // accumulated contact impulse (velocity solve)
    }

    class PairList
    {
        public Pair[] items = new Pair[256];
        public int count;
    }

    readonly List<Body> bodies = new List<Body>();
    readonly List<Transform> visualSpheres = new List<Transform>();
    GameObject floorVisual;
    Shader visualShader;
    float timer;
    float maxRadius;

    // per-substep scratch (grown on demand, never shrunk, so no garbage after warm-up)
    readonly PairList velPairs = new PairList();   // candidates for the velocity solve
    readonly PairList posPairs = new PairList();   // candidates for the displacement bound
    float[] floorLambda = new float[0];
    float[] reach = new float[0];                  // how far each body can travel this substep
    float[] moveCap = new float[0];

    // spatial hash grid (counting sort, no per-frame allocations)
    int[] cellStart = new int[0];                  // bucket b owns cellItems[cellStart[b] .. cellStart[b+1])
    int[] cellItems = new int[0];
    int[] cellStamp = new int[0];                  // marks buckets already visited by the current query
    int[] bodyCell = new int[0];
    int stampId, hashMask;
    float invCell;

    float HardSkin => skin * 0.5f;   // the position stage never lets surfaces get closer than this

    // ------------------------------------------------------------------ setup

    void Start() { ResetScene(); }

    [ContextMenu("Reset Scene")]
    void ResetScene()
    {
        bodies.Clear();
        foreach (var t in visualSpheres) Destroy(t.gameObject);
        visualSpheres.Clear();
        timer = 0f; fallbackCount = 0; minGapEver = float.MaxValue; maxRadius = 0f;

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
        maxRadius = Mathf.Max(maxRadius, r);
    }

    // ------------------------------------------------------------------ simulation

    void FixedUpdate()
    {
        if ((timer += Time.fixedDeltaTime) > shootDelay) { Shoot(); timer = 0; }

        float h = Time.fixedDeltaTime / substeps;
        for (int s = 0; s < substeps; s++) Substep(h);

        SyncVisuals();
    }

    void Substep(float h)
    {
        int n = bodies.Count;
        bodyCount = n;
        EnsureCapacity(n);

        foreach (var b in bodies) b.v += b.p.normalized * -10f * h;

        // 0) Broadphase. Positions p stay fixed for the whole velocity solve, so the grid and the pair data
        //    (normal, gap) are computed ONCE here instead of in every iteration.
        BuildGrid();
        for (int i = 0; i < n; i++) reach[i] = bodies[i].v.magnitude * h;
        BuildPairs(reach, velPairs);
        candidatePairs = velPairs.count;
        System.Array.Clear(floorLambda, 0, n);

        // 1) Velocity solve (projected Gauss-Seidel).
        SolveVelocities(h, velocityIterations);

        // 1b) Close the pair list. The list above was built from the velocities BEFORE the solve, but the solve
        //     itself hands speed to bodies that were at rest (the wave in a chain / pile). Their pairs are not in
        //     the list, so without this the wave stops after one hop and stage 2 / 3 have to delete the leftover
        //     approach speed, which is where the momentum used to disappear (and why a thick skin was needed:
        //     only 2*skin of margin let the next neighbour into the list).
        //     A pair can only matter if the two bodies' reach grew by more than the skin margin, so tiny
        //     speed changes do not trigger anything and a settled pile pays only this O(n) check.
        closureUsed = 0;
        for (int round = 0; round < closureRounds; round++)
        {
            bool grown = false;
            for (int i = 0; i < n; i++)
            {
                float r = bodies[i].v.magnitude * h;
                if (r > reach[i] + 0.5f * skin) { reach[i] = r; grown = true; }
            }
            if (!grown) break;
            BuildPairs(reach, velPairs);
            candidatePairs = velPairs.count;
            SolveVelocities(h, closureIterations);
            closureUsed++;
        }

        // 2) Integrate + 3/4) bound the displacements
        foreach (var b in bodies) b.u = b.v * h;
        LimitDisplacements();

        // (XPBD internal constraints - joints, cloth, ... - would be solved between steps 2 and 3.)

        // 5) Apply motion, velocity comes from what actually happened
        foreach (var b in bodies) { b.p += b.u; b.v = b.u / h; }

        minGapEver = Mathf.Min(minGapEver, MinGap());
    }

    void SolveVelocities(float h, int iterations)
    {
        int n = bodies.Count;
        for (int it = 0; it < iterations; it++)
        {
            for (int i = 0; i < n; i++)
            {
                Body a = bodies[i];
                SolveContact(a, null, Vector3.up, a.p.y - a.r, h, ref floorLambda[i]);      // floor
            }

            for (int k = 0; k < velPairs.count; k++)
            {
                ref Pair pr = ref velPairs.items[k];
                SolveContact(bodies[pr.a], bodies[pr.b], pr.n, pr.gap, h, ref pr.lambda);
            }
        }
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

        // Pairs that are NOT in this list satisfy gap >= |u_a| + |u_b| + 2*skin, so even if they moved straight
        // at each other they could not get closer than the hard skin. Both stages below only ever shrink |u|,
        // so that stays true and these pairs can be ignored without weakening the guarantee.
        for (int i = 0; i < n; i++) reach[i] = bodies[i].u.magnitude;
        BuildPairs(reach, posPairs);


        // Run a few iterations just to iron out the deflections.
        for (int it = 0; it < positionIterations; it++)
        {
            bool violated = false;

            // Floor sliding
            for (int i = 0; i < n; i++)
            {
                Body a = bodies[i];
                float drop = Mathf.Max(a.p.y - a.r - HardSkin, 0f);
                if (a.u.y < -drop) { a.u.y = -drop; violated = true; }
            }

            for (int k = 0; k < posPairs.count; k++)
            {
                Pair pr = posPairs.items[k];
                Body a = bodies[pr.a];
                Body b = bodies[pr.b];

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
                if (pushA < 0f) remA -= pushA * normal;

                // 5. Slide Particle B (If it pushes ALONG the normal, it's penetrating A)
                float pushB = Vector3.Dot(remB, normal);
                if (pushB > 0f) remB -= pushB * normal;

                // 6. Apply to BOTH
                a.u = safeU_A + remA;
                b.u = safeU_B + remB;
            }

            // If we made it through all pairs and nobody crashed, we can exit early!
            if (!violated) return;
        }

        // Planar truncation: non-iterative, order-independent, only limits motion TOWARD a neighbour.
        // TO TEST!!
        // I've changed - HardSkin ti + HardSkin to get a little bit more slack in.
        for (int i = 0; i < n; i++)
        {
            Body a = bodies[i];
            moveCap[i] = 1f;                                      // per-body scale t_i
            float room = Mathf.Max(a.p.y - a.r + HardSkin, 0f);   // floor
            float down = -a.u.y;
            if (down > room) moveCap[i] = room / down;
        }

        for (int k = 0; k < posPairs.count; k++)
        {
            Pair pr = posPairs.items[k];                          // n points from b to a, gap at start of substep
            Body a = bodies[pr.a], b = bodies[pr.b];

            float room = Mathf.Max(pr.gap + HardSkin, 0f);        // space between the two surfaces
            float da = Mathf.Max(-Vector3.Dot(a.u, pr.n), 0f);    // a moving toward b
            float db = Mathf.Max( Vector3.Dot(b.u, pr.n), 0f);    // b moving toward a
            if (da + db <= room) continue;                        // can't close the gap: no limit at all

            float roomA = room * da / (da + db);                  // split by approach, not by mass
            float roomB = room * db / (da + db);
            if (da > roomA) moveCap[pr.a] = Mathf.Min(moveCap[pr.a], roomA / da);
            if (db > roomB) moveCap[pr.b] = Mathf.Min(moveCap[pr.b], roomB / db);
        }

        for (int i = 0; i < n; i++) bodies[i].u *= moveCap[i];
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

    /// Debug only. Looks at the floor and at the pairs from the last bound stage; every other pair was at least
    /// 2*skin apart before the move and could not close that gap, so it can't be the minimum.
    float MinGap()
    {
        float m = float.MaxValue;
        for (int i = 0; i < bodies.Count; i++)
            m = Mathf.Min(m, bodies[i].p.y - bodies[i].r);

        for (int k = 0; k < posPairs.count; k++)
        {
            Body a = bodies[posPairs.items[k].a];
            Body b = bodies[posPairs.items[k].b];
            m = Mathf.Min(m, Vector3.Distance(a.p, b.p) - a.r - b.r);
        }
        return m;
    }

    // ------------------------------------------------------------------ broadphase

    void EnsureCapacity(int n)
    {
        if (reach.Length >= n) return;
        int size = Mathf.Max(n, reach.Length * 2);
        reach = new float[size];
        moveCap = new float[size];
        floorLambda = new float[size];
    }

    int CellHash(int x, int y, int z)
    {
        unchecked { return ((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)) & hashMask; }
    }

    /// Spatial hash of the body centres (cell size = largest diameter). Counting sort: O(n), no allocations.
    void BuildGrid()
    {
        int n = bodies.Count;

        int size = 64;
        while (size < n * 2) size <<= 1;
        hashMask = size - 1;
        if (cellStart.Length != size + 1)
        {
            cellStart = new int[size + 1];
            cellStamp = new int[size];
            stampId = 0;
        }
        if (bodyCell.Length < n)
        {
            bodyCell = new int[n * 2];
            cellItems = new int[n * 2];
        }

        invCell = 1f / Mathf.Max(2f * maxRadius, 0.01f);
        System.Array.Clear(cellStart, 0, cellStart.Length);

        for (int i = 0; i < n; i++)
        {
            Vector3 p = bodies[i].p;
            int c = CellHash(Mathf.FloorToInt(p.x * invCell), Mathf.FloorToInt(p.y * invCell), Mathf.FloorToInt(p.z * invCell));
            bodyCell[i] = c;
            cellStart[c]++;
        }
        for (int c = 1; c < size; c++) cellStart[c] += cellStart[c - 1];      // inclusive prefix sum = bucket ends
        cellStart[size] = n;
        for (int i = n - 1; i >= 0; i--) cellItems[--cellStart[bodyCell[i]]] = i;   // bucket starts
    }

    /// Fills `list` with every pair that could interact this substep: gap < reach_a + reach_b + 2*skin.
    /// `reach[i]` is how far body i can move (velocity * h for the velocity solve, |u| for the bound stage).
    void BuildPairs(float[] reachOf, PairList list)
    {
        int n = bodies.Count;
        list.count = 0;

        float reachMax = 0f;
        for (int i = 0; i < n; i++) reachMax = Mathf.Max(reachMax, reachOf[i]);
        float rhoMax = maxRadius + reachMax + skin;            // largest "inflated radius" of any body

        for (int i = 0; i < n; i++)
        {
            Body a = bodies[i];
            float q = (a.r + reachOf[i] + skin) + rhoMax;      // any partner's centre is within q of ours

            int x0 = Mathf.FloorToInt((a.p.x - q) * invCell), x1 = Mathf.FloorToInt((a.p.x + q) * invCell);
            int y0 = Mathf.FloorToInt((a.p.y - q) * invCell), y1 = Mathf.FloorToInt((a.p.y + q) * invCell);
            int z0 = Mathf.FloorToInt((a.p.z - q) * invCell), z1 = Mathf.FloorToInt((a.p.z + q) * invCell);

            long cells = (long)(x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1);
            if (cells > n)
            {
                // Absurdly fast body: scanning the grid would cost more than just testing everybody.
                for (int j = i + 1; j < n; j++) TryAddPair(i, j, reachOf, list);
                continue;
            }

            stampId++;
            for (int x = x0; x <= x1; x++)
            for (int y = y0; y <= y1; y++)
            for (int z = z0; z <= z1; z++)
            {
                int c = CellHash(x, y, z);
                if (cellStamp[c] == stampId) continue;         // hash collision: bucket already scanned
                cellStamp[c] = stampId;

                for (int k = cellStart[c]; k < cellStart[c + 1]; k++)
                {
                    int j = cellItems[k];
                    if (j > i) TryAddPair(i, j, reachOf, list);   // j > i: every pair is found once
                }
            }
        }
    }

    void TryAddPair(int i, int j, float[] reachOf, PairList list)
    {
        Body a = bodies[i], b = bodies[j];
        Vector3 d = a.p - b.p;
        float dist = d.magnitude;
        float gap = dist - a.r - b.r;
        if (gap >= reachOf[i] + reachOf[j] + 2f * skin) return;

        if (list.count == list.items.Length) System.Array.Resize(ref list.items, list.count * 2);
        list.items[list.count++] = new Pair
        {
            a = i, b = j,
            n = dist > 1e-6f ? d / dist : Vector3.up,
            gap = gap,
            lambda = 0f
        };
    }

    // ------------------------------------------------------------------ drawing

    void InitializeVisuals()
    {
        visualShader = Shader.Find("Universal Render Pipeline/Lit");
        if (visualShader == null) visualShader = Shader.Find("Standard");

        floorVisual = GameObject.CreatePrimitive(PrimitiveType.Cube);
        floorVisual.transform.position = new Vector3(0f, -0.05f, 0f);
        floorVisual.transform.localScale = new Vector3(40f, 0.1f, 40f);
        floorVisual.GetComponent<Renderer>().material = new Material(visualShader) { color = new Color(0.25f, 0.25f, 0.25f) };

        // Remove floor collider so it doesn't interfere with your custom physics
        Destroy(floorVisual.GetComponent<Collider>());
    }

    void AddVisual(Body body)
    {
        GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);

        // CRITICAL: Unity scale is diameter, so we multiply radius by 2
        sphere.transform.localScale = Vector3.one * (body.r * 2f);
        Destroy(sphere.GetComponent<Collider>());

        var mat = new Material(visualShader) { color = Color.HSVToRGB(Mathf.Repeat(body.r * 3f, 1f), 0.6f, 1f) };
        if (mat.HasProperty("_Smoothness")) mat.SetFloat("_Smoothness", 0.5f);
        sphere.GetComponent<Renderer>().material = mat;

        visualSpheres.Add(sphere.transform);
    }

    /// Creates visuals only for NEW bodies (the old version destroyed and rebuilt everything, plus a new floor,
    /// every time a bullet was added).
    void SyncVisuals()
    {
        if (floorVisual == null) InitializeVisuals();
        while (visualSpheres.Count < bodies.Count) AddVisual(bodies[visualSpheres.Count]);
        for (int i = 0; i < bodies.Count; i++) visualSpheres[i].position = bodies[i].p;
    }
}