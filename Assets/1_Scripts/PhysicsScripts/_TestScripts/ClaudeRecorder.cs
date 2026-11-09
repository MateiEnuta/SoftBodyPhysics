#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using CollisionHandler;
using SolverStructs;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using static System.FormattableString;

/*
Added to the scene by ClaudeBridge, never placed by hand. Runs after
PhysicsSolver every FixedUpdate and writes to Temp/ClaudeBridge:

    steps.csv     one line per FixedUpdate, the worst values in the whole scene
    bodies.csv    one line per FixedUpdate and body

Only reads the solver, through reflection so PhysicsSolver needs no changes.
*/
[DefaultExecutionOrder(32000)]
public class ClaudeRecorder : MonoBehaviour
{
    public static ClaudeRecorder Instance { get; private set; }
    public static readonly Stopwatch StepClock = new Stopwatch();

    public int FixedSteps { get; private set; }
    public int NanCount { get; private set; }
    public string LastLine { get; private set; } = "no steps yet";

    private const string OutputDir = "Temp/ClaudeBridge";
    private const BindingFlags Private = BindingFlags.NonPublic | BindingFlags.Instance;
    private const BindingFlags PrivateStatic = BindingFlags.NonPublic | BindingFlags.Static;

    private static readonly int4[] CellTets =
    {
        new int4(0, 3, 6, 5), new int4(0, 5, 6, 4), new int4(0, 6, 3, 2), new int4(5, 3, 6, 7), new int4(0, 3, 5, 1)
    };
    private static readonly int2[] CellEdges =
    {
        new int2(0, 1), new int2(2, 3), new int2(4, 5), new int2(6, 7),
        new int2(0, 2), new int2(1, 3), new int2(4, 6), new int2(5, 7),
        new int2(0, 4), new int2(1, 5), new int2(2, 6), new int2(3, 7)
    };

    private PhysicsSolver solver;
    private StreamWriter stepsCsv;
    private StreamWriter bodiesCsv;

    private struct BodyStats
    {
        public float3 centre, lowest, highest;
        public float maxSpeed, maxEdge, minJ, volume, restVolume;
        public int inverted, count;
    }


    private void Awake()
    {
        Instance = this;
        solver = FindFirstObjectByType<PhysicsSolver>();
        stepsCsv = new StreamWriter(OutputDir + "/steps.csv", false) { AutoFlush = true };
        bodiesCsv = new StreamWriter(OutputDir + "/bodies.csv", false) { AutoFlush = true };
        stepsCsv.WriteLine("step,time,ms,pairs,nan,maxSpeed,speedBody,maxEdge,edgeBody,edgeX,edgeY,edgeZ,inverted,minJ");
        bodiesCsv.WriteLine("step,body,x,y,z,minY,maxY,sizeX,sizeZ,maxSpeed,maxEdge,inverted,minJ,volumeRatio");
    }

    private void OnDestroy()
    {
        stepsCsv?.Dispose();
        bodiesCsv?.Dispose();
        Instance = null;
    }

    private T Field<T>(string name) => (T)typeof(PhysicsSolver).GetField(name, Private).GetValue(solver);

    public string Describe()
    {
        if (solver == null) return "no PhysicsSolver in the scene";
        NativeList<Particle> particles = Field<NativeList<Particle>>("particles");
        NativeList<LatticeCell> cells = Field<NativeList<LatticeCell>>("latticeCells");
        NativeList<int> active = Field<NativeList<int>>("activeCollisionLattices");
        return Invariant($"bodies={Field<List<BodyWraper>>("bodyWrapers").Count} particles={particles.Length} cells={cells.Length} collisionCells={active.Length} substeps={solver.substeps} fixedDt={Time.fixedDeltaTime}");
    }

    private void FixedUpdate()
    {
        if (solver == null) return;
        NativeList<Particle> particles = Field<NativeList<Particle>>("particles");
        if (!particles.IsCreated || particles.Length == 0) return;

        NativeList<LatticeCell> cells = Field<NativeList<LatticeCell>>("latticeCells");
        NativeList<int> cellBody = Field<NativeList<int>>("latticeCellBodyRef");
        List<BodyWraper> wrappers = Field<List<BodyWraper>>("bodyWrapers");

        float h = solver.targetLatticeDimensions;
        float dt = Time.fixedDeltaTime / solver.substeps;
        BodyStats[] stats = new BodyStats[wrappers.Count];
        int nan = 0;

        for (int b = 0; b < wrappers.Count; b++)
        {
            BodyStats s = new BodyStats { lowest = new float3(float.MaxValue), highest = new float3(float.MinValue), minJ = float.MaxValue };
            for (int i = wrappers[b].particlesRange.x; i < wrappers[b].particlesRange.y; i++)
            {
                Particle p = particles[i];
                if (!math.all(math.isfinite(p.currentPosition))) { nan++; continue; }
                s.centre += p.currentPosition;
                s.lowest = math.min(s.lowest, p.currentPosition);
                s.highest = math.max(s.highest, p.currentPosition);
                s.maxSpeed = math.max(s.maxSpeed, math.length(p.currentPosition - p.previousPosition) / dt);
                s.count++;
            }
            s.centre /= math.max(s.count, 1);
            stats[b] = s;
        }

        float maxEdge = 0f;
        int edgeBody = -1;
        float3 edgePos = 0f;
        for (int c = 0; c < cells.Length; c++)
        {
            int b = cellBody[c];
            BodyStats s = stats[b];
            LatticeCell cell = cells[c];

            foreach (int2 e in CellEdges)
            {
                float3 p0 = particles[Node(cell, e.x)].currentPosition;
                float len = math.length(particles[Node(cell, e.y)].currentPosition - p0);
                if (len > s.maxEdge) s.maxEdge = len;
                if (len > maxEdge) { maxEdge = len; edgeBody = b; edgePos = p0; }
            }

            for (int t = 0; t < 5; t++)
            {
                int4 tet = CellTets[t];
                float rest = Volume6(Corner(tet.x) * h, Corner(tet.y) * h, Corner(tet.z) * h, Corner(tet.w) * h);
                float now = Volume6(
                    particles[Node(cell, tet.x)].currentPosition, particles[Node(cell, tet.y)].currentPosition,
                    particles[Node(cell, tet.z)].currentPosition, particles[Node(cell, tet.w)].currentPosition);
                float j = now / rest;
                s.minJ = math.min(s.minJ, j);
                if (j <= 0f) s.inverted++;
                s.volume += now / 6f * math.sign(rest);
                s.restVolume += math.abs(rest) / 6f;
            }
            stats[b] = s;
        }

        FixedSteps++;
        NanCount = nan;

        float maxSpeed = 0f, minJ = float.MaxValue;
        int speedBody = -1, inverted = 0;
        for (int b = 0; b < stats.Length; b++)
        {
            BodyStats s = stats[b];
            string name = wrappers[b].transform.name.Replace(',', ' ');
            bodiesCsv.WriteLine(Invariant($"{FixedSteps},{name},{s.centre.x:F3},{s.centre.y:F3},{s.centre.z:F3},{s.lowest.y:F3},{s.highest.y:F3},{s.highest.x - s.lowest.x:F3},{s.highest.z - s.lowest.z:F3},{s.maxSpeed:F2},{s.maxEdge:F3},{s.inverted},{s.minJ:F3},{s.volume / math.max(s.restVolume, 1e-9f):F3}"));
            if (s.maxSpeed > maxSpeed) { maxSpeed = s.maxSpeed; speedBody = b; }
            minJ = math.min(minJ, s.minJ);
            inverted += s.inverted;
        }

        NativeList<int2> pairs = (NativeList<int2>)typeof(FullCollisionJob).GetField("candidatePairs", PrivateStatic).GetValue(null);
        long ms = StepClock.ElapsedMilliseconds;
        string speedName = speedBody >= 0 ? wrappers[speedBody].transform.name.Replace(',', ' ') : "-";
        string edgeName = edgeBody >= 0 ? wrappers[edgeBody].transform.name.Replace(',', ' ') : "-";

        stepsCsv.WriteLine(Invariant($"{FixedSteps},{Time.fixedTime:F3},{ms},{pairs.Length},{nan},{maxSpeed:F2},{speedName},{maxEdge:F3},{edgeName},{edgePos.x:F3},{edgePos.y:F3},{edgePos.z:F3},{inverted},{minJ:F3}"));
        LastLine = Invariant($"step={FixedSteps} ms={ms} pairs={pairs.Length} nan={nan} maxSpeed={maxSpeed:F1} ({speedName}) maxEdge={maxEdge:F2} ({edgeName}) inverted={inverted} minJ={minJ:F2}");
    }

    public string Snapshot(string name)
    {
        NativeList<Particle> particles = Field<NativeList<Particle>>("particles");
        NativeList<LatticeCell> cells = Field<NativeList<LatticeCell>>("latticeCells");
        NativeList<int> cellBody = Field<NativeList<int>>("latticeCellBodyRef");
        float dt = Time.fixedDeltaTime / solver.substeps;

        string particlesPath = $"{OutputDir}/{name}_particles.csv";
        using (StreamWriter w = new StreamWriter(particlesPath))
        {
            w.WriteLine("i,x,y,z,vx,vy,vz,invMass");
            for (int i = 0; i < particles.Length; i++)
            {
                Particle p = particles[i];
                float3 v = (p.currentPosition - p.previousPosition) / dt;
                w.WriteLine(Invariant($"{i},{p.currentPosition.x:F4},{p.currentPosition.y:F4},{p.currentPosition.z:F4},{v.x:F3},{v.y:F3},{v.z:F3},{p.invMass:G6}"));
            }
        }

        string cellsPath = $"{OutputDir}/{name}_cells.csv";
        using (StreamWriter w = new StreamWriter(cellsPath))
        {
            w.WriteLine("body,v0,v1,v2,v3,v4,v5,v6,v7");
            for (int c = 0; c < cells.Length; c++)
            {
                LatticeCell l = cells[c];
                w.WriteLine($"{cellBody[c]},{l.v0},{l.v1},{l.v2},{l.v3},{l.v4},{l.v5},{l.v6},{l.v7}");
            }
        }
        return particlesPath;
    }

    // Real distance between the collision meshes of different bodies and edges through triangles. Slow, on request only
    public string Check()
    {
        NativeList<Particle> particles = Field<NativeList<Particle>>("particles");
        NativeList<LatticeCell> cells = Field<NativeList<LatticeCell>>("latticeCells");
        NativeList<int> cellBody = Field<NativeList<int>>("latticeCellBodyRef");
        NativeList<CollisionWraper> wrappers = Field<NativeList<CollisionWraper>>("collisionWrapers");
        NativeList<DynamicVertex> verts = Field<NativeList<DynamicVertex>>("collisionVertices");
        NativeList<int3> tris = Field<NativeList<int3>>("collisionTriangles");
        NativeList<int2> edges = Field<NativeList<int2>>("collisionEdges");

        float3[] pos = new float3[verts.Length];
        float3[] lo = new float3[wrappers.Length], hi = new float3[wrappers.Length];
        for (int w = 0; w < wrappers.Length; w++)
        {
            LatticeCell cell = cells[wrappers[w].cellIdx];
            lo[w] = float.MaxValue;
            hi[w] = float.MinValue;
            for (int v = wrappers[w].colVertsRange.x; v < wrappers[w].colVertsRange.y; v++)
            {
                int4 tet = CellTets[verts[v].tet];
                float4 b = verts[v].bary;
                pos[v] = b.x * particles[Node(cell, tet.x)].currentPosition + b.y * particles[Node(cell, tet.y)].currentPosition
                       + b.z * particles[Node(cell, tet.z)].currentPosition + b.w * particles[Node(cell, tet.w)].currentPosition;
                lo[w] = math.min(lo[w], pos[v]);
                hi[w] = math.max(hi[w], pos[v]);
            }
        }

        float best = float.MaxValue;
        float floor = float.MaxValue;
        int crossings = 0, near = 0;
        for (int v = 0; v < pos.Length; v++) floor = math.min(floor, pos[v].y);
        for (int a = 0; a < wrappers.Length; a++)
        {
            for (int b = a + 1; b < wrappers.Length; b++)
            {
                if (cellBody[wrappers[a].cellIdx] == cellBody[wrappers[b].cellIdx]) continue;
                float3 gap = math.max(lo[a] - hi[b], lo[b] - hi[a]);
                if (math.cmax(gap) > 0.05f) continue;
                near++;
                best = math.min(best, MeshDistance(wrappers[a], wrappers[b], pos, tris, edges, ref crossings));
            }
        }
        return Invariant($"step={FixedSteps} minDistance={(best == float.MaxValue ? -1f : best):F4} crossings={crossings} nearPairs={near} lowestVertex={floor:F4}");
    }

    private static float MeshDistance(in CollisionWraper A, in CollisionWraper B, float3[] pos, NativeList<int3> tris, NativeList<int2> edges, ref int crossings)
    {
        float best = float.MaxValue;
        for (int side = 0; side < 2; side++)
        {
            CollisionWraper P = side == 0 ? A : B, Q = side == 0 ? B : A;
            for (int t = Q.colTrisRange.x; t < Q.colTrisRange.y; t++)
            {
                float3 t0 = pos[tris[t].x], t1 = pos[tris[t].y], t2 = pos[tris[t].z];
                for (int v = P.colVertsRange.x; v < P.colVertsRange.y; v++)
                    best = math.min(best, math.length(pos[v] - FullCollisionJob.ClosestPointOnTriangle(pos[v], t0, t1, t2)));
                for (int e = P.colEdgesRange.x; e < P.colEdgesRange.y; e++)
                    if (SegmentHitsTriangle(pos[edges[e].x], pos[edges[e].y], t0, t1, t2)) crossings++;
            }
        }
        return best;
    }

    private static bool SegmentHitsTriangle(float3 p, float3 q, float3 a, float3 b, float3 c)
    {
        float3 n = math.cross(b - a, c - a);
        float dp = math.dot(n, p - a), dq = math.dot(n, q - a);
        if (dp * dq >= 0f) return false;
        float3 x = p + (q - p) * (dp / (dp - dq));
        float s0 = math.dot(n, math.cross(b - a, x - a)), s1 = math.dot(n, math.cross(c - b, x - b)), s2 = math.dot(n, math.cross(a - c, x - c));
        return s0 >= 0f && s1 >= 0f && s2 >= 0f;
    }

    // parts: id, "shot", name, then optionally camera position and target
    public string Shot(string name, string[] parts)
    {
        const int Width = 1280, Height = 720;
        string path = $"{OutputDir}/{name}.png";

        GameObject holder = new GameObject("ClaudeShotCamera");
        Camera camera = holder.AddComponent<Camera>();
        if (Camera.main != null) camera.CopyFrom(Camera.main);
        if (parts.Length >= 9)
        {
            Vector3 position = new Vector3(Parse(parts[3]), Parse(parts[4]), Parse(parts[5]));
            Vector3 target = new Vector3(Parse(parts[6]), Parse(parts[7]), Parse(parts[8]));
            holder.transform.SetPositionAndRotation(position, Quaternion.LookRotation(target - position));
        }

        RenderTexture texture = new RenderTexture(Width, Height, 24);
        RenderTexture previous = RenderTexture.active;
        camera.targetTexture = texture;
        camera.Render();

        RenderTexture.active = texture;
        Texture2D image = new Texture2D(Width, Height, TextureFormat.RGB24, false);
        image.ReadPixels(new Rect(0, 0, Width, Height), 0, 0);
        image.Apply();
        File.WriteAllBytes(path, image.EncodeToPNG());

        RenderTexture.active = previous;
        camera.targetTexture = null;
        Destroy(image);
        Destroy(texture);
        Destroy(holder);
        return path;
    }

    private static float Parse(string text) => float.Parse(text, System.Globalization.CultureInfo.InvariantCulture);

    private static float Volume6(float3 a, float3 b, float3 c, float3 d) => math.dot(b - a, math.cross(c - a, d - a));

    private static float3 Corner(int k) => new float3(k & 1, (k >> 1) & 1, (k >> 2) & 1);

    private static int Node(in LatticeCell c, int k)
    {
        switch (k)
        {
            case 0: return c.v0; case 1: return c.v1; case 2: return c.v2; case 3: return c.v3;
            case 4: return c.v4; case 5: return c.v5; case 6: return c.v6; default: return c.v7;
        }
    }
}
#endif
