using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using SolverStructs;
using Unity.Collections;
using static Unity.Mathematics.math;
using Unity.Jobs;
using Unity.Burst;
using UnityEngine.XR;

#region Jobs
[BurstCompile]
public struct DynamicCellsAABBcount : IJobParallelFor
{
    [ReadOnly] public float invCellSize;
    [ReadOnly] public NativeArray<float3> currentParticles;
    [ReadOnly] public NativeArray<LatticeCell> currentCells;
    public NativeArray<int3> dCellAABBmin;
    public NativeArray<int3> dCellAABBmax;
    
    public void Execute(int index)
    {
        LatticeCell c = currentCells[index];
        float3 p0 = currentParticles[c.v0];
        float3 p1 = currentParticles[c.v1];
        float3 p2 = currentParticles[c.v2];
        float3 p3 = currentParticles[c.v3];
        float3 p4 = currentParticles[c.v4];
        float3 p5 = currentParticles[c.v5];
        float3 p6 = currentParticles[c.v6];
        float3 p7 = currentParticles[c.v7];

        float3 minPos = min(min(min(p0, p1), min(p2, p3)), min(min(p4, p5), min(p6, p7)));
        float3 maxPos = max(max(max(p0, p1), max(p2, p3)), max(max(p4, p5), max(p6, p7)));

        dCellAABBmin[index] = (int3)floor(minPos * invCellSize);
        dCellAABBmax[index] = (int3)floor(maxPos * invCellSize);
    }
}

[BurstCompile]
public struct CountAABBentries : IJob
{
    [ReadOnly] public NativeArray<int3> dCellAABBmin;
    [ReadOnly] public NativeArray<int3> dCellAABBmax;
    public NativeReference<int> totalCount;

    public void Execute()
    {
        int total = 0;
        for (int i = 0; i < dCellAABBmin.Length; i++)
        {
            int3 size = dCellAABBmax[i] - dCellAABBmin[i] + 1;
            total += size.x * size.y * size.z;
        }

        totalCount.Value = total;
    }
}

[BurstCompile]
public struct AddHashCells : IJobParallelFor
{
    [ReadOnly] public NativeArray<int3> dCellAABBmin;
    [ReadOnly] public NativeArray<int3> dCellAABBmax;

    public NativeList<int4>.ParallelWriter hashCellsEntries;
    
    public void Execute(int index)
    {
        for (int x = dCellAABBmin[index].x; x <= dCellAABBmax[index].x; ++x)
        for (int y = dCellAABBmin[index].y; y <= dCellAABBmax[index].y; ++y)
        for (int z = dCellAABBmin[index].z; z <= dCellAABBmax[index].z; ++z)
        {
            hashCellsEntries.AddNoResize(new int4(x, y, z, index));
        }
    }
}

[BurstCompile]
public struct CellEntryComparer : IComparer<int4>
{
    public int Compare(int4 a, int4 b)
    {
        int cmp = a.x.CompareTo(b.x);
        if (cmp != 0) return cmp;

        cmp = a.y.CompareTo(b.y);
        if (cmp != 0) return cmp;

        return a.z.CompareTo(b.z);
    }
}
#endregion

public class CollisionAlgoTest : MonoBehaviour
{
#region Variables
    public List<LatticeBody> collisionBodies;
    public float3 targetLatticeDimensions = new float3(0.5f);
    public float hashCellSize = 1f;
    public bool debugLatticeCells = false;

    private List<LatticePreprocessor> preprocessors;
    private NativeList<float3> currentParticles;
    private NativeList<LatticeCell> currentCells;

    // Collision test data
    private List<int4> cellCoords;
    private List<int4> touchedCells = new List<int4>();
    private HashSet<int2> collisionPairs = new HashSet<int2>();

    // Parallel collision test data
    private NativeList<int3> dCellAABBmin;
    private NativeList<int3> dCellAABBmax;
    private NativeList<int4> hashCellsEntries;
#endregion

    private void Awake()
    {
        if (collisionBodies == null) return;
        targetLatticeDimensions = math.max(targetLatticeDimensions, new float3(0.2f));
        preprocessors = new List<LatticePreprocessor>(collisionBodies.Count);

        foreach (var body in collisionBodies)
        {
            body.Initialize();
            preprocessors.Add(new LatticePreprocessor(body, targetLatticeDimensions));
        }

        // Populate data lists
        currentParticles = new NativeList<float3>(Allocator.Persistent);
        currentCells = new NativeList<LatticeCell>(Allocator.Persistent);
        dCellAABBmin = new NativeList<int3>(Allocator.Persistent);
        dCellAABBmax = new NativeList<int3>(Allocator.Persistent);
        hashCellsEntries = new NativeList<int4>(Allocator.Persistent);
        cellCoords = new List<int4>();
        int currentBody = 0;
        foreach (var body in preprocessors)
        {
            int initialParticleCount = currentParticles.Length;
            
            for (int i = 0; i < body.activeParticles.Count; i++)
            {
                // Calculate the particle position in the lattice
                int3 particleCoords = body.to3D_Particle(body.activeParticles[i]);
                float3 posInLattice = body.latticeNegativeCorner + particleCoords * body.latticeDimensions;
                
                // Transform the position to world space
                float3 posInWorld = body.latticeBody.transform.TransformPoint(posInLattice);
                currentParticles.Add(posInWorld);
            }

            for (int i = 0; i < body.finalLatticeCells.Length; i++)
            {
                LatticeCell currentCell = body.finalLatticeCells[i];
                currentCell.v0 += initialParticleCount;
                currentCell.v1 += initialParticleCount;
                currentCell.v2 += initialParticleCount;
                currentCell.v3 += initialParticleCount;
                currentCell.v4 += initialParticleCount;
                currentCell.v5 += initialParticleCount;
                currentCell.v6 += initialParticleCount;
                currentCell.v7 += initialParticleCount;
                currentCells.Add(currentCell);
                dCellAABBmin.Add(0);
                dCellAABBmax.Add(0);
                cellCoords.Add(new int4(currentBody, body.to3D(body.latticeCellCoords[i])));
            }

            ++currentBody;
        }
    }


#region Simulation
    private void Update()
    {
        if (collisionBodies == null || preprocessors == null ||
            !currentParticles.IsCreated || !currentCells.IsCreated) return;

        // Update lists
        int pIdx = 0;
        foreach (var body in preprocessors)
        for (int i = 0; i < body.activeParticles.Count; i++)
        {
            // Calculate the particle position in the lattice
            int3 particleCoords = body.to3D_Particle(body.activeParticles[i]);
            float3 posInLattice = body.latticeNegativeCorner + particleCoords * body.latticeDimensions;
            
            // Transform the position to world space
            float3 posInWorld = body.latticeBody.transform.TransformPoint(posInLattice);
            currentParticles[pIdx++] = posInWorld;
        }

        // SingleThreadTest();
        MultiTreadedTest();
    }

    private void MultiTreadedTest()
    {
        JobHandle handle = default;
        int numberOfCells = currentCells.Length;

        // Calculate the total number of occupied cells
        float invCellSize = 1.0f / hashCellSize;
        DynamicCellsAABBcount AABBcount = new DynamicCellsAABBcount
        {
            invCellSize = invCellSize, currentParticles = currentParticles.AsArray(),
            currentCells = currentCells.AsArray(), dCellAABBmin = dCellAABBmin.AsArray(),
            dCellAABBmax = dCellAABBmax.AsArray()
        };
        handle = AABBcount.Schedule(numberOfCells, 16, handle);
        var totalCountRef = new NativeReference<int>(Allocator.TempJob);
        CountAABBentries countAABB = new CountAABBentries
        {
            totalCount = totalCountRef, dCellAABBmin = dCellAABBmin.AsArray(),
            dCellAABBmax = dCellAABBmax.AsArray()
        };
        handle = countAABB.Schedule(handle);
        handle.Complete();


        // List initialisation
        if (hashCellsEntries.Capacity < totalCountRef.Value)
            hashCellsEntries.Capacity = totalCountRef.Value;
        hashCellsEntries.Clear();
        Debug.Log(totalCountRef.Value);
        totalCountRef.Dispose();


        // Add and sort entries
        AddHashCells addHashCells = new AddHashCells
        {
            dCellAABBmin = dCellAABBmin.AsArray(), dCellAABBmax = dCellAABBmax.AsArray(),
            hashCellsEntries = hashCellsEntries.AsParallelWriter()
        };
        handle = addHashCells.Schedule(numberOfCells, 8, default);
        // Sort based on x then y then z.
        handle = hashCellsEntries.AsDeferredJobArray().SortJob(new CellEntryComparer()).Schedule(handle);


        handle.Complete();
    }

    private void SingleThreadTest()
    {
        // Biuld cell AABB
        touchedCells.Clear();
        float invCellSize = 1.0f / hashCellSize;
        for (int i = 0; i < currentCells.Length; i++)
        {
            var c = currentCells[i];
            float3 p0 = currentParticles[c.v0];
            float3 p1 = currentParticles[c.v1];
            float3 p2 = currentParticles[c.v2];
            float3 p3 = currentParticles[c.v3];
            float3 p4 = currentParticles[c.v4];
            float3 p5 = currentParticles[c.v5];
            float3 p6 = currentParticles[c.v6];
            float3 p7 = currentParticles[c.v7];

            float3 minPos = min(min(min(p0, p1), min(p2, p3)), min(min(p4, p5), min(p6, p7)));
            float3 maxPos = max(max(max(p0, p1), max(p2, p3)), max(max(p4, p5), max(p6, p7)));

            int3 minCell = (int3)floor(minPos * invCellSize);
            int3 maxCell = (int3)floor(maxPos * invCellSize);

            for (int z = minCell.z; z <= maxCell.z; z++)
            for (int y = minCell.y; y <= maxCell.y; y++)
            for (int x = minCell.x; x <= maxCell.x; x++)
            {
                touchedCells.Add(new int4(x, y, z, i));
            }
        }

        // Sort based on the cell coordinates
        touchedCells.Sort((a, b) =>
        {
            int cmp = a.x.CompareTo(b.x);
            if (cmp != 0) return cmp;

            cmp = a.y.CompareTo(b.y);
            if (cmp != 0) return cmp;

            return a.z.CompareTo(b.z);
        });

        // Create unique collision pairs
        collisionPairs.Clear();
        for (int start = 0; start < touchedCells.Count;)
        {
            int end = start + 1;

            while (end < touchedCells.Count &&
                touchedCells[end].x == touchedCells[start].x &&
                touchedCells[end].y == touchedCells[start].y &&
                touchedCells[end].z == touchedCells[start].z)
            {
                end++;
            }

            for (int i = start; i < end - 1; i++)
            for (int j = i + 1; j < end; j++)
            {
                int cellAidx = touchedCells[i].w;
                int cellBidx = touchedCells[j].w;

                if (cellAidx == cellBidx) continue;
                LatticeCell cellA = currentCells[cellAidx];
                LatticeCell cellB = currentCells[cellBidx];
                if (cellCoords[cellAidx].x == cellCoords[cellBidx].x)
                {
                    int3 difference = cellCoords[cellAidx].yzw - cellCoords[cellBidx].yzw;
                    if (abs(difference.x) <= 1 && abs(difference.y) <= 1 && abs(difference.z) <= 1) continue;
                }

                // Create only if they are not the same cells or touching in ascending index order
                if (cellAidx < cellBidx)
                    collisionPairs.Add(new int2(cellAidx, cellBidx));
                else
                    collisionPairs.Add(new int2(cellBidx, cellAidx));
            }

            start = end;
        }

        // Check for collision
        foreach (var colPair in collisionPairs)
        {
            LatticeCell cellA = currentCells[colPair.x];
            LatticeCell cellB = currentCells[colPair.y];

            float3 posA = (currentParticles[cellA.v0] + currentParticles[cellA.v7]) * 0.5f;
            float3 posB = (currentParticles[cellB.v0] + currentParticles[cellB.v7]) * 0.5f;

            if (!CellsIntersectSAT(cellA, cellB)) continue;

            Debug.DrawLine(posA, posB, Color.pink, Time.deltaTime);
        }
    }
#endregion


    private void OnDrawGizmos()
    {
        if (collisionBodies == null || preprocessors == null ||
            !currentParticles.IsCreated || !currentCells.IsCreated) return;
        if (!debugLatticeCells) return;
        // ------particle drawing------
        Gizmos.color = Color.coral;
        foreach (var particle in currentParticles) Gizmos.DrawSphere(particle, 0.05f);

        // ------Cell drawing------
        Gizmos.color = Color.green;
        foreach (var cell in currentCells)
        {
            Gizmos.DrawLine(currentParticles[cell.v0], currentParticles[cell.v1]);
            Gizmos.DrawLine(currentParticles[cell.v2], currentParticles[cell.v3]);
            Gizmos.DrawLine(currentParticles[cell.v4], currentParticles[cell.v5]);
            Gizmos.DrawLine(currentParticles[cell.v6], currentParticles[cell.v7]);

            Gizmos.DrawLine(currentParticles[cell.v0], currentParticles[cell.v2]);
            Gizmos.DrawLine(currentParticles[cell.v1], currentParticles[cell.v3]);
            Gizmos.DrawLine(currentParticles[cell.v4], currentParticles[cell.v6]);
            Gizmos.DrawLine(currentParticles[cell.v5], currentParticles[cell.v7]);

            Gizmos.DrawLine(currentParticles[cell.v0], currentParticles[cell.v4]);
            Gizmos.DrawLine(currentParticles[cell.v1], currentParticles[cell.v5]);
            Gizmos.DrawLine(currentParticles[cell.v2], currentParticles[cell.v6]);
            Gizmos.DrawLine(currentParticles[cell.v3], currentParticles[cell.v7]);
        }
    }


#region Helper functions
    private int GetCellVertex(LatticeCell c, int i)
    {
        switch (i)
        {
            case 0: return c.v0;
            case 1: return c.v1;
            case 2: return c.v2;
            case 3: return c.v3;
            case 4: return c.v4;
            case 5: return c.v5;
            case 6: return c.v6;
            default:return c.v7;
        }
    }
    // Assumes your cell indexing is box-like and that from vertex 0,
    // vertices 1, 3, and 4 are the 3 adjacent corners.
    // If your lattice indexing differs, just change those 3 picks.
    private void GetCellVertices(LatticeCell c, float3[] v)
    {
        for (int i = 0; i < 8; i++)
            v[i] = currentParticles[GetCellVertex(c, i)];
    }

    private bool OverlapOnAxis(float3[] a, float3[] b, float3 axis)
    {
        float lenSq = lengthsq(axis);
        if (lenSq < 1e-8f)
            return true; // degenerate cross-product axis, ignore

        axis *= rsqrt(lenSq); // normalize

        float minA = dot(a[0], axis);
        float maxA = minA;

        for (int i = 1; i < 8; i++)
        {
            float p = dot(a[i], axis);
            minA = min(minA, p);
            maxA = max(maxA, p);
        }

        float minB = dot(b[0], axis);
        float maxB = minB;

        for (int i = 1; i < 8; i++)
        {
            float p = dot(b[i], axis);
            minB = min(minB, p);
            maxB = max(maxB, p);
        }

        return !(maxA < minB || maxB < minA);
    }

    private bool CellsIntersectSAT(
        LatticeCell aCell,
        LatticeCell bCell)
    {
        float3[] a = new float3[8];
        float3[] b = new float3[8];

        GetCellVertices(aCell, a);
        GetCellVertices(bCell, b);

        float3 ax = normalize(a[1] - a[0]);
        float3 ay = normalize(a[3] - a[0]);
        float3 az = normalize(a[4] - a[0]);

        float3 bx = normalize(b[1] - b[0]);
        float3 by = normalize(b[3] - b[0]);
        float3 bz = normalize(b[4] - b[0]);

        // 3 face-normal axes from A
        if (!OverlapOnAxis(a, b, ax)) return false;
        if (!OverlapOnAxis(a, b, ay)) return false;
        if (!OverlapOnAxis(a, b, az)) return false;

        // 3 face-normal axes from B
        if (!OverlapOnAxis(a, b, bx)) return false;
        if (!OverlapOnAxis(a, b, by)) return false;
        if (!OverlapOnAxis(a, b, bz)) return false;

        // 9 cross-product axes
        if (!OverlapOnAxis(a, b, cross(ax, bx))) return false;
        if (!OverlapOnAxis(a, b, cross(ax, by))) return false;
        if (!OverlapOnAxis(a, b, cross(ax, bz))) return false;

        if (!OverlapOnAxis(a, b, cross(ay, bx))) return false;
        if (!OverlapOnAxis(a, b, cross(ay, by))) return false;
        if (!OverlapOnAxis(a, b, cross(ay, bz))) return false;

        if (!OverlapOnAxis(a, b, cross(az, bx))) return false;
        if (!OverlapOnAxis(a, b, cross(az, by))) return false;
        if (!OverlapOnAxis(a, b, cross(az, bz))) return false;

        return true;
    }

    void OnDestroy()
    {
        currentParticles.Dispose();
        currentCells.Dispose();
        dCellAABBmin.Dispose();
        dCellAABBmax.Dispose();
        hashCellsEntries.Dispose();
    }
}
#endregion
