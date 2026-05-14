using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using SolverStructs;
using Unity.Collections;
using static Unity.Mathematics.math;
using Unity.Jobs;
using Unity.Burst;
using HelperScripts;

namespace collisionTest
{
#region Data structures
public struct HashCellEntry64
{
    public ulong key;
    public int cellIndex;

    public HashCellEntry64(ulong key, int cellIndex)
    {
        this.key = key;
        this.cellIndex = cellIndex;
    }
}
#endregion

#region Jobs
[BurstCompile]
public struct CellsAABBcount : IJobParallelFor
{
    [ReadOnly] public float invCellSize;
    [ReadOnly] public NativeArray<float3> currentParticles;
    [ReadOnly] public NativeArray<LatticeCell> currentCells;
    [WriteOnly] public NativeArray<ulong> dCellAABBmin;
    [WriteOnly] public NativeArray<uint> dCellAABBspan;
    
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

        int3 minCorner = (int3)floor(minPos * invCellSize);
        int3 maxCorner = (int3)floor(maxPos * invCellSize);

        dCellAABBmin[index] = PackWrappedLong(minCorner);
        dCellAABBspan[index] = PackWrappedInt(maxCorner - minCorner + 1);
    }

    public static uint PackWrappedInt(int3 xyz, int bitsX = 11, int bitsY = 10, int bitsZ = 11)
    {
        uint sx = (uint)(xyz.x - 1);
        uint sy = (uint)(xyz.y - 1);
        uint sz = (uint)(xyz.z - 1);

        return sx | (sy << bitsX) | (sz << (bitsX + bitsY));
    }

    public static ulong PackWrappedLong(int3 xyz, int bitsX = 22, int bitsY = 20, int bitsZ = 22)
    {
        ulong px = WrapToBits(xyz.x, bitsX);
        ulong py = WrapToBits(xyz.y, bitsY);
        ulong pz = WrapToBits(xyz.z, bitsZ);

        return px | (py << bitsX) | (pz << (bitsX + bitsY));
    }

    static ulong WrapToBits(int value, int bits)
    {
        long period = (long)(1UL << bits);
        long r = value % period;
        if (r < 0) r += period;
        return (ulong)r;
    }
}

[BurstCompile]
public struct CountAABBentries : IJob
{
    [ReadOnly] public NativeArray<uint> dCellAABBspan;
    [WriteOnly] public NativeReference<int> totalCount;
    [WriteOnly] public NativeArray<int> cellWriteStart;

    public void Execute()
    {
        int total = 0;
        int length = dCellAABBspan.Length;

        for (int i = 0; i < length; i++)
        {
            cellWriteStart[i] = total;

            int3 span = UnpackSpan(dCellAABBspan[i]);
            total += span.x * span.y * span.z;
        }

        totalCount.Value = total;
    }

    static int3 UnpackSpan(uint span)
    {
        int sx = (int)( span         & ((1u << 11) - 1u)) + 1;
        int sy = (int)((span >> 11)  & ((1u << 10) - 1u)) + 1;
        int sz = (int)((span >> 21)  & ((1u << 11) - 1u)) + 1;
        return new int3(sx, sy, sz);
    }
}

[BurstCompile]
public struct AddHashCells : IJobParallelFor
{
    [ReadOnly] public NativeArray<ulong> dCellAABBmin;
    [ReadOnly] public NativeArray<uint> dCellAABBspan;
    [ReadOnly] public NativeArray<int> cellWriteStart;
    [NativeDisableParallelForRestriction]
    [WriteOnly] public NativeArray<HashCellEntry64> hashCellsEntries;
    
    public void Execute(int index)
    {
        int start = cellWriteStart[index];
        int3 span = UnpackSpan(dCellAABBspan[index]);
        int3 minCorner = UnpackMinCorner(dCellAABBmin[index]);

        for (int dx = 0; dx < span.x; dx++)
        for (int dy = 0; dy < span.y; dy++)
        for (int dz = 0; dz < span.z; dz++)
        {
            hashCellsEntries[start++] =
                new HashCellEntry64(
                    PackWrapped(minCorner.x + dx, minCorner.y + dy, minCorner.z + dz),
                    index);
        }
    }

    static int3 UnpackSpan(uint span)
    {
        int sx = (int)( span         & ((1u << 11) - 1u)) + 1;
        int sy = (int)((span >> 11)  & ((1u << 10) - 1u)) + 1;
        int sz = (int)((span >> 21)  & ((1u << 11) - 1u)) + 1;
        return new int3(sx, sy, sz);
    }


    static int3 UnpackMinCorner(ulong minCorner)
    {
        int x = (int)( minCorner         & ((1UL << 22) - 1UL));
        int y = (int)((minCorner >> 22)  & ((1UL << 20) - 1UL));
        int z = (int)((minCorner >> 42)  & ((1UL << 22) - 1UL));
        return new int3(x, y, z);
    }

    public static ulong PackWrapped(int x, int y, int z, int bitsX = 22, int bitsY = 20, int bitsZ = 22)
    {
        ulong px = WrapToBits(x, bitsX);
        ulong py = WrapToBits(y, bitsY);
        ulong pz = WrapToBits(z, bitsZ);

        return px | (py << bitsX) | (pz << (bitsX + bitsY));
    }

    static ulong WrapToBits(int value, int bits)
    {
        long period = (long)(1UL << bits);
        long r = value % period;
        if (r < 0) r += period;
        return (ulong)r;
    }
}

[BurstCompile]
public struct CountCellRanges : IJob
{
    [ReadOnly] public int numberOfEntries;
    [ReadOnly] public NativeArray<HashCellEntry64> hashCellsEntries;
    [ReadOnly] public NativeArray<int4> cellGridPosition;
    [WriteOnly] public NativeReference<int> totalPossiblePairs;
    public NativeList<int2> hashCellRanges;
    public NativeList<int> rangesStart;

    public void Execute()
    {
        int possiblePairs = 0;

        for (int start = 0; start < numberOfEntries;)
        {
            int end = start + 1;
            ulong key = hashCellsEntries[start].key;
            while (end < numberOfEntries && hashCellsEntries[end].key == key)
                end++;

            if (end - start <= 1)
            {
                start = end;
                continue;
            }

            int initialPossiblePairs = possiblePairs;

            for (int i = start; i < end - 1; i++)
            {
                int cellAidx = hashCellsEntries[i].cellIndex;
                int4 cellA = cellGridPosition[cellAidx];

                for (int j = i + 1; j < end; j++)
                {
                    int cellBidx = hashCellsEntries[j].cellIndex;
                    int4 cellB = cellGridPosition[cellBidx];

                    if (cellA.x == cellB.x)
                    {
                        int3 d = cellA.yzw - cellB.yzw;
                        if (abs(d.x) <= 1 && abs(d.y) <= 1 && abs(d.z) <= 1) 
                            continue;
                    }

                    possiblePairs++;
                }
            }

            hashCellRanges.AddNoResize(new int2(start, end));
            rangesStart.AddNoResize(initialPossiblePairs);
            start = end;
        }
        totalPossiblePairs.Value = possiblePairs;
    }
}

[BurstCompile]
public struct AddCollisionPairs : IJobParallelFor
{
    [ReadOnly] public NativeArray<HashCellEntry64> hashCellsEntries;
    [ReadOnly] public NativeArray<int4> cellGridPosition;
    [ReadOnly] public NativeArray<int2> hashCellRanges;
    [ReadOnly] public NativeArray<int> rangesStart;
    [NativeDisableParallelForRestriction]
    [WriteOnly] public NativeArray<ulong> initialColPairs;

    public void Execute(int index)
    {
        int start = rangesStart[index];

        for (int i = hashCellRanges[index].x; i < hashCellRanges[index].y - 1; i++)
        {
            int cellAidx = hashCellsEntries[i].cellIndex;
            int4 cellA = cellGridPosition[cellAidx];

            for (int j = i + 1; j < hashCellRanges[index].y; j++)
            {
                int cellBidx = hashCellsEntries[j].cellIndex;
                int4 cellB = cellGridPosition[cellBidx];

                if (cellA.x == cellB.x)
                {
                    int3 d = cellA.yzw - cellB.yzw;
                    if (abs(d.x) <= 1 && abs(d.y) <= 1 && abs(d.z) <= 1) 
                        continue;
                }

                int a = min(cellAidx, cellBidx);
                int b = max(cellAidx, cellBidx);

                ulong pairKey = ((ulong)(uint)a << 32) | (uint)b;
                initialColPairs[start++] = pairKey;
            }
        }
    }
}

[BurstCompile]
public struct PairUniquer : IJob
{
    [ReadOnly] public NativeArray<ulong> initialColPairs;
    public NativeList<int2> uniqueColPairs;

    public void Execute()
    {
        for (int i = 0; i < initialColPairs.Length; i++)
        {
            ulong p = initialColPairs[i];

            if (i == 0 || !initialColPairs[i - 1].Equals(p))
            {
                int a = (int)(p >> 32);
                int b = (int)(p & 0xFFFFFFFFUL);
                uniqueColPairs.AddNoResize(new int2(a, b));
            }
        }
    }
}

[BurstCompile]
public struct ULongComparer : IComparer<ulong>
{
    public int Compare(ulong a, ulong b)
    {
        if (a < b) return -1;
        if (a > b) return 1;
        return 0;
    }
}
#endregion

public class CollisionAlgoTest : MonoBehaviour
{
#region Variables
    public List<LatticeBody> collisionBodies;
    public float3 targetLatticeDimensions = new float3(0.5f);
    public float hashCellSize = 1f;
    public int numberOfRadixChunks = 10;
    public bool debugLatticeCells = false;

    private List<LatticePreprocessor> preprocessors;
    private NativeList<LatticeCell> currentCells;
    private NativeList<float3> currentParticles;
    private List<float3> particleOffsets;
    private List<Transform> particleTransformRef;

    // Collision test data
    private List<int4> cellCoords;
    private List<int4> touchedCells = new List<int4>();
    private HashSet<int2> collisionPairs = new HashSet<int2>();

    // Parallel collision test data
    private NativeList<ulong> dCellAABBmin;
    private NativeList<uint> dCellAABBspan;
    private NativeList<int> cellWriteStart;

    private NativeList<HashCellEntry64> cellsEntries64;
    private NativeList<HashCellEntry64> auxCellsEntries64;
    private NativeArray<int> histograms;
    private NativeArray<int> histogramsOffsets;
    private NativeList<int2> hashCellRanges;
    private NativeList<int4> cellGridPosition;
    private NativeList<int> rangesStart;

    private NativeList<ulong> initialColPairs;
    private NativeList<ulong> auxInitialColPairs;
    private NativeList<int2> uniqueColPairs;
#endregion

    private void Awake()
    {
        if (collisionBodies == null) return;
        targetLatticeDimensions = max(targetLatticeDimensions, new float3(0.2f));
        preprocessors = new List<LatticePreprocessor>(collisionBodies.Count);

        foreach (var body in collisionBodies)
        {
            body.Initialize();
            preprocessors.Add(new LatticePreprocessor(body, targetLatticeDimensions));
        }

        // Populate data lists
        currentParticles = new NativeList<float3>(Allocator.Persistent);
        particleTransformRef = new List<Transform>();
        currentCells = new NativeList<LatticeCell>(Allocator.Persistent);
        particleOffsets = new List<float3>();
        cellCoords = new List<int4>();

        dCellAABBmin = new NativeList<ulong>(Allocator.Persistent);
        dCellAABBspan = new NativeList<uint>(Allocator.Persistent);
        cellWriteStart = new NativeList<int>(Allocator.Persistent);

        cellsEntries64 = new NativeList<HashCellEntry64>(Allocator.Persistent);
        auxCellsEntries64 = new NativeList<HashCellEntry64>(Allocator.Persistent);
        histograms = new NativeArray<int>(numberOfRadixChunks * 256, Allocator.Persistent);
        histogramsOffsets = new NativeArray<int>(numberOfRadixChunks * 256, Allocator.Persistent);
        hashCellRanges = new NativeList<int2>(Allocator.Persistent);
        cellGridPosition = new NativeList<int4>(Allocator.Persistent);
        rangesStart = new NativeList<int>(Allocator.Persistent);

        initialColPairs = new NativeList<ulong>(Allocator.Persistent);
        auxInitialColPairs = new NativeList<ulong>(Allocator.Persistent);
        uniqueColPairs = new NativeList<int2>(Allocator.Persistent);

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

                float3 offset = posInWorld - (float3)body.latticeBody.transform.position;
                particleTransformRef.Add(body.latticeBody.transform);
                particleOffsets.Add(offset);
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
                dCellAABBspan.Add(0);
                cellWriteStart.Add(0);
                cellCoords.Add(new int4(currentBody, body.to3D(body.latticeCellCoords[i])));
                cellGridPosition.Add(new int4(currentBody, body.to3D(body.latticeCellCoords[i])));
            }

            ++currentBody;
        }
    }


#region Simulation
    private void Update()
    {
        if (collisionBodies == null || preprocessors == null ||
            !currentParticles.IsCreated || !currentCells.IsCreated) return;

        // Update particle positions
        for (int i = 0; i < currentParticles.Length; i++)
        {
            float3 pos = particleTransformRef[i].position;
            pos += (float3)(particleTransformRef[i].rotation * particleOffsets[i]);

            currentParticles[i] = pos;
        }

        // SingleThreadTest();
        MultiTreadedTest();
    }

    private void MultiTreadedTest()
    {
        JobHandle handle = default;
        int numberOfCells = currentCells.Length;

        var totalCellCount = new NativeReference<int>(Allocator.TempJob);
        var totalPossiblePairs = new NativeReference<int>(Allocator.TempJob);

        // Calculate the total number of occupied cells
        float invCellSize = 1.0f / hashCellSize;
        var AABBcount = new CellsAABBcount
        {
            invCellSize = invCellSize,
            currentParticles = currentParticles.AsArray(),
            currentCells = currentCells.AsArray(),
            dCellAABBmin = dCellAABBmin.AsArray(),
            dCellAABBspan = dCellAABBspan.AsArray()
        };
        handle = AABBcount.Schedule(numberOfCells, 8, handle);
        CountAABBentries countAABB = new CountAABBentries
        {
            totalCount = totalCellCount,
            dCellAABBspan = dCellAABBspan.AsArray(),
            cellWriteStart = cellWriteStart.AsArray()
        };
        handle = countAABB.Schedule(handle);
        handle.Complete();


        // Lists initialisation
        if (cellsEntries64.Capacity < totalCellCount.Value)
            cellsEntries64.Capacity = totalCellCount.Value;
        cellsEntries64.ResizeUninitialized(totalCellCount.Value);
        if (auxCellsEntries64.Capacity < totalCellCount.Value)
            auxCellsEntries64.Capacity = totalCellCount.Value;
        auxCellsEntries64.ResizeUninitialized(totalCellCount.Value);
        
        if (hashCellRanges.Capacity < totalCellCount.Value)
            hashCellRanges.Capacity = totalCellCount.Value;
        hashCellRanges.Clear();
        if (rangesStart.Capacity < totalCellCount.Value)
            rangesStart.Capacity = totalCellCount.Value;
        rangesStart.Clear();


        // Add and sort entries
        var addHashCells = new AddHashCells
        {
            dCellAABBmin = dCellAABBmin.AsArray(),
            dCellAABBspan = dCellAABBspan.AsArray(),
            hashCellsEntries = cellsEntries64.AsArray(),
            cellWriteStart = cellWriteStart.AsArray()
        };
        handle = addHashCells.Schedule(numberOfCells, 4, handle);
        // Sort pairs by cell
        RadixSorter.SortCellEntries(
            ref handle,
            numberOfRadixChunks,
            totalCellCount.Value,
            cellsEntries64,
            auxCellsEntries64,
            histograms,
            histogramsOffsets
        );

        CountCellRanges countCellRanges = new CountCellRanges
        {
            numberOfEntries = totalCellCount.Value, 
            hashCellsEntries = cellsEntries64.AsArray(),
            cellGridPosition = cellGridPosition.AsArray(),
            totalPossiblePairs = totalPossiblePairs,
            hashCellRanges = hashCellRanges,
            rangesStart = rangesStart
        };
        handle = countCellRanges.Schedule(handle);
        handle.Complete();


        if (initialColPairs.Capacity < totalPossiblePairs.Value)
            initialColPairs.Capacity = totalPossiblePairs.Value;
        initialColPairs.ResizeUninitialized(totalPossiblePairs.Value);
        if (auxInitialColPairs.Capacity < totalPossiblePairs.Value)
            auxInitialColPairs.Capacity = totalPossiblePairs.Value;
        auxInitialColPairs.ResizeUninitialized(totalPossiblePairs.Value);

        if (uniqueColPairs.Capacity < totalPossiblePairs.Value)
            uniqueColPairs.Capacity = totalPossiblePairs.Value;
        uniqueColPairs.Clear();


        var AddCollisionPairs = new AddCollisionPairs
        {
            hashCellsEntries = cellsEntries64.AsArray(),
            cellGridPosition = cellGridPosition.AsArray(),
            hashCellRanges = hashCellRanges.AsArray(),
            rangesStart = rangesStart.AsArray(),
            initialColPairs = initialColPairs.AsArray()
        };
        handle = AddCollisionPairs.Schedule(hashCellRanges.Length, 2, handle);
        // Sort to make the same pairs be next to each other
        if (totalPossiblePairs.Value < 50000)
            handle = initialColPairs.AsArray().SortJob(new ULongComparer()).Schedule(handle);
        else
            RadixSorter.SortPairEntries(
                ref handle,
                numberOfRadixChunks,
                totalPossiblePairs.Value,
                initialColPairs,
                auxInitialColPairs,
                histograms,
                histogramsOffsets
            );

        var PairUniquer = new PairUniquer
        {
            initialColPairs = initialColPairs.AsDeferredJobArray(),
            uniqueColPairs = uniqueColPairs  
        };
        handle = PairUniquer.Schedule(handle);
        handle.Complete();


        // Visually show collisions
        for (int i = 0; i < uniqueColPairs.Length; i++)
        {
            // Debug.Log(uniqueColPairs[i]);
            LatticeCell cellA = currentCells[uniqueColPairs[i].x];
            LatticeCell cellB = currentCells[uniqueColPairs[i].y];

            float3 posA = (currentParticles[cellA.v0] + currentParticles[cellA.v7]) * 0.5f;
            float3 posB = (currentParticles[cellB.v0] + currentParticles[cellB.v7]) * 0.5f;

            // if (!CellsIntersectSAT(cellA, cellB)) continue;

            Debug.DrawLine(posA, posB, Color.pink, Time.deltaTime);
        }


        Debug.Log("Total cell entries: " + totalCellCount.Value + "\n" + 
            "Number of initial collisions pairs: " + totalPossiblePairs.Value + "\n" +  
            "Number of unique collision pairs: " + uniqueColPairs.Length);

        // Temporary helper values
        totalCellCount.Dispose();
        totalPossiblePairs.Dispose();
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
        dCellAABBspan.Dispose();
        cellWriteStart.Dispose();
        cellsEntries64.Dispose();
        auxCellsEntries64.Dispose();
        histograms.Dispose();
        histogramsOffsets.Dispose();
        hashCellRanges.Dispose();
        cellGridPosition.Dispose();
        rangesStart.Dispose();
        initialColPairs.Dispose();
        auxInitialColPairs.Dispose();
        uniqueColPairs.Dispose();
    }
}
#endregion
}