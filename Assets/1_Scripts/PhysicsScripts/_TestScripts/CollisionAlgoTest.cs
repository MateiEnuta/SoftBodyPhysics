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
    [ReadOnly] public int channelStart;
    [ReadOnly] public float invCellSize;
    [ReadOnly] public NativeArray<float3> currentParticles;
    [ReadOnly] public NativeArray<LatticeCell> currentCells;
    [NativeDisableParallelForRestriction]
    [WriteOnly] public NativeArray<int3> dCellAABBmin;
    [NativeDisableParallelForRestriction]
    [WriteOnly] public NativeArray<int3> dCellAABBmax;
    
    public void Execute(int index)
    {
        // Move to the ciresponding position
        index += channelStart;

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
    [ReadOnly] public int channelStart;
    [ReadOnly] public int channelLength;
    [ReadOnly] public NativeArray<int3> dCellAABBmin;
    [ReadOnly] public NativeArray<int3> dCellAABBmax;
    [WriteOnly] public NativeReference<int> totalCount;
    [WriteOnly] public NativeArray<int> cellWriteStart;

    public void Execute()
    {
        int total = 0;
        for (int i = channelStart; i < channelStart + channelLength; i++)
        {
            cellWriteStart[i] = total;
            int3 size = dCellAABBmax[i] - dCellAABBmin[i] + 1;
            total += size.x * size.y * size.z;
        }

        totalCount.Value = total;
    }
}

[BurstCompile]
public struct AddHashCells : IJobParallelFor
{
    [ReadOnly] public int channelStart;
    [ReadOnly] public NativeArray<int3> dCellAABBmin;
    [ReadOnly] public NativeArray<int3> dCellAABBmax;
    [ReadOnly] public NativeArray<int> cellWriteStart;
    [NativeDisableParallelForRestriction]
    [WriteOnly] public NativeArray<HashCellEntry64> hashCellsEntries;
    
    public void Execute(int index)
    {
        // Move to the ciresponding position
        index += channelStart;
        
        int start = cellWriteStart[index];
        for (int x = dCellAABBmin[index].x; x <= dCellAABBmax[index].x; x++)
        for (int y = dCellAABBmin[index].y; y <= dCellAABBmax[index].y; y++)
        for (int z = dCellAABBmin[index].z; z <= dCellAABBmax[index].z; z++)
        {
            hashCellsEntries[start++] = new HashCellEntry64(PackWrapped(x, y, z), index);
        }
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
        int period = 1 << bits;
        int r = value % period;
        if (r < 0) r += period;
        return (ulong)r;
    }
}

[BurstCompile]
public struct CountCellRanges : IJob
{
    [ReadOnly] public int numberOfEntries;
    [ReadOnly] public NativeArray<HashCellEntry64> hashCellsEntries;
    public NativeReference<int> totalPossiblePairs;
    public NativeList<int2> hashCellRanges;
    public NativeList<int> rangesStart;

    public void Execute()
    {
        int possiblePairs = 0;

        for (int start = 0; start < numberOfEntries;)
        {
            int end = start + 1;
            while (end < numberOfEntries && hashCellsEntries[end].key == hashCellsEntries[start].key)
                end++;

            int rangeSize = end - start;
            if (rangeSize > 1) 
            {
                hashCellRanges.AddNoResize(new int2(start, end));
                rangesStart.AddNoResize(possiblePairs);
                possiblePairs += rangeSize * (rangeSize - 1) / 2;
            }
            start = end;
        }
        totalPossiblePairs.Value = possiblePairs;
    }
}

[BurstCompile]
public struct AddCollisionPairs : IJobParallelFor
{
    [ReadOnly] public NativeArray<HashCellEntry64> hashCellsEntries;
    [ReadOnly] public NativeArray<int2> hashCellRanges;
    [ReadOnly] public NativeArray<int> rangesStart;
    [NativeDisableParallelForRestriction][WriteOnly] public NativeArray<ulong> initialColPairs;

    public void Execute(int index)
    {
        int start = rangesStart[index];
        for (int i = hashCellRanges[index].x; i < hashCellRanges[index].y - 1; i++)
        for (int j = i + 1; j < hashCellRanges[index].y; j++)
        {
            int cellAidx = hashCellsEntries[i].cellIndex;
            int cellBidx = hashCellsEntries[j].cellIndex;

            int a = min(cellAidx, cellBidx);
            int b = max(cellAidx, cellBidx);

            ulong pairKey = ((ulong)(uint)a << 32) | (uint)b;

            initialColPairs[start++] = pairKey;
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
    private int[] globalChlCount;
    private NativeList<LatticeCell> currentCells;
    private NativeList<float3> currentParticles;
    private List<float3> particleOffsets;
    private List<Transform> particleTransformRef;

    // Collision test data
    private List<int4> touchedCells = new List<int4>();
    private HashSet<int2> collisionPairs = new HashSet<int2>();

    // Parallel collision test data
    private NativeList<int3> dCellAABBmin;
    private NativeList<int3> dCellAABBmax;
    private NativeList<int> cellWriteStart;

    private NativeList<HashCellEntry64> cellsEntries64;
    private NativeList<HashCellEntry64> auxCellsEntries64;
    private NativeArray<int> histograms;
    private NativeArray<int> histogramsOffsets;
    private NativeList<int2> hashCellRanges;
    private NativeList<int> rangesStart;

    private NativeList<ulong> initialColPairs;
    private NativeList<ulong> auxInitialColPairs;
    private NativeList<int2> uniqueColPairs;
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

        globalChlCount = new int[8];

        // Populate data lists
        currentParticles = new NativeList<float3>(Allocator.Persistent);
        particleTransformRef = new List<Transform>();
        currentCells = new NativeList<LatticeCell>(Allocator.Persistent);
        particleOffsets = new List<float3>();

        dCellAABBmin = new NativeList<int3>(Allocator.Persistent);
        dCellAABBmax = new NativeList<int3>(Allocator.Persistent);
        cellWriteStart = new NativeList<int>(Allocator.Persistent);

        cellsEntries64 = new NativeList<HashCellEntry64>(Allocator.Persistent);
        auxCellsEntries64 = new NativeList<HashCellEntry64>(Allocator.Persistent);
        histograms = new NativeArray<int>(numberOfRadixChunks * 256, Allocator.Persistent);
        histogramsOffsets = new NativeArray<int>(numberOfRadixChunks * 256, Allocator.Persistent);
        hashCellRanges = new NativeList<int2>(Allocator.Persistent);
        rangesStart = new NativeList<int>(Allocator.Persistent);

        initialColPairs = new NativeList<ulong>(Allocator.Persistent);
        auxInitialColPairs = new NativeList<ulong>(Allocator.Persistent);
        uniqueColPairs = new NativeList<int2>(Allocator.Persistent);

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

            int currentLatticeCell = 0;
            int insertionPos = 0; // where the next channel’s block starts globally
            for (int currentChl = 0; currentChl < 8; currentChl++)
            {
                int existingCount = globalChlCount[currentChl];  // how many cells of this color are already in latticeCells
                int addCount      = body.chlCount[currentChl];
                if (addCount == 0)
                {
                    // Skip to next channel, but the insertion position must jump
                    insertionPos += existingCount;
                    continue;
                }
                
                // Fill the gap [insertionPos + existingCount .. insertionPos + existingCount + addCount)
                int writeStart = insertionPos + existingCount;
                currentCells.InsertRange(writeStart, addCount);

                for (int i = 0; i < addCount; i++)
                {
                    // Create new cell and offset it's particle references
                    LatticeCell newCell = body.finalLatticeCells[currentLatticeCell++];
                    newCell.v0 += initialParticleCount;
                    newCell.v1 += initialParticleCount;
                    newCell.v2 += initialParticleCount;
                    newCell.v3 += initialParticleCount;
                    newCell.v4 += initialParticleCount;
                    newCell.v5 += initialParticleCount;
                    newCell.v6 += initialParticleCount;
                    newCell.v7 += initialParticleCount;

                    currentCells[writeStart + i] = newCell;
                    dCellAABBmin.Add(0);
                    dCellAABBmax.Add(0);
                    cellWriteStart.Add(0);
                }

                // Move insertionPos past this channel’s whole block (old + new)
                insertionPos += existingCount + addCount;
                globalChlCount[currentChl] += addCount;
            }
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
        var cellEntriesCount = new NativeReference<int>(Allocator.TempJob);
        var totalPossiblePairs = new NativeReference<int>(Allocator.TempJob);

        int channelStart = 0;
        for (int chl = 0; chl < 8; chl++)
        {
            int channelLength = globalChlCount[chl];

            // Calculate the total number of occupied cells
            float invCellSize = 1.0f / hashCellSize;
            var AABBcount = new CellsAABBcount
            {
                channelStart = channelStart,
                invCellSize = invCellSize,
                currentParticles = currentParticles.AsArray(),
                currentCells = currentCells.AsArray(),
                dCellAABBmin = dCellAABBmin.AsArray(),
                dCellAABBmax = dCellAABBmax.AsArray()
            };
            handle = AABBcount.Schedule(channelLength, 16, handle);
            CountAABBentries countAABB = new CountAABBentries
            {
                channelStart = channelStart,
                channelLength = channelLength,
                totalCount = cellEntriesCount, 
                dCellAABBmin = dCellAABBmin.AsArray(),
                dCellAABBmax = dCellAABBmax.AsArray(),
                cellWriteStart = cellWriteStart.AsArray()
            };
            handle = countAABB.Schedule(handle);
            handle.Complete();


            // Lists initialisation
            if (cellsEntries64.Capacity < cellEntriesCount.Value)
                cellsEntries64.Capacity = cellEntriesCount.Value;
            cellsEntries64.ResizeUninitialized(cellEntriesCount.Value);
            if (auxCellsEntries64.Capacity < cellEntriesCount.Value)
                auxCellsEntries64.Capacity = cellEntriesCount.Value;
            auxCellsEntries64.ResizeUninitialized(cellEntriesCount.Value);
            
            if (hashCellRanges.Capacity < cellEntriesCount.Value)
                hashCellRanges.Capacity = cellEntriesCount.Value;
            hashCellRanges.Clear();
            if (rangesStart.Capacity < cellEntriesCount.Value)
                rangesStart.Capacity = cellEntriesCount.Value;
            rangesStart.Clear();


            // Add and sort entries
            var addHashCells = new AddHashCells
            {
                channelStart = channelStart,
                dCellAABBmin = dCellAABBmin.AsArray(), 
                dCellAABBmax = dCellAABBmax.AsArray(),
                hashCellsEntries = cellsEntries64.AsArray(),
                cellWriteStart = cellWriteStart.AsArray()
            };
            handle = addHashCells.Schedule(channelLength, 4, handle);
            // Sort pairs by cell
            RadixSorter.SortCellEntries(
                ref handle,
                numberOfRadixChunks,
                cellEntriesCount.Value,
                cellsEntries64,
                auxCellsEntries64,
                histograms,
                histogramsOffsets
            );

            CountCellRanges countCellRanges = new CountCellRanges
            {
                numberOfEntries = cellEntriesCount.Value, 
                hashCellsEntries = cellsEntries64.AsArray(),
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
                LatticeCell cellA = currentCells[uniqueColPairs[i].x];
                LatticeCell cellB = currentCells[uniqueColPairs[i].y];

                float3 posA = (currentParticles[cellA.v0] + currentParticles[cellA.v7]) * 0.5f;
                float3 posB = (currentParticles[cellB.v0] + currentParticles[cellB.v7]) * 0.5f;

                // if (!CellsIntersectSAT(cellA, cellB)) continue;

                Debug.DrawLine(posA, posB, Color.pink, Time.deltaTime);
            }

            Debug.Log(
                "Current channel : " + chl + "\n" + 
                "Total cell entries: " + cellEntriesCount.Value + "\n" + 
                "Number of initial collisions pairs: " + totalPossiblePairs.Value + "\n" +  
                "Number of unique collision pairs: " + uniqueColPairs.Length);
            channelStart += channelLength;
        }


        // Dispose of temporary helper values
        cellEntriesCount.Dispose();
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
                // if (cellCoords[cellAidx].x == cellCoords[cellBidx].x)
                // {
                //     int3 difference = cellCoords[cellAidx].yzw - cellCoords[cellBidx].yzw;
                //     if (abs(difference.x) <= 1 && abs(difference.y) <= 1 && abs(difference.z) <= 1) continue;
                // }

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
        cellWriteStart.Dispose();
        cellsEntries64.Dispose();
        auxCellsEntries64.Dispose();
        histograms.Dispose();
        histogramsOffsets.Dispose();
        hashCellRanges.Dispose();
        rangesStart.Dispose();
        initialColPairs.Dispose();
        auxInitialColPairs.Dispose();
        uniqueColPairs.Dispose();
    }
}
#endregion
}