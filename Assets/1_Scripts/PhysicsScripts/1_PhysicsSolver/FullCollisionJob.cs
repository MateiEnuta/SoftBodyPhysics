using UnityEngine;
using SolverStructs;
using Unity.Collections;
using Unity.Mathematics;
using static Unity.Mathematics.math;
using Unity.Jobs;
using Unity.Burst;
using System.Collections.Generic;
using HelperScripts;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Linq;


namespace CollisionHandler
{
#region Jobs

/*
Creates one conservative broad-phase AABB for every physical lattice cell.

Input:
    - Eight current particle positions belonging to the lattice cell.
    - Inverse broad-phase cell size.

Output:
    latticeCellAabbMinWorldCell[cellIndex]
        Unwrapped minimum world broad-phase cell coordinate.

    latticeCellAabbCellSpan[cellIndex]
        Positive number of broad-phase cells covered on each axis.

Important:
    - The AABB is expressed in unwrapped world-grid coordinates.
    - It is intentionally not wrapped here.
    - Wrapping is only used later to select a broad-phase storage bucket.
    - maxCorner - minCorner + 1 is inclusive, so a lattice cell touching
      a broad-phase cell boundary is conservatively inserted into both sides.
*/
[BurstCompile]
public struct CreateLatticeCellsAABB : IJobParallelFor
{
    [ReadOnly] public float invBroadphaseCellSize;
    [ReadOnly] public NativeList<Particle> particles;
    [ReadOnly] public NativeList<LatticeCell> latticeCells;

    [WriteOnly] public NativeArray<int3> latticeCellAABBmin;
    [WriteOnly] public NativeArray<uint> latticeCellAABBspan;

    
    public void Execute(int cellIndex)
    {
        LatticeCell c = latticeCells[cellIndex];
        float3 p0 = particles[c.v0].currentPosition;
        float3 p1 = particles[c.v1].currentPosition;
        float3 p2 = particles[c.v2].currentPosition;
        float3 p3 = particles[c.v3].currentPosition;
        float3 p4 = particles[c.v4].currentPosition;
        float3 p5 = particles[c.v5].currentPosition;
        float3 p6 = particles[c.v6].currentPosition;
        float3 p7 = particles[c.v7].currentPosition;

        float3 minPos = min(min(min(p0, p1), min(p2, p3)), min(min(p4, p5), min(p6, p7)));
        float3 maxPos = max(max(max(p0, p1), max(p2, p3)), max(max(p4, p5), max(p6, p7)));

        int3 minCorner = (int3)floor(minPos * invBroadphaseCellSize);
        int3 maxCorner = (int3)floor(maxPos * invBroadphaseCellSize);

        latticeCellAABBmin[cellIndex] = minCorner;
        latticeCellAABBspan[cellIndex] = FullCollisionJob.PackSpanUint(maxCorner - minCorner + 1);
    }
}

/*
Counts broad-phase entries before output allocation/scatter.

Each worker chunk owns one disjoint segment of broadphaseCellEntryCounts:

    [workerChunkId * totalBroadphaseCellCount + bucketIndex]

For every lattice-cell AABB:
    - Iterate all unwrapped world-grid cells covered by its AABB.
    - Wrap each world-grid cell into a broad-phase storage bucket.
    - Increment that worker chunk's count for the bucket.

The job does not write entries yet.
It only determines how much storage every worker/bucket combination needs.

Important:
    - Different unwrapped world cells may map to the same wrapped bucket.
    - This is intentional for a wrapped fixed-size broad-phase grid.
    - The next prefix-sum stage converts these counts to write positions.

Required precondition:
    broadphaseCellEntryCounts must be zero before this job runs.

Required postcondition:
    Sum of all values in broadphaseCellEntryCounts equals the total
    number of AABB broad-phase entries to be scattered.
*/
[BurstCompile]
public struct CountLatticeCellsAABB : IJobParallelFor
{
    [ReadOnly] public int3 broadphaseGridSize;
    [ReadOnly] public int chunkSize;
    [ReadOnly] public NativeList<int3> latticeCellAABBmin;
    [ReadOnly] public NativeList<uint> latticeCellAABBspan;
    [NativeDisableParallelForRestriction]
    public NativeArray<int> simCellsEntries;


    public void Execute(int chunkIndex)
    {
        int start = chunkIndex * chunkSize;
        int end = min(start + chunkSize, latticeCellAABBmin.Length);
        int totalbroadphaseGridSize = broadphaseGridSize.x * broadphaseGridSize.y * broadphaseGridSize.z;

        for (int i = start; i < end; i++)
        {
            int3 cellMin = latticeCellAABBmin[i];
            int3 cellSpan = FullCollisionJob.UnpackSpanUint(latticeCellAABBspan[i]);

            int gridSizeX = broadphaseGridSize.x;
            int gridSizeY = broadphaseGridSize.y;
            int gridSizeZ = broadphaseGridSize.z;
            int gridStrideZ = gridSizeX * gridSizeY;

            for (int x = 0; x < cellSpan.x; x++)
            {
                int wrappedX = FullCollisionJob.WrapToGrid(cellMin.x + x, gridSizeX);

                for (int y = 0; y < cellSpan.y; y++)
                {
                    int wrappedY = FullCollisionJob.WrapToGrid(cellMin.y + y, gridSizeY);
                    int xyOffset = wrappedX + wrappedY * gridSizeX;

                    for (int z = 0; z < cellSpan.z; z++)
                    {
                        int wrappedZ = FullCollisionJob.WrapToGrid(cellMin.z + z, gridSizeZ);

                        int bucketIndex =
                            xyOffset +
                            wrappedZ * gridStrideZ;

                        simCellsEntries[chunkIndex * totalbroadphaseGridSize + bucketIndex]++;
                    }
                }
            }
        }
    }
}

/*
Builds local prefix sums for broad-phase entry scatter.

Input layout:
    broadphaseCellEntryCounts[
        workerChunkId * totalBroadphaseCellCount + bucketIndex
    ]

For every prefix block:
    - Visit broad-phase buckets in ascending index order.
    - Visit worker chunks in ascending workerChunkId order.
    - Convert each count into a local write offset.
    - Store the local offset in broadphaseCellEntryOffsets.
    - Clear broadphaseCellEntryCounts for reuse next frame.
    - Store the total number of entries in this prefix block.

After this job:
    prefixBlockEntryTotals[prefixBlockId]
        contains the number of entries in that prefix block.

The main thread then converts prefixBlockEntryTotals in-place into:
    prefixBlockBaseOffsets[prefixBlockId]

That conversion provides the global output offset used by ScatterAABBEntries.

Important:
    broadphaseCellEntryOffsets are local to their prefix block.
    ScatterAABBEntries must add the corresponding prefix block base offset.
*/
[BurstCompile]
public struct PrefixSumSimCellsEntries : IJobParallelFor
{
    [ReadOnly] public int numberOfBroadphaseCells;
    [ReadOnly] public int broadphaseWorkerChunkCount;
    [ReadOnly] public int prefixBlockEntryTotals;
    [NativeDisableParallelForRestriction]
    public NativeArray<int> simCellsEntries;
    [NativeDisableParallelForRestriction]
    [WriteOnly] public NativeArray<int> simCellsEntriesOffset;
    [WriteOnly] public NativeArray<int> prefixChunkSizes;


    public void Execute(int prefixChunkIndex)
    {
        int writeStart = prefixBlockEntryTotals * prefixChunkIndex;
        int writeEnd = min(prefixBlockEntryTotals * (prefixChunkIndex+1), numberOfBroadphaseCells);
        int previousCount = 0;
        
        for (int i = writeStart; i < writeEnd; i++)
        for (int chunk = 0; chunk < broadphaseWorkerChunkCount; chunk++)
        {
            int cellIndex = chunk * numberOfBroadphaseCells + i;
            int cellCount = simCellsEntries[cellIndex];

            simCellsEntries[cellIndex] = 0;
            simCellsEntriesOffset[cellIndex] = previousCount;
            previousCount += cellCount;
        }

        prefixChunkSizes[prefixChunkIndex] = previousCount;
    }
}

/*
Scatters one broad-phase entry for every lattice-cell AABB coverage cell.

For every lattice cell:
    - Iterate every unwrapped world-grid cell covered by its AABB.
    - Wrap the world-grid cell to choose a broad-phase storage bucket.
    - Use the worker-local bucket write cursor produced by prefix sum.
    - Add the prefix block's global base offset.
    - Write:

        int4(
            unwrappedWorldCell.x,
            unwrappedWorldCell.y,
            unwrappedWorldCell.z,
            latticeCellId
        )

The entry stores the unwrapped world-grid coordinate because multiple
world-grid cells can share one wrapped storage bucket.

After this job:
    Every broad-phase bucket contains a contiguous unsorted range of
    entries belonging to one or more unwrapped world-grid cells.
*/
[BurstCompile]
public struct ScatterAABBentries : IJobParallelFor
{
    [ReadOnly] public int3 broadphaseGridSize;
    [ReadOnly] public int chunkSize;
    [ReadOnly] public NativeList<int3> latticeCellAABBmin;
    [ReadOnly] public NativeList<uint> latticeCellAABBspan;
    [ReadOnly] public NativeArray<int> prefixChunkSizes;
    [ReadOnly] public int prefixChunkSize;
    [NativeDisableParallelForRestriction]
    public NativeArray<int> simCellsEntriesOffset;
    [NativeDisableParallelForRestriction]
    [WriteOnly] public NativeArray<int4> latticeCellAABBentry;


    public void Execute(int chunkIndex)
    {
        int start = chunkIndex * chunkSize;
        int end = min(start + chunkSize, latticeCellAABBmin.Length);
        int totalbroadphaseGridSize = broadphaseGridSize.x * broadphaseGridSize.y * broadphaseGridSize.z;

        for (int i = start; i < end; i++)
        {
            int3 cellMin = latticeCellAABBmin[i];
            int3 cellSpan = FullCollisionJob.UnpackSpanUint(latticeCellAABBspan[i]);

            int gridSizeX = broadphaseGridSize.x;
            int gridSizeY = broadphaseGridSize.y;
            int gridSizeZ = broadphaseGridSize.z;
            int gridStrideZ = gridSizeX * gridSizeY;

            for (int x = 0; x < cellSpan.x; x++)
            {
                int wrappedX = FullCollisionJob.WrapToGrid(cellMin.x + x, gridSizeX);

                for (int y = 0; y < cellSpan.y; y++)
                {
                    int wrappedY = FullCollisionJob.WrapToGrid(cellMin.y + y, gridSizeY);
                    int xyOffset = wrappedX + wrappedY * gridSizeX;

                    for (int z = 0; z < cellSpan.z; z++)
                    {
                        int wrappedZ = FullCollisionJob.WrapToGrid(cellMin.z + z, gridSizeZ);

                        int bucketIndex =
                            xyOffset +
                            wrappedZ * gridStrideZ;

                        latticeCellAABBentry[simCellsEntriesOffset[chunkIndex * totalbroadphaseGridSize + bucketIndex]++
                            + prefixChunkSizes[bucketIndex / prefixChunkSize]]
                            = new int4(cellMin + new int3(x,y,z), i);
                    }
                }
            }
        }
    }
}

/*
Sorts each wrapped broad-phase bucket and counts unique candidate pairs.

For one wrapped broad-phase bucket:
    1. Recover its contiguous broadphaseEntries range from the final
       worker-chunk offsets and prefix-block base offsets.

    2. Sort entries lexicographically by their unwrapped world-grid
       coordinate (x, then y, then z).

    3. Process each contiguous equal-coordinate group independently.

    4. For every unordered pair of lattice-cell entries in that group:
         - Reject same-body pairs under the current inter-body-only policy.
         - Compute the pair's unique owner world-grid cell:
               max(aabbMinWorldCellA, aabbMinWorldCellB)
         - Count the pair only if the current group's unwrapped world
           coordinate equals this owner cell.

The output is one candidate-pair count per wrapped broad-phase bucket.

This job must use exactly the same candidate-pair predicate as
AddCandidatePairs.
*/
[BurstCompile]
public struct SortSimCellsEntries : IJobParallelFor
{
    [ReadOnly] public NativeArray<int> prefixChunkSizes;
    [ReadOnly] public int prefixChunkSize;
    [ReadOnly] public NativeList<int3> latticeCellAABBmin;
    [ReadOnly] public NativeList<uint2> latticeCellLatticeCoords;
    [ReadOnly] public NativeSlice<int> simCellsEntriesOffset;
    [NativeDisableParallelForRestriction]
    public NativeArray<int4> broadphaseEntries;
    [WriteOnly] public NativeArray<int> simCellsNumberOfPairs;
    

    public void Execute(int gridCellIndex)
    {
        int cellStart = gridCellIndex == 0 ? 0 : (simCellsEntriesOffset[gridCellIndex - 1] + 
            prefixChunkSizes[(gridCellIndex - 1) / prefixChunkSize]);
        int cellEnd = simCellsEntriesOffset[gridCellIndex] + prefixChunkSizes[gridCellIndex / prefixChunkSize];
        int count = cellEnd - cellStart;

        if (count <= 1) { simCellsNumberOfPairs[gridCellIndex] = 0; return; }

        InsertionSort(broadphaseEntries, cellStart, count);


        int totalPairCount = 0;
        for (int start = cellStart; start < cellEnd;)
        {
            int end = start + 1;
            int3 key = broadphaseEntries[start].xyz;
            while (end < cellEnd && all(broadphaseEntries[end].xyz == key))
                end++;

            if (end - start <= 1)
            {
                start = end;
                continue;
            }

            for (int i = start; i < end - 1; i++)
            {
                int latticeCellAId = broadphaseEntries[i].w;
                int3 minACell = latticeCellAABBmin[latticeCellAId];
                uint2 latticeACoords = latticeCellLatticeCoords[latticeCellAId];

                for (int j = i + 1; j < end; j++)
                {
                    int latticeCellBId = broadphaseEntries[j].w;
                    uint2 latticeBCoords = latticeCellLatticeCoords[latticeCellBId];
                    if (latticeACoords.x == latticeBCoords.x) continue;

                    int3 minBCell = latticeCellAABBmin[latticeCellBId];
                    int3 commonMinCell = max(minACell, minBCell);

                    if (!all(commonMinCell == key)) continue;
                    // if (latticeACoords.x == latticeBCoords.x && 
                    //     FullCollisionJob.AreLocalNeighbours10(latticeACoords.y, latticeBCoords.y)) continue;

                    totalPairCount++;
                }
            }

            start = end;
        }
        simCellsNumberOfPairs[gridCellIndex] = totalPairCount;
    }

    // [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void InsertionSort(NativeArray<int4> arr, int start, int count)
    {
        for (int i = 1; i < count; i++)
        {
            int4 key = arr[start + i];
            int j = i - 1;

            while (j >= 0)
            {
                int4 candidate = arr[start + j];
                
                if (CompareWorldCell(candidate.xyz, key.xyz) > 0)
                {
                    arr[start + j + 1] = candidate;
                    j--;
                }
                else
                {
                    break;
                }
            }

            arr[start + j + 1] = key;
        }
    }

    // Returns: >0 if a > b, <0 if a < b, 0 if equal
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static int CompareWorldCell(int3 a, int3 b)
    {
        if (a.x < b.x) return -1;
        if (a.x > b.x) return 1;

        if (a.y < b.y) return -1;
        if (a.y > b.y) return 1;

        if (a.z < b.z) return -1;
        if (a.z > b.z) return 1;

        return 0;
    }
}

/*
Converts per-bucket candidate-pair counts to candidate-pair output offsets.

Before this job:
    candidatePairCounts[bucketIndex]
        is the number of candidate pairs that AddCandidatePairs will write
        for that broad-phase bucket.

After this job:
    candidatePairStartOffsets[bucketIndex]
        is the start index in candidatePairs where that bucket writes.

numberOfCandidatePairs.Value:
    total output candidate-pair count.

Important:
    This job changes the semantic meaning of the array in-place:
        count -> write offset

AddCandidatePairs must use the resulting offsets and must write exactly
the number of pairs predicted by the original counting pass.
*/
[BurstCompile]
public struct prefixSumSimCellPairs : IJob
{
    [ReadOnly] public int totalbroadphaseGridSize;
    public NativeArray<int> simCellsNumberOfPairs;
    [WriteOnly] public NativeReference<int> numberOfCandidatePairs;


    public void Execute()
    {
        int previousCount = 0;
        for (int i = 0; i < totalbroadphaseGridSize; i++)
        {
            int cellCount = simCellsNumberOfPairs[i];
            if (cellCount == 0) continue;
            simCellsNumberOfPairs[i] = previousCount;
            previousCount += cellCount;
        }
        numberOfCandidatePairs.Value = previousCount;
    }
}

/*
Writes the unique inter-body candidate pairs counted by SortSimCellsEntries.

For every wrapped broad-phase bucket:
    - Recover its contiguous broadphaseEntries range.
    - Iterate each equal unwrapped-world-cell group.
    - Re-evaluate the exact same candidate-pair predicate used by
      SortSimCellsEntries.
    - Write accepted pairs into candidatePairs starting at the bucket's
      candidatePairStartOffset.

Output:
    candidatePairs[pairIndex] = int2(latticeCellIdA, latticeCellIdB)

Required invariant:
    The number of writes from this job must equal numberOfCandidatePairs.

Important:
    candidatePairs are broad-phase candidates only.
    They still require narrow-phase collision/contact generation before
    they become solver contact constraints.
*/
[BurstCompile]
public struct AddCandidatePairs : IJobParallelFor
{
    [ReadOnly] public NativeArray<int> prefixChunkSizes;
    [ReadOnly] public int prefixChunkSize;
    [ReadOnly] public NativeList<int3> latticeCellAABBmin;
    [ReadOnly] public NativeList<uint2> latticeCellLatticeCoords;
    [ReadOnly] public NativeSlice<int> simCellsEntriesOffset;
    [ReadOnly] public NativeArray<int4> latticeCellAABBentry;
    [ReadOnly] public NativeArray<int> simCellsNumberOfPairs;
    [NativeDisableParallelForRestriction]
    [WriteOnly] public NativeArray<int2> candidatePairs;
    

    public void Execute(int gridCellIndex)
    {
        int cellStart = gridCellIndex == 0 ? 0 : (simCellsEntriesOffset[gridCellIndex - 1] + 
            prefixChunkSizes[(gridCellIndex - 1) / prefixChunkSize]);
        int cellEnd = simCellsEntriesOffset[gridCellIndex] + prefixChunkSizes[gridCellIndex / prefixChunkSize];
        int count = cellEnd - cellStart;

        if (count <= 1) return;

        int writeStart = simCellsNumberOfPairs[gridCellIndex];
        for (int start = cellStart; start < cellEnd;)
        {
            int end = start + 1;
            int3 key = latticeCellAABBentry[start].xyz;
            while (end < cellEnd && all(latticeCellAABBentry[end].xyz == key))
                end++;

            if (end - start <= 1)
            {
                start = end;
                continue;
            }

            for (int i = start; i < end - 1; i++)
            {
                int latticeCellAId = latticeCellAABBentry[i].w;
                int3 minACell = latticeCellAABBmin[latticeCellAId];
                uint2 latticeACoords = latticeCellLatticeCoords[latticeCellAId];

                for (int j = i + 1; j < end; j++)
                {
                    int latticeCellBId = latticeCellAABBentry[j].w;
                    uint2 latticeBCoords = latticeCellLatticeCoords[latticeCellBId];
                    if (latticeACoords.x == latticeBCoords.x) continue;

                    int3 minBCell = latticeCellAABBmin[latticeCellBId];
                    int3 commonMinCell = max(minACell, minBCell);

                    if (!all(commonMinCell == key)) continue;
                    // if (latticeACoords.x == latticeBCoords.x && 
                    //     FullCollisionJob.AreLocalNeighbours10(latticeACoords.y, latticeBCoords.y)) continue;

                    candidatePairs[writeStart++] = new int2(latticeCellAId, latticeCellBId);
                }
            }

            start = end;
        }
    }
}

#endregion



public static class FullCollisionJob
{
#region Job functions
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int3 UnpackULong(ulong minCorner)
    {
        int x = (int)( minCorner         & ((1UL << 22) - 1UL));
        int y = (int)((minCorner >> 22)  & ((1UL << 20) - 1UL));
        int z = (int)((minCorner >> 42)  & ((1UL << 22) - 1UL));
        return new int3(x, y, z);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong PackWrappedLong(int3 xyz, int bitsX = 22, int bitsY = 20, int bitsZ = 22)
    {
        ulong px = WrapToBits(xyz.x, bitsX);
        ulong py = WrapToBits(xyz.y, bitsY);
        ulong pz = WrapToBits(xyz.z, bitsZ);

        return px | (py << bitsX) | (pz << (bitsX + bitsY));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong WrapToBits(int value, int bits)
    {
        long period = (long)(1UL << bits);
        long r = value % period;
        if (r < 0) r += period;
        return (ulong)r;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int3 UnpackSpanUint(uint span)
    {
        int sx = (int)( span         & ((1u << 10) - 1u)) + 1;
        int sy = (int)((span >> 10)  & ((1u << 10) - 1u)) + 1;
        int sz = (int)((span >> 20)  & ((1u << 10) - 1u)) + 1;
        return new int3(sx, sy, sz);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint PackSpanUint(int3 xyz, int bitsX = 10, int bitsY = 10, int bitsZ = 10)
    {
        uint sx = (uint)(xyz.x - 1);
        uint sy = (uint)(xyz.y - 1);
        uint sz = (uint)(xyz.z - 1);

        return sx | (sy << bitsX) | (sz << (bitsX + bitsY));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool AreLocalNeighbours10(uint packedA, uint packedB)
    {
        const uint Mask = 0x3FFu;

        int dx = (int)( packedA        & Mask) -
                (int)( packedB        & Mask);

        int dy = (int)((packedA >> 10) & Mask) -
                (int)((packedB >> 10) & Mask);

        int dz = (int)((packedA >> 20) & Mask) -
                (int)((packedB >> 20) & Mask);

        return abs(dx) <= 1 &&
               abs(dy) <= 1 &&
               abs(dz) <= 1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int WrapToGrid(int value, int gridSize)
    {
        int wrapped = value % gridSize;
        return wrapped < 0 ? wrapped + gridSize : wrapped;
    }
#endregion


#region Collision data
    private static bool isInitialised = false;
    private static int3 broadphaseGridSize = new int3(24);
    private static int broadphaseWorkerChunkCount = 16;
    private static int prefixScanBlockCount = 8;

    private static NativeList<int3> latticeCellAABBmin;
    private static NativeList<uint> latticeCellAABBspan;
    private static NativeArray<int> simCellsEntries;
    private static NativeArray<int> simCellsEntriesOffset;
    private static NativeArray<int> prefixChunkSizes;

    private static NativeList<int4> broadphaseEntries;
    private static NativeArray<int> simCellsNumberOfPairs;
    private static NativeReference<int> numberOfCandidatePairs;
    private static NativeList<int2> candidatePairs;

#endregion


#region Main function
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void RunCollision(
        float dt,
        float simCellSize,
        ref JobHandle handle,
        NativeList<Particle>        particles,
        NativeList<LatticeCell>     latticeCells,
        NativeList<uint2>           latticeCellLatticeCoords,
        NativeList<DynamicVertex>   collisionVertices,
        NativeList<int3>            collisionTriangles,
        NativeList<int2>            collisionEdges,
        NativeList<CollisionWraper> collisionWrapers
    )
    {
        Initialise();

        
        int numberOfLatticeCells = latticeCells.Length;
        int numberOfBroadphaseCells = broadphaseGridSize.x * broadphaseGridSize.y * broadphaseGridSize.z;
        int prefixChunkSize = (numberOfBroadphaseCells + prefixScanBlockCount - 1) / prefixScanBlockCount;
        numberOfCandidatePairs.Value = 0;

        // Resize the lattice cells helper containers if needed
        latticeCellAABBmin.ResizeUninitialized(numberOfLatticeCells);
        latticeCellAABBspan.ResizeUninitialized(numberOfLatticeCells);

        // -------------------------------------------------
        //                   Step 1
        // For each lattice cell create it's AABB extends,
        // then count the total span and for each sim cell
        // or grid cell that it touches increment it's count
        // at the end prefix sum using a parallel algorithm

        CreateLatticeCellsAABB createCellAABB = new CreateLatticeCellsAABB
        {
            invBroadphaseCellSize = 1f / (simCellSize * 1),
            particles = particles,
            latticeCells = latticeCells,
            latticeCellAABBmin = latticeCellAABBmin.AsArray(),
            latticeCellAABBspan = latticeCellAABBspan.AsArray()
        };
        handle = createCellAABB.Schedule(numberOfLatticeCells, 8, handle);
        CountLatticeCellsAABB countAABBGridPos = new CountLatticeCellsAABB
        {
            broadphaseGridSize = broadphaseGridSize,
            chunkSize = (numberOfLatticeCells + broadphaseWorkerChunkCount - 1) / broadphaseWorkerChunkCount,
            latticeCellAABBmin = latticeCellAABBmin,
            latticeCellAABBspan = latticeCellAABBspan,
            simCellsEntries = simCellsEntries
        };
        handle = countAABBGridPos.Schedule(broadphaseWorkerChunkCount, 1, handle);
        PrefixSumSimCellsEntries prefixSumGridCells = new PrefixSumSimCellsEntries
        {
            numberOfBroadphaseCells = numberOfBroadphaseCells,
            broadphaseWorkerChunkCount = broadphaseWorkerChunkCount,
            prefixBlockEntryTotals = prefixChunkSize,
            simCellsEntries = simCellsEntries,
            simCellsEntriesOffset = simCellsEntriesOffset,
            prefixChunkSizes = prefixChunkSizes
        };
        handle = prefixSumGridCells.Schedule(prefixScanBlockCount, 1, handle);
        handle.Complete();

        // Simple prefix sum to get the number of AABB entries
        int AABBentriesCount = 0;
        for (int i = 0; i < prefixScanBlockCount; i++)
        {
            int chunkSize = prefixChunkSizes[i];
            prefixChunkSizes[i] = AABBentriesCount;
            AABBentriesCount += chunkSize;
        }
        // Resize the entry list
        broadphaseEntries.ResizeUninitialized(AABBentriesCount);
        // Store only the last chunk of sim cells offsets to know the total offset for each sim cell
        NativeSlice<int> finalChunkCellEndOffsets = new NativeSlice<int>(simCellsEntriesOffset, numberOfBroadphaseCells * (broadphaseWorkerChunkCount-1));


        // -------------------------------------------------
        //                   Step 2
        // Scatter the entries based on each sim cell start
        // using the last chunk of the simCellsEntriesOffset
        // then for each cell even if it's not used count the 
        // actual collision pairs based on the coordinates
        // common min corner and if they are in the same body,
        // Last step is to prefix sum to get the total number
        // of candidate pairs and also the per sim cell offset

        ScatterAABBentries scatterAABBCells = new ScatterAABBentries
        {
            broadphaseGridSize = broadphaseGridSize,
            chunkSize = (numberOfLatticeCells + broadphaseWorkerChunkCount - 1) / broadphaseWorkerChunkCount,
            latticeCellAABBmin = latticeCellAABBmin,
            latticeCellAABBspan = latticeCellAABBspan,
            prefixChunkSize = prefixChunkSize,
            prefixChunkSizes = prefixChunkSizes,
            simCellsEntriesOffset = simCellsEntriesOffset,
            latticeCellAABBentry = broadphaseEntries.AsArray()
        };
        handle = scatterAABBCells.Schedule(broadphaseWorkerChunkCount, 1, handle);
        SortSimCellsEntries sortGridCellEntries = new SortSimCellsEntries
        {
            prefixChunkSize = prefixChunkSize,
            prefixChunkSizes = prefixChunkSizes,
            latticeCellAABBmin = latticeCellAABBmin,
            latticeCellLatticeCoords = latticeCellLatticeCoords,
            simCellsEntriesOffset = finalChunkCellEndOffsets,
            broadphaseEntries = broadphaseEntries.AsArray(),
            simCellsNumberOfPairs = simCellsNumberOfPairs
        };
        handle = sortGridCellEntries.Schedule(numberOfBroadphaseCells, 1, handle);
        prefixSumSimCellPairs prefixSumCellPairs = new prefixSumSimCellPairs
        {
            totalbroadphaseGridSize = numberOfBroadphaseCells,
            simCellsNumberOfPairs = simCellsNumberOfPairs,
            numberOfCandidatePairs = numberOfCandidatePairs
        };
        handle = prefixSumCellPairs.Schedule(handle);
        handle.Complete();


        // Resize the final candidate pairs list
        candidatePairs.ResizeUninitialized(numberOfCandidatePairs.Value);


        // -------------------------------------------------
        //                   Step 3
        // The last step for broadphase is to use the same
        // algorithm as the sorting part but instead of just
        // counting you actually add the pairs.

        AddCandidatePairs addPairs = new AddCandidatePairs
        {
            prefixChunkSize = prefixChunkSize,
            prefixChunkSizes = prefixChunkSizes,
            latticeCellAABBmin = latticeCellAABBmin,
            latticeCellLatticeCoords = latticeCellLatticeCoords,
            simCellsEntriesOffset = finalChunkCellEndOffsets,
            latticeCellAABBentry = broadphaseEntries.AsArray(),
            simCellsNumberOfPairs = simCellsNumberOfPairs,
            candidatePairs = candidatePairs.AsArray()
        };
        handle = addPairs.Schedule(numberOfBroadphaseCells, 1, handle);
        handle.Complete();

        
        DebugCollision(particles, latticeCells);
    }
#endregion
#region Helper Functions

    private static void DebugCollision(
        NativeList<Particle>        particles,
        NativeList<LatticeCell>     latticeCells
    )
    {
        Debug.Log(numberOfCandidatePairs.Value);
        for (int i = 0; i < numberOfCandidatePairs.Value; i++)
        {
            int2 pair = candidatePairs[i];

            LatticeCell latticeCellA = latticeCells[pair.x];
            LatticeCell latticeCellB = latticeCells[pair.y];

            float3 MidPoint(LatticeCell latticeCell)
            {
                float3 midPoint = new float3(0);

                midPoint += particles[latticeCell.v0].currentPosition;
                midPoint += particles[latticeCell.v1].currentPosition;
                midPoint += particles[latticeCell.v2].currentPosition;
                midPoint += particles[latticeCell.v3].currentPosition;
                midPoint += particles[latticeCell.v4].currentPosition;
                midPoint += particles[latticeCell.v5].currentPosition;
                midPoint += particles[latticeCell.v6].currentPosition;
                midPoint += particles[latticeCell.v7].currentPosition;
                midPoint *= 1f/8f;

                return midPoint;
            }

            Debug.DrawLine(MidPoint(latticeCellA), MidPoint(latticeCellB), Color.red, Time.fixedDeltaTime);
        }


        // int maxGridCellSize = 0;
        // float averageGridCellSize = 0;
        // int under2 = 0;
        // for (int i = 0; i < activeGridCells.Length; i++)
        // {
        //     int gridCellSize = activeGridCells[i] - (i==0 ? 0 : activeGridCells[i-1]);
        //     if (gridCellSize > maxGridCellSize) maxGridCellSize = gridCellSize;
        //     if (gridCellSize < 2) under2++;
        //     averageGridCellSize += gridCellSize;
        // }
        // averageGridCellSize /= activeGridCells.Length;
        // Debug.Log(
        //     "Number of AABB entries: " + activeGridCells[activeGridCells.Length-1] +
        //     "\nNumber of active grid cells: " + activeGridCells.Length +
        //     "\nMax grid cell size: " + maxGridCellSize +
        //     "\nAverage grid cell size: " + averageGridCellSize +
        //     "\nUnder 2: " + under2
        // );
    }

    public static void Initialise()
    {
        if (isInitialised) return;
        isInitialised = true;

        latticeCellAABBmin = new (Allocator.Persistent);
        latticeCellAABBspan = new (Allocator.Persistent);
        simCellsEntries = new NativeArray<int>(
            broadphaseWorkerChunkCount * broadphaseGridSize.x * broadphaseGridSize.y * broadphaseGridSize.z,
        Allocator.Persistent);
        simCellsEntriesOffset = new NativeArray<int>(
            broadphaseWorkerChunkCount * broadphaseGridSize.x * broadphaseGridSize.y * broadphaseGridSize.z,
        Allocator.Persistent);
        prefixChunkSizes = new NativeArray<int>(prefixScanBlockCount, Allocator.Persistent);
        broadphaseEntries = new (Allocator.Persistent);
        simCellsNumberOfPairs = new (broadphaseGridSize.x * broadphaseGridSize.y * broadphaseGridSize.z, Allocator.Persistent);
        numberOfCandidatePairs = new (0, Allocator.Persistent);
        candidatePairs = new (Allocator.Persistent);
    }

    public static void Disable()
    {
        if (!isInitialised) return;
        isInitialised = false;

        latticeCellAABBmin.Dispose();
        latticeCellAABBspan.Dispose();
        simCellsEntries.Dispose();
        simCellsEntriesOffset.Dispose();
        prefixChunkSizes.Dispose();
        broadphaseEntries.Dispose();
        simCellsNumberOfPairs.Dispose();
        numberOfCandidatePairs.Dispose();
        candidatePairs.Dispose();
    }

#endregion
}
}