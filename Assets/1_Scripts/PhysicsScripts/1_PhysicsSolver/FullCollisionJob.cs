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
    [ReadOnly] public NativeList<int> activeCollisionLattices;
    [ReadOnly] public NativeList<LatticeCell> latticeCells;

    [WriteOnly] public NativeArray<int3> latticeCellAABBmin;
    [WriteOnly] public NativeArray<uint> latticeCellAABBspan;

    
    public void Execute(int activeCellIndex)
    {
        LatticeCell c = latticeCells[activeCollisionLattices[activeCellIndex]];
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

        latticeCellAABBmin[activeCellIndex] = minCorner;
        latticeCellAABBspan[activeCellIndex] = FullCollisionJob.PackSpanUint(maxCorner - minCorner + 1);
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
    [WriteOnly] public NativeArray<int4> broadphaseEntries;


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

                        broadphaseEntries[simCellsEntriesOffset[chunkIndex * totalbroadphaseGridSize + bucketIndex]++
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
    [ReadOnly] public NativeList<int> activeCollCellBodyRef;
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
                int latticeACoords = activeCollCellBodyRef[latticeCellAId];

                for (int j = i + 1; j < end; j++)
                {
                    int latticeCellBId = broadphaseEntries[j].w;
                    int latticeBCoords = activeCollCellBodyRef[latticeCellBId];
                    if (latticeACoords == latticeBCoords) continue;

                    int3 minBCell = latticeCellAABBmin[latticeCellBId];
                    int3 commonMinCell = max(minACell, minBCell);

                    if (!all(commonMinCell == key)) continue;

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
    [ReadOnly] public NativeList<int> activeCollCellBodyRef;
    [ReadOnly] public NativeSlice<int> simCellsEntriesOffset;
    [ReadOnly] public NativeArray<int4> latticeCellAABBentry;
    [ReadOnly] public NativeArray<int> simCellsNumberOfPairs;
    [ReadOnly] public NativeList<int> activeCollisionLattices;

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
                int latticeACoords = activeCollCellBodyRef[latticeCellAId];

                for (int j = i + 1; j < end; j++)
                {
                    int latticeCellBId = latticeCellAABBentry[j].w;
                    int latticeBCoords = activeCollCellBodyRef[latticeCellBId];
                    if (latticeACoords == latticeBCoords) continue;

                    int3 minBCell = latticeCellAABBmin[latticeCellBId];
                    int3 commonMinCell = max(minACell, minBCell);

                    if (!all(commonMinCell == key)) continue;

                    candidatePairs[writeStart++] = 
                        new int2(activeCollisionLattices[latticeCellAId], 
                                 activeCollisionLattices[latticeCellBId]);
                }
            }

            start = end;
        }
    }
}

/*
Evaluates every collision vertex at the start of the substep
(previousPosition) from its tet and barycentric weights.

One work item per collision wrapper. Collision vertices are not shared
between cells, so every wrapper writes a disjoint range.
*/
[BurstCompile]
public struct ColVertsPrevPositions : IJobParallelFor
{
    [ReadOnly] public NativeList<Particle> particles;
    [ReadOnly] public NativeList<LatticeCell> latticeCells;
    [ReadOnly] public NativeList<DynamicVertex> collisionVertices;
    [ReadOnly] public NativeList<CollisionWraper> collisionWrapers;
    [NativeDisableParallelForRestriction]
    [WriteOnly] public NativeArray<float3> colVertPrevPositions;

    // Tetra order, the same with pre processor
    private static readonly int4[] s_CellTets =
    {
        new int4(0, 3, 6, 5),
        new int4(0, 5, 6, 4),
        new int4(0, 6, 3, 2),
        new int4(5, 3, 6, 7),
        new int4(0, 3, 5, 1)
    };


    public void Execute(int wraperIndex)
    {
        CollisionWraper wraper = collisionWrapers[wraperIndex];
        LatticeCell cell = latticeCells[wraper.cellIdx];

        for (int v = wraper.colVertsRange.x; v < wraper.colVertsRange.y; v++)
        {
            DynamicVertex vertex = collisionVertices[v];
            int4 tet = s_CellTets[vertex.tet];

            colVertPrevPositions[v] =
                vertex.bary.x * particles[FullCollisionJob.CellNode(cell, tet.x)].previousPosition +
                vertex.bary.y * particles[FullCollisionJob.CellNode(cell, tet.y)].previousPosition +
                vertex.bary.z * particles[FullCollisionJob.CellNode(cell, tet.z)].previousPosition +
                vertex.bary.w * particles[FullCollisionJob.CellNode(cell, tet.w)].previousPosition;
        }
    }
}

/*
Narrow phase: one speculative contact per candidate cell pair.

For a pair (A, B) with both cells owning a collision mesh:
    - Vertices of A against triangles of B.
    - Vertices of B against triangles of A.
    - Edges of A against edges of B.
    The closest of all these feature pairs gives:

    pairContacts[pairIndex] = float4(n, gap)
        n   = unit direction from B's closest point to A's closest point
        gap = distance between the two inner meshes

All positions are start-of-substep, so stage 3 guarantees gap >= hardSkin
in normal operation.

Special values:
    gap = +inf     at least one cell has no collision mesh, pair is inactive
    n   = 0        meshes touch (gap ~ 0), normal is undefined and the
                   solver falls back to the cell centre direction
*/
[BurstCompile]
public struct CellPairsNarrowPhase : IJobParallelFor
{
    [ReadOnly] public NativeArray<int2> candidatePairs;
    [ReadOnly] public NativeList<int> latticeColReferences;   // cell -> wrapper, -1 = no collision mesh
    [ReadOnly] public NativeList<CollisionWraper> collisionWrapers;
    [ReadOnly] public NativeList<int3> collisionTriangles;
    [ReadOnly] public NativeList<int2> collisionEdges;
    [ReadOnly] public NativeArray<float3> colVertPrevPositions;
    [WriteOnly] public NativeArray<float4> pairContacts;


    public void Execute(int pairIndex)
    {
        int2 pair = candidatePairs[pairIndex];
        int wraperAId = latticeColReferences[pair.x];
        int wraperBId = latticeColReferences[pair.y];

        CollisionWraper A = collisionWrapers[wraperAId];
        CollisionWraper B = collisionWrapers[wraperBId];

        float bestDistSq = float.MaxValue;
        float3 bestA = new float3(0);
        float3 bestB = new float3(0);

        // Vertices of A against triangles of B
        for (int t = B.colTrisRange.x; t < B.colTrisRange.y; t++)
        {
            int3 tri = collisionTriangles[t];
            float3 t0 = colVertPrevPositions[tri.x];
            float3 t1 = colVertPrevPositions[tri.y];
            float3 t2 = colVertPrevPositions[tri.z];

            for (int v = A.colVertsRange.x; v < A.colVertsRange.y; v++)
            {
                float3 p = colVertPrevPositions[v];
                float3 q = FullCollisionJob.ClosestPointOnTriangle(p, t0, t1, t2);
                float distSq = lengthsq(p - q);
                if (distSq < bestDistSq) { bestDistSq = distSq; bestA = p; bestB = q; }
            }
        }

        // Vertices of B against triangles of A
        for (int t = A.colTrisRange.x; t < A.colTrisRange.y; t++)
        {
            int3 tri = collisionTriangles[t];
            float3 t0 = colVertPrevPositions[tri.x];
            float3 t1 = colVertPrevPositions[tri.y];
            float3 t2 = colVertPrevPositions[tri.z];

            for (int v = B.colVertsRange.x; v < B.colVertsRange.y; v++)
            {
                float3 p = colVertPrevPositions[v];
                float3 q = FullCollisionJob.ClosestPointOnTriangle(p, t0, t1, t2);
                float distSq = lengthsq(p - q);
                if (distSq < bestDistSq) { bestDistSq = distSq; bestA = q; bestB = p; }
            }
        }

        // Edges of A against edges of B
        for (int ea = A.colEdgesRange.x; ea < A.colEdgesRange.y; ea++)
        {
            int2 edgeA = collisionEdges[ea];
            float3 a0 = colVertPrevPositions[edgeA.x];
            float3 a1 = colVertPrevPositions[edgeA.y];

            for (int eb = B.colEdgesRange.x; eb < B.colEdgesRange.y; eb++)
            {
                int2 edgeB = collisionEdges[eb];
                FullCollisionJob.ClosestPointsSegmentSegment(
                    a0, a1, colVertPrevPositions[edgeB.x], colVertPrevPositions[edgeB.y],
                    out float3 ca, out float3 cb);
                float distSq = lengthsq(ca - cb);
                if (distSq < bestDistSq) { bestDistSq = distSq; bestA = ca; bestB = cb; }
            }
        }

        float gap = sqrt(bestDistSq);
        float3 n = gap > 1e-7f ? (bestA - bestB) / gap : new float3(0);
        pairContacts[pairIndex] = new float4(n, gap);
    }
}

/*
Stage 1: speculative pair-relative velocity contacts on cells.

Cells act as rigid bodies here. A correction is added uniformly to all
8 nodes, so with cell mass = sum of node masses momentum is conserved
exactly. Contacts only push (c >= 0 is skipped), there is no hard stop.

Contact normal and gap come from CellPairsNarrowPhase (start of substep),
the approach comes from the cell centre displacement this substep.
*/
[BurstCompile]
public struct LatticeCellsVelTransfer : IJob
{
    public float skin;       // soft skin, stages 2 and 3 use skin / 2
    public int sweeps;       // passes over all pairs, around 8

    [ReadOnly] public NativeList<LatticeCell> latticeCells;
    [ReadOnly] public NativeArray<int2> candidatePairs;
    [ReadOnly] public NativeArray<float4> pairContacts;

    public NativeList<Particle> particles;

    public void Execute()
    {
        for (int sweep = 0; sweep < sweeps; sweep++)
        {
            for (int i = 0; i < candidatePairs.Length; i++)
            {
                float4 contact = pairContacts[i];
                if (math.isinf(contact.w)) continue;   // a cell without a collision mesh

                LatticeCell A = latticeCells[candidatePairs[i].x];
                LatticeCell B = latticeCells[candidatePairs[i].y];

                float wA = CellInvMass(A);
                float wB = CellInvMass(B);
                if (wA + wB <= 0f) continue;

                float3 prevA = CellCenter(A, true);
                float3 prevB = CellCenter(B, true);

                // contact from the START of the substep, n points from B to A
                float3 n = contact.xyz;
                float gap = contact.w;
                if (math.lengthsq(n) == 0f)
                {
                    // meshes touch, no feature normal, use the centre direction
                    n = prevA - prevB;
                    float dist = math.length(n);
                    if (dist < 1e-6f) continue;
                    n /= dist;
                }

                // displacement of each cell this substep (current is re-read, so earlier corrections are seen)
                float3 dispA = CellCenter(A, false) - prevA;
                float3 dispB = CellCenter(B, false) - prevB;

                float vn = math.dot(dispA - dispB, n);
                float c = vn + math.max(gap - skin, 0f);   // all lengths, no dt
                if (c >= 0f) continue;                      // contacts only push

                float dl = -c / (wA + wB);

                AddToCell(A, n * (wA * dl));
                AddToCell(B, n * (-wB * dl));
            }
        }
    }

    private float3 CellCenter(in LatticeCell c, bool previous)
    {
        float3 s = new float3(0);
        for (int k = 0; k < 8; k++)
        {
            Particle p = particles[FullCollisionJob.CellNode(c, k)];
            s += previous ? p.previousPosition : p.currentPosition;
        }
        return s * 0.125f;
    }

    // 1 / (sum of node masses). A cell with a pinned node does not move.
    private float CellInvMass(in LatticeCell c)
    {
        float mass = 0f;
        for (int k = 0; k < 8; k++)
        {
            float w = particles[FullCollisionJob.CellNode(c, k)].invMass;
            if (w <= 0f) return 0f;
            mass += 1f / w;
        }
        return 1f / mass;
    }

    private void AddToCell(in LatticeCell c, float3 d)
    {
        for (int k = 0; k < 8; k++)
        {
            int idx = FullCollisionJob.CellNode(c, k);
            Particle p = particles[idx];
            p.currentPosition += d;
            particles[idx] = p;
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

    // Particle index of local corner k (0..7) of a lattice cell
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int CellNode(in LatticeCell c, int k)
    {
        switch (k)
        {
            case 0: return c.v0; case 1: return c.v1; case 2: return c.v2; case 3: return c.v3;
            case 4: return c.v4; case 5: return c.v5; case 6: return c.v6; default: return c.v7;
        }
    }

    /*
    Closest point to p on triangle (a, b, c), Ericson 5.1.5.
    A degenerate (collinear) triangle returns a, its edges are still
    covered by the edge-edge tests, so the pair distance stays exact.
    */
    public static float3 ClosestPointOnTriangle(float3 p, float3 a, float3 b, float3 c)
    {
        float3 ab = b - a;
        float3 ac = c - a;
        float3 ap = p - a;
        float d1 = dot(ab, ap);
        float d2 = dot(ac, ap);
        if (d1 <= 0f && d2 <= 0f) return a;

        float3 bp = p - b;
        float d3 = dot(ab, bp);
        float d4 = dot(ac, bp);
        if (d3 >= 0f && d4 <= d3) return b;

        float vc = d1 * d4 - d3 * d2;
        if (vc <= 0f && d1 >= 0f && d3 <= 0f) return a + (d1 / (d1 - d3)) * ab;

        float3 cp = p - c;
        float d5 = dot(ab, cp);
        float d6 = dot(ac, cp);
        if (d6 >= 0f && d5 <= d6) return c;

        float vb = d5 * d2 - d1 * d6;
        if (vb <= 0f && d2 >= 0f && d6 <= 0f) return a + (d2 / (d2 - d6)) * ac;

        float va = d3 * d6 - d5 * d4;
        if (va <= 0f && (d4 - d3) >= 0f && (d5 - d6) >= 0f)
            return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b);

        float sum = va + vb + vc;
        if (sum <= 1e-30f) return a;
        float v = vb / sum;
        float w = vc / sum;
        return a + ab * v + ac * w;
    }

    /*
    Closest points between segments (p1, q1) and (p2, q2), Ericson 5.1.9.
    Near-parallel segments take s = 0 first, the clamping that follows
    still gives a valid closest pair.
    */
    public static void ClosestPointsSegmentSegment(
        float3 p1, float3 q1, float3 p2, float3 q2, out float3 c1, out float3 c2)
    {
        const float Eps = 1e-12f;
        float3 d1 = q1 - p1;
        float3 d2 = q2 - p2;
        float3 r = p1 - p2;
        float a = dot(d1, d1);
        float e = dot(d2, d2);
        float f = dot(d2, r);
        float s, t;

        if (a <= Eps && e <= Eps) { c1 = p1; c2 = p2; return; }

        if (a <= Eps)
        {
            s = 0f;
            t = saturate(f / e);
        }
        else
        {
            float c = dot(d1, r);
            if (e <= Eps)
            {
                t = 0f;
                s = saturate(-c / a);
            }
            else
            {
                float b = dot(d1, d2);
                float denom = a * e - b * b;
                s = denom > 1e-6f * a * e ? saturate((b * f - c * e) / denom) : 0f;
                t = (b * s + f) / e;

                if (t < 0f)      { t = 0f; s = saturate(-c / a); }
                else if (t > 1f) { t = 1f; s = saturate((b - c) / a); }
            }
        }

        c1 = p1 + d1 * s;
        c2 = p2 + d2 * t;
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

    private static NativeList<float3> colVertPrevPositions;
    private static NativeList<float4> pairContacts;

#endregion


#region Main function
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void BroadPhaseRun(
        float simCellSize,
        ref JobHandle handle,
        NativeList<Particle>        particles,
        NativeList<LatticeCell>     latticeCells,
        NativeList<int>             activeCollisionLattices,
        NativeList<int>             activeCollCellBodyRef
    )
    {
        int numberOfActiveLatticeCells = activeCollisionLattices.Length;
        int numberOfParticles = particles.Length;
        int numberOfBroadphaseCells = broadphaseGridSize.x * broadphaseGridSize.y * broadphaseGridSize.z;
        int prefixChunkSize = (numberOfBroadphaseCells + prefixScanBlockCount - 1) / prefixScanBlockCount;
        numberOfCandidatePairs.Value = 0;

        // Resize the lattice cells helper containers if needed
        latticeCellAABBmin.ResizeUninitialized(numberOfActiveLatticeCells);
        latticeCellAABBspan.ResizeUninitialized(numberOfActiveLatticeCells);

        // ==========================================
        //                   Step 1
        // For each lattice cell create it's AABB extends,
        // then count the total span and for each sim cell
        // or grid cell that it touches increment it's count
        // at the end prefix sum using a parallel algorithm
        // ==========================================

        CreateLatticeCellsAABB createCellAABB = new CreateLatticeCellsAABB
        {
            invBroadphaseCellSize = 1f / (simCellSize * 1),
            particles = particles,
            activeCollisionLattices = activeCollisionLattices,
            latticeCells = latticeCells,
            latticeCellAABBmin = latticeCellAABBmin.AsArray(),
            latticeCellAABBspan = latticeCellAABBspan.AsArray()
        };
        handle = createCellAABB.Schedule(numberOfActiveLatticeCells, 8, handle);
        CountLatticeCellsAABB countAABBGridPos = new CountLatticeCellsAABB
        {
            broadphaseGridSize = broadphaseGridSize,
            chunkSize = (numberOfActiveLatticeCells + broadphaseWorkerChunkCount - 1) / broadphaseWorkerChunkCount,
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


        // ==========================================
        //                   Step 2
        // Scatter the entries based on each sim cell start
        // using the last chunk of the simCellsEntriesOffset
        // then for each cell even if it's not used count the 
        // actual collision pairs based on the coordinates
        // common min corner and if they are in the same body,
        // Last step is to prefix sum to get the total number
        // of candidate pairs and also the per sim cell offset
        // ==========================================

        ScatterAABBentries scatterAABBCells = new ScatterAABBentries
        {
            broadphaseGridSize = broadphaseGridSize,
            chunkSize = (numberOfActiveLatticeCells + broadphaseWorkerChunkCount - 1) / broadphaseWorkerChunkCount,
            latticeCellAABBmin = latticeCellAABBmin,
            latticeCellAABBspan = latticeCellAABBspan,
            prefixChunkSize = prefixChunkSize,
            prefixChunkSizes = prefixChunkSizes,
            simCellsEntriesOffset = simCellsEntriesOffset,
            broadphaseEntries = broadphaseEntries.AsArray()
        };
        handle = scatterAABBCells.Schedule(broadphaseWorkerChunkCount, 1, handle);
        SortSimCellsEntries sortGridCellEntries = new SortSimCellsEntries
        {
            prefixChunkSize = prefixChunkSize,
            prefixChunkSizes = prefixChunkSizes,
            latticeCellAABBmin = latticeCellAABBmin,
            activeCollCellBodyRef = activeCollCellBodyRef,
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
        // (contacts too, so the per substep path never resizes)
        candidatePairs.ResizeUninitialized(numberOfCandidatePairs.Value);
        pairContacts.ResizeUninitialized(numberOfCandidatePairs.Value);


        // ==========================================
        //                   Step 3
        // The last step for broadphase is to use the same
        // algorithm as the sorting part but instead of just
        // counting you actually add the pairs.
        // ==========================================

        AddCandidatePairs addPairs = new AddCandidatePairs
        {
            prefixChunkSize = prefixChunkSize,
            prefixChunkSizes = prefixChunkSizes,
            latticeCellAABBmin = latticeCellAABBmin,
            activeCollCellBodyRef = activeCollCellBodyRef,
            simCellsEntriesOffset = finalChunkCellEndOffsets,
            latticeCellAABBentry = broadphaseEntries.AsArray(),
            simCellsNumberOfPairs = simCellsNumberOfPairs,
            activeCollisionLattices = activeCollisionLattices,
            candidatePairs = candidatePairs.AsArray()
        };
        handle = addPairs.Schedule(numberOfBroadphaseCells, 1, handle);

        // The complete is needed only when debugging.
        handle.Complete();
        DebugBroadPhase(particles, latticeCells);
    }

    /*
    Per substep collision. Call after the particle predict step,
    BroadPhaseRun must have run this frame (it sizes every list used here).

    1. Collision vertices at the start of the substep.
    2. Narrow phase, one (n, gap) contact per candidate pair.
    3. Stage 1 velocity transfer.
    */
    public static void RunCollision(
        float simCellSize,
        float skin,
        int sweeps,
        ref JobHandle handle,
        NativeList<Particle>        particles,
        NativeList<LatticeCell>     latticeCells,
        NativeList<int>             activeCollisionLattices,
        NativeList<int>             activeCollCellBodyRef,
        NativeList<int>             latticeColReferences,
        NativeList<DynamicVertex>   collisionVertices,
        NativeList<int3>            collisionTriangles,
        NativeList<int2>            collisionEdges,
        NativeList<CollisionWraper> collisionWrapers
    )
    {
        Initialise();

        colVertPrevPositions.ResizeUninitialized(collisionVertices.Length);

        BroadPhaseRun(
            simCellSize,
            ref handle,
            particles,
            latticeCells,
            activeCollisionLattices,
            activeCollCellBodyRef
        );
        
        ColVertsPrevPositions colVertsPositions = new ColVertsPrevPositions
        {
            particles = particles,
            latticeCells = latticeCells,
            collisionVertices = collisionVertices,
            collisionWrapers = collisionWrapers,
            colVertPrevPositions = colVertPrevPositions.AsArray()
        };
        handle = colVertsPositions.Schedule(collisionWrapers.Length, 8, handle);
        CellPairsNarrowPhase narrowPhase = new CellPairsNarrowPhase
        {
            candidatePairs = candidatePairs.AsArray(),
            latticeColReferences = latticeColReferences,
            collisionWrapers = collisionWrapers,
            collisionTriangles = collisionTriangles,
            collisionEdges = collisionEdges,
            colVertPrevPositions = colVertPrevPositions.AsArray(),
            pairContacts = pairContacts.AsArray()
        };
        handle = narrowPhase.Schedule(candidatePairs.Length, 8, handle);
        LatticeCellsVelTransfer velTransfer = new LatticeCellsVelTransfer
        {
            skin = skin,
            sweeps = sweeps,
            latticeCells = latticeCells,
            candidatePairs = candidatePairs.AsArray(),
            pairContacts = pairContacts.AsArray(),
            particles = particles
        };
        handle = velTransfer.Schedule(handle);
        handle.Complete();
    }
#endregion
#region Helper Functions

    private static void DebugBroadPhase(
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
        colVertPrevPositions = new (Allocator.Persistent);
        pairContacts = new (Allocator.Persistent);
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
        colVertPrevPositions.Dispose();
        pairContacts.Dispose();
    }

#endregion
}
}