using UnityEngine;
using SolverStructs;
using Unity.Collections;
using Unity.Mathematics;
using static Unity.Mathematics.math;
using Unity.Jobs;
using Unity.Burst;
using System.Runtime.CompilerServices;


/*
Collision between soft bodies, run once per substep by FullCollisionJob.RunCollision.

Pipeline:
    Broad phase   swept cell boxes -> wrapped grid -> candidate cell pairs
    Narrow phase  collision vertex positions -> one (normal, gap) per pair
    Stage 1       speculative velocity contacts (momentum exchange)
    Stage 2       slide along the contact planes, the lighter side gives way
    Stage 3       hard planar Divide-and-Truncate (no penetration)

Positions:
    previousPosition  start of the substep (separated, guaranteed by stage 3)
    currentPosition   predicted end of the substep, the only one changed here

Index spaces (the easiest thing to mix up):
    lattice cell   index into latticeCells
    active cell    index into activeCollisionLattices, only cells that own a
                   collision mesh, the broad phase works in this space
    wrapper        index into collisionWrapers (latticeColReferences maps
                   lattice cell -> wrapper)
    collision vert global index into collisionVertices, every wrapper owns
                   a contiguous range, vertices are never shared between cells
    bucket         one cell of the fixed size, wrapped broad-phase grid
*/
namespace CollisionHandler
{
#region Broad phase jobs

/*
Broad phase 1/7: one swept, skin inflated box per active collision cell.

The box covers the 8 nodes at the start (previous) and end (current) of the
substep, grown by skin / 2 on every side. Two cells that can get closer than
skin during the substep therefore always share a grid cell.

Output, in unwrapped grid coordinates (wrapping happens later):
    latticeCellAABBmin[active]    lowest covered grid cell
    latticeCellAABBspan[active]   covered grid cells per axis, packed
*/
[BurstCompile]
public struct CreateLatticeCellsAABB : IJobParallelFor
{
    [ReadOnly] public float invBroadphaseCellSize;
    [ReadOnly] public float skin;
    [ReadOnly] public NativeList<Particle> particles;
    [ReadOnly] public NativeList<int> activeCollisionLattices;
    [ReadOnly] public NativeList<LatticeCell> latticeCells;

    [WriteOnly] public NativeArray<int3> latticeCellAABBmin;
    [WriteOnly] public NativeArray<uint> latticeCellAABBspan;


    public void Execute(int activeCellIndex)
    {
        LatticeCell c = latticeCells[activeCollisionLattices[activeCellIndex]];

        float3 minPos = new float3(float.MaxValue);
        float3 maxPos = new float3(float.MinValue);
        for (int k = 0; k < 8; k++)
        {
            Particle p = particles[FullCollisionJob.CellNode(c, k)];
            minPos = min(minPos, min(p.previousPosition, p.currentPosition));
            maxPos = max(maxPos, max(p.previousPosition, p.currentPosition));
        }
        minPos -= 0.5f * skin;
        maxPos += 0.5f * skin;

        // inclusive on both ends, a box touching a grid line goes into both sides
        int3 minCorner = (int3)floor(minPos * invBroadphaseCellSize);
        int3 maxCorner = (int3)floor(maxPos * invBroadphaseCellSize);

        latticeCellAABBmin[activeCellIndex] = minCorner;
        latticeCellAABBspan[activeCellIndex] = FullCollisionJob.PackSpanUint(maxCorner - minCorner + 1);
    }
}

/*
Broad phase 2/7: counts how many entries every bucket gets.

One work item per worker chunk (a contiguous range of active cells). Every
grid cell a box covers is wrapped into a bucket and counted in the chunk's
own row, so two chunks never write the same counter:

    simCellsEntries[chunkIndex * totalGridSize + bucketIndex]

simCellsEntries must be zero before this job, the prefix sum clears it again.
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
                        int bucketIndex = xyOffset + wrappedZ * gridStrideZ;

                        simCellsEntries[chunkIndex * totalbroadphaseGridSize + bucketIndex]++;
                    }
                }
            }
        }
    }
}

/*
Broad phase 3/7: turns the entry counts into write offsets.

One work item per prefix block (a contiguous range of buckets). Inside its
block it walks bucket by bucket, and inside a bucket chunk by chunk, so every
(chunk, bucket) gets a start offset local to the block:

    simCellsEntriesOffset[chunk * totalGridSize + bucket]   local offset
    prefixChunkSizes[block]                                  entries in the block

The counts are cleared for the next run. The main thread then turns
prefixChunkSizes into block base offsets, local offset + base = global index.
*/
[BurstCompile]
public struct PrefixSumSimCellsEntries : IJobParallelFor
{
    [ReadOnly] public int numberOfBroadphaseCells;
    [ReadOnly] public int broadphaseWorkerChunkCount;
    [ReadOnly] public int prefixChunkSize;      // buckets per prefix block
    [NativeDisableParallelForRestriction]
    public NativeArray<int> simCellsEntries;
    [NativeDisableParallelForRestriction]
    [WriteOnly] public NativeArray<int> simCellsEntriesOffset;
    [WriteOnly] public NativeArray<int> prefixChunkSizes;


    public void Execute(int prefixChunkIndex)
    {
        int bucketStart = prefixChunkSize * prefixChunkIndex;
        int bucketEnd = min(prefixChunkSize * (prefixChunkIndex + 1), numberOfBroadphaseCells);
        int runningOffset = 0;

        for (int bucket = bucketStart; bucket < bucketEnd; bucket++)
        for (int chunk = 0; chunk < broadphaseWorkerChunkCount; chunk++)
        {
            int counterIndex = chunk * numberOfBroadphaseCells + bucket;
            int count = simCellsEntries[counterIndex];

            simCellsEntries[counterIndex] = 0;
            simCellsEntriesOffset[counterIndex] = runningOffset;
            runningOffset += count;
        }

        prefixChunkSizes[prefixChunkIndex] = runningOffset;
    }
}

/*
Broad phase 4/7: writes one entry per covered grid cell.

Same walk as the count job. Every entry is written at its (chunk, bucket)
cursor in simCellsEntriesOffset, advanced after each write, plus the base
offset of the bucket's prefix block:

    entry = int4(unwrapped grid cell xyz, active cell index)

The unwrapped coordinate is kept because different grid cells can wrap into
the same bucket. Afterwards every bucket is a contiguous, unsorted range and
the last chunk's cursors hold the block-local end of every bucket.
*/
[BurstCompile]
public struct ScatterAABBentries : IJobParallelFor
{
    [ReadOnly] public int3 broadphaseGridSize;
    [ReadOnly] public int chunkSize;
    [ReadOnly] public NativeList<int3> latticeCellAABBmin;
    [ReadOnly] public NativeList<uint> latticeCellAABBspan;
    [ReadOnly] public NativeArray<int> prefixChunkSizes;    // block base offsets by now
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
                        int bucketIndex = xyOffset + wrappedZ * gridStrideZ;

                        int writeIndex = simCellsEntriesOffset[chunkIndex * totalbroadphaseGridSize + bucketIndex]++
                            + prefixChunkSizes[bucketIndex / prefixChunkSize];
                        broadphaseEntries[writeIndex] = new int4(cellMin + new int3(x, y, z), i);
                    }
                }
            }
        }
    }
}

/*
Broad phase 5/7: sorts every bucket and counts its candidate pairs.

One work item per bucket. The bucket ends at the last chunk's cursor plus its
block base, and starts where the previous bucket ends. Entries are sorted by
unwrapped grid cell, then every group in the same grid cell is tested pair
by pair:

    same body      rejected, only collision between bodies for now
    owner cell     max(minCornerA, minCornerB), a pair is only counted in
                   that grid cell, so pairs sharing several cells count once

Must use exactly the same test as AddCandidatePairs.
*/
[BurstCompile]
public struct SortSimCellsEntries : IJobParallelFor
{
    [ReadOnly] public NativeArray<int> prefixChunkSizes;
    [ReadOnly] public int prefixChunkSize;
    [ReadOnly] public NativeList<int3> latticeCellAABBmin;
    [ReadOnly] public NativeList<int> activeCollCellBodyRef;
    [ReadOnly] public NativeSlice<int> simCellsEntriesOffset;   // last chunk's cursors
    [NativeDisableParallelForRestriction]
    public NativeArray<int4> broadphaseEntries;
    [WriteOnly] public NativeArray<int> simCellsNumberOfPairs;


    public void Execute(int bucketIndex)
    {
        int bucketStart = bucketIndex == 0 ? 0 : (simCellsEntriesOffset[bucketIndex - 1] +
            prefixChunkSizes[(bucketIndex - 1) / prefixChunkSize]);
        int bucketEnd = simCellsEntriesOffset[bucketIndex] + prefixChunkSizes[bucketIndex / prefixChunkSize];
        int count = bucketEnd - bucketStart;

        if (count <= 1) { simCellsNumberOfPairs[bucketIndex] = 0; return; }

        InsertionSort(broadphaseEntries, bucketStart, count);

        int totalPairCount = 0;
        for (int start = bucketStart; start < bucketEnd;)
        {
            // group of entries in the same unwrapped grid cell
            int end = start + 1;
            int3 key = broadphaseEntries[start].xyz;
            while (end < bucketEnd && all(broadphaseEntries[end].xyz == key))
                end++;

            if (end - start <= 1)
            {
                start = end;
                continue;
            }

            for (int i = start; i < end - 1; i++)
            {
                int activeA = broadphaseEntries[i].w;
                int3 minCornerA = latticeCellAABBmin[activeA];
                int bodyA = activeCollCellBodyRef[activeA];

                for (int j = i + 1; j < end; j++)
                {
                    int activeB = broadphaseEntries[j].w;
                    int bodyB = activeCollCellBodyRef[activeB];
                    if (bodyA == bodyB) continue;

                    int3 ownerCell = max(minCornerA, latticeCellAABBmin[activeB]);
                    if (!all(ownerCell == key)) continue;

                    totalPairCount++;
                }
            }

            start = end;
        }
        simCellsNumberOfPairs[bucketIndex] = totalPairCount;
    }

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

    // Lexicographic x, then y, then z. >0 if a > b, <0 if a < b, 0 if equal
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
Broad phase 6/7: turns the pair count of every bucket into its write offset,
in place, and stores the total in numberOfCandidatePairs.
*/
[BurstCompile]
public struct prefixSumSimCellPairs : IJob
{
    [ReadOnly] public int totalbroadphaseGridSize;
    public NativeArray<int> simCellsNumberOfPairs;
    [WriteOnly] public NativeReference<int> numberOfCandidatePairs;


    public void Execute()
    {
        int runningOffset = 0;
        for (int i = 0; i < totalbroadphaseGridSize; i++)
        {
            int count = simCellsNumberOfPairs[i];
            if (count == 0) continue;
            simCellsNumberOfPairs[i] = runningOffset;
            runningOffset += count;
        }
        numberOfCandidatePairs.Value = runningOffset;
    }
}

/*
Broad phase 7/7: writes the pairs counted by SortSimCellsEntries.

Same walk and same test, every accepted pair is written from the bucket's
offset onward and converted from active to lattice cell indices:

    candidatePairs[i] = int2(latticeCellA, latticeCellB)
*/
[BurstCompile]
public struct AddCandidatePairs : IJobParallelFor
{
    [ReadOnly] public NativeArray<int> prefixChunkSizes;
    [ReadOnly] public int prefixChunkSize;
    [ReadOnly] public NativeList<int3> latticeCellAABBmin;
    [ReadOnly] public NativeList<int> activeCollCellBodyRef;
    [ReadOnly] public NativeSlice<int> simCellsEntriesOffset;   // last chunk's cursors
    [ReadOnly] public NativeArray<int4> broadphaseEntries;      // sorted by now
    [ReadOnly] public NativeArray<int> simCellsNumberOfPairs;   // write offsets by now
    [ReadOnly] public NativeList<int> activeCollisionLattices;

    [NativeDisableParallelForRestriction]
    [WriteOnly] public NativeArray<int2> candidatePairs;


    public void Execute(int bucketIndex)
    {
        int bucketStart = bucketIndex == 0 ? 0 : (simCellsEntriesOffset[bucketIndex - 1] +
            prefixChunkSizes[(bucketIndex - 1) / prefixChunkSize]);
        int bucketEnd = simCellsEntriesOffset[bucketIndex] + prefixChunkSizes[bucketIndex / prefixChunkSize];
        int count = bucketEnd - bucketStart;

        if (count <= 1) return;

        int writeIndex = simCellsNumberOfPairs[bucketIndex];
        for (int start = bucketStart; start < bucketEnd;)
        {
            // group of entries in the same unwrapped grid cell
            int end = start + 1;
            int3 key = broadphaseEntries[start].xyz;
            while (end < bucketEnd && all(broadphaseEntries[end].xyz == key))
                end++;

            if (end - start <= 1)
            {
                start = end;
                continue;
            }

            for (int i = start; i < end - 1; i++)
            {
                int activeA = broadphaseEntries[i].w;
                int3 minCornerA = latticeCellAABBmin[activeA];
                int bodyA = activeCollCellBodyRef[activeA];

                for (int j = i + 1; j < end; j++)
                {
                    int activeB = broadphaseEntries[j].w;
                    int bodyB = activeCollCellBodyRef[activeB];
                    if (bodyA == bodyB) continue;

                    int3 ownerCell = max(minCornerA, latticeCellAABBmin[activeB]);
                    if (!all(ownerCell == key)) continue;

                    candidatePairs[writeIndex++] =
                        new int2(activeCollisionLattices[activeA],
                                 activeCollisionLattices[activeB]);
                }
            }

            start = end;
        }
    }
}

#endregion
#region Narrow phase jobs

/*
Narrow phase 1/2: every collision vertex at the start of the substep.

One work item per wrapper. A vertex is its tet's 4 previous positions mixed
by its barycentric weights. Wrappers own disjoint vertex ranges, so the
parallel writes never overlap.
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

    // Tet corners as local cell nodes, same order as the preprocessor
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
Narrow phase 2/2: one contact per candidate pair, at the start of the substep.

Brute force over the two collision meshes, the closest of
    vertices of A against triangles of B
    vertices of B against triangles of A
    edges of A against edges of B
gives (exact for meshes that do not intersect)

    pairContacts[pair] = float4(n, gap)
        n    unit direction from B's closest point to A's closest point
        gap  distance between the two meshes

n = 0 when the meshes touch (gap ~ 0), the solvers then fall back to the
direction between the cell centres.
*/
[BurstCompile]
public struct CellPairsNarrowPhase : IJobParallelFor
{
    [ReadOnly] public NativeArray<int2> candidatePairs;
    [ReadOnly] public NativeList<int> latticeColReferences;
    [ReadOnly] public NativeList<CollisionWraper> collisionWrapers;
    [ReadOnly] public NativeList<int3> collisionTriangles;
    [ReadOnly] public NativeList<int2> collisionEdges;
    [ReadOnly] public NativeArray<float3> colVertPrevPositions;
    [WriteOnly] public NativeArray<float4> pairContacts;


    public void Execute(int pairIndex)
    {
        int2 pair = candidatePairs[pairIndex];
        CollisionWraper A = collisionWrapers[latticeColReferences[pair.x]];
        CollisionWraper B = collisionWrapers[latticeColReferences[pair.y]];

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

#endregion
#region Solver jobs

/*
Stage 1: speculative velocity contacts, cells act as rigid bodies.

For every pair, with d = current - previous of each cell centre:

    C = n.(dA - dB) + max(gap - skin, 0)

C >= 0 means the pair cannot get closer than skin this substep, nothing
happens (contacts only push). Otherwise both cells are pushed apart along n:

    dl = -C / (wA + wB),   cell A += wA dl n,   cell B -= wB dl n

w = 1 / sum of the cell's node masses and the push is added to all 8 nodes,
so momentum is conserved exactly. Gauss-Seidel: every sweep re-reads the
current positions, so earlier pushes are seen and spread through chains.

Floor (temporary, for testing): the same constraint against a static plane
y = floorY, n = up, gap = lowest collision vertex, the whole push goes to
the cell. Tested after the pairs in every sweep.
*/
[BurstCompile]
public struct LatticeCellsVelTransfer : IJob
{
    public float skin;
    public int sweeps;
    public float floorY;

    [ReadOnly] public NativeList<LatticeCell> latticeCells;
    [ReadOnly] public NativeArray<int2> candidatePairs;
    [ReadOnly] public NativeArray<float4> pairContacts;

    [ReadOnly] public NativeList<int> activeCollisionLattices;
    [ReadOnly] public NativeList<int> latticeColReferences;
    [ReadOnly] public NativeList<CollisionWraper> collisionWrapers;
    [ReadOnly] public NativeArray<float3> colVertPrevPositions;

    public NativeList<Particle> particles;


    public void Execute()
    {
        for (int sweep = 0; sweep < sweeps; sweep++)
        {
            // Pair contacts
            for (int i = 0; i < candidatePairs.Length; i++)
            {
                float4 contact = pairContacts[i];

                LatticeCell A = latticeCells[candidatePairs[i].x];
                LatticeCell B = latticeCells[candidatePairs[i].y];

                float wA = CellInvMass(A);
                float wB = CellInvMass(B);
                if (wA + wB <= 0f) continue;

                float3 prevA = CellCenter(A, true);
                float3 prevB = CellCenter(B, true);

                float3 n = contact.xyz;
                float gap = contact.w;
                if (math.lengthsq(n) == 0f)
                {
                    n = prevA - prevB;
                    float dist = math.length(n);
                    if (dist < 1e-6f) continue;
                    n /= dist;
                }

                float3 dispA = CellCenter(A, false) - prevA;
                float3 dispB = CellCenter(B, false) - prevB;

                float c = math.dot(dispA - dispB, n) + math.max(gap - skin, 0f);
                if (c >= 0f) continue;

                float dl = -c / (wA + wB);
                AddToCell(A, n * (wA * dl));
                AddToCell(B, n * (-wB * dl));
            }

            // Floor contacts
            for (int a = 0; a < activeCollisionLattices.Length; a++)
            {
                int cellIdx = activeCollisionLattices[a];
                LatticeCell cell = latticeCells[cellIdx];
                if (CellInvMass(cell) <= 0f) continue;

                CollisionWraper wraper = collisionWrapers[latticeColReferences[cellIdx]];
                float lowestY = float.MaxValue;
                for (int v = wraper.colVertsRange.x; v < wraper.colVertsRange.y; v++)
                    lowestY = math.min(lowestY, colVertPrevPositions[v].y);
                float gap = lowestY - floorY;

                float dispY = CellCenter(cell, false).y - CellCenter(cell, true).y;
                float c = dispY + math.max(gap - skin, 0f);
                if (c >= 0f) continue;

                AddToCell(cell, new float3(0f, -c, 0f));
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

    // 1 / (sum of node masses), 0 when a node is pinned so the cell never moves
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

/*
Stage 2: slide, gives the excess motion to the lighter side before stage 3 cuts it.

Every pair has a plane fixed at the start of the substep, the same one stage 3
uses: n from the narrow phase (B to A) and slab, the gap along n between A's
lowest collision vertex and B's highest. The cheap test takes the node of each
side that approaches most, a side that moves away gives extra room:

    excess = approachA + approachB - max(slab - hardSkin, 0)

excess > 0 only says the pair might get too close. Stretched cells make it true
for almost every pair in a crushed pile, and acting on those pairs stretches the
cells further, so a pair is skipped when

    the meshes cannot meet     gap - hardSkin > largest node motion of A + of B
    its vertices end apart     deficit = hardSkin - (A's lowest end - B's highest
                               end, along n) <= 0

A real deficit is split by cell inverse mass, so the lighter side gives way.
Vertices past their side's share are lifted to it along n through their tet's
nodes, sideways motion is untouched.

A start slab below hardSkin leaves nothing to measure the end against. Those
pairs only stop the approach: excess is split the same way and every node over
its side's limit moves back to it, a limit below 0 pushes a light side away.

Floor (temporary, for testing): vertices ending below min(start gap, hardSkin)
are lifted to it.

A move can start another violation, so everything repeats until a whole pass
finds none or the passes run out. Stage 3 handles the rest.
*/
[BurstCompile]
public struct LatticeCellsSlide : IJob
{
    private const int SlidePasses = 4;
    private const float Tolerance = 1e-5f;

    public float hardSkin;
    public float floorY;

    [ReadOnly] public NativeList<LatticeCell> latticeCells;
    [ReadOnly] public NativeArray<int2> candidatePairs;
    [ReadOnly] public NativeArray<float4> pairContacts;

    [ReadOnly] public NativeList<int> activeCollisionLattices;
    [ReadOnly] public NativeList<int> latticeColReferences;
    [ReadOnly] public NativeList<CollisionWraper> collisionWrapers;
    [ReadOnly] public NativeArray<float3> colVertPrevPositions;
    [ReadOnly] public NativeList<DynamicVertex> collisionVertices;

    public NativeList<Particle> particles;


    public void Execute()
    {
        // Pair planes, (n, slab at the start of the substep)
        NativeArray<float4> planes = new NativeArray<float4>(candidatePairs.Length, Allocator.Temp);
        for (int i = 0; i < candidatePairs.Length; i++)
        {
            int2 pair = candidatePairs[i];
            LatticeCell A = latticeCells[pair.x];
            LatticeCell B = latticeCells[pair.y];

            float3 n = pairContacts[i].xyz;
            if (math.lengthsq(n) == 0f)
            {
                n = CellPrevCenter(A) - CellPrevCenter(B);
                float dist = math.length(n);
                n = dist > 1e-6f ? n / dist : new float3(0f, 1f, 0f);
            }

            CollisionWraper wraperA = collisionWrapers[latticeColReferences[pair.x]];
            CollisionWraper wraperB = collisionWrapers[latticeColReferences[pair.y]];
            float minA = float.MaxValue;
            for (int v = wraperA.colVertsRange.x; v < wraperA.colVertsRange.y; v++)
                minA = math.min(minA, math.dot(n, colVertPrevPositions[v]));
            float maxB = float.MinValue;
            for (int v = wraperB.colVertsRange.x; v < wraperB.colVertsRange.y; v++)
                maxB = math.max(maxB, math.dot(n, colVertPrevPositions[v]));

            planes[i] = new float4(n, minA - maxB);
        }

        // Floor rooms
        NativeArray<float> floorRooms = new NativeArray<float>(activeCollisionLattices.Length, Allocator.Temp);
        for (int a = 0; a < floorRooms.Length; a++)
        {
            CollisionWraper wraper = collisionWrapers[latticeColReferences[activeCollisionLattices[a]]];
            float lowestY = float.MaxValue;
            for (int v = wraper.colVertsRange.x; v < wraper.colVertsRange.y; v++)
                lowestY = math.min(lowestY, colVertPrevPositions[v].y);
            floorRooms[a] = lowestY - floorY;
        }

        bool violated = true;
        for (int pass = 0; pass < SlidePasses && violated; pass++)
        {
            violated = false;

            // Floor
            for (int a = 0; a < floorRooms.Length; a++)
            {
                int cellIdx = activeCollisionLattices[a];
                float target = floorY + math.min(floorRooms[a], hardSkin);
                if (RaiseVerts(latticeCells[cellIdx], collisionWrapers[latticeColReferences[cellIdx]], new float3(0f, 1f, 0f), target)) violated = true;
            }

            // Pairs
            for (int i = 0; i < candidatePairs.Length; i++)
            {
                float3 n = planes[i].xyz;
                float slab = planes[i].w;
                float room = math.max(slab - hardSkin, 0f);
                LatticeCell A = latticeCells[candidatePairs[i].x];
                LatticeCell B = latticeCells[candidatePairs[i].y];

                float approachA = MaxApproach(A, -n);
                float approachB = MaxApproach(B, n);
                float excess = approachA + approachB - room;
                if (excess <= Tolerance) continue;

                // The worst node against the start slab only says it might touch, the rest looks at the real end
                if (pairContacts[i].w - hardSkin > MaxMove(A) + MaxMove(B)) continue;

                float wA = CellInvMass(A);
                float wB = CellInvMass(B);
                if (wA + wB <= 0f) continue;

                if (slab < hardSkin)
                {
                    // The plane has no room to measure against, only stop the approach
                    violated = true;
                    LimitNodes(A, -n, approachA - excess * wA / (wA + wB));
                    LimitNodes(B, n, approachB - excess * wB / (wA + wB));
                    continue;
                }

                CollisionWraper wraperA = collisionWrapers[latticeColReferences[candidatePairs[i].x]];
                CollisionWraper wraperB = collisionWrapers[latticeColReferences[candidatePairs[i].y]];
                float endA = LowestVert(A, wraperA, n);
                float endB = -LowestVert(B, wraperB, -n);
                float deficit = hardSkin - (endA - endB);
                if (deficit <= Tolerance) continue;

                violated = true;
                RaiseVerts(A, wraperA, n, endA + deficit * wA / (wA + wB));
                RaiseVerts(B, wraperB, -n, -endB + deficit * wB / (wA + wB));
            }
        }

        planes.Dispose();
        floorRooms.Dispose();
    }

    // Every free node of the cell moving along dir by more than limit moves back to it
    private void LimitNodes(in LatticeCell c, float3 dir, float limit)
    {
        for (int k = 0; k < 8; k++)
        {
            int idx = FullCollisionJob.CellNode(c, k);
            Particle p = particles[idx];
            if (p.invMass <= 0f) continue;

            float over = math.dot(dir, p.currentPosition - p.previousPosition) - limit;
            if (over <= 0f) continue;

            p.currentPosition -= dir * over;
            particles[idx] = p;
        }
    }

    // 1 / (sum of node masses), 0 when a node is pinned so that side does not give way
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

    // Every collision vertex of the cell below target along dir is lifted to it, through its tet's nodes. True when one moved
    private bool RaiseVerts(in LatticeCell c, in CollisionWraper wraper, float3 dir, float target)
    {
        bool moved = false;
        for (int v = wraper.colVertsRange.x; v < wraper.colVertsRange.y; v++)
        {
            DynamicVertex vertex = collisionVertices[v];
            int4 tet = FullCollisionJob.CellTet(vertex.tet);

            float3 end = 0f;
            float invMassSum = 0f;
            for (int k = 0; k < 4; k++)
            {
                Particle p = particles[FullCollisionJob.CellNode(c, tet[k])];
                end += vertex.bary[k] * p.currentPosition;
                invMassSum += vertex.bary[k] * vertex.bary[k] * p.invMass;
            }

            float missing = target - math.dot(dir, end);
            if (missing <= Tolerance || invMassSum <= 0f) continue;

            for (int k = 0; k < 4; k++)
            {
                int idx = FullCollisionJob.CellNode(c, tet[k]);
                Particle p = particles[idx];
                p.currentPosition += dir * (p.invMass * vertex.bary[k] * missing / invMassSum);
                particles[idx] = p;
            }
            moved = true;
        }
        return moved;
    }

    // The lowest of the cell's collision vertices along dir, where they really end this substep
    private float LowestVert(in LatticeCell c, in CollisionWraper wraper, float3 dir)
    {
        float lowest = float.MaxValue;
        for (int v = wraper.colVertsRange.x; v < wraper.colVertsRange.y; v++)
        {
            DynamicVertex vertex = collisionVertices[v];
            int4 tet = FullCollisionJob.CellTet(vertex.tet);
            float3 end = 0f;
            for (int k = 0; k < 4; k++)
                end += vertex.bary[k] * particles[FullCollisionJob.CellNode(c, tet[k])].currentPosition;
            lowest = math.min(lowest, math.dot(dir, end));
        }
        return lowest;
    }

    // The largest distance any node of the cell moves this substep
    private float MaxMove(in LatticeCell c)
    {
        float most = 0f;
        for (int k = 0; k < 8; k++)
        {
            Particle p = particles[FullCollisionJob.CellNode(c, k)];
            most = math.max(most, math.length(p.currentPosition - p.previousPosition));
        }
        return most;
    }

    // The most any node of the cell moves along dir, negative when all move away
    private float MaxApproach(in LatticeCell c, float3 dir)
    {
        float most = float.MinValue;
        for (int k = 0; k < 8; k++)
        {
            Particle p = particles[FullCollisionJob.CellNode(c, k)];
            most = math.max(most, math.dot(dir, p.currentPosition - p.previousPosition));
        }
        return most;
    }

    private float3 CellPrevCenter(in LatticeCell c)
    {
        float3 s = new float3(0);
        for (int k = 0; k < 8; k++)
            s += particles[FullCollisionJob.CellNode(c, k)].previousPosition;
        return s * 0.125f;
    }
}

/*
Stage 3: hard planar Divide-and-Truncate (DAT), no penetration.

Must be the last thing that moves particles in a substep. The motion of every
node, d = current - previous, is only ever scaled toward the start:

    current = previous + alpha d,    alpha in [0, 1], one per particle

For every pair, n from the narrow phase (B to A):
    slab      min of n.x over A's collision verts - max over B's, at the
              start of the substep. A lower bound on the distance for any
              unit n, a poor n only gives less room
    room      max(slab - hardSkin, 0)
    approach  the most any node of a side moves toward the other side, -n.d for
              A's nodes and n.d for B's. Negative when the whole side moves away

Only relative motion counts: a side that moves away gives that motion as extra
room, so two bodies moving together are never limited. If the approaching sides
fit in the room, the pair is safe. Otherwise the room is split in proportion to
the two approaches and every node taking more than its side's share gets

    alpha = min(alpha, share / approach)

The approach test is only a bound, stretched cells fail it without being close.
Before capping, a pair is left alone if the meshes cannot meet or its collision
vertices really end hardSkin apart along n with the current alphas (the same
two tests as stage 2, for a start slab of at least hardSkin).

Capping a node can take away a retreat or move a vertex another pair was relying
on, so the pairs repeat until nothing changes. If the passes run out, a last pass
ignores retreats and the real end: its caps hold whatever else happens, which is
what stops real pinching.

Floor (temporary, for testing): n = up, slab = lowest collision vertex - floorY,
the whole room goes to the cell, repeated with the pairs for the same reason.

Why it holds:
    - A collision vertex is a weighted average of its tet's nodes (weights >= 0,
      sum 1), so it approaches the plane at most as much as its side's approach
      and the pair ends at least hardSkin apart along n.
    - When the pairs settle, every pair holds with the final motions. In the last
      pass the min only scales motions toward zero, which never makes a node
      approach any plane more than that plane allowed.
    - Along a fixed n the gap between the two sides is a concave function of
      time for straight motions, so it is never below its smaller end value.

Only pairs inside candidatePairs are protected.
*/
[BurstCompile]
public struct LatticeCellsPositionLimit : IJob
{
    private const int RetreatPasses = 4;
    private const float Tolerance = 1e-5f;

    public float hardSkin;
    public float floorY;

    [ReadOnly] public NativeList<LatticeCell> latticeCells;
    [ReadOnly] public NativeArray<int2> candidatePairs;
    [ReadOnly] public NativeArray<float4> pairContacts;

    [ReadOnly] public NativeList<int> activeCollisionLattices;
    [ReadOnly] public NativeList<int> latticeColReferences;
    [ReadOnly] public NativeList<CollisionWraper> collisionWrapers;
    [ReadOnly] public NativeArray<float3> colVertPrevPositions;
    [ReadOnly] public NativeList<DynamicVertex> collisionVertices;

    public NativeList<Particle> particles;


    public void Execute()
    {
        NativeArray<float> alpha = new NativeArray<float>(particles.Length, Allocator.Temp);
        for (int i = 0; i < alpha.Length; i++) alpha[i] = 1f;

        // Pair planes, (n, room)
        NativeArray<float4> planes = new NativeArray<float4>(candidatePairs.Length, Allocator.Temp);
        NativeArray<float> slabs = new NativeArray<float>(candidatePairs.Length, Allocator.Temp);
        for (int i = 0; i < candidatePairs.Length; i++)
        {
            int2 pair = candidatePairs[i];
            LatticeCell A = latticeCells[pair.x];
            LatticeCell B = latticeCells[pair.y];

            float3 n = pairContacts[i].xyz;
            if (math.lengthsq(n) == 0f)
            {
                n = CellPrevCenter(A) - CellPrevCenter(B);
                float dist = math.length(n);
                n = dist > 1e-6f ? n / dist : new float3(0f, 1f, 0f);
            }

            CollisionWraper wraperA = collisionWrapers[latticeColReferences[pair.x]];
            CollisionWraper wraperB = collisionWrapers[latticeColReferences[pair.y]];
            float minA = float.MaxValue;
            for (int v = wraperA.colVertsRange.x; v < wraperA.colVertsRange.y; v++)
                minA = math.min(minA, math.dot(n, colVertPrevPositions[v]));
            float maxB = float.MinValue;
            for (int v = wraperB.colVertsRange.x; v < wraperB.colVertsRange.y; v++)
                maxB = math.max(maxB, math.dot(n, colVertPrevPositions[v]));

            slabs[i] = minA - maxB;
            planes[i] = new float4(n, math.max(slabs[i] - hardSkin, 0f));
        }

        // Floor gaps at the start of the substep
        NativeArray<float> floorGaps = new NativeArray<float>(activeCollisionLattices.Length, Allocator.Temp);
        for (int a = 0; a < floorGaps.Length; a++)
        {
            CollisionWraper wraper = collisionWrapers[latticeColReferences[activeCollisionLattices[a]]];
            float lowestY = float.MaxValue;
            for (int v = wraper.colVertsRange.x; v < wraper.colVertsRange.y; v++)
                lowestY = math.min(lowestY, colVertPrevPositions[v].y);
            floorGaps[a] = lowestY - floorY;
        }

        // Caps, a retreating side gives room until nothing changes
        bool settled = false;
        for (int pass = 0; pass < RetreatPasses && !settled; pass++)
        {
            settled = true;
            for (int a = 0; a < floorGaps.Length; a++)
                if (CapFloor(alpha, activeCollisionLattices[a], floorGaps[a], true)) settled = false;
            for (int i = 0; i < candidatePairs.Length; i++)
                if (CapPair(alpha, i, planes[i], slabs[i], true)) settled = false;
        }

        // Last pass, no retreats and no real end test: its caps hold whatever else happens
        if (!settled)
        {
            for (int a = 0; a < floorGaps.Length; a++)
                CapFloor(alpha, activeCollisionLattices[a], floorGaps[a], false);
            for (int i = 0; i < candidatePairs.Length; i++)
                CapPair(alpha, i, planes[i], slabs[i], false);
        }

        // Apply
        for (int i = 0; i < particles.Length; i++)
        {
            if (alpha[i] >= 1f) continue;
            Particle p = particles[i];
            p.currentPosition = p.previousPosition + alpha[i] * (p.currentPosition - p.previousPosition);
            particles[i] = p;
        }

        alpha.Dispose();
        planes.Dispose();
        slabs.Dispose();
        floorGaps.Dispose();
    }

    // True when a node got capped
    private bool CapPair(NativeArray<float> alpha, int index, float4 plane, float slab, bool allowRetreat)
    {
        int2 pair = candidatePairs[index];
        LatticeCell A = latticeCells[pair.x];
        LatticeCell B = latticeCells[pair.y];
        float3 n = plane.xyz;

        float approachA = MaxApproach(alpha, A, -n);
        float approachB = MaxApproach(alpha, B, n);

        float available = plane.w;
        if (allowRetreat) available -= math.min(approachA, 0f) + math.min(approachB, 0f);

        approachA = math.max(approachA, 0f);
        approachB = math.max(approachB, 0f);
        if (approachA + approachB <= available) return false;
        if (allowRetreat && EndsApart(alpha, index, n, slab)) return false;

        float shareA = available * approachA / (approachA + approachB);
        float shareB = math.max(available - shareA, 0f);    // rounding can push it just below 0
        bool cappedA = CapCell(alpha, A, -n, shareA);
        bool cappedB = CapCell(alpha, B, n, shareB);
        return cappedA || cappedB;
    }

    // The worst node only says the pair might touch, this looks at where the meshes really end up
    private bool EndsApart(NativeArray<float> alpha, int index, float3 n, float slab)
    {
        int2 pair = candidatePairs[index];
        LatticeCell A = latticeCells[pair.x];
        LatticeCell B = latticeCells[pair.y];
        if (pairContacts[index].w - hardSkin > MaxMove(alpha, A) + MaxMove(alpha, B)) return true;
        if (slab < hardSkin) return false;

        float endA = LowestVert(alpha, A, collisionWrapers[latticeColReferences[pair.x]], n);
        float endB = -LowestVert(alpha, B, collisionWrapers[latticeColReferences[pair.y]], -n);
        return endA - endB >= hardSkin - Tolerance;
    }

    // True when a node got capped
    private bool CapFloor(NativeArray<float> alpha, int cellIdx, float gap, bool realEnd)
    {
        LatticeCell cell = latticeCells[cellIdx];
        float3 down = new float3(0f, -1f, 0f);
        float room = math.max(gap - hardSkin, 0f);
        if (MaxApproach(alpha, cell, down) <= room) return false;

        CollisionWraper wraper = collisionWrapers[latticeColReferences[cellIdx]];
        if (realEnd && LowestVert(alpha, cell, wraper, -down) >= floorY + math.min(gap, hardSkin) - Tolerance) return false;
        return CapCell(alpha, cell, down, room);
    }

    // The lowest of the cell's collision vertices along dir, at the end of the capped motion
    private float LowestVert(NativeArray<float> alpha, in LatticeCell c, in CollisionWraper wraper, float3 dir)
    {
        float lowest = float.MaxValue;
        for (int v = wraper.colVertsRange.x; v < wraper.colVertsRange.y; v++)
        {
            DynamicVertex vertex = collisionVertices[v];
            int4 tet = FullCollisionJob.CellTet(vertex.tet);
            float3 end = 0f;
            for (int k = 0; k < 4; k++)
            {
                int idx = FullCollisionJob.CellNode(c, tet[k]);
                Particle p = particles[idx];
                end += vertex.bary[k] * (p.previousPosition + alpha[idx] * (p.currentPosition - p.previousPosition));
            }
            lowest = math.min(lowest, math.dot(dir, end));
        }
        return lowest;
    }

    // The largest distance any node of the cell moves with its current alpha
    private float MaxMove(NativeArray<float> alpha, in LatticeCell c)
    {
        float most = 0f;
        for (int k = 0; k < 8; k++)
        {
            int idx = FullCollisionJob.CellNode(c, k);
            Particle p = particles[idx];
            most = math.max(most, alpha[idx] * math.length(p.currentPosition - p.previousPosition));
        }
        return most;
    }

    // Every node of the cell moving along dir by more than share is capped to share. True when one got capped
    private bool CapCell(NativeArray<float> alpha, in LatticeCell c, float3 dir, float share)
    {
        bool capped = false;
        for (int k = 0; k < 8; k++)
        {
            int idx = FullCollisionJob.CellNode(c, k);
            float approach = math.dot(dir, NodeDisplacement(c, k));
            if (approach * alpha[idx] <= share) continue;

            alpha[idx] = share / approach;
            capped = true;
        }
        return capped;
    }

    // The most any node of the cell moves along dir with its current alpha, negative when all move away
    private float MaxApproach(NativeArray<float> alpha, in LatticeCell c, float3 dir)
    {
        float most = float.MinValue;
        for (int k = 0; k < 8; k++)
        {
            int idx = FullCollisionJob.CellNode(c, k);
            most = math.max(most, alpha[idx] * math.dot(dir, NodeDisplacement(c, k)));
        }
        return most;
    }

    private float3 NodeDisplacement(in LatticeCell c, int k)
    {
        Particle p = particles[FullCollisionJob.CellNode(c, k)];
        return p.currentPosition - p.previousPosition;
    }

    private float3 CellPrevCenter(in LatticeCell c)
    {
        float3 s = new float3(0);
        for (int k = 0; k < 8; k++)
            s += particles[FullCollisionJob.CellNode(c, k)].previousPosition;
        return s * 0.125f;
    }
}

#endregion



public static class FullCollisionJob
{
#region Settings and persistent data

    private static bool isInitialised = false;

    // Broad phase grid, the grid wraps so any world position fits
    private static int3 broadphaseGridSize = new int3(24);
    private static int broadphaseWorkerChunkCount = 16;    // parallel counting / scatter chunks
    private static int prefixScanBlockCount = 8;           // parallel prefix sum blocks

    // Broad phase, per active cell
    private static NativeList<int3> latticeCellAABBmin;
    private static NativeList<uint> latticeCellAABBspan;

    // Broad phase, per (worker chunk, bucket) and per prefix block
    private static NativeArray<int> simCellsEntries;           // counts
    private static NativeArray<int> simCellsEntriesOffset;     // offsets, then scatter cursors
    private static NativeArray<int> prefixChunkSizes;          // block totals, then block base offsets

    // Broad phase, entries and pairs
    private static NativeList<int4> broadphaseEntries;         // (grid cell xyz, active cell)
    private static NativeArray<int> simCellsNumberOfPairs;     // per bucket counts, then offsets
    private static NativeReference<int> numberOfCandidatePairs;
    private static NativeList<int2> candidatePairs;            // lattice cell pairs

    // Narrow phase
    private static NativeList<float3> colVertPrevPositions;    // per collision vertex
    private static NativeList<float4> pairContacts;            // per candidate pair, (n, gap)

#endregion
#region Entry point

    /*
    Full collision for one substep. Call after the particle predict step and the
    elastic solve, nothing may move particles after it in the same substep.

        1. Broad phase       candidate cell pairs
        2. Narrow phase      collision vertices, then one (n, gap) per pair
        3. Stage 1           velocity contacts, `sweeps` passes
        4. Stage 2           slide, the lighter side gives way before stage 3 cuts
        5. Stage 3           position limit, guarantees >= skin / 2 between
                             the collision meshes of every candidate pair

    Completes the handle before returning.
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

        // 1. Broad phase
        // Completed for now because of the dependencies, would be nicer without
        BroadPhaseRun(
            simCellSize,
            skin,
            ref handle,
            particles,
            latticeCells,
            activeCollisionLattices,
            activeCollCellBodyRef
        );
        handle.Complete();

        // 2. Narrow phase
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

        // 3. Stage 1, velocity contacts
        LatticeCellsVelTransfer velTransfer = new LatticeCellsVelTransfer
        {
            skin = skin,
            sweeps = sweeps,
            floorY = 0f,    // temporary test floor
            latticeCells = latticeCells,
            candidatePairs = candidatePairs.AsArray(),
            pairContacts = pairContacts.AsArray(),
            activeCollisionLattices = activeCollisionLattices,
            latticeColReferences = latticeColReferences,
            collisionWrapers = collisionWrapers,
            colVertPrevPositions = colVertPrevPositions.AsArray(),
            particles = particles
        };
        handle = velTransfer.Schedule(handle);

        // 4. Stage 2, slide
        LatticeCellsSlide slide = new LatticeCellsSlide
        {
            hardSkin = 0.5f * skin,
            floorY = 0f,    // temporary test floor
            latticeCells = latticeCells,
            candidatePairs = candidatePairs.AsArray(),
            pairContacts = pairContacts.AsArray(),
            activeCollisionLattices = activeCollisionLattices,
            latticeColReferences = latticeColReferences,
            collisionWrapers = collisionWrapers,
            colVertPrevPositions = colVertPrevPositions.AsArray(),
            collisionVertices = collisionVertices,
            particles = particles
        };
        handle = slide.Schedule(handle);

        // 5. Stage 3, position limit
        LatticeCellsPositionLimit positionLimit = new LatticeCellsPositionLimit
        {
            hardSkin = 0.5f * skin,
            floorY = 0f,    // temporary test floor
            latticeCells = latticeCells,
            candidatePairs = candidatePairs.AsArray(),
            pairContacts = pairContacts.AsArray(),
            activeCollisionLattices = activeCollisionLattices,
            latticeColReferences = latticeColReferences,
            collisionWrapers = collisionWrapers,
            colVertPrevPositions = colVertPrevPositions.AsArray(),
            collisionVertices = collisionVertices,
            particles = particles
        };
        handle = positionLimit.Schedule(handle);
        handle.Complete();
    }

    /*
    Builds candidatePairs (lattice cell pairs from different bodies whose swept
    boxes share a grid cell) and sizes pairContacts to match. Completes the
    handle twice on the way, the entry and pair counts are needed on the main
    thread to size the lists.
    */
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void BroadPhaseRun(
        float simCellSize,
        float skin,
        ref JobHandle handle,
        NativeList<Particle>        particles,
        NativeList<LatticeCell>     latticeCells,
        NativeList<int>             activeCollisionLattices,
        NativeList<int>             activeCollCellBodyRef
    )
    {
        int numberOfActiveLatticeCells = activeCollisionLattices.Length;
        int numberOfBroadphaseCells = broadphaseGridSize.x * broadphaseGridSize.y * broadphaseGridSize.z;
        int prefixChunkSize = (numberOfBroadphaseCells + prefixScanBlockCount - 1) / prefixScanBlockCount;
        int workerChunkSize = (numberOfActiveLatticeCells + broadphaseWorkerChunkCount - 1) / broadphaseWorkerChunkCount;
        numberOfCandidatePairs.Value = 0;

        latticeCellAABBmin.ResizeUninitialized(numberOfActiveLatticeCells);
        latticeCellAABBspan.ResizeUninitialized(numberOfActiveLatticeCells);

        // ==========================================
        // Step 1: boxes, entry counts, entry offsets
        // ==========================================

        CreateLatticeCellsAABB createCellAABB = new CreateLatticeCellsAABB
        {
            invBroadphaseCellSize = 1f / (simCellSize * 1),
            skin = skin,
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
            chunkSize = workerChunkSize,
            latticeCellAABBmin = latticeCellAABBmin,
            latticeCellAABBspan = latticeCellAABBspan,
            simCellsEntries = simCellsEntries
        };
        handle = countAABBGridPos.Schedule(broadphaseWorkerChunkCount, 1, handle);
        PrefixSumSimCellsEntries prefixSumGridCells = new PrefixSumSimCellsEntries
        {
            numberOfBroadphaseCells = numberOfBroadphaseCells,
            broadphaseWorkerChunkCount = broadphaseWorkerChunkCount,
            prefixChunkSize = prefixChunkSize,
            simCellsEntries = simCellsEntries,
            simCellsEntriesOffset = simCellsEntriesOffset,
            prefixChunkSizes = prefixChunkSizes
        };
        handle = prefixSumGridCells.Schedule(prefixScanBlockCount, 1, handle);
        handle.Complete();

        // Block totals -> block base offsets, the last one ends at the entry count
        int AABBentriesCount = 0;
        for (int i = 0; i < prefixScanBlockCount; i++)
        {
            int blockSize = prefixChunkSizes[i];
            prefixChunkSizes[i] = AABBentriesCount;
            AABBentriesCount += blockSize;
        }
        broadphaseEntries.ResizeUninitialized(AABBentriesCount);

        // After the scatter, the last chunk's cursors are the block-local end of every bucket
        NativeSlice<int> finalChunkCellEndOffsets = new NativeSlice<int>(simCellsEntriesOffset, numberOfBroadphaseCells * (broadphaseWorkerChunkCount - 1));

        // ==========================================
        // Step 2: scatter, sort buckets, count pairs
        // ==========================================

        ScatterAABBentries scatterAABBCells = new ScatterAABBentries
        {
            broadphaseGridSize = broadphaseGridSize,
            chunkSize = workerChunkSize,
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

        // Contacts are sized here too, so the solver path never resizes
        candidatePairs.ResizeUninitialized(numberOfCandidatePairs.Value);
        pairContacts.ResizeUninitialized(numberOfCandidatePairs.Value);

        // ==========================================
        // Step 3: write the pairs
        // ==========================================

        AddCandidatePairs addPairs = new AddCandidatePairs
        {
            prefixChunkSize = prefixChunkSize,
            prefixChunkSizes = prefixChunkSizes,
            latticeCellAABBmin = latticeCellAABBmin,
            activeCollCellBodyRef = activeCollCellBodyRef,
            simCellsEntriesOffset = finalChunkCellEndOffsets,
            broadphaseEntries = broadphaseEntries.AsArray(),
            simCellsNumberOfPairs = simCellsNumberOfPairs,
            activeCollisionLattices = activeCollisionLattices,
            candidatePairs = candidatePairs.AsArray()
        };
        handle = addPairs.Schedule(numberOfBroadphaseCells, 1, handle);

        // Debugging only
        // handle.Complete();
        // DebugBroadPhase(particles, latticeCells);
    }

#endregion
#region Lifetime

    public static void Initialise()
    {
        if (isInitialised) return;
        isInitialised = true;

        int totalGridSize = broadphaseGridSize.x * broadphaseGridSize.y * broadphaseGridSize.z;

        latticeCellAABBmin = new (Allocator.Persistent);
        latticeCellAABBspan = new (Allocator.Persistent);
        simCellsEntries = new NativeArray<int>(broadphaseWorkerChunkCount * totalGridSize, Allocator.Persistent);
        simCellsEntriesOffset = new NativeArray<int>(broadphaseWorkerChunkCount * totalGridSize, Allocator.Persistent);
        prefixChunkSizes = new NativeArray<int>(prefixScanBlockCount, Allocator.Persistent);
        broadphaseEntries = new (Allocator.Persistent);
        simCellsNumberOfPairs = new (totalGridSize, Allocator.Persistent);
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
#region Job helpers

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

    // Corners (local cell nodes) of tet t of the 5-tet split, same order as the preprocessor
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int4 CellTet(int t)
    {
        switch (t)
        {
            case 0: return new int4(0, 3, 6, 5);
            case 1: return new int4(0, 5, 6, 4);
            case 2: return new int4(0, 6, 3, 2);
            case 3: return new int4(5, 3, 6, 7);
            default: return new int4(0, 3, 5, 1);
        }
    }

    // Wraps a world grid coordinate into [0, gridSize)
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int WrapToGrid(int value, int gridSize)
    {
        int wrapped = value % gridSize;
        return wrapped < 0 ? wrapped + gridSize : wrapped;
    }

    // Box span (1..1024 per axis) packed as 10 bits per axis, stored minus one
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint PackSpanUint(int3 xyz, int bitsX = 10, int bitsY = 10, int bitsZ = 10)
    {
        uint sx = (uint)(xyz.x - 1);
        uint sy = (uint)(xyz.y - 1);
        uint sz = (uint)(xyz.z - 1);

        return sx | (sy << bitsX) | (sz << (bitsX + bitsY));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int3 UnpackSpanUint(uint span)
    {
        int sx = (int)( span         & ((1u << 10) - 1u)) + 1;
        int sy = (int)((span >> 10)  & ((1u << 10) - 1u)) + 1;
        int sz = (int)((span >> 20)  & ((1u << 10) - 1u)) + 1;
        return new int3(sx, sy, sz);
    }

#endregion
#region Geometry

    /*
    Closest point to p on triangle (a, b, c), Ericson 5.1.5.
    A degenerate (collinear) triangle returns a, its edges are still covered
    by the edge-edge tests, so the pair distance stays exact.
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
    Near-parallel segments take s = 0 first, the clamping that follows still
    gives a valid closest pair.
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
#region Debug

    // Draws a red line between the centres of every candidate pair, needs a completed broad phase
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
                for (int k = 0; k < 8; k++)
                    midPoint += particles[CellNode(latticeCell, k)].currentPosition;
                return midPoint * 0.125f;
            }

            Debug.DrawLine(MidPoint(latticeCellA), MidPoint(latticeCellB), Color.red, Time.fixedDeltaTime);
        }
    }

#endregion
#region Unused helpers

    // Not called anywhere at the moment, kept from earlier broad phase versions

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

    // True when two packed 10 bit lattice coordinates are at most one cell apart on every axis
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

#endregion
}
}
