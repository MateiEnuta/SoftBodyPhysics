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


namespace CollisionHandler
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
    [ReadOnly] public NativeArray<Particle> currentParticles;
    [ReadOnly] public NativeArray<LatticeCell> currentCells;
    [WriteOnly] public NativeArray<ulong> dCellAABBmin;
    [WriteOnly] public NativeArray<uint> dCellAABBspan;
    
    public void Execute(int index)
    {
        LatticeCell c = currentCells[index];
        float3 p0 = currentParticles[c.v0].currentPosition;
        float3 p1 = currentParticles[c.v1].currentPosition;
        float3 p2 = currentParticles[c.v2].currentPosition;
        float3 p3 = currentParticles[c.v3].currentPosition;
        float3 p4 = currentParticles[c.v4].currentPosition;
        float3 p5 = currentParticles[c.v5].currentPosition;
        float3 p6 = currentParticles[c.v6].currentPosition;
        float3 p7 = currentParticles[c.v7].currentPosition;

        float3 minPos = min(min(min(p0, p1), min(p2, p3)), min(min(p4, p5), min(p6, p7)));
        float3 maxPos = max(max(max(p0, p1), max(p2, p3)), max(max(p4, p5), max(p6, p7)));

        int3 minCorner = (int3)floor(minPos * invCellSize);
        int3 maxCorner = (int3)floor(maxPos * invCellSize);

        dCellAABBmin[index] = FullCollisionJob.PackWrappedLong(minCorner);
        dCellAABBspan[index] = FullCollisionJob.PackWrappedInt(maxCorner - minCorner + 1);
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

            int3 span = FullCollisionJob.UnpackSpan(dCellAABBspan[i]);
            total += span.x * span.y * span.z;
        }

        totalCount.Value = total;
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
        int3 span = FullCollisionJob.UnpackSpan(dCellAABBspan[index]);
        int3 minCorner = FullCollisionJob.UnpackMinCorner(dCellAABBmin[index]);

        for (int dx = 0; dx < span.x; dx++)
        for (int dy = 0; dy < span.y; dy++)
        for (int dz = 0; dz < span.z; dz++)
        {
            hashCellsEntries[start++] =
                new HashCellEntry64(
                    FullCollisionJob.PackWrappedLong(new int3(minCorner.x + dx, minCorner.y + dy, minCorner.z + dz)),
                    index);
        }
    }
}

[BurstCompile]
public struct CountCellRanges : IJob
{
    [ReadOnly] public int numberOfEntries;
    [ReadOnly] public NativeArray<HashCellEntry64> hashCellsEntries;
    // [ReadOnly] public NativeArray<int4> cellGridPosition;
    [ReadOnly] public NativeArray<ulong> dCellAABBmin;
    [WriteOnly] public NativeReference<int> totalPairsCount;
    public NativeList<int2> hashCellRanges;
    public NativeList<int> rangesStart;

    public void Execute()
    {
        int currentPairsCount = 0;

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

            int initialPairsCount = currentPairsCount;
            int3 currentCellCoords = FullCollisionJob.UnpackMinCorner(hashCellsEntries[start].key);

            for (int i = start; i < end - 1; i++)
            {
                int cellAidx = hashCellsEntries[i].cellIndex;
                // int4 cellA = cellGridPosition[cellAidx];
                int3 cellAcorner = FullCollisionJob.UnpackMinCorner(dCellAABBmin[cellAidx]);

                for (int j = i + 1; j < end; j++)
                {
                    int cellBidx = hashCellsEntries[j].cellIndex;
                    // int4 cellB = cellGridPosition[cellBidx];
                    int3 cellBcorner = FullCollisionJob.UnpackMinCorner(dCellAABBmin[cellBidx]);

                    // if (cellA.x == cellB.x)
                    // {
                    //     int3 d = cellA.yzw - cellB.yzw;
                    //     if (abs(d.x) <= 1 && abs(d.y) <= 1 && abs(d.z) <= 1) 
                    //         continue;
                    // }
                    if (!all(max(cellAcorner, cellBcorner) == currentCellCoords)) continue;

                    currentPairsCount++;
                }
            }

            hashCellRanges.AddNoResize(new int2(start, end));
            rangesStart.AddNoResize(initialPairsCount);
            start = end;
        }

        totalPairsCount.Value = currentPairsCount;
    }
}

[BurstCompile]
public struct AddCollisionPairs : IJobParallelFor
{
    [ReadOnly] public NativeArray<HashCellEntry64> hashCellsEntries;
    // [ReadOnly] public NativeArray<int4> cellGridPosition;
    [ReadOnly] public NativeArray<ulong> dCellAABBmin;
    [ReadOnly] public NativeArray<int2> hashCellRanges;
    [ReadOnly] public NativeArray<int> rangesStart;
    [NativeDisableParallelForRestriction]
    [WriteOnly] public NativeArray<int2> collisionPairs;

    public void Execute(int index)
    {
        int start = rangesStart[index];
        int3 currentCellCoords = FullCollisionJob.UnpackMinCorner(hashCellsEntries[hashCellRanges[index].x].key);

        for (int i = hashCellRanges[index].x; i < hashCellRanges[index].y - 1; i++)
        {
            int cellAidx = hashCellsEntries[i].cellIndex;
            // int4 cellA = cellGridPosition[cellAidx];
            int3 cellAcorner = FullCollisionJob.UnpackMinCorner(dCellAABBmin[cellAidx]);

            for (int j = i + 1; j < hashCellRanges[index].y; j++)
            {
                int cellBidx = hashCellsEntries[j].cellIndex;
                // int4 cellB = cellGridPosition[cellBidx];
                int3 cellBcorner = FullCollisionJob.UnpackMinCorner(dCellAABBmin[cellBidx]);

                // if (cellA.x == cellB.x)
                // {
                //     int3 d = cellA.yzw - cellB.yzw;
                //     if (abs(d.x) <= 1 && abs(d.y) <= 1 && abs(d.z) <= 1) 
                //         continue;
                // }
                if (!all(max(cellAcorner, cellBcorner) == currentCellCoords)) continue;

                int a = min(cellAidx, cellBidx);
                int b = max(cellAidx, cellBidx);

                collisionPairs[start++] = new int2(a, b);
            }
        }
    }
}
#endregion



public static class FullCollisionJob
{
#region Jobs common functions
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int3 UnpackMinCorner(ulong minCorner)
    {
        int x = (int)( minCorner         & ((1UL << 22) - 1UL));
        int y = (int)((minCorner >> 22)  & ((1UL << 20) - 1UL));
        int z = (int)((minCorner >> 42)  & ((1UL << 22) - 1UL));
        return new int3(x, y, z);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int3 UnpackSpan(uint span)
    {
        int sx = (int)( span         & ((1u << 11) - 1u)) + 1;
        int sy = (int)((span >> 11)  & ((1u << 10) - 1u)) + 1;
        int sz = (int)((span >> 21)  & ((1u << 11) - 1u)) + 1;
        return new int3(sx, sy, sz);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint PackWrappedInt(int3 xyz, int bitsX = 11, int bitsY = 10, int bitsZ = 11)
    {
        uint sx = (uint)(xyz.x - 1);
        uint sy = (uint)(xyz.y - 1);
        uint sz = (uint)(xyz.z - 1);

        return sx | (sy << bitsX) | (sz << (bitsX + bitsY));
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
#endregion


#region Collision data
    private static bool isInitialised = false;
    private static int numberOfRadixChunks = 32;

    private static NativeReference<int>         AABBcellCount;
    private static NativeList<ulong>            dCellAABBmin;
    private static NativeList<uint>             dCellAABBspan;
    private static NativeList<int>              cellWriteStart;
    
    private static NativeReference<int>         totalPairsCount;
    private static NativeList<HashCellEntry64>  cellsEntries64;
    private static NativeList<HashCellEntry64>  auxCellsEntries64;
    private static NativeArray<int>             histograms;
    private static NativeArray<int>             histogramsOffsets;
    private static NativeList<int2>             hashCellRanges;
    private static NativeList<int>              rangesStart;

    private static NativeList<int2>             collisionPairs;
#endregion


#region Main function
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void RunCollision(
        float dt,
        float hashCellSize,
        ref JobHandle handle,
        NativeList<Particle>        particles,
        NativeList<LatticeCell>     latticeCells,
        NativeList<DynamicVertex>   collisionVertices,
        NativeList<int3>            collisionTriangles,
        NativeList<int2>            collisionEdges,
        NativeList<CollisionWraper> collisionWrapers
    )
    {
        int numberOfCells = latticeCells.Length;
        Initialise();

        // Reset counters
        AABBcellCount.Value = 0;
        totalPairsCount.Value = 0;


        // Lists resizing
        dCellAABBmin.ResizeUninitialized(numberOfCells);
        dCellAABBspan.ResizeUninitialized(numberOfCells);
        cellWriteStart.ResizeUninitialized(numberOfCells);


        // Calculate the total number of occupied cells
        float invCellSize = 1.0f / hashCellSize;
        var AABBcount = new CellsAABBcount
        {
            invCellSize = invCellSize,
            currentParticles = particles.AsArray(),
            currentCells = latticeCells.AsArray(),
            dCellAABBmin = dCellAABBmin.AsArray(),
            dCellAABBspan = dCellAABBspan.AsArray()
        };
        handle = AABBcount.Schedule(numberOfCells, 16, handle);
        CountAABBentries countAABB = new CountAABBentries
        {
            totalCount = AABBcellCount,
            dCellAABBspan = dCellAABBspan.AsArray(),
            cellWriteStart = cellWriteStart.AsArray()
        };
        handle = countAABB.Schedule(handle);
        handle.Complete();


        // Lists resizing
        cellsEntries64.ResizeUninitialized(AABBcellCount.Value);
        auxCellsEntries64.ResizeUninitialized(AABBcellCount.Value);
        
        if (hashCellRanges.Capacity < AABBcellCount.Value)
            hashCellRanges.Capacity = AABBcellCount.Value;
        hashCellRanges.Clear();
        if (rangesStart.Capacity < AABBcellCount.Value)
            rangesStart.Capacity = AABBcellCount.Value;
        rangesStart.Clear();
        Debug.Log(AABBcellCount.Value);


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
            AABBcellCount.Value,
            cellsEntries64,
            auxCellsEntries64,
            histograms,
            histogramsOffsets
        );
        CountCellRanges countCellRanges = new CountCellRanges
        {
            numberOfEntries = AABBcellCount.Value, 
            hashCellsEntries = cellsEntries64.AsArray(),
            // cellGridPosition = cellGridPosition.AsArray(),
            dCellAABBmin = dCellAABBmin.AsArray(),
            totalPairsCount = totalPairsCount,
            hashCellRanges = hashCellRanges,
            rangesStart = rangesStart
        };
        handle = countCellRanges.Schedule(handle);
        handle.Complete();

        
        // Lists resizing
        collisionPairs.ResizeUninitialized(totalPairsCount.Value);


        AddCollisionPairs addCollisionPairs = new AddCollisionPairs
        {
            hashCellsEntries = cellsEntries64.AsArray(),
            // cellGridPosition = cellGridPosition.AsArray(),
            dCellAABBmin = dCellAABBmin.AsArray(),
            hashCellRanges = hashCellRanges.AsArray(),
            rangesStart = rangesStart.AsArray(),
            collisionPairs = collisionPairs.AsArray()
        };
        handle = addCollisionPairs.Schedule(hashCellRanges.Length, 2, handle);
        // WHY ??????
        handle.Complete();
    }
#endregion

    public static void Initialise()
    {
        if (isInitialised) return;
        isInitialised = true;

        AABBcellCount = new NativeReference<int>(Allocator.Persistent);
        dCellAABBmin = new NativeList<ulong>(Allocator.Persistent);
        dCellAABBspan = new NativeList<uint>(Allocator.Persistent);
        cellWriteStart = new NativeList<int>(Allocator.Persistent);
        
        totalPairsCount = new NativeReference<int>(Allocator.Persistent);
        cellsEntries64 = new NativeList<HashCellEntry64>(Allocator.Persistent);
        auxCellsEntries64 = new NativeList<HashCellEntry64>(Allocator.Persistent);
        histograms = new NativeArray<int>(numberOfRadixChunks * 256, Allocator.Persistent);
        histogramsOffsets = new NativeArray<int>(numberOfRadixChunks * 256, Allocator.Persistent);
        hashCellRanges = new NativeList<int2>(Allocator.Persistent);
        rangesStart = new NativeList<int>(Allocator.Persistent);

        collisionPairs = new NativeList<int2>(Allocator.Persistent);
    }

    public static void Disable()
    {
        if (AABBcellCount.IsCreated) AABBcellCount.Dispose();
        if (dCellAABBmin.IsCreated) dCellAABBmin.Dispose();
        if (dCellAABBspan.IsCreated) dCellAABBspan.Dispose();
        if (cellWriteStart.IsCreated) cellWriteStart.Dispose();

        if (totalPairsCount.IsCreated) totalPairsCount.Dispose();
        if (cellsEntries64.IsCreated) cellsEntries64.Dispose();
        if (auxCellsEntries64.IsCreated) auxCellsEntries64.Dispose();
        if (histograms.IsCreated) histograms.Dispose();
        if (histogramsOffsets.IsCreated) histogramsOffsets.Dispose();
        if (hashCellRanges.IsCreated) hashCellRanges.Dispose();
        if (rangesStart.IsCreated) rangesStart.Dispose();

        if (collisionPairs.IsCreated) collisionPairs.Dispose();
    }

}
}