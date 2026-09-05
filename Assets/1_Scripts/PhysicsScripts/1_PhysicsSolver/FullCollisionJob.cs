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

[BurstCompile]
public struct CreateCellAABB : IJobParallelFor
{
    [ReadOnly] public float invCellSize;
    [ReadOnly] public int3 hashGridSize;
    [ReadOnly] public NativeList<Particle> particles;
    [ReadOnly] public NativeList<LatticeCell> latticeCells;
    [WriteOnly] public NativeArray<int3> cellAABBmin;
    [WriteOnly] public NativeArray<uint> cellAABBspan;

    
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

        int3 minCorner = (int3)floor(minPos * invCellSize);
        int3 maxCorner = (int3)floor(maxPos * invCellSize);

        cellAABBmin[cellIndex] = minCorner;
        cellAABBspan[cellIndex] = FullCollisionJob.PackWrappedInt(maxCorner - minCorner + 1);
    }
}

[BurstCompile]
public struct CountAABBGridPos : IJobParallelFor
{
    [ReadOnly] public int3 hashGridSize;
    [ReadOnly] public int chunkSize;
    [ReadOnly] public NativeList<int3> cellAABBmin;
    [ReadOnly] public NativeList<uint> cellAABBspan;
    [NativeDisableParallelForRestriction]
    public NativeArray<int> hashGridCells;


    public void Execute(int chunkIndex)
    {
        int start = chunkIndex * chunkSize;
        int end = min(start + chunkSize, cellAABBmin.Length);
        int totalHashGridSize = hashGridSize.x * hashGridSize.y * hashGridSize.z;

        for (int i = start; i < end; i++)
        {
            int3 cellMin = cellAABBmin[i];
            int3 cellSpan = FullCollisionJob.UnpackUInt(cellAABBspan[i]);

            for (int x = 0; x < cellSpan.x; x++)
            for (int y = 0; y < cellSpan.y; y++)
            for (int z = 0; z < cellSpan.z; z++)
            {
                int3 currentCellPos = cellMin + new int3(x,y,z);
                int3 hashGridPos = ((currentCellPos % hashGridSize) + hashGridSize) % hashGridSize;
                
                int cellAABBindex = 
                    hashGridPos.x +
                    hashGridPos.y * hashGridSize.x +
                    hashGridPos.z * hashGridSize.x * hashGridSize.y;
                
                hashGridCells[chunkIndex * totalHashGridSize + cellAABBindex]++;
            }
        }
    }
}

[BurstCompile]
public struct PrefixSumGridCells : IJob
{
    [ReadOnly] public int3 hashGridSize;
    [ReadOnly] public int numberOfChunks;
    public NativeArray<int> hashGridCells;
    [WriteOnly] public NativeArray<int> hashGridCellOffsets;
    [WriteOnly] public NativeList<int> activeGridCells;


    public void Execute()
    {
        int totalHashGridSize = hashGridSize.x * hashGridSize.y * hashGridSize.z;

        int previousCount = 0;
        
        for (int i = 0; i < totalHashGridSize; i++)
        {
            int initialCount = previousCount;

            for (int chunk = 0; chunk < numberOfChunks; chunk++)
            {
                int currentCellCount = hashGridCells[chunk * totalHashGridSize + i];
                if (currentCellCount == 0) continue;
                hashGridCells[chunk * totalHashGridSize + i] = 0;
                hashGridCellOffsets[chunk * totalHashGridSize + i] = previousCount;
                previousCount += currentCellCount;
            }

            if (initialCount != previousCount) activeGridCells.AddNoResize(previousCount);
        }
    }
}

[BurstCompile]
public struct ScatterAABBCells : IJobParallelFor
{
    [ReadOnly] public int3 hashGridSize;
    [ReadOnly] public int chunkSize;
    [ReadOnly] public NativeList<int3> cellAABBmin;
    [ReadOnly] public NativeList<uint> cellAABBspan;
    [NativeDisableParallelForRestriction]
    public NativeArray<int> hashGridCellOffsets;
    [NativeDisableParallelForRestriction]
    [WriteOnly] public NativeArray<uint2> cellAABBentries;


    public void Execute(int chunkIndex)
    {
        int start = chunkIndex * chunkSize;
        int end = min(start + chunkSize, cellAABBmin.Length);
        int totalHashGridSize = hashGridSize.x * hashGridSize.y * hashGridSize.z;

        for (int i = start; i < end; i++)
        {
            int3 cellMin = cellAABBmin[i];
            int3 cellSpan = FullCollisionJob.UnpackUInt(cellAABBspan[i]);

            for (int x = 0; x < cellSpan.x; x++)
            for (int y = 0; y < cellSpan.y; y++)
            for (int z = 0; z < cellSpan.z; z++)
            {
                int3 currentCellPos = cellMin + new int3(x,y,z);
                int3 hashGridPos = ((currentCellPos % hashGridSize) + hashGridSize) % hashGridSize;
                
                int cellAABBindex = 
                    hashGridPos.x +
                    hashGridPos.y * hashGridSize.x +
                    hashGridPos.z * hashGridSize.x * hashGridSize.y;
                
                cellAABBentries
                    [hashGridCellOffsets[chunkIndex * totalHashGridSize + cellAABBindex]++]
                    = new uint2(FullCollisionJob.PackWrappedInt(currentCellPos), (uint)i);
            }
        }
    }
}

[BurstCompile]
public struct SortGridCellEntries : IJobParallelFor
{
    [ReadOnly] public NativeList<int> activeGridCells;
    [NativeDisableParallelForRestriction]
    public NativeArray<uint2> cellAABBentries;

    public void Execute(int gridCellIndex)
    {
        int cellStart = gridCellIndex == 0 ? 0 : activeGridCells[gridCellIndex - 1];
        int cellEnd = activeGridCells[gridCellIndex];
        int count = cellEnd - cellStart;

        if (count <= 1) return;

        // Insertion sort is faster for n < 64
        InsertionSort(cellAABBentries, cellStart, count);
    }

    // [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void InsertionSort(NativeArray<uint2> arr, int start, int count)
    {
        for (int i = 1; i < count; i++)
        {
            uint2 key = arr[start + i];
            int j = i - 1;

            // Inline comparison (no function call)
            while (j >= 0)
            {
                uint2 candidate = arr[start + j];
                
                // Compare packed grid IDs directly
                if (candidate.x > key.x)
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
    public static int3 UnpackUInt(uint span)
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
    private static int3 hashGridSize = new int3(24);
    private static int numberOfChunks = 16;

    private static NativeList<int3> cellAABBmin;
    private static NativeList<uint> cellAABBspan;
    private static NativeArray<int> hashGridCells;
    private static NativeArray<int> hashGridCellOffsets;
    private static NativeList<int> activeGridCells;
    private static NativeList<uint2> cellAABBentries;

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


        cellAABBmin.ResizeUninitialized(numberOfCells);
        cellAABBspan.ResizeUninitialized(numberOfCells);
        activeGridCells.Clear();


        CreateCellAABB createCellAABB = new CreateCellAABB
        {
            invCellSize = 1f / (hashCellSize * 1),
            hashGridSize = hashGridSize,
            particles = particles,
            latticeCells = latticeCells,
            cellAABBmin = cellAABBmin.AsDeferredJobArray(),
            cellAABBspan = cellAABBspan.AsDeferredJobArray()
        };
        handle = createCellAABB.Schedule(numberOfCells, 8, handle);
        CountAABBGridPos countAABBGridPos = new CountAABBGridPos
        {
            hashGridSize = hashGridSize,
            chunkSize = (numberOfCells + numberOfChunks - 1) / numberOfChunks,
            cellAABBmin = cellAABBmin,
            cellAABBspan = cellAABBspan,
            hashGridCells = hashGridCells
        };
        handle = countAABBGridPos.Schedule(numberOfChunks, 1, handle);
        PrefixSumGridCells prefixSumGridCells = new PrefixSumGridCells
        {
            hashGridSize = hashGridSize,
            numberOfChunks = numberOfChunks,
            hashGridCells = hashGridCells,
            hashGridCellOffsets = hashGridCellOffsets,
            activeGridCells = activeGridCells
        };
        handle = prefixSumGridCells.Schedule(handle);
        handle.Complete();


        cellAABBentries.ResizeUninitialized(activeGridCells[activeGridCells.Length-1]);


        ScatterAABBCells scatterAABBCells = new ScatterAABBCells
        {
            hashGridSize = hashGridSize,
            chunkSize = (numberOfCells + numberOfChunks - 1) / numberOfChunks,
            cellAABBmin = cellAABBmin,
            cellAABBspan = cellAABBspan,
            hashGridCellOffsets = hashGridCellOffsets,
            cellAABBentries = cellAABBentries.AsDeferredJobArray()
        };
        handle = scatterAABBCells.Schedule(numberOfChunks, 1, handle);
        SortGridCellEntries sortGridCellEntries = new SortGridCellEntries
        {
            activeGridCells = activeGridCells,
            cellAABBentries = cellAABBentries.AsDeferredJobArray()
        };
        handle = sortGridCellEntries.Schedule(activeGridCells.Length, 1, handle);
        handle.Complete();


        DebugCollision();
    }
#endregion
#region Helper Functions

    private static void DebugCollision()
    {
        int maxGridCellSize = 0;
        float averageGridCellSize = 0;
        for (int i = 0; i < activeGridCells.Length; i++)
        {
            int gridCellSize = activeGridCells[i] - (i==0 ? 0 : activeGridCells[i-1]);
            if (gridCellSize > maxGridCellSize) maxGridCellSize = gridCellSize;
            averageGridCellSize += gridCellSize;
        }
        averageGridCellSize /= activeGridCells.Length;
        Debug.Log(
            "Number of AABB entries: " + activeGridCells[activeGridCells.Length-1] +
            "\nNumber of active grid cells: " + activeGridCells.Length +
            "\nMax grid cell size: " + maxGridCellSize +
            "\nAverage grid cell size: " + averageGridCellSize
        );
    }

    public static void Initialise()
    {
        if (isInitialised) return;
        isInitialised = true;

        cellAABBmin = new (Allocator.Persistent);
        cellAABBspan = new (Allocator.Persistent);
        hashGridCells = new NativeArray<int>(
            numberOfChunks * hashGridSize.x * hashGridSize.y * hashGridSize.z,
        Allocator.Persistent);
        hashGridCellOffsets = new NativeArray<int>(
            numberOfChunks * hashGridSize.x * hashGridSize.y * hashGridSize.z,
        Allocator.Persistent);
        activeGridCells = new (hashGridSize.x * hashGridSize.y * hashGridSize.z, Allocator.Persistent);
        cellAABBentries = new (Allocator.Persistent);
    }

    public static void Disable()
    {
        if (!isInitialised) return;
        isInitialised = false;

        cellAABBmin.Dispose();
        cellAABBspan.Dispose();
        hashGridCells.Dispose();
        hashGridCellOffsets.Dispose();
        activeGridCells.Dispose();
        cellAABBentries.Dispose();
    }

#endregion
}
}