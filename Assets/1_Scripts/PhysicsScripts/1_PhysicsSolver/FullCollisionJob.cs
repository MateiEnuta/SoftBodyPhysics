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
    [WriteOnly] public NativeArray<int> cellAABBmin;
    [WriteOnly] public NativeArray<uint> cellAABBspan;
    [WriteOnly] public NativeArray<uint> cellAABBgrid;

    
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

        // Wrap into hash grid [0, hashGridSize)
        int3 hashCell;
        hashCell.x = ((minCorner.x % hashGridSize.x) + hashGridSize.x) % hashGridSize.x;
        hashCell.y = ((minCorner.y % hashGridSize.y) + hashGridSize.y) % hashGridSize.y;
        hashCell.z = ((minCorner.z % hashGridSize.z) + hashGridSize.z) % hashGridSize.z;

        // Convert to linear index
        int hashIndex = (
            hashCell.x +
            hashCell.y * hashGridSize.x +
            hashCell.z * hashGridSize.x * hashGridSize.y
        );

        int3 gridId;
        gridId.x = minCorner.x / hashGridSize.x;
        gridId.y = minCorner.y / hashGridSize.y;
        gridId.z = minCorner.z / hashGridSize.z;

        cellAABBmin[cellIndex] = hashIndex;
        cellAABBspan[cellIndex] = FullCollisionJob.PackWrappedInt(maxCorner - minCorner + 1);
        cellAABBgrid[cellIndex] = FullCollisionJob.PackWrappedInt(gridId);
    }
}

[BurstCompile]
public struct CountAABBGridPos : IJobParallelFor
{
    [ReadOnly] public int3 hashGridSize;
    [ReadOnly] public int chunkSize;
    [ReadOnly] public NativeList<int> cellAABBmin;
    [ReadOnly] public NativeList<uint> cellAABBspan;
    [NativeDisableParallelForRestriction]
    public NativeArray<int> hashGridCells;
    [NativeDisableParallelForRestriction]
    [WriteOnly] public NativeArray<int> hashGridCellSizes;
    [WriteOnly] public NativeArray<int> chunkTotalGridSize;


    public void Execute(int chunkIndex)
    {
        int start = chunkIndex * chunkSize;
        int end = min(start + chunkSize, cellAABBmin.Length);
        int totalHashGridSize = hashGridSize.x * hashGridSize.y * hashGridSize.z;

        for (int i = start; i < end; i++)
        {
            int cellStart = cellAABBmin[i];
            int3 cellSpan = FullCollisionJob.UnpackUInt(cellAABBspan[i]);

            // Pre-wrap the start coordinate
            int startX = cellStart % hashGridSize.x;
            int startY = cellStart % hashGridSize.y;
            int startZ = cellStart % hashGridSize.z;

            for (int x = 0; x < cellSpan.x; x++)
            for (int y = 0; y < cellSpan.y; y++)
            for (int z = 0; z < cellSpan.z; z++)
            {
                int wrappedX = (startX + x) % hashGridSize.x;
                int wrappedY = (startY + y) % hashGridSize.y;
                int wrappedZ = (startZ + z) % hashGridSize.z;
                
                int cellAABBindex = 
                    wrappedX +
                    wrappedY * hashGridSize.x +
                    wrappedZ * hashGridSize.x * hashGridSize.y;
                
                hashGridCells[chunkIndex * totalHashGridSize + cellAABBindex]++;
            }
        }

        int previousCount = 0;
        for (int i = 0; i < totalHashGridSize; i++)
        {
            int currentCellCount = hashGridCells[chunkIndex * totalHashGridSize + i];
            hashGridCellSizes[chunkIndex * totalHashGridSize + i] = currentCellCount;
            if (currentCellCount == 0) continue;
            hashGridCells[chunkIndex * totalHashGridSize + i] += previousCount;
            previousCount += currentCellCount;
        }
        chunkTotalGridSize[chunkIndex] = previousCount;
    }
}

[BurstCompile]
public struct ScatterAABBCells : IJobParallelFor
{
    [ReadOnly] public int3 hashGridSize;
    [ReadOnly] public int chunkSize;
    [ReadOnly] public NativeList<int> cellAABBmin;
    [ReadOnly] public NativeList<uint> cellAABBspan;
    [ReadOnly] public NativeList<uint> cellAABBgrid;
    [ReadOnly] public NativeArray<int> chunkTotalGridSize;
    [NativeDisableParallelForRestriction]
    public NativeArray<int> hashGridCells;
    [NativeDisableParallelForRestriction]
    [WriteOnly] public NativeArray<uint2> cellAABBentries;


    public void Execute(int chunkIndex)
    {
        int start = chunkIndex * chunkSize;
        int end = min(start + chunkSize, cellAABBmin.Length);
        int totalHashGridSize = hashGridSize.x * hashGridSize.y * hashGridSize.z;

        for (int i = start; i < end; i++)
        {
            int cellStart = cellAABBmin[i];
            int3 cellSpan = FullCollisionJob.UnpackUInt(cellAABBspan[i]);
            int3 startWorldCell = FullCollisionJob.UnpackUInt(cellAABBgrid[i]);

            // Compute initial grid ID and local cell
            int3 gridId;
            gridId.x = startWorldCell.x / hashGridSize.x;
            gridId.y = startWorldCell.y / hashGridSize.y;
            gridId.z = startWorldCell.z / hashGridSize.z;

            // Pre-wrap the start coordinate
            int startX = cellStart % hashGridSize.x;
            int startY = cellStart % hashGridSize.y;
            int startZ = cellStart % hashGridSize.z;

            for (int x = 0; x < cellSpan.x; x++)
            for (int y = 0; y < cellSpan.y; y++)
            for (int z = 0; z < cellSpan.z; z++)
            {
                // Update grid ID when wrapping
                int currentGridX = gridId.x + (startX + x) / hashGridSize.x;
                int currentGridY = gridId.y + (startY + y) / hashGridSize.y;
                int currentGridZ = gridId.z + (startZ + z) / hashGridSize.z;
                
                uint currentGridId = FullCollisionJob.PackWrappedInt(new int3(currentGridX, currentGridY, currentGridZ));
                
                int wrappedX = (startX + x) % hashGridSize.x;
                int wrappedY = (startY + y) % hashGridSize.y;
                int wrappedZ = (startZ + z) % hashGridSize.z;
                
                int cellAABBindex = 
                    wrappedX +
                    wrappedY * hashGridSize.x +
                    wrappedZ * hashGridSize.x * hashGridSize.y;
                
                cellAABBentries
                    [--hashGridCells[chunkIndex * totalHashGridSize + cellAABBindex] + chunkTotalGridSize[chunkIndex]] 
                    = new uint2(currentGridId, (uint)i);
            }
        }
    }
}

[BurstCompile]
public struct FindActiveGridCells : IJob
{
    [ReadOnly] public int3 hashGridSize;
    [ReadOnly] public int numberOfChunks;
    [ReadOnly] public NativeArray<int> hashGridCellSizes;
    [WriteOnly] public NativeList<int> activeGridCells;

    public void Execute()
    {
        int totalHashGridSize = hashGridSize.x * hashGridSize.y * hashGridSize.z;
        
        for (int i = 0; i < totalHashGridSize; i++)
        for (int chunk = 0; chunk < numberOfChunks; chunk++)
        {
            if (hashGridCellSizes[chunk * totalHashGridSize + i] != 0)
            {
                activeGridCells.AddNoResize(i);
                break;
            }
        }
    }
}

[BurstCompile]
public struct JustCount : IJob
{
    [ReadOnly] public int3 hashGridSize;
    [ReadOnly] public int numberOfChunks;
    public NativeArray<int> hashGridCells;
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
                hashGridCells[chunk * totalHashGridSize + i] = previousCount;
                previousCount += currentCellCount;
            }

            if (initialCount != previousCount) activeGridCells.AddNoResize(i);
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
    private static int3 hashGridSize = new int3(20);
    private static int numberOfChunks = 16;

    private static NativeList<int> cellAABBmin;
    private static NativeList<uint> cellAABBspan;
    private static NativeList<uint> cellAABBgrid;
    private static NativeArray<int> hashGridCells;
    private static NativeArray<int> hashGridCellSizes;
    private static NativeArray<int> chunkTotalGridSize;
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
        cellAABBgrid.ResizeUninitialized(numberOfCells);
        activeGridCells.Clear();

        CreateCellAABB createCellAABB = new CreateCellAABB
        {
            invCellSize = 1f / hashCellSize,
            hashGridSize = hashGridSize,
            particles = particles,
            latticeCells = latticeCells,
            cellAABBmin = cellAABBmin.AsDeferredJobArray(),
            cellAABBspan = cellAABBspan.AsDeferredJobArray(),
            cellAABBgrid = cellAABBgrid.AsDeferredJobArray()
        };
        handle = createCellAABB.Schedule(numberOfCells, 8, handle);
        CountAABBGridPos countAABBGridPos = new CountAABBGridPos
        {
            hashGridSize = hashGridSize,
            chunkSize = (numberOfCells + numberOfChunks - 1) / numberOfChunks,
            cellAABBmin = cellAABBmin,
            cellAABBspan = cellAABBspan,
            hashGridCells = hashGridCells,
            hashGridCellSizes = hashGridCellSizes,
            chunkTotalGridSize = chunkTotalGridSize
        };
        handle = countAABBGridPos.Schedule(numberOfChunks, 1, handle);
        // JustCount justCount = new JustCount
        // {
        //     hashGridSize = hashGridSize,
        //     numberOfChunks = numberOfChunks,
        //     hashGridCells = hashGridCellSizes,
        //     activeGridCells = activeGridCells
        // };
        // handle = justCount.Schedule(handle);
        handle.Complete();


        int totalAABBCount = 0;
        for (int i = 0; i < numberOfChunks; i++)
        {
            int currentChunkCount = chunkTotalGridSize[i];
            chunkTotalGridSize[i] = totalAABBCount;
            totalAABBCount += currentChunkCount;
        }


        cellAABBentries.ResizeUninitialized(totalAABBCount);


        ScatterAABBCells scatterAABBCells = new ScatterAABBCells
        {
            hashGridSize = hashGridSize,
            chunkSize = (numberOfCells + numberOfChunks - 1) / numberOfChunks,
            cellAABBmin = cellAABBmin,
            cellAABBspan = cellAABBspan,
            cellAABBgrid = cellAABBgrid,
            chunkTotalGridSize = chunkTotalGridSize,
            hashGridCells = hashGridCells,
            cellAABBentries = cellAABBentries.AsDeferredJobArray()
        };
        handle = scatterAABBCells.Schedule(numberOfChunks, 1, handle);
        FindActiveGridCells findActiveGridCells = new FindActiveGridCells
        {
            hashGridSize = hashGridSize,
            numberOfChunks = numberOfChunks,
            hashGridCellSizes = hashGridCellSizes,
            activeGridCells = activeGridCells
        };
        handle = findActiveGridCells.Schedule(handle);
        handle.Complete();
        
        
        int max = 0;
        float averageNon0 = 0;
        int numberGridCells = 0;
        int larger100 = 0;
        int larger50 = 0;
        int smaller10 = 0;
        for (int i = 0; i < hashGridSize.x * hashGridSize.y * hashGridSize.z * numberOfChunks; i++)
        {
            int currentGridSize = hashGridCellSizes[i];
            if (currentGridSize != 0) { averageNon0 += currentGridSize; numberGridCells++; }
            if (currentGridSize > max) max = currentGridSize;
            if (currentGridSize > 100) larger100++;
            if (currentGridSize > 50) larger50++;
            if (currentGridSize < 10) smaller10++;
            hashGridCells[i] = hashGridCellSizes[i] = 0;
        }
        averageNon0 /= numberGridCells;
        Debug.Log(
            "Active cells: " + numberGridCells +
            "\nMax number in a cell: " + max +
            "\nAverage non 0: " + averageNon0 +
            "\nLarger than 100: " + larger100 +
            "\nLarger than 50: " + larger50 +
            "\nSmaller than 10: " + smaller10
        );
    }
#endregion

    public static void Initialise()
    {
        if (isInitialised) return;
        isInitialised = true;

        cellAABBmin = new (Allocator.Persistent);
        cellAABBspan = new (Allocator.Persistent);
        cellAABBgrid = new (Allocator.Persistent);
        hashGridCells = new NativeArray<int>(
            numberOfChunks * hashGridSize.x * hashGridSize.y * hashGridSize.z,
        Allocator.Persistent);
        hashGridCellSizes = new NativeArray<int>(
            numberOfChunks * hashGridSize.x * hashGridSize.y * hashGridSize.z,
        Allocator.Persistent);
        chunkTotalGridSize = new NativeArray<int>(numberOfChunks, Allocator.Persistent);
        activeGridCells = new (hashGridSize.x * hashGridSize.y * hashGridSize.z, Allocator.Persistent);
        cellAABBentries = new (Allocator.Persistent);
    }

    public static void Disable()
    {
        if (!isInitialised) return;
        isInitialised = false;

        cellAABBmin.Dispose();
        cellAABBspan.Dispose();
        cellAABBgrid.Dispose();
        hashGridCells.Dispose();
        hashGridCellSizes.Dispose();
        chunkTotalGridSize.Dispose();
        activeGridCells.Dispose();
        cellAABBentries.Dispose();
    }

}
}