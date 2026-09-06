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
public struct PrefixSumGridCells : IJobParallelFor
{
    [ReadOnly] public int numberOfGridCells;
    [ReadOnly] public int numberOfChunks;
    [ReadOnly] public int prefixChunkSize;
    [NativeDisableParallelForRestriction]
    public NativeArray<int> hashGridCells;
    [NativeDisableParallelForRestriction]
    [WriteOnly] public NativeArray<int> hashGridCellOffsets;
    [WriteOnly] public NativeArray<int> prefixChunkSizes;


    public void Execute(int prefixChunkIndex)
    {
        int writeStart = prefixChunkSize * prefixChunkIndex;
        int writeEnd = min(prefixChunkSize * (prefixChunkIndex+1), numberOfGridCells);
        int previousCount = 0;
        
        for (int i = writeStart; i < writeEnd; i++)
        for (int chunk = 0; chunk < numberOfChunks; chunk++)
        {
            int cellIndex = chunk * numberOfGridCells + i;
            int cellCount = hashGridCells[cellIndex];

            hashGridCells[cellIndex] = 0;
            hashGridCellOffsets[cellIndex] = previousCount;
            previousCount += cellCount;
        }

        prefixChunkSizes[prefixChunkIndex] = previousCount;
    }
}

[BurstCompile]
public struct ScatterAABBCells : IJobParallelFor
{
    [ReadOnly] public int3 hashGridSize;
    [ReadOnly] public int chunkSize;
    [ReadOnly] public NativeList<int3> cellAABBmin;
    [ReadOnly] public NativeList<uint> cellAABBspan;
    [ReadOnly] public NativeArray<int> prefixChunkSizes;
    [ReadOnly] public int prefixChunkSize;
    [NativeDisableParallelForRestriction]
    public NativeArray<int> hashGridCellOffsets;
    [NativeDisableParallelForRestriction]
    [WriteOnly] public NativeArray<int4> cellAABBentries;


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
                
                cellAABBentries[hashGridCellOffsets[chunkIndex * totalHashGridSize + cellAABBindex]++
                    + prefixChunkSizes[cellAABBindex / prefixChunkSize]]
                    = new int4(currentCellPos, i);
            }
        }
    }
}

[BurstCompile]
public struct SortGridCellEntries : IJobParallelFor
{
    [ReadOnly] public NativeArray<int> prefixChunkSizes;
    [ReadOnly] public int prefixChunkSize;
    [ReadOnly] public NativeList<int3> cellAABBmin;
    [ReadOnly] public NativeSlice<int> hashGridCellOffsets;
    [NativeDisableParallelForRestriction]
    public NativeArray<int4> cellAABBentries;
    [WriteOnly] public NativeArray<int> gridCellPairs;
    

    public void Execute(int gridCellIndex)
    {
        int cellStart = gridCellIndex == 0 ? 0 : (hashGridCellOffsets[gridCellIndex - 1] + 
            prefixChunkSizes[(gridCellIndex - 1) / prefixChunkSize]);
        int cellEnd = hashGridCellOffsets[gridCellIndex] + prefixChunkSizes[gridCellIndex / prefixChunkSize];
        int count = cellEnd - cellStart;

        if (count <= 1) { gridCellPairs[gridCellIndex] = 0; return; }

        InsertionSort(cellAABBentries, cellStart, count);


        int totalPairCount = 0;
        for (int start = cellStart; start < cellEnd;)
        {
            int end = start + 1;
            int3 key = cellAABBentries[start].xyz;
            while (end < cellEnd && all(cellAABBentries[end].xyz == key))
                end++;

            if (end - start <= 1)
            {
                start = end;
                continue;
            }

            for (int i = start; i < end - 1; i++)
            {
                int bodyAIndex = cellAABBentries[i].w;
                int3 minACell = cellAABBmin[bodyAIndex];

                for (int j = i + 1; j < end; j++)
                {
                    int bodyBIndex = cellAABBentries[j].w;
                    int3 minBCell = cellAABBmin[bodyBIndex];

                    int3 commonMinCell = max(minACell, minBCell);

                    if (all(commonMinCell == key))
                    {
                        totalPairCount++;
                    }
                }
            }

            start = end;
        }
        gridCellPairs[gridCellIndex] = totalPairCount;
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
                
                if (Compare(candidate.xyz, key.xyz) > 0)
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
    static int Compare(int3 va, int3 vb)
    {
        // Compare x first
        if (va.x != vb.x)
            return va.x - vb.x;

        // Then y
        if (va.y != vb.y)
            return va.y - vb.y;

        // Then z
        if (va.z != vb.z)
            return va.z - vb.z;

        // All equal (w/bodyID doesn't matter for sorting)
        return 0;
    }
}

[BurstCompile]
public struct prefixSumCellPairs : IJob
{
    [ReadOnly] public int totalHashGridSize;
    public NativeArray<int> gridCellPairs;
    [WriteOnly] public NativeReference<int> numberOfPairs;


    public void Execute()
    {
        int previousCount = 0;
        for (int i = 0; i < totalHashGridSize; i++)
        {
            int cellCount = gridCellPairs[i];
            if (cellCount == 0) continue;
            gridCellPairs[i] = previousCount;
            previousCount += cellCount;
        }
        numberOfPairs.Value = previousCount;
    }
}

[BurstCompile]
public struct AddPairs : IJobParallelFor
{
    [ReadOnly] public NativeArray<int> prefixChunkSizes;
    [ReadOnly] public int prefixChunkSize;
    [ReadOnly] public NativeList<int3> cellAABBmin;
    [ReadOnly] public NativeSlice<int> hashGridCellOffsets;
    [ReadOnly] public NativeArray<int4> cellAABBentries;
    [ReadOnly] public NativeArray<int> gridCellPairs;
    [NativeDisableParallelForRestriction]
    [WriteOnly] public NativeArray<int2> finalPairs;
    

    public void Execute(int gridCellIndex)
    {
        int cellStart = gridCellIndex == 0 ? 0 : (hashGridCellOffsets[gridCellIndex - 1] + 
            prefixChunkSizes[(gridCellIndex - 1) / prefixChunkSize]);
        int cellEnd = hashGridCellOffsets[gridCellIndex] + prefixChunkSizes[gridCellIndex / prefixChunkSize];
        int count = cellEnd - cellStart;

        if (count <= 1) return;

        int writeStart = gridCellPairs[gridCellIndex];
        for (int start = cellStart; start < cellEnd;)
        {
            int end = start + 1;
            int3 key = cellAABBentries[start].xyz;
            while (end < cellEnd && all(cellAABBentries[end].xyz == key))
                end++;

            if (end - start <= 1)
            {
                start = end;
                continue;
            }

            for (int i = start; i < end - 1; i++)
            {
                int bodyAIndex = cellAABBentries[i].w;
                int3 minACell = cellAABBmin[bodyAIndex];

                for (int j = i + 1; j < end; j++)
                {
                    int bodyBIndex = cellAABBentries[j].w;
                    int3 minBCell = cellAABBmin[bodyBIndex];

                    int3 commonMinCell = max(minACell, minBCell);

                    if (all(commonMinCell == key))
                    {
                        finalPairs[writeStart++] = new int2(bodyAIndex, bodyBIndex);
                    }
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
    private static int numberOfPrefixChunks = 8;

    private static NativeList<int3> cellAABBmin;
    private static NativeList<uint> cellAABBspan;
    private static NativeArray<int> hashGridCells;
    private static NativeArray<int> hashGridCellOffsets;
    private static NativeArray<int> prefixChunkSizes;
    private static NativeList<int4> cellAABBentries;
    private static NativeArray<int> gridCellPairs;
    private static NativeReference<int> numberOfPairs;
    private static NativeList<int2> finalPairs;

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
        Initialise();
        int numberOfCells = latticeCells.Length;
        int numberOfGridCells = hashGridSize.x * hashGridSize.y * hashGridSize.z;
        int prefixChunkSize = (numberOfGridCells + numberOfPrefixChunks - 1) / numberOfPrefixChunks;
        numberOfPairs.Value = 0;


        cellAABBmin.ResizeUninitialized(numberOfCells);
        cellAABBspan.ResizeUninitialized(numberOfCells);


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
            numberOfGridCells = numberOfGridCells,
            numberOfChunks = numberOfChunks,
            prefixChunkSize = prefixChunkSize,
            hashGridCells = hashGridCells,
            hashGridCellOffsets = hashGridCellOffsets,
            prefixChunkSizes = prefixChunkSizes
        };
        handle = prefixSumGridCells.Schedule(numberOfPrefixChunks, 1, handle);
        handle.Complete();
        int AABBentriesCount = 0;
        for (int i = 0; i < numberOfPrefixChunks; i++)
        {
            int chunkSize = prefixChunkSizes[i];
            prefixChunkSizes[i] = AABBentriesCount;
            AABBentriesCount += chunkSize;
        }


        cellAABBentries.ResizeUninitialized(AABBentriesCount);


        ScatterAABBCells scatterAABBCells = new ScatterAABBCells
        {
            hashGridSize = hashGridSize,
            chunkSize = (numberOfCells + numberOfChunks - 1) / numberOfChunks,
            cellAABBmin = cellAABBmin,
            cellAABBspan = cellAABBspan,
            prefixChunkSize = prefixChunkSize,
            prefixChunkSizes = prefixChunkSizes,
            hashGridCellOffsets = hashGridCellOffsets,
            cellAABBentries = cellAABBentries.AsDeferredJobArray()
        };
        handle = scatterAABBCells.Schedule(numberOfChunks, 1, handle);
        SortGridCellEntries sortGridCellEntries = new SortGridCellEntries
        {
            prefixChunkSize = prefixChunkSize,
            prefixChunkSizes = prefixChunkSizes,
            cellAABBmin = cellAABBmin,
            hashGridCellOffsets = new NativeSlice<int>(hashGridCellOffsets, numberOfGridCells * (numberOfChunks-1)),
            cellAABBentries = cellAABBentries.AsDeferredJobArray(),
            gridCellPairs = gridCellPairs
        };
        handle = sortGridCellEntries.Schedule(numberOfGridCells, 1, handle);
        prefixSumCellPairs prefixSumCellPairs = new prefixSumCellPairs
        {
            totalHashGridSize = numberOfGridCells,
            gridCellPairs = gridCellPairs,
            numberOfPairs = numberOfPairs
        };
        handle = prefixSumCellPairs.Schedule(handle);
        handle.Complete();


        finalPairs.ResizeUninitialized(numberOfPairs.Value);


        AddPairs addPairs = new AddPairs
        {
            prefixChunkSize = prefixChunkSize,
            prefixChunkSizes = prefixChunkSizes,
            cellAABBmin = cellAABBmin,
            hashGridCellOffsets = new NativeSlice<int>(hashGridCellOffsets, numberOfGridCells * (numberOfChunks-1)),
            cellAABBentries = cellAABBentries.AsDeferredJobArray(),
            gridCellPairs = gridCellPairs,
            finalPairs = finalPairs.AsDeferredJobArray()
        };
        handle = addPairs.Schedule(numberOfGridCells, 1, handle);
        handle.Complete();

        
        // DebugCollision(particles, latticeCells);
    }
#endregion
#region Helper Functions

    private static void DebugCollision(
        NativeList<Particle>        particles,
        NativeList<LatticeCell>     latticeCells
    )
    {
        Debug.Log(numberOfPairs.Value);
        for (int i = 0; i < numberOfPairs.Value; i++)
        {
            int2 pair = finalPairs[i];

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

        cellAABBmin = new (Allocator.Persistent);
        cellAABBspan = new (Allocator.Persistent);
        hashGridCells = new NativeArray<int>(
            numberOfChunks * hashGridSize.x * hashGridSize.y * hashGridSize.z,
        Allocator.Persistent);
        hashGridCellOffsets = new NativeArray<int>(
            numberOfChunks * hashGridSize.x * hashGridSize.y * hashGridSize.z,
        Allocator.Persistent);
        prefixChunkSizes = new NativeArray<int>(numberOfPrefixChunks, Allocator.Persistent);
        cellAABBentries = new (Allocator.Persistent);
        gridCellPairs = new (hashGridSize.x * hashGridSize.y * hashGridSize.z, Allocator.Persistent);
        numberOfPairs = new (0, Allocator.Persistent);
        finalPairs = new (Allocator.Persistent);
    }

    public static void Disable()
    {
        if (!isInitialised) return;
        isInitialised = false;

        cellAABBmin.Dispose();
        cellAABBspan.Dispose();
        hashGridCells.Dispose();
        hashGridCellOffsets.Dispose();
        prefixChunkSizes.Dispose();
        cellAABBentries.Dispose();
        gridCellPairs.Dispose();
        numberOfPairs.Dispose();
        finalPairs.Dispose();
    }

#endregion
}
}