using SolverStructs;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

[BurstCompile]
public struct VisVertsUpdateJob : IJobParallelFor
{
    [ReadOnly] public NativeArray<Particle>      particles;
    [ReadOnly] public NativeArray<LatticeCell>   latticeCells;
    [ReadOnly] public NativeArray<DynamicVertex> visualVerticesData;
    [ReadOnly] public NativeArray<int>           visualVerticesOffset;
    [NativeDisableParallelForRestriction] public NativeArray<float3> visualVertices;
    [ReadOnly] public NativeArray<int3>          visualWrapers;

    private static readonly int4[] s_CellTets =
    {
        new int4(0, 3, 6, 5),
        new int4(0, 5, 6, 4),
        new int4(0, 6, 3, 2),
        new int4(5, 3, 6, 7),
        new int4(0, 3, 5, 1)
    };


    public void Execute(int index)
    {
        LatticeCell currentCell = latticeCells[visualWrapers[index].x];

        for (int i = visualWrapers[index].y; i < visualWrapers[index].z; i++)
        {
            int4 tet = s_CellTets[visualVerticesData[i].tet];
            float4 bary = visualVerticesData[i].bary;
            tet = new int4(
                GetCellVertIndex(currentCell, tet.x),
                GetCellVertIndex(currentCell, tet.y),
                GetCellVertIndex(currentCell, tet.z),
                GetCellVertIndex(currentCell, tet.w)
            );

            visualVertices[visualVerticesOffset[i]] = 
                bary.x * particles[tet.x].currentPosition + 
                bary.y * particles[tet.y].currentPosition +
                bary.z * particles[tet.z].currentPosition +
                bary.w * particles[tet.w].currentPosition;
        }
    }

    private static int GetCellVertIndex(LatticeCell c, int localIndex)
    {
        switch (localIndex)
        {
            case 0: return c.v0;
            case 1: return c.v1;
            case 2: return c.v2;
            case 3: return c.v3;
            case 4: return c.v4;
            case 5: return c.v5;
            case 6: return c.v6;
            case 7: return c.v7;
            default: return -1;
        }
    }
}
