using SolverStructs;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

[BurstCompile]
public struct VolConsJob : IJobParallelFor
{
    [ReadOnly] public float dt;
    [ReadOnly] public float damping;
    [ReadOnly] public int channelStart;
    [ReadOnly] public NativeArray<LatticeCell> latticeCells;
    [ReadOnly] public NativeArray<int>         latticeCellBodyRef;
    [ReadOnly] public NativeArray<float3>      bodyLatticeDimensions;
    [ReadOnly] public NativeArray<float>       bodyVolConsCompliance;
    [NativeDisableParallelForRestriction] public NativeArray<Particle> particles;

    public void Execute(int index)
    {
        // Add the current channel offset
        index += channelStart;
        float3 latticeCellDimensions = bodyLatticeDimensions[latticeCellBodyRef[index]];
        float cellCompliance = bodyVolConsCompliance[latticeCellBodyRef[index]];
        float cellVolume = latticeCellDimensions.x * latticeCellDimensions.y * latticeCellDimensions.z;

        for (int i = 0; i < 5; i++)
        {
            float cellPercentage = i == 0 ? 1f / 3f : 1f / 6f;
            
        }
    }

    static int4 GetCellTet(int tetIndex)
    {
        switch (tetIndex)
        {
            case 0: return new int4(0, 3, 6, 5);
            case 1: return new int4(0, 5, 6, 4);
            case 2: return new int4(0, 6, 3, 2);
            case 3: return new int4(5, 3, 6, 7);
            default:return new int4(0, 3, 5, 1);
        }
    }
}
