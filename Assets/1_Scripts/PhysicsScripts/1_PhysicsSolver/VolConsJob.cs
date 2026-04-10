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

        float bigTetVol = cellVolume * (1f/3f);
        float smallTetsVol = cellVolume * (1f/6f);

        for (int i = 0; i < 5; i++)
        {
            int4 currentTetIdx = GetCellTet(latticeCells[index], i);
            float initialTetVolume = i == 0 ? bigTetVol : smallTetsVol;

            SolveVolumeConstraint(currentTetIdx, initialTetVolume, cellCompliance);
        }
    }

    private void SolveVolumeConstraint(int4 tetIdx, float initialVolume, float compliance)
    {
        
    }

    static int4 GetCellTet(LatticeCell c, int tetIndex)
    {
        switch (tetIndex)
        {
            case 0: return new int4(c.v0, c.v3, c.v6, c.v5);
            case 1: return new int4(c.v0, c.v5, c.v6, c.v4);
            case 2: return new int4(c.v0, c.v6, c.v3, c.v2);
            case 3: return new int4(c.v5, c.v3, c.v6, c.v7);
            default:return new int4(c.v0, c.v3, c.v5, c.v1);
        }
    }
}
